from transformers import BertModel, AutoModel
from fane.models.base import PretrainModel
from fane.modules.sentence_pool import SentenceAttentionPool
# from prior.modules.sparse_local_attention import LocalCrossAttention
from fane.modules.gather import SentenceGather
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fane.utils.scheduler import CosineAnnealingWarmupRestarts

from fane.modules.sparse_local_attention import SparseLocalCrossAttention

from fane.utils.semantic_similarity import SemanticSimilarityNormalizer


class Mcga(PretrainModel):
    task = 'pretrain'

    def __init__(self, text_encoder, image_encoder, gpus, num_heads = 1, threshold=0.6, stage1_epochs=20, stage2_epochs=30, stage3_epochs=50,
                 max_epochs=100, stage1_warmup_epochs=1, stage2_warmup_epochs=1, stage3_warmup_epochs=5, batch_size=56,
                 optim='adam', scheduler='linear_warmup_cosine_annealing', stage1_learning_rate=1e-5,
                 stage1_learning_rate_start=1e-7, stage1_learning_rate_end=0, stage1_weight_decay=1e-6,
                 temperature=0.01, local_temperature=0.01, intra_temperature=0.01,
                 embed_dim=128, image_rec_drop_out_rate=0.5, spb_k=512, num_queries=16, gahter_pool='avg',
                 lambda_proto=10, exclude_bn_bias=False, train_dataset=None, validation_dataset=None, num_workers=16,
                 temp_decay='fixed', frozen_text_encoder=False, ckpt_path='checkpoints/'):
        super().__init__(text_encoder=text_encoder, image_encoder=image_encoder)

        self.threshold = threshold
        self.stage1_epochs = stage1_epochs

        # define tool for distinguish the possitive and negative instances
        self.tool_bert = AutoModel.from_pretrained("/Bio_ClinicalBERT")
        self.normalizer = SemanticSimilarityNormalizer(alpha=0.05)

        # Get embedding space from language model
        self.text_width = text_encoder.get_width()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # define hyperparam
        self.lambda_global = 1.0     # 10.0
        self.lambda_intra =  0.1         #   3.0
        self.lambda_local_text = 1.0   # 0.3
        self.lambda_regular = 0.5       # 0.05

        self.vision_width = 768

        # Define text global pooling over sentences
        self.global_text_attention = SentenceAttentionPool(16, embed_dim, pos_embed=False)    # Max sentence num: 32


        # Define project
        self.local_text_width = text_encoder.get_width()
        self.global_text_projection =  nn.Linear(self.text_width, self.embed_dim)
        self.local_text_projection =  nn.Linear(self.text_width, self.embed_dim)

        self.global_image_projection = nn.Linear(self.vision_width, self.embed_dim)
        self.local_image_projection = nn.Linear(self.vision_width, self.embed_dim)

        # Define local-interaction （  局部对齐  ）
        self.local_cross_attention = SparseLocalCrossAttention(embed_dim)


        self.logit_scale = 0.1
        self.local_logit_scale = 0.07
        self.intra_logit_scale = 0.07
        self.softmax_temperature = 0.07
        self.negative_scale = 0.07

        # Define hyper-params for optimization
        self.exclude_bn_bias = exclude_bn_bias
        self.batch_size = batch_size
        self.optim = optim
        self.scheduler = scheduler
        self.stage1_warmup_epochs = stage1_warmup_epochs
        self.stage1_learning_rate = stage1_learning_rate
        self.stage1_learning_rate_start = stage1_learning_rate_start
        self.stage1_learning_rate_end = stage1_learning_rate_end
        self.stage1_weight_decay = stage1_weight_decay

        self.max_epochs = 100

        # Define loss hyper-params
        self.temp_decay = temp_decay

        # Define NLP gather
        self.item_gather = SentenceGather(gahter_pool, embed_dim)

        # cache for loss
        self.last_local_batch_size = None
        self.global_alignment_labels = None

        # Define dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.num_workers = num_workers
        # self.train_iters_per_epoch = len(self.train_dataset) // (len(gpus) * batch_size)      # comment in finetuning

        # for dist-training, log...
        self.gpus = gpus
        self.ckpt_path = ckpt_path

        # freeze/finetuning params
        if frozen_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        for param in self.tool_bert.parameters():
            param.requires_grad = False

        self.inititalize_parameters()


    def inititalize_parameters(self):
        # Initialize parameters
        nn.init.normal_(self.global_text_projection.weight, std=self.text_width ** -0.5)
        nn.init.normal_(self.local_text_projection.weight, std=self.local_text_width ** -0.5)
        nn.init.normal_(self.global_image_projection.weight, std=self.vision_width ** -0.5)
        nn.init.normal_(self.local_image_projection.weight, std=self.vision_width ** -0.5)


    def train_dataloader(self):
        print(f"batch_size: {self.batch_size}")
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) :
        if self.validation_dataset is not None:
            return DataLoader(self.validation_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def encode_image(self, image):
        # image shape:    [ batchsize, (width / patch) * (height / patch), channel * patch * patch ]    ----->   16, 196, 768
        # print(f"image shape: {image.shape}")

        global_image_features, local_image_features = self.image_encoder(image)     #   global:  bs * 768      local: bs * 196 * 768
        return self.local_image_projection(local_image_features), self.global_image_projection(global_image_features)

    def encode_text(self, text):
        x = self.text_encoder(text)
        local_text_features = x['last_hidden_state']
        # global_text_features = x['pooler_output'] # Although we get the global features, we do not use it
        return self.local_text_projection(local_text_features)

    def getTextSoftTarget(self, raw_txt, threshold):
        margin = 1.0
        self.tool_bert.eval()
        with torch.no_grad():
            f_txt = self.tool_bert(**raw_txt)
            f_txt = f_txt['pooler_output']

            # # l2 normalization
            # f_txt = F.normalize(f_txt, p=2, dim=1)
            # # cosin score
            # scores = torch.matmul(f_txt,f_txt.transpose(-2,-1))
            scores = self.normalizer.normalize_similarity(f_txt)
            scores.fill_diagonal_(threshold + margin)  # 保证对角线是正样本对

            return (scores, threshold)


    def softXEnt(self, target, logits):

        logprobs = torch.nn.functional.log_softmax(logits, dim = -1)
        loss = -torch.sum(target * logprobs, dim=-1).mean()
        return loss

    def get_imc_loss(self, embedding_matrix: torch.Tensor):
        """
        embedding_matrix: extra similarity matrix served as denominator in clip loss
        """

        logtis_matrix = embedding_matrix
        labels = torch.zeros(logtis_matrix.shape[0], device=logtis_matrix.device, dtype=torch.long)
        imc_loss = F.cross_entropy(logtis_matrix, labels)
        return imc_loss

    def global_alignment_and_intra_modality_loss(self, f_img, f_txt, labels):

        scores, threshold = labels

        f_img = F.normalize(f_img, dim=-1)
        f_txt = F.normalize(f_txt, dim=-1)

        logits_i2t = f_img @ f_txt.T / self.logit_scale
        logits_t2i = f_txt @ f_img.T / self.logit_scale
        logits_text = f_txt @ f_txt.T / self.intra_logit_scale
        logits_image = f_img @ f_img.T / self.intra_logit_scale

        batch_size = logits_i2t.size(0)
        global_loss = 0
        intra_loss = 0
        pos_number = 0
        neg_number = 0

        for layer, score in enumerate(scores):
            pos_mask = score > threshold
            neg_mask = ~pos_mask
            pos_indices = pos_mask.nonzero(as_tuple=True)[0]
            neg_indices = neg_mask.nonzero(as_tuple=True)[0]
            pos_number += len(pos_indices)
            neg_number += len(neg_indices)


            neg_i2t = logits_i2t[layer][neg_indices]
            neg_t2i = logits_t2i[layer][neg_indices]
            neg_text = logits_text[layer][neg_indices]
            neg_image = logits_image[layer][neg_indices]

            # scaler = batch_size - len(pos_indices)
            # print(f"scaler: {scaler}")

            neg_i2t_weighted = torch.softmax(neg_i2t / self.negative_scale, dim=-1) * neg_i2t
            neg_t2i_weighted = torch.softmax(neg_t2i / self.negative_scale, dim=-1) * neg_t2i
            neg_text_weighted = torch.softmax(neg_text / self.negative_scale, dim=-1) * neg_text
            neg_image_weighted = torch.softmax(neg_image / self.negative_scale, dim=-1) * neg_image


            pos_i2t = logits_i2t[layer][pos_indices].unsqueeze(1)  # [P, 1]
            pos_t2i = logits_t2i[layer][pos_indices].unsqueeze(1)


            # 扩展负样本维度： [P, N]，
            neg_i2t_expand = neg_i2t_weighted.unsqueeze(0).expand(len(pos_indices), -1)   # [P, N]
            neg_t2i_expand = neg_t2i_weighted.unsqueeze(0).expand(len(pos_indices), -1)


            logits_i2t_all = torch.cat([pos_i2t, neg_i2t_expand], dim=1)
            logits_t2i_all = torch.cat([pos_t2i, neg_t2i_expand], dim=1)

            targets = torch.zeros_like(logits_i2t_all)
            targets[:, 0] = 1.0


            loss_i2t = self.softXEnt(targets, logits_i2t_all)
            loss_t2i = self.softXEnt(targets, logits_t2i_all)
            loss_text = self.get_imc_loss(neg_text_weighted.unsqueeze(0))
            loss_image = self.get_imc_loss(neg_image_weighted.unsqueeze(0))

            global_loss += (loss_i2t + loss_t2i) / 2
            intra_loss += (loss_text + loss_image) / 2

        global_alignment_loss = global_loss / batch_size
        intra_modality_loss = intra_loss / batch_size

        return {
            'global_alignment_loss': global_alignment_loss,
            'intra_modality_loss': intra_modality_loss,
            'num_pos_pairs rate': pos_number / batch_size,
            'num_neg_pairs rate': neg_number / batch_size
        }


    def local_loss_fn(self, embed_A, embed_B, norm=True):

        # logit_scale = self.local_logit_scale.exp()
        if norm:
            embed_A = F.normalize(embed_A, dim=-1, p=2)
            embed_B = F.normalize(embed_B, dim=-1, p=2)
        self.lc_labels = torch.arange(embed_B.size(0), device=embed_B.device).long()
        logits_per_image = embed_B @ embed_A.t() / self.local_logit_scale
        logits_per_text = embed_A @ embed_B.t() /  self.local_logit_scale
        image_loss = F.cross_entropy(logits_per_image, self.lc_labels)
        text_loss = F.cross_entropy(logits_per_text, self.lc_labels)
        loss = (image_loss + text_loss) / 2
        return loss





    def get_global_text_representation(self, local_text_embed_stacks):
        batch_stacks = []
        for local_text_embed in local_text_embed_stacks:
            batch_stacks.append(self.global_text_attention(local_text_embed.unsqueeze(dim=0)))
        return torch.cat(batch_stacks, dim=0)

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def compute_retrieval_acc(self, global_image, global_text):
        bz = global_image.size(0)
        labels = torch.arange(bz).type_as(global_text).long()
        global_image_embed = F.normalize(global_image, dim=-1)
        global_text_embed = F.normalize(global_text, dim=-1)

        scores = global_image_embed.mm(global_text_embed.t())
        scores /= self.softmax_temperature
        scores1 = scores.transpose(0, 1)

        i2t_acc1, i2t_acc5 = self.precision_at_k(
            scores, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            scores1, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        return  acc1, acc5


    def forward(self, batch):
        pass


    def training_step(self, batch, batch_idx):
        image = batch['image']
        text = batch['text']

        '''
        =================================================================
        Obtain soft label
        =================================================================
        '''
        labels = self.getTextSoftTarget(text, self.threshold)
        '''
        =================================================================
        Encode image and text and get the local and global representation
        =================================================================
        '''
        # Embed image
        local_image_embed, global_image_embed = self.encode_image(image)       #   global:  bs * 128          local: bs * 196 * 128
        # print(f"local_image_embed shape: {local_image_embed.shape}")
        # print(f"global_image_embed shape: {global_image_embed.shape}")

        # Embed text
        local_text_embed = self.encode_text(text)     # 98 * 192 * 128      bs * length * 128

        # gather local text embedding on sentence level
        local_text_embed_stacks = self.item_gather(local_text_embed, batch)

        # get global text embedding
        global_text_embed = self.get_global_text_representation(local_text_embed_stacks)   # bs * 128   ( 98 * 128 )

        '''
        =================================================================
        Calculate the alignment loss
        =================================================================
        '''
        # local alignment loss
        local_loss_dict = self.local_alignment_loss(local_image_embed, local_text_embed_stacks)  # shared local image embedding

        # global contrastive loss  and  intra modality loss
        gi = self.global_alignment_and_intra_modality_loss(global_image_embed, global_text_embed, labels)

        # Compute retrieval accuracy
        acc1, acc5 = self.compute_retrieval_acc(global_image_embed, global_text_embed)

        loss_dict = {}


        loss_dict['loss']  = gi['global_alignment_loss'] * self.lambda_global + gi['intra_modality_loss'] * self.lambda_intra \
                                  +  self.lambda_local_text * local_loss_dict['local_align_loss'] \
                                  + self.lambda_regular * local_loss_dict['sparse_regular_loss']



        for k, v in gi.items():
            loss_dict[k] = v
            # if 'loss' in k:
            #     loss_dict['train_loss'] += v
        for k, v in local_loss_dict.items():
            loss_dict[k] = v
            # if 'loss' in k:
            #     loss_dict['train_loss'] += v

        loss_dict['train_acc1'] = acc1
        loss_dict['train_acc5'] = acc5
        # loss_dict['loss'] = loss_dict['train_loss']

        self.log_dict(loss_dict, on_step=True, on_epoch=False, prog_bar=True,sync_dist=True)
        torch.cuda.empty_cache()  # 清理显存
        return loss_dict



    def on_train_epoch_end(self) -> None:
        if self.current_epoch == self.max_epochs - 1:
            if self.global_rank == 0:
                self.trainer.save_checkpoint(f"{self.ckpt_path}/HAMeR.ckpt")


    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=["bias", "bn"]):
        params = []
        excluded_params = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]

    def exclude_from_text_encoder(self, named_params, weight_decay):
        # exclude discriminator param
        params = []
        excluded_params = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif 'text_encoder' in name:
                excluded_params.append(param)
            else:
                params.append(param)
        return params, excluded_params



    def configure_optimizers(self):
        total_steps = self.train_iters_per_epoch * self.max_epochs
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=4e-4,             # 2e-5
            betas=(0.9, 0.999),
            weight_decay=1e-6                   #  MGCA:  0.05
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=total_steps,
            cycle_mult=1.0,
            max_lr=4e-4,          # 2e-5
            min_lr=1e-8,
            warmup_steps=int(total_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}






