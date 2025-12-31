import torch.nn as nn
import torch
import math
import torch.nn.functional as F



class SparseLocalCrossAttention(nn.Module):
    def __init__(self, embed_dim, text_width = 15, image_width = 49, drop_rate=0):     #  text_width: max lenth of sentence  ,  image_width: max patch of image
        super(SparseLocalCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop_rate)


        self.text_width = text_width
        self.image_width = image_width

        # learnable sparse attention mask
        self.mask_linear = nn.Sequential( nn.Linear(2 * embed_dim, embed_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(embed_dim, 1))

        self.out_project = nn.Sequential(nn.Linear(embed_dim,embed_dim),
                                         nn.LayerNorm(embed_dim))

        # self.learn_sparse_image_att = torch.nn.Embedding(text_width * image_width, 1)
        # self.learn_sparse_text_att = torch.nn.Embedding(text_width * image_width, 1)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(
            self,
            image_embedding,  # image_embedding
            text_embedding,  # text_embedding
            attention_mask1=None,
            sparse = True,
            attention_mask2=None
    ):

        seq_len_image, embed_dim = image_embedding.shape
        seq_len_text, _ = text_embedding.shape
        # print(f"seq_len_image: {seq_len_image}, image_shape: {input_tensor1.shape}, seq_len_text: {seq_len_text}, text_shape: {input_tensor2.shape}")

        text_query = self.query(text_embedding)
        image_key = self.key(image_embedding)
        image_value = self.value(image_embedding)

        if sparse:
            #  image sparse attention mask  (text as query)
            text_expanded = text_query.unsqueeze(1).expand(seq_len_text, seq_len_image, embed_dim)  # (T, I, D)
            image_expanded = image_key.unsqueeze(0).expand(seq_len_text, seq_len_image, embed_dim)  # (T, I, D)
            qk_concat = torch.cat([text_expanded, image_expanded], dim=-1)  # (T, I, 2*D)
            mask_logits = self.mask_linear(qk_concat).squeeze(-1)  # (T, I)
            sparse_mask = self.sigmoid(mask_logits)  # (T, I)


        attention_scores = text_query @ image_key.T  # [T, D] @ [D, I] = [T, I]
        # print(f"attention_scores1: {attention_scores1.shape}, learn_image_att: {learn_image_att.shape}")
        if sparse:
            attention_scores = attention_scores * sparse_mask      # apply sparse mask to attention scores to filter irrelevant areas
        attention_scores = attention_scores / math.sqrt(self.embed_dim)


        # Sigmoid is better in this case
        # TODO: pre-normalize vs. post-normalize
        attention_probs = F.sigmoid(attention_scores)

        # attention_probs1 = self.dropout1(attention_probs1)
        context_layer = attention_probs @ image_value  # [T, I] @ [I, D] = [T, D]
        context_layer = self.out_project(context_layer)

        if sparse:
            # sparse loss
            mask_sparsity = self.get_loss_sparsity(sparse_mask)

        if sparse:
            return context_layer, attention_probs, mask_sparsity
        else:
            return context_layer, attention_probs

    def get_loss_sparsity(self, mask):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(mask)))
        return sparsity_loss


