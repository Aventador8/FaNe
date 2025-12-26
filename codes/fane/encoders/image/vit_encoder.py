import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertConfig, BertTokenizer, logging
# from prior.encoders.image.vits import create_vit
from codes.fane.encoders.image.vits import create_vit

logging.set_verbosity_error()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)

        return x.permute(0, 2, 1)


class VitEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "vit_base",
                 output_dim: int = 128,
                 hidden_dim: int = 2048,
                 image_size = 224,
                 weights_path = None
                 ):
        super(VitEncoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim

        vit_grad_ckpt = False
        vit_ckpt_layer = 0


        vit_name = model_name[4:]
        self.model, vision_width = create_vit(
            vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

        self.feature_dim = vision_width  # 768

        checkpoint = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(checkpoint["model"], strict=False)

        self.global_embed = GlobalEmbedding(
            vision_width, hidden_dim, output_dim
        )


    def vit_forward(self, x):
        return self.model(x, register_blk=11)

    def forward(self, x, get_local=False):
        img_feat = self.vit_forward(x)

        # print(f"image shape: {img_feat.shape}")
        return img_feat[:, 0].contiguous(), img_feat[:, 1:].contiguous()    #       bs * (patch * patch) * 768