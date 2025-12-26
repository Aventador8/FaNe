import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from prior.encoders.image import cnn_backbones
from codes.fane.encoders.image import cnn_backbones
from transformers import AutoTokenizer, BertConfig, BertTokenizer, logging


logging.set_verbosity_error()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]



class ResEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "resnet_50",
                 pretrained: bool = True,
                 weights_path=None
                 ):
        super(ResEncoder, self).__init__()
        self.model_name = model_name


        model_function = getattr(
            cnn_backbones, model_name)
        self.model, self.feature_dim, self.interm_feature_dim = model_function(
            pretrained=pretrained, weights_path = weights_path
        )

        # # Average pooling
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Attention pooling
        self.pool = AttentionPool2d(spacial_dim=10, embed_dim=2048, num_heads=8)


    def resnet_forward(self, x):
        x = nn.Upsample(size=(299, 299), mode="bilinear",
                        align_corners=True)(x)
        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # (batch_size, 256, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 512, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 1024, 19, 19)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 2048, 10, 10)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        local_features = rearrange(local_features, "b c w h -> b (w h) c")

        return x, local_features.contiguous()


    def forward(self, x):
        return self.resnet_forward(x)





