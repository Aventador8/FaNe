import torch.nn as nn
import torch
from torchvision import models as models_2d
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


################################################################################
# ResNet Family
################################################################################


def resnet_18(pretrained=True):
    model = models_2d.resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_34(pretrained=True):
    model = models_2d.resnet34(weights=ResNet34_Weights.DEFAULT)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_50(pretrained=True, weights_path=None):
    if pretrained and weights_path:
        model = models_2d.resnet50(weights=None)     # 不使用在线权重
        # 加载本地权重文件
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
    else:
        # 如果 pretrained=True 但无 weights_path，使用默认权重（可能触发在线下载）
        model = models_2d.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)


    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024
