import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 7, pretrained: bool = True):
    """
    ResNet18 with ImageNet weights, final FC replaced for num_classes.
    Returns (model, param_groups) with two-speed LRs: backbone 1e-4, head 1e-3.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model   = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc    = nn.Linear(in_features, num_classes)

    backbone_params = [p for name, p in model.named_parameters()
                       if not name.startswith("fc")]
    head_params     = list(model.fc.parameters())

    param_groups = [
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params,     "lr": 1e-3},
    ]

    return model, param_groups


def build_efficientnet_b0(num_classes: int = 7, pretrained: bool = True):
    """
    EfficientNet-B0 with ImageNet weights, classifier head replaced for num_classes.
    Returns (model, param_groups) with two-speed LRs: backbone 1e-4, head 1e-3.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model   = models.efficientnet_b0(weights=weights)

    in_features          = model.classifier[1].in_features
    model.classifier[1]  = nn.Linear(in_features, num_classes)

    backbone_params = [p for name, p in model.named_parameters()
                       if not name.startswith("classifier")]
    head_params     = list(model.classifier.parameters())

    param_groups = [
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params,     "lr": 1e-3},
    ]

    return model, param_groups
