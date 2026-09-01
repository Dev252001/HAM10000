"""
src/models/transfer_models.py
──────────────────────────────
ResNet18 and EfficientNet-B0 with ImageNet pretrained weights, classifier
heads replaced for HAM10000's 7-class output.

Freeze vs. fine-tune decision
──────────────────────────────
Two strategies exist:

  A) Freeze backbone, train head only
     - Only the new classifier layer is updated.
     - Fast (few gradients), very low overfitting risk.
     - Works well when the source domain (ImageNet) is very similar to the
       target domain. Dermoscopic images are NOT natural photos — textures,
       colours, and structures are quite different from ImageNet content.
     - Risk: the frozen features may not be expressive enough for fine-grained
       lesion texture discrimination.

  B) Fine-tune end-to-end (whole network)
     - All layers are updated, backbone features adapt to dermoscopy.
     - More expressive — the backbone can learn lesion-specific patterns.
     - Needs a lower learning rate for the backbone (pretrained weights are
       already good; aggressive updates would destroy them — "catastrophic
       forgetting").
     - More parameters = slightly higher overfitting risk, but we have class
       weights + dropout + augmentation already in the pipeline.

  Choice: FINE-TUNE END-TO-END with a two-speed learning rate
     - Backbone:  lr = 1e-4  (10× lower than the head — protects pretrained features)
     - New head:  lr = 1e-3  (same as Stage 3 baseline)
     - Rationale: HAM10000 dermoscopic images differ enough from ImageNet that
       the backbone needs to adapt, not just the head. With only ~7k training
       images, we use the lower backbone LR to avoid catastrophic forgetting
       while still allowing adaptation. This is the standard fine-tuning recipe
       (used in the original ResNet and EfficientNet fine-tuning literature).

Why these two backbones?
  ResNet18   — smallest ResNet; ~11M params; simple skip-connection architecture
               widely cited as a baseline in medical imaging literature;
               fast to train; easy to explain in interviews.
  EfficientNet-B0 — compound-scaled; ~5.3M params; state-of-the-art accuracy/
               efficiency tradeoff at the time of release; commonly used in
               recent dermoscopy classification papers. Comparing it to ResNet18
               tests whether a more efficient architecture helps on this dataset.

Both use the same train() loop, class weights, splits, and evaluate_model()
from Stages 2–3 — ensuring a fair comparison with the baseline CNN.
"""

import torch
import torch.nn as nn
from torchvision import models


# ── ResNet18 ──────────────────────────────────────────────────────────────────

def build_resnet18(num_classes: int = 7, pretrained: bool = True):
    """
    ResNet18 with ImageNet weights, final FC replaced for num_classes.

    Architecture change:
      Original: Linear(512 → 1000)  [ImageNet classes]
      Replaced: Linear(512 → num_classes)

    The new head is randomly initialised; the backbone starts from ImageNet
    weights. All layers are trainable (fine-tune end-to-end).

    Parameters
    ----------
    num_classes : int  (default 7)
    pretrained  : bool (default True) — False useful for ablation studies

    Returns
    -------
    (model, param_groups) where param_groups is a list of dicts suitable for
    passing directly to torch.optim.Adam, with different LRs for backbone
    and head.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model   = models.resnet18(weights=weights)

    # Replace classifier head
    in_features = model.fc.in_features        # 512 for ResNet18
    model.fc    = nn.Linear(in_features, num_classes)

    # Two-speed parameter groups:
    #   backbone (all layers except fc) → lr = 1e-4
    #   new head (fc)                   → lr = 1e-3
    backbone_params = [p for name, p in model.named_parameters()
                       if not name.startswith("fc")]
    head_params     = list(model.fc.parameters())

    param_groups = [
        {"params": backbone_params, "lr": 1e-4},   # protect pretrained features
        {"params": head_params,     "lr": 1e-3},   # train head at full speed
    ]

    return model, param_groups


# ── EfficientNet-B0 ───────────────────────────────────────────────────────────

def build_efficientnet_b0(num_classes: int = 7, pretrained: bool = True):
    """
    EfficientNet-B0 with ImageNet weights, classifier head replaced for
    num_classes.

    Architecture change:
      Original: Sequential(Dropout(0.2), Linear(1280 → 1000))
      Replaced: Sequential(Dropout(0.2), Linear(1280 → num_classes))
    We keep the existing dropout — EfficientNet's head already has it.

    Parameters
    ----------
    num_classes : int  (default 7)
    pretrained  : bool (default True)

    Returns
    -------
    (model, param_groups) — same two-speed LR structure as ResNet18.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model   = models.efficientnet_b0(weights=weights)

    # EfficientNet's classifier is model.classifier: [Dropout, Linear]
    in_features = model.classifier[1].in_features   # 1280 for B0
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Two-speed parameter groups
    backbone_params = [p for name, p in model.named_parameters()
                       if not name.startswith("classifier")]
    head_params     = list(model.classifier.parameters())

    param_groups = [
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params,     "lr": 1e-3},
    ]

    return model, param_groups
