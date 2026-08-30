"""
src/models/transfer_models.py
──────────────────────────────
ResNet18 and EfficientNet-B0 fine-tuning setup using torchvision pretrained
weights. Backbone choice and head modifications documented in Stage 4.

STAGE STATUS: stub — implementation added in Stage 4.
"""


def build_resnet18(num_classes: int = 7):
    """Return a ResNet18 with its final FC layer replaced for num_classes."""
    raise NotImplementedError("Implemented in Stage 4.")


def build_efficientnet_b0(num_classes: int = 7):
    """Return an EfficientNet-B0 with its classifier head replaced for num_classes."""
    raise NotImplementedError("Implemented in Stage 4.")
