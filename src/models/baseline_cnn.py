"""
src/models/baseline_cnn.py
──────────────────────────
Baseline CNN trained from scratch on HAM10000.
No pretrained weights — this is the baseline we will beat with transfer
learning in Stage 4.

Architecture design rationale
──────────────────────────────
Goal: simple enough to fully explain in an interview, deep enough to learn
meaningful features from 224×224 dermoscopic images.

Structure: 4 convolutional blocks → global average pooling → classifier head

  Block 1: 3→32 filters, 3×3
  Block 2: 32→64 filters, 3×3
  Block 3: 64→128 filters, 3×3
  Block 4: 128→256 filters, 3×3

Each block: Conv2d → BatchNorm2d → ReLU → MaxPool2d(2×2)
After 4 max-pools: 224 → 112 → 56 → 28 → 14 spatial resolution

Classifier head: GAP(14×14×256 → 256) → Dropout(0.5) → Linear(256→7)

Why 4 blocks?
  - 2 blocks: too shallow to capture texture patterns in lesion images
    (dermoscopy relies on fine-grained texture, colour gradients, borders).
  - 5+ blocks: diminishing returns for a from-scratch model on 10k images —
    deeper networks need more data or pretraining to avoid underfitting.
  - 4 blocks is a standard depth for this dataset size.

Why filter sizes double each block (32→64→128→256)?
  Early layers detect low-level features (edges, colour blobs) that need
  fewer channels. Later layers combine those into higher-level patterns
  (irregular borders, colour variegation) that benefit from more channels.
  Doubling is the standard convention (VGG, ResNet).

Why Global Average Pooling instead of Flatten + FC?
  GAP collapses each feature map to a single number, dramatically reducing
  parameters (256 vs 256×14×14 = 50,176 before the final FC). This lowers
  overfitting risk on a small dataset and makes the model resolution-flexible.

Why Dropout(0.5) before the classifier?
  With only ~7k training images, the dense classifier head is the most
  likely place to overfit. 0.5 dropout is the standard starting point.

Why 3×3 conv kernels throughout?
  Two stacked 3×3 convolutions cover the same receptive field as one 5×5
  but with fewer parameters and an extra non-linearity — the standard choice
  since VGG (2014).

Why BatchNorm after every conv?
  Stabilises gradient flow, allows higher learning rates, and acts as mild
  regularisation — especially important when training from scratch on small data.

Parameter count (approximate): ~1.2M — small enough to fit on a T4 with
batch size 32 and fast enough to train in ~30–60 min on Colab free tier.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Single convolutional block: Conv → BatchNorm → ReLU → MaxPool.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int  (default 3, padding=1 to preserve spatial dims before pooling)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, bias=False),  # bias=False: BN has its own bias
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            # halve spatial dims
        )

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    4-block CNN baseline for HAM10000 7-class skin lesion classification.

    Input:  (B, 3, 224, 224)  — ImageNet-normalised RGB images
    Output: (B, 7)            — raw logits (no softmax; use with CrossEntropyLoss)

    Parameters
    ----------
    num_classes : int  (default 7 — HAM10000 classes)
    dropout_p   : float (default 0.5 — dropout probability before classifier)
    """
    def __init__(self, num_classes: int = 7, dropout_p: float = 0.5):
        super().__init__()

        # ── Feature extractor: 4 conv blocks ──────────────────────────────────
        # Spatial progression: 224 → 112 → 56 → 28 → 14
        self.features = nn.Sequential(
            ConvBlock(3,   32),   # Block 1: RGB → 32 feature maps
            ConvBlock(32,  64),   # Block 2: low-level → mid-level features
            ConvBlock(64,  128),  # Block 3: mid-level features
            ConvBlock(128, 256),  # Block 4: high-level features
        )

        # ── Classifier head ───────────────────────────────────────────────────
        # Global Average Pooling: (B, 256, 14, 14) → (B, 256)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)        # (B, 256, 14, 14)
        x = self.gap(x)             # (B, 256, 1, 1)
        x = x.flatten(start_dim=1) # (B, 256)
        x = self.classifier(x)     # (B, 7)
        return x


def build_baseline_cnn(num_classes: int = 7, dropout_p: float = 0.5) -> BaselineCNN:
    """
    Convenience factory function — returns an initialised BaselineCNN.
    Weights are randomly initialised (Xavier uniform for conv layers via
    PyTorch default init); no pretrained weights.
    """
    return BaselineCNN(num_classes=num_classes, dropout_p=dropout_p)
