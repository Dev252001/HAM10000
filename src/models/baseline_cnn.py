import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    4-block CNN trained from scratch on HAM10000.

    Input:  (B, 3, 224, 224)
    Output: (B, 7) raw logits

    4 conv blocks (32→64→128→256 filters), global average pooling,
    dropout(0.5), linear head. ~1.2M parameters.
    """

    def __init__(self, num_classes: int = 7, dropout_p: float = 0.5):
        super().__init__()

        # Spatial: 224 → 112 → 56 → 28 → 14
        self.features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


def build_baseline_cnn(num_classes: int = 7, dropout_p: float = 0.5) -> BaselineCNN:
    return BaselineCNN(num_classes=num_classes, dropout_p=dropout_p)
