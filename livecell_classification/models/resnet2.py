"""ResNet2 — a custom CNN with residual skip connections.

Same convolutional stem as :class:`~livecell_classification.models.custom_cnn.CustomCNN`
(three conv blocks, 32→64→128), but adds three residual blocks after the
stem.  Each residual block applies two 3×3 convolutions with a skip
connection, matching the standard pre-activation residual design.

The name "ResNet2" distinguishes this from standard ResNets (which use
downsampling residual blocks with projection shortcuts).  This
architecture was designed to test whether adding skip connections to the
custom CNN baseline improves single-cell classification.

Parameter count: ~0.8M.
"""

from __future__ import annotations

import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with identity skip connection.

    Two 3×3 convolutions with batch norm and ReLU, plus a dropout layer
    after the skip addition.  Channels are preserved (no downsampling).
    """

    def __init__(self, channels: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ResNet2(nn.Module):
    """Custom CNN with three residual blocks appended to the convolutional stem.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    """

    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.res_block4 = ResidualBlock(128)
        self.res_block5 = ResidualBlock(128)
        self.res_block6 = ResidualBlock(128)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        return self.classifier(x)
