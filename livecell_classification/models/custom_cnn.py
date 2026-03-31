"""Custom CNN baseline — a plain convolutional network with no skip connections.

This is the simplest architecture in the benchmark, included to establish
a lower bound.  Three conv blocks (32→64→128 channels), each with two
3×3 convolutions, batch norm, ReLU, max-pool, and dropout.  The
classifier head uses adaptive average pooling followed by two fully
connected layers with batch norm and dropout.

Parameter count: ~0.5M (vs 5.3M for EN-B0, 86M for EVA-02).
"""

from __future__ import annotations

import torch.nn as nn


class CustomCNN(nn.Module):
    """Plain convolutional baseline (no residual connections).

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    """

    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))
