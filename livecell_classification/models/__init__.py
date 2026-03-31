"""Unified model factory.

Supports both ``timm`` models (any string that ``timm.create_model``
accepts) and the two custom architectures defined in this package.

Usage::

    from livecell_classification.models import create_model

    model = create_model("custom_cnn", num_classes=8)
    model = create_model("resnet2", num_classes=8)
    model = create_model("eva02_base_patch14_224.mim_in22k", num_classes=8, pretrained=True)
"""

from __future__ import annotations

import torch.nn as nn
import timm

from livecell_classification.config import NUM_CLASSES
from livecell_classification.models.custom_cnn import CustomCNN
from livecell_classification.models.resnet2 import ResNet2, ResidualBlock

# Registry of custom (non-timm) architectures.
_CUSTOM_MODELS = {
    "custom_cnn": CustomCNN,
    "resnet2": ResNet2,
}


def create_model(
    name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    **kwargs,
) -> nn.Module:
    """Create a model by name.

    Parameters
    ----------
    name : str
        Either a key in the custom registry (``"custom_cnn"``,
        ``"resnet2"``) or any valid ``timm`` model string.
    num_classes : int
        Number of output classes.
    pretrained : bool
        For timm models, whether to load pretrained weights.
        Ignored for custom models (they have no pretrained weights).
    **kwargs
        Additional keyword arguments forwarded to ``timm.create_model``
        (e.g. ``drop_path_rate``).

    Returns
    -------
    nn.Module
    """
    if name in _CUSTOM_MODELS:
        if pretrained:
            print(f"[warn] pretrained=True ignored for custom model '{name}'")
        return _CUSTOM_MODELS[name](num_classes=num_classes)

    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kwargs)


def is_custom_model(name: str) -> bool:
    """Check whether *name* refers to a custom (non-timm) architecture."""
    return name in _CUSTOM_MODELS


__all__ = [
    "create_model",
    "is_custom_model",
    "CustomCNN",
    "ResNet2",
    "ResidualBlock",
]
