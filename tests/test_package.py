"""Tests for livecell_classification."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from livecell_classification.config import CELL_TYPES, NUM_CLASSES
from livecell_classification.models import create_model, is_custom_model, CustomCNN, ResNet2
from livecell_classification.models.resnet2 import ResidualBlock
from livecell_classification.evaluation import compute_ece
from livecell_classification.optim import EMA, get_vit_param_groups
from livecell_classification.utils import set_seeds


class TestConfig:
    def test_cell_types(self) -> None:
        assert len(CELL_TYPES) == 8
        assert NUM_CLASSES == 8


class TestModels:
    def test_custom_cnn_forward(self) -> None:
        m = create_model("custom_cnn")
        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        assert out.shape == (2, 8)

    def test_resnet2_forward(self) -> None:
        m = create_model("resnet2")
        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        assert out.shape == (2, 8)

    def test_resnet2_has_residual_blocks(self) -> None:
        m = ResNet2()
        assert isinstance(m.res_block4, ResidualBlock)

    def test_custom_cnn_is_custom(self) -> None:
        assert is_custom_model("custom_cnn")
        assert is_custom_model("resnet2")
        assert not is_custom_model("efficientnet_b0")

    def test_custom_cnn_num_classes(self) -> None:
        m = CustomCNN(num_classes=4)
        m.eval()  # BatchNorm1d requires batch>1 in train mode
        x = torch.randn(1, 3, 64, 64)
        assert m(x).shape == (1, 4)

    def test_residual_block_identity(self) -> None:
        """Output shape should match input shape."""
        block = ResidualBlock(64)
        x = torch.randn(2, 64, 16, 16)
        assert block(x).shape == x.shape

    def test_custom_cnn_param_count(self) -> None:
        m = CustomCNN()
        params = sum(p.numel() for p in m.parameters())
        assert 100_000 < params < 2_000_000  # ~0.5M expected

    def test_resnet2_more_params_than_cnn(self) -> None:
        cnn = CustomCNN()
        res = ResNet2()
        cnn_p = sum(p.numel() for p in cnn.parameters())
        res_p = sum(p.numel() for p in res.parameters())
        assert res_p > cnn_p


class TestECE:
    def test_perfect_calibration(self) -> None:
        n = 100
        labels = np.arange(n) % 5
        probs = np.zeros((n, 5))
        for i in range(n):
            probs[i, labels[i]] = 1.0
        assert compute_ece(probs, labels) < 0.01

    def test_overconfident(self) -> None:
        n = 100
        labels = np.zeros(n, dtype=int)
        probs = np.zeros((n, 5))
        probs[:, 1] = 0.95
        probs[:, 0] = 0.05
        assert compute_ece(probs, labels) > 0.5


class TestOptim:
    def test_ema_update(self) -> None:
        m = CustomCNN(num_classes=4)
        ema = EMA(m, decay=0.99)
        x = torch.randn(2, 3, 32, 32)
        m(x).sum().backward()
        # Simulate a param update
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        ema.update(m)
        # Shadow should differ from model
        for k in ema.shadow:
            if ema.shadow[k].is_floating_point():
                assert not torch.equal(ema.shadow[k], m.state_dict()[k])

    def test_vit_param_groups_no_llrd(self) -> None:
        m = CustomCNN()
        groups = get_vit_param_groups(m, base_lr=1e-3, weight_decay=0.01, llrd_factor=1.0)
        assert len(groups) == 2  # decay + no-decay
        assert groups[0]["weight_decay"] == 0.01
        assert groups[1]["weight_decay"] == 0.0


class TestSetSeeds:
    def test_determinism(self) -> None:
        set_seeds(42)
        a = torch.randn(5)
        set_seeds(42)
        b = torch.randn(5)
        assert torch.equal(a, b)
