"""Optimiser utilities: layer-wise learning rate decay and EMA.

**Layer-Wise Learning Rate Decay (LLRD)** assigns geometrically
decreasing learning rates to transformer layers, with the lowest rate
at the patch embedding (closest to the input) and the highest at the
classification head.  This is standard practice for fine-tuning ViTs
on new domains.

**However**, our benchmark found that LLRD *consistently hurts*
performance when transferring from natural images to phase-contrast
microscopy — a finding with implications for how NLP-derived
fine-tuning recipes are adopted in biomedical vision.  The default
``llrd_factor=1.0`` disables LLRD.

**Exponential Moving Average (EMA)** maintains a running average of
model weights for smoother convergence.  Used primarily in teacher
training runs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from livecell_classification.config import NO_DECAY_KEYWORDS


# ──────────────────────────────────────────────────────────────────────
# LLRD
# ──────────────────────────────────────────────────────────────────────

def get_vit_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    llrd_factor: float = 1.0,
    num_blocks: int = 12,
) -> List[Dict[str, Any]]:
    """Build per-parameter groups with optional LLRD.

    When ``llrd_factor == 1.0`` all layers share the same LR and the
    function only separates decay vs no-decay parameters.  When
    ``llrd_factor < 1.0``, earlier layers get geometrically smaller LRs.

    Parameters
    ----------
    model : nn.Module
        A ViT-style model with ``patch_embed``, ``blocks.N``, etc.
    base_lr : float
        Learning rate for the top (classifier) layer.
    weight_decay : float
        Weight decay for parameters not in :data:`NO_DECAY_KEYWORDS`.
    llrd_factor : float
        Multiplicative factor per layer (1.0 = no decay).
    num_blocks : int
        Number of transformer blocks (12 for ViT-B, 24 for ViT-L).
    """
    if llrd_factor >= 1.0:
        # No LLRD — just separate decay / no-decay
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(kw in name.lower() for kw in NO_DECAY_KEYWORDS):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # Full LLRD
    num_layers = num_blocks + 2  # embed + N blocks + head

    def _layer_id(name: str) -> int:
        if any(name.startswith(p) for p in ("patch_embed", "cls_token", "pos_embed")):
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        return num_layers - 1

    scales = {i: llrd_factor ** (num_layers - 1 - i) for i in range(num_layers)}
    groups: List[Dict[str, Any]] = []
    seen = set()

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        lid = _layer_id(name)
        skip_wd = any(kw in name.lower() for kw in NO_DECAY_KEYWORDS)
        groups.append({
            "params": [p],
            "lr": base_lr * scales[lid],
            "weight_decay": 0.0 if skip_wd else weight_decay,
        })
    return groups


# ──────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of model parameters.

    Parameters
    ----------
    model : nn.Module
        Source model whose weights are tracked.
    decay : float
        EMA decay rate (e.g. 0.9999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow
