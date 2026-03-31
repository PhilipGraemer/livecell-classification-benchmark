"""Shared constants and defaults for the classification benchmark."""

from __future__ import annotations

from typing import List, Tuple

CELL_TYPES: List[str] = [
    "A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SKOV3", "SkBr3"
]
NUM_CLASSES: int = len(CELL_TYPES)

# Parameters excluded from weight decay in all optimiser configurations.
NO_DECAY_KEYWORDS: Tuple[str, ...] = ("bias", "norm", "cls_token", "pos_embed")
