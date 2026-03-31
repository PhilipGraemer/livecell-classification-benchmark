"""Small utility functions."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from livecell_classification.config import CELL_TYPES


def set_seeds(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def save_results_csv(
    results: List[Dict],
    path: str | Path,
    cell_types: List[str] = CELL_TYPES,
    batch_size: int = 128,
) -> None:
    """Write experiment results to CSV, including per-class recall and timing."""
    if not results:
        return

    fieldnames = [
        "model", "method", "run_seed", "subset_seed", "data_fraction", "img_size",
        "best_epoch", "training_time_min", "time_to_best_epoch_min",
        "test_accuracy", "macro_f1", "balanced_accuracy", "ece",
        "agreement_with_teacher",
        "e2e_ms_per_image", "e2e_img_per_s", "e2e_timed_images",
        "total_parameters", "gmacs", "max_gpu_memory_mb",
        "grad_accum_steps", "effective_batch_size", "llrd_factor",
        "amp_dtype", "activation_checkpointing", "max_grad_norm", "weight_decay",
        "timestamp",
        "gpu_ms_img_b1", "gpu_img_s_b1",
        "gpu_batch_median_ms_b1", "gpu_batch_q25_ms_b1", "gpu_batch_q75_ms_b1",
        "gpu_ms_img_bN", "gpu_img_s_bN",
        "gpu_batch_median_ms_bN", "gpu_batch_q25_ms_bN", "gpu_batch_q75_ms_bN",
    ] + [f"recall_{ct}" for ct in cell_types]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(str(path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fieldnames}
            for i, ct in enumerate(cell_types):
                row[f"recall_{ct}"] = r.get("per_class_recall", [None] * len(cell_types))[i]

            gpu = r.get("gpu_only_timing") or {}
            for prefix, bs in [("b1", 1), ("bN", batch_size)]:
                if bs in gpu:
                    g = gpu[bs]
                    row[f"gpu_ms_img_{prefix}"] = g["gpu_only_ms_per_image"]
                    row[f"gpu_img_s_{prefix}"] = g["gpu_only_img_per_s"]
                    row[f"gpu_batch_median_ms_{prefix}"] = g["gpu_only_ms_per_batch_median"]
                    row[f"gpu_batch_q25_ms_{prefix}"] = g["gpu_only_ms_per_batch_q25"]
                    row[f"gpu_batch_q75_ms_{prefix}"] = g["gpu_only_ms_per_batch_q75"]
            w.writerow(row)
