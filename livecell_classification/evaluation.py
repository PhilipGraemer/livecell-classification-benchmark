"""Evaluation metrics: accuracy, macro F1, ECE, teacher agreement."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error.

    Returns ECE as a fraction in [0, 1].
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        prop = mask.mean()
        if prop > 0:
            ece += abs(accuracies[mask].mean() - confidences[mask].mean()) * prop
    return float(ece)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    teacher_preds: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Comprehensive evaluation on a data split.

    Returns a dict with accuracy, macro_f1, balanced_accuracy, ECE,
    teacher agreement, per-class recall, and raw arrays.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(images)
        probs = F.softmax(logits, dim=1)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.float().cpu().numpy())

    preds = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)

    acc = float((preds == labels_np).mean() * 100)
    macro_f1 = float(f1_score(labels_np, preds, average="macro") * 100)
    bal_acc = float(balanced_accuracy_score(labels_np, preds) * 100)
    per_class = recall_score(labels_np, preds, average=None, zero_division=0) * 100
    ece = compute_ece(probs_np, labels_np) * 100
    agreement = (
        float((preds == teacher_preds).mean() * 100)
        if teacher_preds is not None else None
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "per_class_recall": per_class,
        "ece": float(ece),
        "agreement_with_teacher": agreement,
    }
