"""Training loop with gradient accumulation, AMP, EMA, and dual checkpointing.

This module contains the epoch-level functions and the high-level
``run_train`` orchestrator.

**Dual checkpointing:** We save both the best-by-loss and best-by-accuracy
checkpoints.  Best-by-loss is the primary checkpoint used for
distillation (where the teacher's calibration matters).  Best-by-accuracy
is saved as a secondary checkpoint for direct comparison.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from livecell_classification.data import (
    HDF5CellDataset,
    load_sample_lists,
    make_eval_transform,
    make_subset,
    make_train_transform,
)
from livecell_classification.models import create_model
from livecell_classification.optim import EMA, get_vit_param_groups
from livecell_classification.config import NO_DECAY_KEYWORDS


# ──────────────────────────────────────────────────────────────────────
# Epoch-level functions
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 0.0,
    ema: Optional[EMA] = None,
) -> float:
    """One training epoch.  Returns average loss."""
    model.train()
    running = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)
            loss_scaled = loss / grad_accum_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        running += loss.item() * images.size(0)
        n += images.size(0)

    return running / n


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[float, float]:
    """Validate.  Returns ``(val_loss, val_accuracy_pct)``."""
    model.eval()
    running = correct = total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)
        running += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running / total, (correct / total) * 100.0


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────

def run_train(
    *,
    model_name: str,
    method: str,
    run_seed: int,
    subset_seed: int,
    data_fraction: float,
    img_size: int,
    batch_size: int,
    effective_batch_size: int,
    num_workers: int,
    epochs: int,
    patience: int,
    warmup_epochs: int,
    lr: float,
    weight_decay: float,
    llrd_factor: float,
    max_grad_norm: float,
    rot_deg: int,
    cache_dir: str,
    out_dir: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_torch_compile: bool = False,
    torch_compile_backend: Optional[str] = None,
    activation_checkpointing: bool = True,
    drop_path_rate: float = 0.0,
    no_early_stopping: bool = False,
    label_smoothing: float = 0.0,
    ema_decay: float = 0.0,
    teacher_max: bool = False,
) -> None:
    """Train a model and save checkpoints + metadata.

    This is the main training entrypoint, called once per
    (method, seed, data_fraction) combination.  It saves:

    - ``best_model.pth`` — best by validation loss
    - ``best_model_acc.pth`` — best by validation accuracy
    - ``best_model_ema.pth`` — EMA weights (if enabled)
    - ``train_meta.json`` — all metadata needed by the eval phase
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_accum_steps = max(1, effective_batch_size // batch_size)
    actual_ebs = batch_size * grad_accum_steps
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Train] batch={batch_size} accum={grad_accum_steps} effective={actual_ebs}")

    # ── Data ──────────────────────────────────────────────────────
    splits = load_sample_lists(os.path.join(cache_dir, "sample_lists.pkl"))
    train_samples = make_subset(splits["train_samples"], data_fraction, subset_seed)
    val_samples = splits["val_samples"]

    train_tf = make_train_transform(img_size, rot_deg, teacher_max)
    eval_tf = make_eval_transform(img_size, teacher_max)

    g = torch.Generator().manual_seed(run_seed)
    train_loader = DataLoader(
        HDF5CellDataset(train_samples, train_tf),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), generator=g,
    )
    val_loader = DataLoader(
        HDF5CellDataset(val_samples, eval_tf),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────
    pretrained = method == "pretrained"
    create_kw = {}
    if drop_path_rate > 0:
        create_kw["drop_path_rate"] = drop_path_rate
    model = create_model(model_name, pretrained=pretrained, **create_kw).to(device)

    if activation_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(enable=True)
        print("[Model] Activation checkpointing enabled")

    if use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model, backend=torch_compile_backend)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {model_name}  params={total_params:,}  pretrained={pretrained}")

    # ── Optimiser ─────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing if label_smoothing > 0 else 0.0
    )

    if pretrained and llrd_factor < 1.0:
        print(f"[Optim] LLRD factor={llrd_factor}")
        param_groups = get_vit_param_groups(model, lr, weight_decay, llrd_factor)
    else:
        # Standard decay / no-decay split
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(kw in name.lower() for kw in NO_DECAY_KEYWORDS):
                no_decay.append(p)
            else:
                decay.append(p)
        param_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        t = (epoch_idx - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    # ── Training loop ─────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch_loss = 0
    best_epoch_acc = 0
    trigger = 0
    cumulative_s = 0.0
    time_to_best_s = 0.0

    t0 = time.time()
    last_epoch = 0

    for epoch in range(epochs):
        last_epoch = epoch
        t_ep = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            use_amp=use_amp, amp_dtype=amp_dtype,
            grad_accum_steps=grad_accum_steps, max_grad_norm=max_grad_norm,
            ema=ema,
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, use_amp, amp_dtype
        )

        ep_s = time.time() - t_ep
        cumulative_s += ep_s
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch + 1:3d}/{epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.2f}% | "
            f"lr={current_lr:.2e} | {ep_s:.1f}s"
        )

        # Best by loss (primary)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            best_epoch_loss = epoch + 1
            time_to_best_s = cumulative_s
            trigger = 0
        else:
            trigger += 1

        # Best by accuracy (secondary)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model_acc.pth"))
            best_epoch_acc = epoch + 1

        if not no_early_stopping and trigger >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

        scheduler.step()

    total_min = (time.time() - t0) / 60.0

    # EMA checkpoint
    if ema is not None:
        torch.save(ema.state_dict(), os.path.join(out_dir, "best_model_ema.pth"))

    # ── Save metadata ─────────────────────────────────────────────
    meta = {
        "model": model_name,
        "method": method,
        "run_seed": run_seed,
        "subset_seed": subset_seed,
        "data_fraction": data_fraction,
        "img_size": img_size,
        "best_epoch_loss": best_epoch_loss,
        "best_epoch_acc": best_epoch_acc,
        "best_epoch": best_epoch_loss,
        "training_time_min": total_min,
        "time_to_best_epoch_min": time_to_best_s / 60.0,
        "total_parameters": total_params,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": actual_ebs,
        "llrd_factor": llrd_factor if pretrained else None,
        "amp_dtype": str(amp_dtype),
        "activation_checkpointing": activation_checkpointing,
        "max_grad_norm": max_grad_norm,
        "weight_decay": weight_decay,
        "drop_path_rate": drop_path_rate,
        "no_early_stopping": no_early_stopping,
        "epochs_completed": last_epoch + 1,
        "best_val_acc": best_val_acc,
        "best_val_loss": float(best_val_loss),
        "teacher_max": teacher_max,
        "label_smoothing": label_smoothing,
        "ema_decay": ema_decay if ema is not None else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Train] Done. best_loss@{best_epoch_loss} best_acc@{best_epoch_acc} time={total_min:.1f}min")
