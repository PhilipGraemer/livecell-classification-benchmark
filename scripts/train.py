#!/usr/bin/env python3
"""Unified training script for the LIVECell classification benchmark.

Replaces five original scripts (CNN_reruns.py, train_resnet2_baseline.py,
train_b0_baseline.py, enb5_reruns.py, nigoki_reruns.py) with a single
entry point.

Two-phase design to avoid CPU RAM OOM on HPC nodes:
  Phase 1 (train):  python scripts/train.py --phase train  [args]
  Phase 2 (eval):   python scripts/train.py --phase eval   [args]
  Both sequential:  python scripts/train.py --phase both   [args]

'both' runs train, then spawns a fresh subprocess for eval so the OS
fully reclaims training memory before evaluation begins.

Examples
--------
Custom CNN::

    python scripts/train.py --model custom_cnn --methods scratch \\
        --cache-dir /path/to/distillation --output-dir /path/to/output/custom_cnn

ResNet2::

    python scripts/train.py --model resnet2 --methods scratch \\
        --cache-dir /path/to/distillation --output-dir /path/to/output/resnet2

EfficientNet-B0 (pretrained)::

    python scripts/train.py --model efficientnet_b0 --methods pretrained \\
        --cache-dir /path/to/distillation --output-dir /path/to/output/enb0

EVA-02::

    python scripts/train.py --model eva02_base_patch14_224.mim_in22k \\
        --methods pretrained --llrd-factor 1.0 --activation-checkpointing 1 \\
        --cache-dir /path/to/distillation --output-dir /path/to/output/eva02

ViT-S/8 (DINO pretrained, grad accum for effective BS 128)::

    python scripts/train.py --model vit_small_patch8_224.dino \\
        --methods pretrained --batch-size 64 --effective-batch-size 128 \\
        --activation-checkpointing 1 \\
        --cache-dir /path/to/distillation --output-dir /path/to/output/vit_s8
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import torch

from livecell_classification.config import CELL_TYPES
from livecell_classification.data import HDF5CellDataset, load_sample_lists, make_eval_transform
from livecell_classification.evaluation import evaluate
from livecell_classification.models import create_model
from livecell_classification.system import log_system_spec
from livecell_classification.timing import benchmark_e2e_eval, benchmark_gpu_forward
from livecell_classification.training import run_train
from livecell_classification.utils import save_results_csv, set_seeds


# ──────────────────────────────────────────────────────────────────────
# AMP dtype selection
# ──────────────────────────────────────────────────────────────────────

def select_amp_dtype(prefer_bf16: bool = True) -> torch.dtype:
    if prefer_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("[AMP] Using bfloat16")
        return torch.bfloat16
    print("[AMP] Using float16")
    return torch.float16


# ──────────────────────────────────────────────────────────────────────
# Eval phase
# ──────────────────────────────────────────────────────────────────────

def run_eval(
    *,
    model_name: str,
    out_dir: str,
    cache_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    do_timing: bool,
    timing_batch_sizes: list[int],
    timing_warmup: int,
    timing_num_blocks: int,
    timing_iters_per_block: int,
    e2e_warmup_batches: int,
    e2e_max_timed_batches: int,
    checkpoint_name: str = "best_model.pth",
) -> dict | None:
    """Load checkpoint, evaluate, time, and save results JSON."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_path = os.path.join(out_dir, "train_meta.json")
    weights_path = os.path.join(out_dir, checkpoint_name)

    if not os.path.exists(meta_path):
        print(f"[Eval] ERROR: {meta_path} not found")
        return None
    if not os.path.exists(weights_path):
        print(f"[Eval] ERROR: {weights_path} not found")
        return None

    with open(meta_path) as f:
        train_meta = json.load(f)

    model = create_model(model_name, pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    # Test data
    splits = load_sample_lists(os.path.join(cache_dir, "sample_lists.pkl"))
    test_samples = splits["test_samples"]

    teacher_preds = None
    teacher_path = os.path.join(cache_dir, "teacher_logits_test.h5")
    if os.path.exists(teacher_path):
        with h5py.File(teacher_path, "r") as hf:
            teacher_preds = hf["predictions"][:]

    eval_tf = make_eval_transform(img_size)
    test_ds = HDF5CellDataset(test_samples, eval_tf)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=min(num_workers, 2), pin_memory=True,
    )

    torch.cuda.reset_peak_memory_stats(device)
    metrics = evaluate(model, test_loader, device, use_amp, amp_dtype, teacher_preds)
    max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else None

    print(f"[Eval] acc={metrics['accuracy']:.2f}% f1={metrics['macro_f1']:.2f}%")

    test_ds.close()
    del test_loader, test_ds
    gc.collect()
    torch.cuda.empty_cache()

    # Timing
    gpu_only = None
    e2e = None
    if do_timing and torch.cuda.is_available():
        gpu_only = benchmark_gpu_forward(
            model, device,
            input_shape=(3, img_size, img_size),
            batch_sizes=tuple(timing_batch_sizes),
            warmup=timing_warmup,
            num_blocks=timing_num_blocks,
            iters_per_block=timing_iters_per_block,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )
        test_ds_t = HDF5CellDataset(test_samples, eval_tf)
        loader_t = torch.utils.data.DataLoader(
            test_ds_t, batch_size=batch_size, shuffle=False,
            num_workers=min(num_workers, 2), pin_memory=True,
        )
        e2e = benchmark_e2e_eval(
            model, loader_t, device,
            warmup_batches=e2e_warmup_batches,
            max_timed_batches=e2e_max_timed_batches,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )
        test_ds_t.close()

    result = {
        **train_meta,
        "test_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "ece": metrics["ece"],
        "agreement_with_teacher": metrics["agreement_with_teacher"],
        "e2e_ms_per_image": e2e["e2e_ms_per_image"] if e2e else None,
        "e2e_img_per_s": e2e["e2e_img_per_s"] if e2e else None,
        "e2e_timed_images": e2e["e2e_timed_images"] if e2e else None,
        "gpu_only_timing": gpu_only,
        "total_parameters": total_params,
        "max_gpu_memory_mb": max_mem,
        "per_class_recall": metrics["per_class_recall"].tolist(),
        "checkpoint": checkpoint_name,
    }

    # Save JSON
    ckpt_stem = checkpoint_name.replace("best_model", "").replace(".pth", "").strip("_")
    fname = f"eval_results_{ckpt_stem}.json" if ckpt_stem else "eval_results.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Unified training for the LIVECell classification benchmark."
    )
    ap.add_argument("--phase", choices=["train", "eval", "both"], default="both")
    ap.add_argument("--model", required=True,
                    help="Model name: 'custom_cnn', 'resnet2', or any timm model string.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint", default="best_model.pth",
                    help="Checkpoint filename for eval phase.")

    # Experiment grid
    ap.add_argument("--methods", nargs="+", default=["pretrained", "scratch"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--data-fractions", nargs="+", type=float, default=[0.1, 1.0])
    ap.add_argument("--subset-seed", type=int, default=123)

    # Training
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--effective-batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--llrd-factor", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=0.0)
    ap.add_argument("--rot-deg", type=int, default=10)
    ap.add_argument("--drop-path-rate", type=float, default=0.0)
    ap.add_argument("--no-early-stopping", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--ema-decay", type=float, default=0.0)
    ap.add_argument("--teacher-max", action="store_true")

    # AMP / compile
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--prefer-bf16", type=int, default=1)
    ap.add_argument("--activation-checkpointing", type=int, default=1)
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--compile-backend", type=str, default=None)

    # Timing
    ap.add_argument("--do-timing", type=int, default=1)
    ap.add_argument("--timing-batch-sizes", nargs="+", type=int, default=[1, 128])
    ap.add_argument("--timing-warmup", type=int, default=80)
    ap.add_argument("--timing-num-blocks", type=int, default=30)
    ap.add_argument("--timing-iters-per-block", type=int, default=20)
    ap.add_argument("--e2e-warmup-batches", type=int, default=10)
    ap.add_argument("--e2e-max-timed-batches", type=int, default=50)

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    use_amp = bool(args.amp)
    do_timing = bool(args.do_timing)
    act_ckpt = bool(args.activation_checkpointing)
    amp_dtype = select_amp_dtype(bool(args.prefer_bf16)) if use_amp else torch.float32

    rerun_root = os.path.join(args.output_dir, "reruns")
    os.makedirs(rerun_root, exist_ok=True)

    # Save system spec (train phase only)
    if args.phase in ("train", "both"):
        spec = log_system_spec(
            amp_enabled=use_amp, amp_dtype=str(amp_dtype),
            torch_compile=bool(args.compile),
            grad_accum_steps=max(1, args.effective_batch_size // args.batch_size),
            llrd_factor=args.llrd_factor,
            activation_checkpointing=act_ckpt,
        )
        with open(os.path.join(rerun_root, "system_spec.json"), "w") as f:
            json.dump(spec, f, indent=2)
        with open(os.path.join(rerun_root, "experiment_config.json"), "w") as f:
            json.dump({**vars(args), "amp_dtype": str(amp_dtype)}, f, indent=2)

    all_results = []

    for method in args.methods:
        for seed in args.seeds:
            set_seeds(seed, deterministic=True)
            for frac in args.data_fractions:
                tag = f"{method} | seed={seed} | data={int(frac * 100)}%"
                print(f"\n{'=' * 60}")
                print(f"RUN: {tag} [phase={args.phase}]")
                print(f"{'=' * 60}")

                run_dir = os.path.join(
                    rerun_root, method, f"seed_{seed}", f"data{int(frac * 100)}"
                )

                # --- TRAIN ---
                if args.phase in ("train", "both"):
                    run_train(
                        model_name=args.model, method=method, run_seed=seed,
                        subset_seed=args.subset_seed, data_fraction=frac,
                        img_size=args.img_size, batch_size=args.batch_size,
                        effective_batch_size=args.effective_batch_size,
                        num_workers=args.num_workers, epochs=args.epochs,
                        patience=args.patience, warmup_epochs=args.warmup_epochs,
                        lr=args.lr, weight_decay=args.weight_decay,
                        llrd_factor=args.llrd_factor, max_grad_norm=args.max_grad_norm,
                        rot_deg=args.rot_deg, cache_dir=args.cache_dir,
                        out_dir=run_dir, use_amp=use_amp, amp_dtype=amp_dtype,
                        use_torch_compile=bool(args.compile),
                        torch_compile_backend=args.compile_backend,
                        activation_checkpointing=act_ckpt,
                        drop_path_rate=args.drop_path_rate,
                        no_early_stopping=args.no_early_stopping,
                        label_smoothing=args.label_smoothing,
                        ema_decay=args.ema_decay,
                        teacher_max=args.teacher_max,
                    )

                # --- EVAL (subprocess for 'both', inline for 'eval') ---
                if args.phase == "both":
                    print("\n[Both] Spawning fresh subprocess for eval...")
                    cmd = [
                        sys.executable, __file__,
                        "--phase", "eval",
                        "--model", args.model,
                        "--cache-dir", args.cache_dir,
                        "--output-dir", args.output_dir,
                        "--checkpoint", args.checkpoint,
                        "--methods", method,
                        "--seeds", str(seed),
                        "--data-fractions", str(frac),
                        "--img-size", str(args.img_size),
                        "--batch-size", str(args.batch_size),
                        "--num-workers", "0",
                        "--amp", str(args.amp),
                        "--prefer-bf16", str(args.prefer_bf16),
                        "--do-timing", str(args.do_timing),
                        "--timing-batch-sizes", *[str(x) for x in args.timing_batch_sizes],
                        "--timing-warmup", str(args.timing_warmup),
                        "--timing-num-blocks", str(args.timing_num_blocks),
                        "--timing-iters-per-block", str(args.timing_iters_per_block),
                        "--e2e-warmup-batches", str(args.e2e_warmup_batches),
                        "--e2e-max-timed-batches", str(args.e2e_max_timed_batches),
                    ]
                    ret = subprocess.run(cmd)
                    if ret.returncode != 0:
                        print(f"[Both] WARNING: eval exited with code {ret.returncode}")

                if args.phase == "eval":
                    res = run_eval(
                        model_name=args.model, out_dir=run_dir,
                        cache_dir=args.cache_dir, img_size=args.img_size,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        use_amp=use_amp, amp_dtype=amp_dtype, do_timing=do_timing,
                        timing_batch_sizes=args.timing_batch_sizes,
                        timing_warmup=args.timing_warmup,
                        timing_num_blocks=args.timing_num_blocks,
                        timing_iters_per_block=args.timing_iters_per_block,
                        e2e_warmup_batches=args.e2e_warmup_batches,
                        e2e_max_timed_batches=args.e2e_max_timed_batches,
                        checkpoint_name=args.checkpoint,
                    )
                    if res:
                        all_results.append(res)

                # Load eval JSON after subprocess
                if args.phase == "both":
                    ckpt_stem = args.checkpoint.replace("best_model", "").replace(".pth", "").strip("_")
                    fname = f"eval_results_{ckpt_stem}.json" if ckpt_stem else "eval_results.json"
                    eval_json = os.path.join(run_dir, fname)
                    if os.path.exists(eval_json):
                        with open(eval_json) as f:
                            res = json.load(f)
                        if res.get("gpu_only_timing"):
                            res["gpu_only_timing"] = {int(k): v for k, v in res["gpu_only_timing"].items()}
                        all_results.append(res)
                    else:
                        print(f"  => WARNING: {eval_json} not found")

    # Combined CSV
    if all_results:
        short = args.model.split(".")[0].replace("/", "_")
        csv_path = os.path.join(rerun_root, f"{short}_results_reruns.csv")
        save_results_csv(all_results, csv_path, CELL_TYPES, args.batch_size)
        print(f"\nResults CSV: {csv_path}")
    elif args.phase == "train":
        print("\nTraining complete. Run --phase eval to generate metrics.")


if __name__ == "__main__":
    main()
