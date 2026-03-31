"""Inference timing benchmarks.

Two complementary measurements:

1. **GPU-only forward latency** — synthetic input, CUDA events, no
   data loading overhead.  Reports median and IQR across blocks of
   iterations for statistical robustness.

2. **End-to-end evaluation throughput** — real data from a DataLoader,
   includes host→device transfer.  Excludes initial warmup batches.
"""

from __future__ import annotations

import gc
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


@torch.inference_mode()
def benchmark_gpu_forward(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    batch_sizes: Sequence[int] = (1, 128),
    warmup: int = 80,
    num_blocks: int = 30,
    iters_per_block: int = 20,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[int, Dict[str, float]]:
    """GPU-only forward latency via CUDA event timing.

    Returns a dict keyed by batch size, each containing timing statistics.
    """
    model.eval()
    torch.cuda.empty_cache()
    gc.collect()

    old_bench = torch.backends.cudnn.benchmark
    old_det = torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    results = {}
    c, h, w = input_shape

    for b in batch_sizes:
        torch.cuda.empty_cache()
        x = torch.randn((b, c, h, w), device=device)

        # Warmup
        for _ in range(warmup):
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                _ = model(x)
        torch.cuda.synchronize()

        # Timed blocks
        block_ms = []
        for _ in range(num_blocks):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters_per_block):
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    _ = model(x)
            end.record()
            torch.cuda.synchronize()
            block_ms.append(start.elapsed_time(end))

        del x
        torch.cuda.empty_cache()

        arr = np.array(block_ms) / float(iters_per_block)
        med = float(np.median(arr))
        results[b] = {
            "gpu_only_ms_per_image": med / b,
            "gpu_only_img_per_s": (1000.0 / med) * b,
            "gpu_only_ms_per_batch_median": med,
            "gpu_only_ms_per_batch_q25": float(np.percentile(arr, 25)),
            "gpu_only_ms_per_batch_q75": float(np.percentile(arr, 75)),
        }

    torch.backends.cudnn.benchmark = old_bench
    torch.backends.cudnn.deterministic = old_det
    return results


@torch.inference_mode()
def benchmark_e2e_eval(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    warmup_batches: int = 10,
    max_timed_batches: Optional[int] = 50,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """End-to-end evaluation throughput including data loading.

    Returns ms/image, images/s, and the number of timed images.
    """
    model.eval()
    torch.cuda.synchronize()
    n_timed = 0
    timed_start = None

    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        if batch_idx == warmup_batches:
            torch.cuda.synchronize()
            timed_start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            _ = model(images)
        if timed_start is not None:
            n_timed += images.size(0)
        if (
            timed_start is not None
            and max_timed_batches is not None
            and (batch_idx - warmup_batches + 1) >= max_timed_batches
        ):
            break

    # Fallback if dataset was smaller than warmup
    if timed_start is None:
        torch.cuda.synchronize()
        timed_start = time.perf_counter()
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                _ = model(images)
            n_timed += images.size(0)

    torch.cuda.synchronize()
    secs = max(time.perf_counter() - timed_start, 1e-9)
    return {
        "e2e_ms_per_image": (secs / n_timed) * 1000.0,
        "e2e_img_per_s": n_timed / secs,
        "e2e_timed_images": int(n_timed),
    }
