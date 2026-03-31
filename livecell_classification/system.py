"""System specification logging for reproducibility."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import torchvision
import timm


def _run_cmd(cmd) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return None


def log_system_spec(**extra: Any) -> Dict[str, Any]:
    """Collect hardware and software information for experiment logs.

    Any keyword arguments are merged into the returned dict (e.g.
    ``amp_enabled=True``, ``llrd_factor=0.85``).
    """
    spec: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        spec["gpu_name"] = props.name
        spec["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        drv = _run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        if drv:
            spec["nvidia_driver"] = drv.splitlines()[0].strip()

    spec.update(extra)
    return spec
