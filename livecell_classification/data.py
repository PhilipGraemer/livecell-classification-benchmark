"""Dataset classes, transforms, and sample-list utilities.

The HDF5 layout matches the per-class crop files produced by the
segmentation-crop-checker pipeline::

    images[i]   : vlen uint8  (flattened pixel data)
    shapes[i]   : (H, W, C)   (per-crop shape)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from livecell_classification.config import CELL_TYPES

Sample = Tuple[str, int, int]  # (hdf5_path, label, index_in_file)


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

class HDF5CellDataset(Dataset):
    """Reads single-cell crop images from per-class HDF5 files.

    Parameters
    ----------
    samples : list[Sample]
        Each entry is ``(hdf5_path, label, index_in_file)``.
    transform : callable or None
        Torchvision transform applied to each PIL image.
    """

    def __init__(self, samples: List[Sample], transform=None) -> None:
        self.samples = samples
        self.transform = transform
        self._h5: Dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, i = self.samples[idx]
        if path not in self._h5:
            self._h5[path] = h5py.File(path, "r")
        hf = self._h5[path]
        arr = hf["images"][i].reshape(hf["shapes"][i])
        img = Image.fromarray(arr)
        if self.transform:
            img = self.transform(img)
        return img, label

    def close(self) -> None:
        """Explicitly close all open HDF5 handles."""
        for f in self._h5.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5 = {}

    def __del__(self) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────────────
# Sample-list I/O
# ──────────────────────────────────────────────────────────────────────

def load_sample_lists(path: str | Path) -> Dict:
    """Load sample lists from a pickle."""
    with open(str(path), "rb") as f:
        return pickle.load(f)


def make_subset(
    samples: List[Sample],
    fraction: float,
    seed: int,
) -> List[Sample]:
    """Return a stratified subset of *samples*."""
    if fraction >= 1.0:
        return samples
    n = int(len(samples) * fraction)
    labels = [s[1] for s in samples]
    idx, _ = train_test_split(
        range(len(samples)), train_size=n, random_state=seed, stratify=labels
    )
    return [samples[i] for i in idx]


# ──────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────

def make_train_transform(
    img_size: int = 224,
    rot_deg: int = 10,
    teacher_max: bool = False,
) -> transforms.Compose:
    """Training transform with configurable augmentation.

    When *teacher_max* is True, adds vertical flip, bicubic resize,
    and contrast jitter — the augmentation bundle used for teacher
    training runs.
    """
    interp = Image.BICUBIC if teacher_max else Image.BILINEAR
    ops = [transforms.Resize((img_size, img_size), interpolation=interp)]
    ops.append(transforms.RandomHorizontalFlip())
    if teacher_max:
        ops.append(transforms.RandomVerticalFlip())
        ops.append(transforms.ColorJitter(contrast=0.2))
    ops.extend([
        transforms.RandomRotation(rot_deg),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    return transforms.Compose(ops)


def make_eval_transform(img_size: int = 224, teacher_max: bool = False) -> transforms.Compose:
    """Evaluation transform: resize and normalise only."""
    interp = Image.BICUBIC if teacher_max else Image.BILINEAR
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=interp),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
