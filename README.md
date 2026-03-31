# livecell-classification-benchmark

A controlled benchmark comparing CNN and Vision Transformer architectures for single-cell classification on the [LIVECell](https://sartorius-research.github.io/LIVECell/) dataset.


This repository accompanies the upcoming paper:

> **Pretraining and the CNN–ViT Ranking for Single-Cell Classification: A Controlled Benchmark on LIVECell**
> Philip Graemer, Giuseppe Di Caprio
> *Manuscript in preparation*

## Key findings

| Architecture | Macro F1 (pretrained) | Note |
|---|---|---|
| EfficientNet-B5 | **92.25%** | Best single model |
| 3×EVA-02 council → EN-B0 | **92.05%** | Council distillation; 21× fewer params than teacher |
| EVA-02 → EN-B0 | 91.01% | Cross-arch distillation |
| Swin-B | 90.53% | |
| EVA-02 | 90.51% | Better distillation teacher than EN-B5 |
| EN-B5 → EN-B0 | 91.01% | Same-family distillation |
| ViT-B/16 | 89.93% |  |
| ViT-S/8 | 90.61% | Exceeds EVA-02 F1 at ¼ parameters (But with EVA-02 having higher Acc), also better than larger ViT-B/16 |

*Selected results from eleven architectures tested pretrained and from scratch with matched conditions. Full results across all architectures in the paper.*


## Architectures

Eleven architectures spanning three families, each tested pretrained (ImageNet-1k/21k) and from scratch:

- **Custom baselines:** Custom CNN, ResNet2
- **CNNs (timm):** EfficientNet-B0, EfficientNet-B5
- **Plain ViTs:** ViT-B/16, ViT-S/8
- **Hierarchical transformers:** Swin-B, EVA-02 (ViT-B/14)

## Installation

```bash
git clone https://github.com/PhilipGraemer/livecell-classification-benchmark.git
cd livecell-classification-benchmark
pip install -e ".[dev]"
```

## Usage

A single unified script replaces five original training scripts. It supports the two custom architectures and any timm model string.

### Two-phase design

Training and evaluation run in separate phases to avoid CPU RAM OOM on HPC nodes. The `--phase both` option runs training, then spawns a fresh subprocess for evaluation.

```bash
# Train + eval in one go
python scripts/train.py --phase both --model <name> [args]

# Or separately
python scripts/train.py --phase train --model <name> [args]
python scripts/train.py --phase eval  --model <name> [args]
```

### Examples

```bash
# Custom CNN baseline (from scratch only)
python scripts/train.py --model custom_cnn --methods scratch \
    --cache-dir /path/to/distillation --output-dir output/custom_cnn

# ResNet2 baseline
python scripts/train.py --model resnet2 --methods scratch \
    --cache-dir /path/to/distillation --output-dir output/resnet2

# EfficientNet-B5 (pretrained + scratch, multi-seed)
python scripts/train.py --model tf_efficientnet_b5.ns_jft_in1k \
    --methods pretrained scratch --seeds 42 43 \
    --cache-dir /path/to/distillation --output-dir output/enb5

# EVA-02 (pretrained, no LLRD)
python scripts/train.py --model eva02_base_patch14_224.mim_in22k \
    --methods pretrained --llrd-factor 1.0 \
    --activation-checkpointing 1 \
    --cache-dir /path/to/distillation --output-dir output/eva02

# ViT-S/8 with DINO weights (grad accum for effective BS 128)
python scripts/train.py --model vit_small_patch8_224.dino \
    --methods pretrained --batch-size 64 --effective-batch-size 128 \
    --activation-checkpointing 1 \
    --cache-dir /path/to/distillation --output-dir output/vit_s8
```

### Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | required | `custom_cnn`, `resnet2`, or any timm model string |
| `--methods` | `pretrained scratch` | Training conditions to run |
| `--seeds` | `42 43` | Random seeds for multi-seed runs |
| `--data-fractions` | `0.1 1.0` | Fraction of training data |
| `--llrd-factor` | `1.0` | Layer-wise LR decay (1.0 = disabled) |
| `--effective-batch-size` | `128` | Gradient accumulation target |
| `--activation-checkpointing` | `1` | Memory-efficient training |
| `--ema-decay` | `0.0` | EMA for teacher training (0.9999 recommended) |
| `--teacher-max` | off | Enhanced augmentation bundle for teachers |
| `--no-early-stopping` | off | Run all epochs (for teacher training) |

## Pipeline overview

- **Framework:** PyTorch + timm + scikit-learn
- **Hardware:** NVIDIA A100 80 GB (ArchieWest HPC)
- **Optimiser:** AdamW with warmup + cosine annealing
- **Precision:** Mixed precision (bf16 on A100, fp16 fallback)
- **Reproducibility:** Multi-seed runs, fixed subset seed for data fraction experiments
- **Checkpointing:** Dual — best-by-loss (for distillation) and best-by-accuracy

## Repository structure

```
├── livecell_classification/
│   ├── __init__.py
│   ├── config.py            # Constants, cell types
│   ├── data.py              # HDF5CellDataset, transforms, sample list I/O
│   ├── evaluation.py        # Metrics: accuracy, macro F1, ECE, teacher agreement
│   ├── models/
│   │   ├── __init__.py      # Model factory (timm + custom)
│   │   ├── custom_cnn.py    # Plain convolutional baseline
│   │   └── resnet2.py       # CNN with residual blocks
│   ├── optim.py             # LLRD parameter groups, EMA
│   ├── plotting.py          # Training curves, confusion matrices
│   ├── system.py            # Hardware/software spec logging
│   ├── timing.py            # GPU-only and end-to-end benchmarks
│   ├── training.py          # Training loop with grad accum, AMP, dual checkpointing
│   └── utils.py             # Seeds, CSV export
├── scripts/
│   └── train.py             # Unified CLI (replaces 5 original scripts)
├── tests/
│   └── test_package.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## Data

This pipeline operates on **single-cell crops** extracted from LIVECell segmentation masks, stored as per-class HDF5 files. The cropping and validation pipeline is available in [`segmentation-crop-checker`](https://github.com/PhilipGraemer/segmentation-crop-checker).

Already cropped LIVECell data as H5 available at: 
https://pureportal.strath.ac.uk/en/datasets/single-cell-crops-from-the-livecell-dataset/

## Tests

```bash
pytest
pytest --cov=livecell_classification
```

## Citation

```bibtex
@article{graemer2026pretraining,
  title={Pretraining Inverts the CNN--ViT Ranking for Single-Cell Classification:
         A Controlled Benchmark on LIVECell},
  author={Graemer, Philip and Di Caprio, Giuseppe},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
