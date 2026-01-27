# YOLO26 Medical Imaging Pipeline

Fine-tuning and evaluation pipeline for YOLO26 on medical imaging datasets (OCT cancer detection).

## Overview

This repository implements a complete pipeline for:
- **Phase 1**: Pretrain evaluation (baseline performance without fine-tuning)
- **Phase 2**: Fine-tuning on medical imaging data
- **Phase 3**: Validation and comparison with YOLOv9

## Features

- Ultralytics YOLO26 integration
- WandB experiment tracking
- Medical-specific hyperparameters (conservative augmentation for OCT)
- Patch-level metrics (sensitivity, F2 score)
- Model comparison framework

## Directory Structure

```
yolo26/
├── configs/
│   ├── data/data_2class.yaml       # Dataset configuration
│   └── hyp/hyp_yolo26_medical.yaml # Medical-optimized hyperparameters
├── scripts/
│   ├── convert_manifest_to_ultralytics.py  # Data format conversion
│   ├── pretrain_eval.py                    # Phase 1: Pretrain evaluation
│   ├── train_yolo26.py                     # Phase 2: Fine-tuning
│   ├── val_yolo26.py                       # Phase 3: Validation
│   └── compute_patch_metrics.py            # Medical patch-level metrics
├── data/                           # Generated data (gitignored)
├── runs/                           # Training outputs (gitignored)
├── weights/                        # Model checkpoints (gitignored)
└── run_training_yolo26.sh          # Training launcher script
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install ultralytics wandb

# Login to WandB (optional but recommended)
wandb login
```

### 2. Prepare Data

```bash
# Convert manifests to Ultralytics format
python scripts/convert_manifest_to_ultralytics.py \
    --manifest-dir /path/to/manifests \
    --output-dir data/
```

### 3. Run Training

```bash
# Default: yolo26s with WandB logging
./run_training_yolo26.sh

# Specific model variant
./run_training_yolo26.sh yolo26n

# Custom options
./run_training_yolo26.sh yolo26s --epochs 100 --batch 16

# Disable WandB
./run_training_yolo26.sh yolo26s --no-wandb
```

### 4. Evaluate

```bash
# Pretrain evaluation (no fine-tuning)
python scripts/pretrain_eval.py --variant yolo26s

# Validation on test set
python scripts/val_yolo26.py \
    --weights runs/finetune/yolo26s_2class_v12/weights/best.pt \
    --split test

# Compare with YOLOv9
python scripts/val_yolo26.py --compare

# Compute patch-level metrics
python scripts/compute_patch_metrics.py \
    --pred-dir runs/val/val_test/labels \
    --manifest /path/to/test_manifest.json
```

## WandB Integration

All scripts support WandB logging:

```bash
# Training with custom project
python scripts/train_yolo26.py --wandb-project MyProject --wandb-entity my-team

# Disable WandB
python scripts/train_yolo26.py --no-wandb
```

Logged metrics:
- Training: loss, mAP, precision, recall
- Validation: comparison tables, model artifacts
- Final results: best checkpoint as artifact

## Configuration

### Hyperparameters (hyp_yolo26_medical.yaml)

Medical-specific settings:
- **No rotation/shear**: OCT images have fixed orientation
- **No mosaic**: Preserves spatial relationships
- **Conservative color augmentation**: Medical imaging consistency
- **High box loss weight**: Accurate localization important

### Data Format

Uses Ultralytics format:
- `train.txt`, `val.txt`, `test.txt`: Image paths
- `labels/`: YOLO format annotations (class x y w h)

## Metrics

### Standard YOLO Metrics
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall

### Medical Patch-Level Metrics
- **Sensitivity (Recall)**: TP/(TP+FN) - critical for cancer detection
- **Specificity**: TN/(TN+FP)
- **F2 Score**: Weights recall 4x more than precision

## Model Variants

| Model | Parameters | Use Case |
|-------|------------|----------|
| yolo26n | ~3M | Fast inference, mobile |
| yolo26s | ~9.5M | Balanced (recommended) |
| yolo26m | ~25M | Higher accuracy |
| yolo26l | ~53M | Best accuracy |
| yolo26x | ~97M | Maximum accuracy |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- ultralytics >= 8.0
- wandb (optional)

## Documentation

Detailed technical documentation is available in the `docs/` folder:

- **[YOLO26 Technical Analysis](docs/YOLO26_TECHNICAL_ANALYSIS.md)**: Deep dive into YOLO26 architecture, innovations (NMS-free, DFL removal, MuSGD), and performance benchmarks
- **[Code Verification Report](docs/YOLO26_VERIFICATION.md)**: Verification that the Ultralytics implementation matches documented specifications
- **[Training Results](docs/TRAINING_RESULTS.md)**: Detailed training results for medical imaging fine-tuning

## Training Results Summary

### Best Model: YOLO26m v2 Optimized

| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.781** |
| mAP@0.5:0.95 | **0.422** |
| Precision | 0.761 |
| Recall | 0.693 |
| Epochs | 130 |
| Image Size | 768 |

**Best weights:** `runs/detect/runs/finetune/yolo26m_v2_optimized/weights/best.pt`

## License

MIT License
