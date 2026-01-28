# YOLO26 Training Results

Training results for YOLO26 fine-tuning on ImgAssist OCT cancer detection.

**Note:** These results are from experiments BEFORE the preprocessing fix. New experiments with Y-axis squash preprocessing are pending.

---

## 1. Best Model: YOLO26m v2 Optimized

**Location:** `runs/finetune/yolo26m_v2_optimized/`

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | yolo26m |
| Pretrained | yolo26m.pt (COCO) |
| Dataset | 2-class (Suspicious) v12 |
| Image Size | 768x768 |
| Batch Size | 24 |
| Epochs | 130 (early stopped) |
| Optimizer | SGD |
| Learning Rate | lr0=0.002, lrf=0.01 |
| Patience | 30 |

### Hyperparameters

```yaml
# Optimizer
optimizer: SGD
lr0: 0.002
lrf: 0.01
momentum: 0.9
weight_decay: 0.0005
warmup_epochs: 5.0

# Loss weights
box: 7.5
cls: 0.5
dfl: 1.5

# Augmentation
mosaic: 0.2
mixup: 0.1
copy_paste: 0.3
degrees: 0.0      # No rotation (OCT fixed orientation)
flipud: 0.0       # No vertical flip
fliplr: 0.5
scale: 0.5
```

---

## 2. Validation Results

### Final Metrics (Epoch 130)

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **0.781** |
| **mAP@0.5:0.95** | **0.422** |
| Precision | 0.761 |
| Recall | 0.693 |
| Box Loss | 1.121 |
| Cls Loss | 0.804 |

### Training Progression

| Epoch | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|---------|--------------|-----------|--------|
| 10 | 0.473 | 0.204 | 0.457 | 0.526 |
| 30 | 0.739 | 0.382 | 0.744 | 0.634 |
| 50 | 0.768 | 0.411 | 0.759 | 0.679 |
| 70 | 0.773 | 0.420 | 0.755 | 0.685 |
| 90 | 0.774 | 0.422 | 0.760 | 0.683 |
| 110 | 0.782 | 0.425 | 0.767 | 0.690 |
| 130 | 0.781 | 0.422 | 0.761 | 0.693 |

---

## 3. Test Set Results (PROBLEMATIC)

**These results show the preprocessing issue:**

| Model | Val mAP@0.5 | Test mAP@0.5 | Gap |
|-------|-------------|--------------|-----|
| YOLO26s | 75.9% | 24.7% | -51.2% |
| YOLO26m | 78.1% | 16.5% | -61.6% |

**Root cause:** Incorrect preprocessing. See [PREPROCESSING.md](PREPROCESSING.md).

---

## 4. YOLO26s Results (Earlier Run)

| Parameter | Value |
|-----------|-------|
| Model | yolo26s |
| Epochs | 43 |
| Image Size | 640 |
| Batch Size | 36 |

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.773 |
| mAP@0.5:0.95 | 0.372 |
| Precision | 0.769 |
| Recall | 0.672 |

---

## 5. YOLO26m vs YOLO26s Comparison

| Metric | YOLO26s | YOLO26m | Improvement |
|--------|---------|---------|-------------|
| mAP@0.5 | 0.773 | 0.781 | +1.0% |
| mAP@0.5:0.95 | 0.372 | 0.422 | +13.4% |
| Precision | 0.769 | 0.761 | -1.0% |
| Recall | 0.672 | 0.693 | +3.1% |
| Parameters | 10M | 22M | +120% |

YOLO26m provides better mAP@0.5:0.95 (+13.4%) at the cost of 2x parameters.

---

## 6. Checkpoints

| File | Size | Description |
|------|------|-------------|
| `best.pt` | 44 MB | Best validation mAP |
| `last.pt` | 44 MB | Final epoch |
| `epoch*.pt` | 88 MB | Periodic checkpoints |

**Best model:**
```
runs/finetune/yolo26m_v2_optimized/weights/best.pt
```

---

## 7. Output Files

```
runs/finetune/yolo26m_v2_optimized/
├── args.yaml                    # Training arguments
├── results.csv                  # Epoch-by-epoch metrics
├── results.png                  # Training curves plot
├── confusion_matrix.png
├── BoxF1_curve.png
├── BoxPR_curve.png
├── labels.jpg                   # Label distribution
├── train_batch*.jpg
├── val_batch*_labels.jpg
├── val_batch*_pred.jpg
└── weights/
    ├── best.pt
    └── last.pt
```

---

## 8. Next Steps

1. **Retrain with correct preprocessing** - Use `data/yaxis_squash_h320/` dataset
2. **Use rectangular input** - `imgsz: [320, 672]`
3. **Validate Val-Test gap** - Target < 10%
4. **Export for deployment** - ONNX/TensorRT

---

## WandB Tracking

Project: `ImgAssist_YOLO26`
