# YOLO26 Training Results - Medical Imaging (OCT)

## Overview

This document summarizes the training results for YOLO26 fine-tuning on the ImgAssist medical imaging dataset (OCT cancer detection).

---

## 1. Training Configuration

### 1.1 Best Model: YOLO26m v2 Optimized

**Location:** `runs/detect/runs/finetune/yolo26m_v2_optimized/`

| Parameter | Value |
|-----------|-------|
| Model | yolo26m |
| Pretrained Weights | yolo26m.pt (COCO) |
| Dataset | 2-class (Suspicious) v12 |
| Image Size | 768 |
| Batch Size | 24 |
| Epochs | 130 (early stopped from 300) |
| Optimizer | SGD |
| Learning Rate | lr0=0.002, lrf=0.01 |
| Patience | 30 |
| Device | GPU 0 |

### 1.2 Hyperparameters

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

## 2. Training Results

### 2.1 Final Metrics (Epoch 130)

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **0.781** |
| **mAP@0.5:0.95** | **0.422** |
| Precision | 0.761 |
| Recall | 0.693 |
| Box Loss | 1.121 |
| Cls Loss | 0.804 |

### 2.2 Training Progression

| Epoch | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|---------|--------------|-----------|--------|
| 10 | 0.473 | 0.204 | 0.457 | 0.526 |
| 30 | 0.739 | 0.382 | 0.744 | 0.634 |
| 50 | 0.768 | 0.411 | 0.759 | 0.679 |
| 70 | 0.773 | 0.420 | 0.755 | 0.685 |
| 90 | 0.774 | 0.422 | 0.760 | 0.683 |
| 110 | 0.782 | 0.425 | 0.767 | 0.690 |
| 130 | 0.781 | 0.422 | 0.761 | 0.693 |

### 2.3 Loss Curves

| Epoch | Box Loss | Cls Loss | DFL Loss |
|-------|----------|----------|----------|
| 1 | 1.915 | 5.111 | 0.023 |
| 50 | 1.343 | 1.133 | 0.015 |
| 100 | 1.202 | 0.908 | 0.013 |
| 130 | 1.121 | 0.804 | 0.012 |

---

## 3. Model Checkpoints

### 3.1 Saved Weights

| File | Size | Description |
|------|------|-------------|
| `best.pt` | 44 MB | Best validation mAP |
| `last.pt` | 44 MB | Final epoch |
| `epoch0.pt` | 88 MB | Initial checkpoint |
| `epoch10.pt` | 88 MB | Epoch 10 |
| `epoch20.pt` | 88 MB | Epoch 20 |
| ... | ... | Every 10 epochs |
| `epoch120.pt` | 88 MB | Epoch 120 |

### 3.2 Best Model Location

```
/home/ubuntu/ImgAssistClone/experiments/yolo26/runs/detect/runs/finetune/yolo26m_v2_optimized/weights/best.pt
```

---

## 4. YOLO26s Training (Earlier Run)

### 4.1 Configuration

| Parameter | Value |
|-----------|-------|
| Model | yolo26s |
| Epochs | 43 |
| Image Size | 640 |
| Batch Size | 36 |

### 4.2 Results (Epoch 43)

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.773 |
| mAP@0.5:0.95 | 0.372 |
| Precision | 0.769 |
| Recall | 0.672 |

---

## 5. Comparison: YOLO26m vs YOLO26s

| Metric | YOLO26s | YOLO26m | Improvement |
|--------|---------|---------|-------------|
| mAP@0.5 | 0.773 | 0.781 | +1.0% |
| mAP@0.5:0.95 | 0.372 | 0.422 | +13.4% |
| Precision | 0.769 | 0.761 | -1.0% |
| Recall | 0.672 | 0.693 | +3.1% |
| Parameters | 10M | 22M | +120% |

**Conclusion:** YOLO26m provides significantly better mAP@0.5:0.95 (+13.4%) with improved recall, at the cost of 2x parameters.

---

## 6. Output Files

### 6.1 Training Artifacts

```
runs/detect/runs/finetune/yolo26m_v2_optimized/
├── args.yaml                    # Training arguments
├── results.csv                  # Epoch-by-epoch metrics
├── results.png                  # Training curves plot
├── confusion_matrix.png         # Confusion matrix
├── confusion_matrix_normalized.png
├── BoxF1_curve.png             # F1 curve
├── BoxPR_curve.png             # Precision-Recall curve
├── BoxP_curve.png              # Precision curve
├── BoxR_curve.png              # Recall curve
├── labels.jpg                  # Label distribution
├── train_batch*.jpg            # Training batch samples
├── val_batch*_labels.jpg       # Validation ground truth
├── val_batch*_pred.jpg         # Validation predictions
└── weights/
    ├── best.pt
    ├── last.pt
    └── epoch*.pt
```

---

## 7. Next Steps

1. **Test Set Evaluation:** Run validation on held-out test set with patch-level metrics (TP/TN/FP/FN, F2 score)

2. **Compare with YOLOv9:** Benchmark against existing YOLOv9 baseline

3. **Export for Deployment:** Export to ONNX/TensorRT for production

4. **Try MuSGD:** Re-train with MuSGD optimizer for potentially faster convergence

---

## 8. WandB Tracking

Training was logged to WandB project: `ImgAssist_YOLO26`

View runs at: [WandB Dashboard](https://wandb.ai/)
