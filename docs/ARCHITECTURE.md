# YOLO26 Architecture

YOLO26 (Released Sept 2025) - Ultralytics' deployment-first YOLO variant.

---

## Core Philosophy: Deployment-First

YOLO26 solves the "Export Gap" - models that run fast on GPU but slow down when exported to edge devices.

Key innovations:
- **NMS-Free:** Deterministic latency
- **DFL Removal:** Clean INT8 quantization
- **MuSGD Optimizer:** Fast convergence

---

## 1. Key Architectural Changes

### 1.1 Native End-to-End NMS-Free

Traditional YOLO uses Non-Maximum Suppression (NMS) for filtering, causing latency fluctuation.

**YOLO26 solution:**
- Uses One-to-One Label Assignment
- Head outputs sparse, deterministic predictions
- No post-processing required

```yaml
# yolo26.yaml
end2end: True  # NMS-free mode
```

**Impact:** CPU latency reduced by ~43%.

### 1.2 DFL Removal (Direct Regression)

Previous versions (v8-v13) used Distribution Focal Loss (DFL) with Softmax integration, unfriendly to edge hardware.

**YOLO26 solution:**
- Returns to direct regression
- Greatly simplifies computation graph
- Nearly lossless INT8 quantization

```yaml
# yolo26.yaml
reg_max: 1  # Direct regression (v8/v9 uses 16 for DFL)
```

---

## 2. Model Variants

| Model | Parameters | GFLOPs | Layers | Use Case |
|-------|------------|--------|--------|----------|
| yolo26n | 2.6M | 6.1 | 260 | Mobile, fast inference |
| yolo26s | 10.0M | 22.8 | 260 | Balanced |
| yolo26m | 21.9M | 75.4 | 280 | Higher accuracy |
| yolo26l | 26.3M | 93.8 | 392 | High accuracy |
| yolo26x | 59.0M | 209.5 | 392 | Maximum accuracy |

---

## 3. Backbone Architecture

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]       # P1/2
  - [-1, 1, Conv, [128, 3, 2]]      # P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]      # P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]      # P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]     # P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]]
  - [-1, 2, C2PSA, [1024]]          # PSA Attention
```

### Key Modules

| Module | Description |
|--------|-------------|
| **C3k2** | CSP-like block with configurable kernel size |
| **C2PSA** | Polarized Self-Attention module |
| **SPPF** | Spatial Pyramid Pooling - Fast |

---

## 4. Training Features

### 4.1 MuSGD Optimizer

Combines SGD with Muon Optimizer (Momentum-Unified).

- Origin: Inspired by Moonshot AI's Kimi K2 training
- Performs orthogonalization on gradient matrices
- Ultra-fast convergence without Warm-up

```python
# Usage
from ultralytics.optim import MuSGD
# Available optimizers: Adam, AdamW, SGD, MuSGD, etc.
```

### 4.2 STAL (Small-Target-Aware Label Assignment)

Solves gradient vanishing for small objects:
- Dynamically adjusts IoU threshold
- Smaller objects get lower thresholds

### 4.3 ProgLoss (Progressive Loss Balancing)

Dynamic loss weight balancing:
- **Early phase:** Focus on classification (semantic learning)
- **Late phase:** Focus on regression (geometric fine-tuning)

---

## 5. Multi-Task Support

YOLO26 natively supports five tasks:

| Task | Config File |
|------|-------------|
| Detection | yolo26.yaml |
| Segmentation | yolo26-seg.yaml |
| Pose | yolo26-pose.yaml |
| OBB | yolo26-obb.yaml |
| Classification | yolo26-cls.yaml |

---

## 6. Performance (COCO)

| Model | mAP (50-95) | CPU (ms) | T4 TensorRT (ms) |
|-------|-------------|----------|------------------|
| YOLOv8-n | 37.3% | 80.4 | 0.99 |
| YOLO11-n | 39.5% | 56.1 | 1.5 |
| **YOLO26-n** | **40.1%** | **38.9** | 1.7 |
| **YOLO26-m** | **53.1%** | 220.0 | 4.7 |
| **YOLO26-x** | **56.9%** | 525.8 | 11.8 |

---

## 7. Deployment

### Export Formats

| Format | File | Metadata |
|--------|------|----------|
| PyTorch | yolo26n.pt | Yes |
| ONNX | yolo26n.onnx | Yes |
| TensorRT | yolo26n.engine | Yes |
| CoreML | yolo26n.mlpackage | Yes |
| TFLite | yolo26n.tflite | Yes |
| OpenVINO | yolo26n_openvino_model/ | Yes |

### Export Advantages

- No NMS node needed in ONNX
- Direct TensorRT export (no plugin required)
- Clean INT8 (no DFL issues)
- Native CoreML support

---

## 8. Code Verification

| Feature | Status |
|---------|--------|
| NMS-Free (end2end: True) | Verified |
| DFL Removal (reg_max: 1) | Verified |
| C3k2 Module | Verified |
| C2PSA Attention | Verified |
| MuSGD Optimizer | Verified |
| Multi-task configs | Verified |
| Model parameters | Verified |

**Ultralytics Version:** 8.4.7

---

## References

- [Ultralytics YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
