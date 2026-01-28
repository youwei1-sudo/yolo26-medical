# YOLO26 vs YOLOv9 Evaluation

Model comparison for ImgAssist OCT cancer detection.

**Status:** Requires re-evaluation after preprocessing fix. See [PREPROCESSING.md](PREPROCESSING.md).

---

## Current Results (Pre-Fix)

### Validation Set

| Metric | YOLOv9-s | YOLO26s | YOLO26m |
|--------|----------|---------|---------|
| **mAP@0.5** | 0.766 | 0.759 | **0.781** |
| **mAP@0.5:0.95** | 0.326 | 0.394 | **0.422** |
| Precision | 0.747 | 0.769 | 0.761 |
| Recall | 0.689 | 0.672 | **0.693** |

### Test Set (PROBLEMATIC)

| Metric | YOLOv9-s | YOLO26s | YOLO26m |
|--------|----------|---------|---------|
| **mAP@0.5** | **0.607** | 0.247 | 0.165 |
| Precision | **0.658** | 0.279 | 0.205 |
| Recall | **0.559** | 0.407 | 0.396 |

**YOLO26 shows severe Val-Test gap due to preprocessing mismatch.**

---

## Configuration Comparison

### Training

| Parameter | YOLOv9-s | YOLO26m |
|-----------|----------|---------|
| Architecture | GELAN | C3k2 + C2PSA |
| Image Size | 320x672 (rect) | 768x768 (square) |
| Batch Size | 32 | 24 |
| Epochs | 109 | 130 |
| NMS | Post-processing | Native end-to-end |

### Augmentation

| Augmentation | YOLOv9-s | YOLO26m |
|--------------|----------|---------|
| Mosaic | 0.0 | 0.2 |
| Mixup | 0.0 | 0.1 |
| Copy-Paste | 0.0 | 0.3 |
| Flip LR | 0.113 | 0.5 |

---

## Architecture Differences

| Feature | YOLOv9-s | YOLO26m |
|---------|----------|---------|
| Backbone | GELAN (RepNCSPELAN4) | C3k2 + C2PSA |
| Head | Dual-branch (aux + main) | Single streamlined |
| NMS | Post-processing | Native end-to-end |
| DFL | reg_max=16 (softmax) | reg_max=1 (direct) |
| Parameters | ~9.6M | ~21.9M |
| GFLOPs | ~26 | ~75 |

---

## YOLOv9 Test Set Metrics (Reference)

From FP analysis at conf=0.001:

| Metric | Value |
|--------|-------|
| Total Patches | 17,785 |
| Positive Patches | 4,399 |
| Negative Patches | 13,386 |
| TP | 4,318 |
| FN | 81 |
| FP | 13,275 |
| TN | 111 |
| **Sensitivity** | **98.16%** |
| Specificity | 0.83% |

Note: Very low threshold (0.001) maximizes sensitivity.

---

## Deployment Considerations

### Export Complexity

| Format | YOLOv9-s | YOLO26m |
|--------|----------|---------|
| ONNX | Requires NMS node | Native, no NMS |
| TensorRT | Requires plugin | Direct export |
| INT8 | DFL may cause issues | Direct regression, clean |

### Latency Expectations

| Scenario | YOLOv9-s | YOLO26m |
|----------|----------|---------|
| GPU (A100) | ~5ms + NMS | ~7ms |
| CPU | ~80ms + NMS | ~220ms |

YOLO26 has deterministic latency (no NMS), but is larger.

---

## Required Re-evaluation

After preprocessing fix, re-run:

```bash
python scripts/compare_yolov9_yolo26.py \
    --yolov9-weights <yolov9_best.pt> \
    --yolo26-weights <yolo26_best.pt> \
    --data data/yaxis_squash_h320/data_2class.yaml \
    --split test \
    --conf 0.001 \
    --eval-mode patch
```

**Expected outcome:**
- YOLO26 Val-Test gap < 10%
- Comparable or better sensitivity than YOLOv9

---

## Model Selection Guidance

| Scenario | Recommended |
|----------|-------------|
| Maximum Accuracy | YOLO26m |
| Latency Critical | YOLOv9-s or YOLO26s |
| Edge Deployment | YOLO26 (NMS-free) |
| INT8 Quantization | YOLO26 (no DFL) |

---

## File Locations

### YOLOv9-s
- Weights: `/lambda/nfs/ImgAssistClone/imgAssist/modelGenerationTool/runs/train/PatientSplit_v12.1_unstretch_320x672_SGD/weights/best.pt`

### YOLO26m
- Weights: `runs/finetune/yolo26m_v2_optimized/weights/best.pt`
