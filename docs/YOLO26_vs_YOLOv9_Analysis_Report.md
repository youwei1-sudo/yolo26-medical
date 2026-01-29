# YOLO26 vs YOLOv9 Comparative Analysis Report

**Version**: 1.0
**Date**: 2025-01-29
**Status**: Final Report

---

## Executive Summary

This report provides a comparative analysis of YOLO26s and YOLOv9 v12.1 performance on the ImgAssist OCT cancer detection task.

### Key Conclusions

| Dimension | Better Model | Details |
|-----------|--------------|---------|
| Box-level mAP | **YOLOv9** | 76.6% vs 67.6% (+9.0%) |
| Patch-level Sensitivity | **YOLO26** | 98.82% vs 98.16% (+0.66%) |
| Patch-level Specificity | **YOLO26** | 8.34% vs 0.83% (+7.51%) |
| Val-Test Stability | **YOLOv9** | Smaller gap, better generalization |

**Current Recommendation**: Continue using YOLOv9; defer YOLO26 model decision.

---

## 1. Model Configuration Comparison

| Configuration | YOLOv9 v12.1 | YOLO26s |
|---------------|--------------|---------|
| **Architecture** | GELAN | C3k2 + C2PSA |
| **Parameters** | ~9.6M | ~10M |
| **Input Size** | 320x672 | 672 (rect=True) |
| **Preprocessing** | Rectangle letterbox (unstretch) | Y-axis squash to 320 |
| **Optimizer** | SGD | SGD |
| **NMS** | Post-processing NMS | Native end-to-end (NMS-free) |
| **Quantization** | Standard | INT8 friendly |

### 1.1 NMS vs NMS-free Explanation

**NMS (Non-Maximum Suppression)** is a post-processing step in object detection used to filter overlapping detection boxes.

#### YOLOv9 - Post-processing NMS

```
Model Output → Many candidate boxes (thousands) → NMS Post-processing → Final detections
                                                        ↑
                                                  Requires hyperparameters:
                                                  - iou_thres (overlap threshold)
                                                  - max_det (max boxes per image)
```

**Characteristics**:
- Requires additional computation step
- Has hyperparameters to tune
- Latency varies (more candidates = slower NMS)

#### YOLO26 - NMS-free (End-to-end)

```
Model Output → Directly final detection boxes (no post-processing needed)
                    ↑
              Model internally learned to suppress duplicate boxes
```

**Advantages**:
- No need to set `max_det`, `iou_thres` hyperparameters
- Deterministic, stable inference latency
- Simpler deployment (cleaner ONNX/TensorRT export)
- More INT8 quantization friendly

#### Comparison Summary

| Aspect | YOLOv9 (Post-processing NMS) | YOLO26 (NMS-free) |
|--------|------------------------------|-------------------|
| **Output** | Raw candidates → need filtering | Direct final boxes |
| **Post-processing** | Requires NMS step | Not needed |
| **Hyperparameters** | Need to tune `max_det`, `iou_thres` | None required |
| **Latency** | Variable (depends on box count) | Deterministic, stable |
| **Deployment** | Requires NMS plugin | Direct export |

---

## 2. Evaluation Modes

This report uses two evaluation modes targeting different clinical needs:

### 2.1 Box-Level - Standard Object Detection

**Definition**: Each detection box is matched with Ground Truth boxes by IoU

**Computed Metrics**:
| Metric | Formula | Description |
|--------|---------|-------------|
| mAP@0.5 | - | Average Precision at IoU=0.5 |
| mAP@0.5:0.95 | - | Average across IoU 0.5 to 0.95 |
| Precision | TP / (TP + FP) | Detection box accuracy |
| Recall | TP / (TP + FN) | Detection box recall |

**Use Case**: Standard detection tasks, evaluating bounding box localization accuracy

**conf threshold**: 0.25 (standard)

### 2.2 Patch-Level (Image-Level) - Medical Imaging Specific

**Definition**: Each image is treated as one sample, determining if any detection exists

**Core Logic**:
```
For each image:
  has_gt = image has GT boxes (has cancer)
  has_pred = image has prediction boxes (conf >= threshold)

  if has_gt and has_pred:      TP  # Has cancer, detected
  elif not has_gt and not has_pred: TN  # No cancer, no false alarm
  elif not has_gt and has_pred:     FP  # No cancer, but false alarm
  else:                             FN  # Has cancer, but missed!
```

**Computed Metrics**:
| Metric | Formula | Significance |
|--------|---------|--------------|
| **Sensitivity** | TP / (TP + FN) | Miss rate (**Most important!**) |
| Specificity | TN / (TN + FP) | False alarm rate |
| Precision | TP / (TP + FP) | Positive predictive value |
| F2 Score | 5PR / (4P + R) | Recall weighted 4x |

**Use Case**: Medical imaging screening, directly answers "Does this image have suspicious areas?"

**conf threshold**: 0.001 (very low, ensuring high sensitivity)

### 2.3 Comparison of Two Modes

| Aspect | Box-Level | Patch-Level |
|--------|-----------|-------------|
| **Evaluation Unit** | Each detection box | Each image |
| **conf threshold** | 0.25 (standard) | 0.001 (high sensitivity) |
| **Primary Metric** | mAP@0.5 | Sensitivity |
| **Clinical Meaning** | Localization accuracy | Miss rate |
| **Use Case** | Precise localization | Medical screening |

### 2.4 Why Use Patch-Level for Medical Imaging?

In medical imaging, **missing a case (FN) is far worse than a false alarm (FP)**:
- Missing cancer → Delayed treatment, potentially fatal
- False alarm → Additional follow-up, but no harm

Patch-level evaluation directly answers the clinical question:
> "Does this image have suspicious areas requiring doctor's attention?" (binary classification)

Rather than:
> "Is this detection box accurately drawn?" (box matching)

### 2.5 Confidence Threshold Selection Logic

#### 2.5.1 Threshold Selection Principles

Based on **Funneling Evaluation Logic** (ref: ImgAssist 3.0 Model Selection Framework v3.1):

| Evaluation Purpose | Recommended Threshold | Rationale |
|--------------------|----------------------|-----------|
| **High-Sensitivity Screening** | conf=0.001 | Minimize missed cases (FN), accept higher FP |
| **Balanced Detection** | conf=0.25 | Standard object detection, balance P/R |
| **High-Precision Detection** | conf=0.50+ | Minimize FP, accept higher FN |

#### 2.5.2 Thresholds Used in This Report

| Evaluation Mode | Threshold | Selection Rationale |
|-----------------|-----------|---------------------|
| **Box-Level (Val)** | 0.25 | Standard YOLO validation config, evaluates localization accuracy |
| **Patch-Level (Test)** | 0.001 | Clinical screening scenario, **Hard on Delivery** principle |

#### 2.5.3 Threshold Alignment with Clinical Goals

Reference: PCCP Regulatory Requirements

```
┌─────────────────────────────────────────────────────────────────┐
│            Threshold Selection → Clinical Goal Alignment         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  conf=0.001 (Patch-Level)                                       │
│  ├── Goal: Sensitivity ≥ 98% (no missed cases)                 │
│  ├── Aligns with: PCCP Margin Recall ≥ 61% Hard Gate           │
│  └── Principle: "Hard on Delivery" - ensure safety floor        │
│                                                                 │
│  conf=0.25 (Box-Level)                                          │
│  ├── Goal: Balance Precision/Recall                            │
│  ├── Aligns with: Signal quality assessment (not Pass/Fail)    │
│  └── Principle: "Weighted on Precision" - optimize localization │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.5.4 Expected Behavior at Different Thresholds

| Threshold | Sensitivity | Specificity | FP Count | Use Case |
|-----------|-------------|-------------|----------|----------|
| 0.001 | **Highest** (~98%+) | Lowest | Most | Initial screening, no misses |
| 0.1 | High | Low | Many | Sensitive screening |
| 0.25 | Medium | Medium | Medium | Standard detection |
| 0.5 | Lower | Higher | Fewer | High-confidence detection |
| 0.75 | Low | High | Few | Confirmation diagnosis |

### 2.6 Fair Comparison Framework

#### 2.6.1 Comparison Principles

Based on **Signal vs Success** principle:

```
Signal (L1 Box-Level) ≠ Success
Success = Delivery (Patch-Level) + Clinical Validation
```

**Fair Comparison Requirements**:
1. **Same Dataset**: Use identical Test Set (17,785 patches)
2. **Same Threshold**: Patch-Level uniformly uses conf=0.001
3. **Same Evaluation Code**: Identical metric calculation logic
4. **Same Preprocessing**: Consider preprocessing differences when comparing

#### 2.6.2 Comparison Configuration in This Report

| Configuration | YOLOv9 v12.1 | YOLO26s | Comparison Fairness |
|---------------|--------------|---------|---------------------|
| **Test Dataset** | 17,785 patches | 17,785 patches | Same |
| **Patch conf** | 0.001 | 0.001 | Same |
| **Preprocessing** | Rectangle letterbox | Y-axis squash | **Different** (note) |
| **Evaluation Code** | Unified patch_metrics | Unified patch_metrics | Same |

#### 2.6.3 Comparison Limitations

This report's comparison has the following limitations:

| Limitation | Impact | Description |
|------------|--------|-------------|
| **Different Preprocessing** | Medium | YOLOv9 uses letterbox, YOLO26 uses squash |
| **Different Training Config** | Low | Both use SGD, minimal differences |
| **Val-Test Gap** | High | YOLO26 has -38.6% gap, generalization needs verification |

---

## 3. Performance Comparison

### 3.1 Validation Set - Box-Level

| Metric | YOLOv9 v12.1 | YOLO26s | Difference |
|--------|--------------|---------|------------|
| **mAP@0.5** | 76.6% | 67.6% | **-9.0%** |

### 3.2 Test Set - Box-Level

| Metric | YOLOv9 v12.1 | YOLO26s | Difference |
|--------|--------------|---------|------------|
| **mAP@0.5** | - | 29.0% | - |

**Val-Test Gap**:
- YOLO26s: **-38.6%** (67.6% → 29.0%)
- YOLOv9 v12.1: Smaller gap

### 3.3 Test Set - Patch-Level (conf=0.001)

| Metric | YOLOv9 v12.1 | YOLO26s | Difference |
|--------|--------------|---------|------------|
| **Total Patches** | 17,785 | 17,785 | - |
| **TP (True Positive)** | 4,318 | 4,347 | +29 |
| **FN (False Negative)** | 81 | 52 | **-29** |
| **FP (False Positive)** | 13,275 | 12,270 | **-1,005** |
| **TN (True Negative)** | 111 | 1,116 | +1,005 |
| **Sensitivity** | 98.16% | **98.82%** | **+0.66%** |
| **Specificity** | 0.83% | **8.34%** | **+7.51%** |

---

## 4. Key Findings

### 4.1 Sensitivity (Recall)

- **YOLO26 slightly better**: 98.82% vs 98.16%
- **Fewer missed cases**: FN reduced from 81 to 52 (29 fewer)
- **Clinical significance**: Higher sensitivity means fewer missed cancer cases

### 4.2 Specificity

- **YOLO26 significantly better**: 8.34% vs 0.83% (~**10x** improvement)
- **Fewer false alarms**: FP reduced from 13,275 to 12,270 (1,005 fewer)
- **Clinical significance**: Higher specificity means fewer unnecessary follow-ups

### 4.3 Val-Test Gap

- **YOLO26 issue**: -38.6% Val-Test gap remains
- **Performance**: Validation 67.6% → Test 29.0%
- **Possible causes**: Overfitting or data distribution mismatch
- **YOLOv9 more stable**: Smaller Val-Test gap, better generalization

### 4.4 Box-level mAP

- **YOLOv9 higher**: 76.6% vs 67.6%
- **Gap**: 9.0 percentage points
- **Significance**: YOLOv9 performs better in bounding box localization

---

## 5. Meeting Decision Points

Based on team meeting discussions, the following decisions were made:

### 5.1 Inference Performance

- **YOLO26 is faster**
- **Conclusion**: Inference time is not a deciding factor currently; TensorRT + NVIDIA GPU provides sufficient acceleration

### 5.2 Commercial Considerations

- **Ultralytics license fee**: ~$15,000/year (Enterprise)
- **Long-term risk**: If company moves away from Ultralytics models, license fees continue for devices running old versions
- **Deployment constraints**: Manual device updates, no OTA capability

### 5.3 Work Priority Shift

- **Current bottleneck**: Data quality (CAM generation), not the model itself
- **Decision**: Defer YOLO26 vs YOLOv9 model decision
- **Next step**: Focus on CAM generation to obtain high-quality "Golden Dataset"

---

## 6. Comprehensive Evaluation

### 6.1 YOLO26 Advantages

| Advantage | Description |
|-----------|-------------|
| **NMS-free** | Native end-to-end, no post-processing NMS needed |
| **INT8 friendly** | Easier quantized deployment |
| **Higher Sensitivity** | 98.82% vs 98.16%, 29 fewer missed cases |
| **Higher Specificity** | 8.34% vs 0.83%, 1,005 fewer false alarms |
| **Faster inference** | More efficient model architecture |

### 6.2 YOLO26 Disadvantages

| Disadvantage | Description |
|--------------|-------------|
| **Val-Test Gap** | -38.6%, generalization needs verification |
| **Lower Box mAP** | 67.6% vs 76.6% |
| **License required** | Ultralytics Enterprise ~$15k/year |
| **Deployment constraints** | Manual device updates, ongoing license fees |

### 6.3 YOLOv9 Advantages

| Advantage | Description |
|-----------|-------------|
| **Higher Box mAP** | 76.6%, more accurate localization |
| **More stable generalization** | Smaller Val-Test gap |
| **MIT license available** | No long-term license burden |

### 6.4 YOLOv9 Disadvantages

| Disadvantage | Description |
|--------------|-------------|
| **Lower Sensitivity** | 98.16%, 29 more missed cases |
| **Lower Specificity** | 0.83%, more false alarms |
| **Requires NMS post-processing** | Additional inference overhead |

---

## 7. Conclusions and Recommendations

### 7.1 Current Recommendation

**Continue using YOLOv9 v12.1 as the primary model**

Rationale:
1. Higher Box-level mAP (76.6%)
2. More stable Val-Test generalization
3. MIT license version available, no long-term license fees

### 7.2 Outstanding YOLO26 Issues

1. **Val-Test Gap**: Need to resolve -38.6% performance gap
2. **Commercial license**: Evaluate long-term cost impact
3. **Data quality**: Prioritize improving CAM generation

### 7.3 Next Steps

1. **Short-term**: Focus on CAM generation and data quality
2. **Medium-term**: Re-evaluate models after obtaining Golden Dataset
3. **Long-term**: Make final decision based on business needs and licensing costs

---

## Appendix: Confusion Matrices

### YOLOv9 v12.1

```
                 Predicted
              Positive  Negative
Actual  Pos    4,318      81      (Total: 4,399)
        Neg   13,275     111      (Total: 13,386)

Sensitivity = 4318 / 4399 = 98.16%
Specificity = 111 / 13386 = 0.83%
```

### YOLO26s

```
                 Predicted
              Positive  Negative
Actual  Pos    4,347      52      (Total: 4,399)
        Neg   12,270   1,116      (Total: 13,386)

Sensitivity = 4347 / 4399 = 98.82%
Specificity = 1116 / 13386 = 8.34%
```

---

## Related Documentation

- [Meeting Decisions](zh/会议决定.md) - Meeting discussion details (Chinese)
- [EVALUATION.md](EVALUATION.md) - Detailed evaluation methods
- [PREPROCESSING.md](PREPROCESSING.md) - Preprocessing method comparison
