# YOLO26 Data Flow & Pipeline

Complete data flow from raw data to inference.

---

## 1. Data Requirements

### Input Format: JSON Manifest

Original data is in JSON manifest format from ImgAssist.

**Manifest structure:**
```json
{
  "items": {
    "patient_id": {
      "images": [
        {
          "image_path": "/path/to/image.png",
          "bboxes": [
            {"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.1}
          ]
        }
      ]
    }
  }
}
```

**Manifest locations:**
```
/datasets/ImgAssist_Data_10142025/SEL_OTIS_2class_PatientSplit_Manifests_v12/
├── train_manifest.json
├── validation_manifest.json
└── test_manifest.json
```

### Output Format: Ultralytics YOLO

**Conversion script:** `scripts/convert_manifest_to_ultralytics.py`

**Key functions:**
| Function | Line | Purpose |
|----------|------|---------|
| `load_manifest()` | 41-44 | Load JSON manifest |
| `create_label_content()` | 47-57 | Convert to YOLO format |
| `YAxisSquashPreprocessor` | 60-109 | Y-axis squash preprocessing |
| `process_manifest()` | 130-263 | Process entire manifest |

**Output structure:**
```
data/yaxis_squash_h320/
├── train.txt              # Image path list
├── val.txt
├── test.txt
├── data_2class.yaml       # Dataset config
├── images/{train,val,test}/*.png   # Preprocessed images (672x320)
└── labels/{train,val,test}/*.txt   # YOLO format labels
```

**YOLO label format:** `class_id x_center y_center width height` (normalized 0-1)

---

## 2. Preprocessing

### Option A: Y-Axis Squash (Recommended)

**Code:** `scripts/convert_manifest_to_ultralytics.py:60-109`

```python
class YAxisSquashPreprocessor:
    def process(self, img, bboxes):
        h0, w0 = img.shape[:2]
        # Keep width, squash height only
        img_squashed = cv2.resize(img, (w0, self.target_height))
        # Labels unchanged - normalized coords still valid
        return img_squashed, bboxes
```

**Pipeline:**
```
Original (672x~420) → cv2.resize(672, 320) → Final (672x320)
```

**Command:**
```bash
python scripts/convert_manifest_to_ultralytics.py --yaxis-squash --target-height 320
```

### Option B: Rectangle Letterbox

**Code:** `scripts/squash_transform.py:30-149`

```python
class RectangleTransform:
    def __call__(self, img, labels):
        # Scale to fit target while maintaining aspect ratio
        scale = min(target_w/w, target_h/h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Pad to target size
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2

        # Transform labels
        x_new = (x * new_w + pad_left) / target_width
        y_new = (y * new_h + pad_top) / target_height
```

---

## 3. Data Augmentation

### Config File: `configs/hyp/hyp_yolo26_medical.yaml`

**Medical imaging strategy (conservative):**

```yaml
# Geometric - lines 19-27
degrees: 0.0          # No rotation (OCT fixed orientation)
translate: 0.014      # 1.4% translation
scale: 0.42           # 42% scale
shear: 0.0            # No shear
flipud: 0.0           # No vertical flip (anatomy matters)
fliplr: 0.113         # 11.3% horizontal flip

# Color - lines 15-17
hsv_h: 0.033          # Hue
hsv_s: 0.098          # Saturation
hsv_v: 0.038          # Value

# Advanced - lines 28-30
mosaic: 0.0           # Disabled (preserve spatial relationships)
mixup: 0.091
copy_paste: 0.24
```

**Why conservative?**
- OCT images have fixed orientation → no rotation
- Vertical position matters clinically → no vertical flip
- Mosaic breaks spatial context → disabled

### Augmentation in Training

**Code:** `scripts/train_yolo26.py:178-191`

```python
train_args = {
    # Augmentation params passed to Ultralytics
    'mosaic': args.mosaic,
    'mixup': args.mixup,
    'copy_paste': args.copy_paste,
    'degrees': args.degrees,
    'translate': args.translate,
    'scale': args.scale,
    'flipud': args.flipud,
    'fliplr': args.fliplr,
    'hsv_h': args.hsv_h,
    'hsv_s': args.hsv_s,
    'hsv_v': args.hsv_v,
}
```

---

## 4. Training

### Script: `scripts/train_yolo26.py`

**Key code locations:**
| Function | Lines | Purpose |
|----------|-------|---------|
| Argument parsing | 264-369 | CLI arguments |
| Model loading | 141-155 | Load YOLO26 weights |
| Training config | 157-214 | Build train_args dict |
| Execute training | 248 | `model.train(**train_args)` |
| Results output | 250-259 | Print and save results |

### Training Parameters

```python
train_args = {
    'epochs': 200,
    'imgsz': [320, 672],      # height, width
    'batch': 36,
    'lr0': 0.00173,
    'lrf': 0.021,
    'momentum': 0.895,
    'weight_decay': 0.000446,
    'warmup_epochs': 4.8,
    'box': 7.2,               # Box loss weight (prioritize localization)
    'cls': 0.31,              # Classification weight
    'patience': 15,           # Early stopping
    'amp': True,              # Mixed precision
}
```

### Training Command

```bash
# Using script
python scripts/train_yolo26.py \
    --model yolo26s \
    --data data/yaxis_squash_h320/data_2class.yaml \
    --imgsz 320 672 \
    --epochs 200 \
    --batch 36 \
    --mosaic 0.0

# Or using shell script
./run_training_yolo26.sh yolo26s
```

### Output

```
runs/finetune/{name}/
├── weights/
│   ├── best.pt          # Best validation mAP
│   └── last.pt          # Final epoch
├── results.csv          # Epoch-by-epoch metrics
├── args.yaml            # Training arguments
└── *.png                # Training curves
```

---

## 5. Inference / Validation

### Script: `scripts/val_yolo26.py`

### Two Evaluation Modes

| Mode | Code Lines | Purpose |
|------|------------|---------|
| Box-Level | 385-446 | Standard detection metrics (mAP) |
| Patch-Level | 240-351 | Medical imaging (Sensitivity/Specificity) |

### Box-Level Validation

```bash
python scripts/val_yolo26.py \
    --weights runs/finetune/yolo26s_yaxis_squash/weights/best.pt \
    --data data/yaxis_squash_h320/data_2class.yaml \
    --split test \
    --conf 0.25
```

**Output metrics:**
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall

### Patch-Level Validation (Medical Key Metrics)

```bash
python scripts/val_yolo26.py \
    --weights runs/finetune/yolo26s_yaxis_squash/weights/best.pt \
    --data data/yaxis_squash_h320/data_2class.yaml \
    --split test \
    --eval-mode patch \
    --conf 0.001 \
    --labels-dir data/yaxis_squash_h320/labels/test
```

**Patch-level logic (`val_yolo26.py:134-212`):**
```python
# Each image is one sample
for img_path in test_images:
    has_detection = any(pred.conf >= conf for pred in predictions)
    is_positive = has_ground_truth_bbox(img_path)

    if is_positive and has_detection: TP += 1
    elif is_positive and not has_detection: FN += 1  # Missed cancer!
    elif not is_positive and has_detection: FP += 1
    else: TN += 1

sensitivity = TP / (TP + FN)  # Critical: don't miss cancer
specificity = TN / (TN + FP)
f2_score = 5 * P * R / (4 * P + R)  # Recall weighted 4x
```

**Output metrics:**
- **Sensitivity (敏感度)**: TP/(TP+FN) - Most important, don't miss cancer
- **Specificity (特异度)**: TN/(TN+FP)
- **F2 Score**: Recall weighted 4x higher than precision

---

## 6. Complete Workflow

```bash
# 1. Data preparation (with Y-axis squash)
python scripts/convert_manifest_to_ultralytics.py \
    --yaxis-squash \
    --target-height 320

# 2. Training
python scripts/train_yolo26.py \
    --model yolo26s \
    --data data/yaxis_squash_h320/data_2class.yaml \
    --imgsz 320 672 \
    --epochs 200 \
    --mosaic 0.0

# 3. Box-level validation
python scripts/val_yolo26.py \
    --weights runs/finetune/*/weights/best.pt \
    --split test \
    --conf 0.25

# 4. Patch-level evaluation (medical metrics)
python scripts/val_yolo26.py \
    --weights runs/finetune/*/weights/best.pt \
    --eval-mode patch \
    --conf 0.001

# 5. Compare with YOLOv9
python scripts/compare_yolov9_yolo26.py \
    --yolov9-weights /path/to/yolov9.pt \
    --yolo26-weights runs/finetune/*/weights/best.pt \
    --eval-mode patch
```

---

## Code Reference Summary

| File | Key Lines | Purpose |
|------|-----------|---------|
| `convert_manifest_to_ultralytics.py` | 60-109 | YAxisSquashPreprocessor |
| `convert_manifest_to_ultralytics.py` | 130-263 | process_manifest() |
| `squash_transform.py` | 30-149 | RectangleTransform |
| `train_yolo26.py` | 141-155 | Model loading |
| `train_yolo26.py` | 157-214 | Training config |
| `train_yolo26.py` | 248 | Execute training |
| `val_yolo26.py` | 134-212 | Patch metrics calculation |
| `val_yolo26.py` | 240-351 | Patch-level evaluation |
| `val_yolo26.py` | 385-446 | Box-level evaluation |
| `hyp_yolo26_medical.yaml` | 15-30 | Augmentation config |
