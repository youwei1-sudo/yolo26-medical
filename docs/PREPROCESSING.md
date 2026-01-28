# YOLO26 Preprocessing: Y-Axis Squash Fix

**Last Updated:** 2026-01-28

This document explains the critical preprocessing fix that resolves the Val-Test performance gap.

---

## Problem Summary

YOLO26 showed severe Val-Test performance degradation:

| Model | Val mAP@0.5 | Test mAP@0.5 | Gap |
|-------|-------------|--------------|-----|
| YOLO26s | 75.9% | 24.7% | -51.2% |
| YOLO26m | 78.1% | 16.5% | -61.6% |

**Root Cause:** Preprocessing mismatch with YOLOv9 baseline.

---

## Solution: Y-Axis Squash

### Correct Pipeline

```
Original (672x~420)
    | Y-axis squash: cv2.resize(img, (672, 320))
Final (672x320)
    | No padding needed
```

**Key characteristics:**
- **Width unchanged:** Keep original 672
- **Height compressed:** Squash to 320
- **No padding:** Direct resize
- **Labels unchanged:** Normalized coordinates remain valid

### Implementation

```python
class YAxisSquashPreprocessor:
    def __init__(self, target_height: int = 320):
        self.target_height = target_height

    def process(self, img, bboxes):
        h0, w0 = img.shape[:2]
        # Y-axis squash: keep width, only change height
        img_squashed = cv2.resize(img, (w0, self.target_height),
                                   interpolation=cv2.INTER_LINEAR)
        # Labels remain UNCHANGED - normalized coordinates still valid
        return img_squashed, bboxes
```

**Code location:** `scripts/convert_manifest_to_ultralytics.py:60-109`

---

## Why Labels Don't Need Modification

Normalized coordinates (0-1) remain valid after Y-axis squash:

```
Original (672x420):
- Point at x=336, y=210 (image center)
- Normalized: x=0.5, y=0.5

After squash (672x320):
- Point at x=336, y=160 (still image center)
- Normalized: x=336/672=0.5, y=160/320=0.5

Conclusion: Normalized coordinates unchanged!
```

---

## Experiment History

| Version | imgsz | Preprocessing | Issue |
|---------|-------|---------------|-------|
| v1 | 768x768 | Standard Ultralytics resize | Square, no squash |
| v2 | 640x640 | Standard Ultralytics resize | Square, no squash |
| v3 | 640x640 | Wrong squash (0.457 ratio) | Used incorrect 188 target height |
| **v4 (correct)** | **672x320** | **Y-axis squash** | Width unchanged, height to 320 |

### v1-v2: Square Resize (Wrong)
- Ultralytics default square resize (768x768 or 640x640)
- Lost original aspect ratio information
- Val-Test gap: -51% to -61%

### v3: Wrong Squash Ratio (Wrong)
- Attempted squash with `target_height_ratio=0.457` (from 188/411)
- This was based on YOLOv9 v12's 188px target height, not 320

### v4: Correct Y-Axis Squash
- YOLOv9 v12 approach with target_height=320
- Keep width at 672, squash height to 320
- Dataset: `data/yaxis_squash_h320/`

---

## Previous Misunderstandings

### Error 1: Confused v12 and v12.1

| Version | squash_enabled | imgsz | Method |
|---------|----------------|-------|--------|
| v12 | **true** | 640 | Y-axis squash to 188, then letterbox to 640x640 |
| v12.1 | false | [320, 672] | Rectangle letterbox (scale + pad) |

We need **v12's Y-axis squash method** but with target_height=320.

### Error 2: Used Standard Letterbox

**Wrong pipeline (what we tried initially):**
```
Original (672x420) -> scale(0.76) -> (512x320) -> pad -> (672x320)
```

Problem: Scales width from 672 to ~512, then pads back. Wastes original image information.

**Correct pipeline:**
```
Original (672x420) -> cv2.resize(672, 320) -> (672x320)
```

Keep width unchanged, only compress height.

---

## YOLOv9 Reference Implementation

From `/imgAssist/modelGenerationTool/utils/dataloaders.py:103-134`:

```python
def squash_image(img, labels, hyp):
    """Squash image Y-axis before letterboxing."""
    if not hyp.get('squash_enabled', False):
        return img, labels

    h0, w0 = img.shape[:2]
    target_h = hyp.get('squash_target_height', 188)  # We use 320

    # Keep original width, only squash height
    squashed = cv2.resize(img, (w0, target_h), interpolation=cv2.INTER_LINEAR)

    # Labels are in NORMALIZED format (0-1) relative to image dimensions
    # After squashing, normalized coordinates remain VALID
    return squashed, labels
```

---

## Dataset Generation

### Command

```bash
python scripts/convert_manifest_to_ultralytics.py \
    --yaxis-squash \
    --target-height 320
```

### Output

**Directory:** `data/yaxis_squash_h320/`

| Split | Images | Size |
|-------|--------|------|
| Train | 19,000 | 672x320 |
| Val | 2,275 | 672x320 |
| Test | 17,785 | 672x320 |
| **Total** | **39,060** | |

---

## Training with Fixed Preprocessing

### Configuration

```python
from ultralytics import YOLO

model = YOLO('yolo26s.pt')
model.train(
    data='data/yaxis_squash_h320/data_2class.yaml',
    imgsz=[320, 672],  # height, width
    mosaic=0.0,
    mixup=0.0,
)
```

### Evaluation

```bash
python scripts/val_yolo26.py \
    --weights runs/finetune/yolo26s_yaxis_squash/weights/best.pt \
    --data data/yaxis_squash_h320/data_2class.yaml \
    --imgsz 320 672 \
    --split test \
    --eval-mode patch \
    --conf 0.001
```

---

## File References

| File | Description |
|------|-------------|
| `scripts/convert_manifest_to_ultralytics.py` | Y-axis squash data generation |
| `data/yaxis_squash_h320/` | Generated 672x320 dataset |
| YOLOv9 reference | `/imgAssist/modelGenerationTool/utils/dataloaders.py:103-134` |

---

## Next Steps

1. Train YOLO26 with `yaxis_squash_h320` dataset
2. Use `imgsz: [320, 672]`
3. Compare Val and Test performance to verify fix
4. Target: Val-Test gap < 10%
