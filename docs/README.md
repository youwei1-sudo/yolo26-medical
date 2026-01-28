# YOLO26 Medical Imaging Experiment Documentation

YOLO26 fine-tuning for ImgAssist OCT cancer detection.

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [PREPROCESSING.md](PREPROCESSING.md) | **Critical Fix**: Y-axis squash preprocessing to fix Val-Test gap |
| [TRAINING.md](TRAINING.md) | Training configuration and results |
| [EVALUATION.md](EVALUATION.md) | YOLOv9 vs YOLO26 comparison |
| [ARCHITECTURE.md](ARCHITECTURE.md) | YOLO26 technical architecture |

---

## Quick Links

### Data

| Dataset | Path | Description |
|---------|------|-------------|
| **yaxis_squash_h320** | `data/yaxis_squash_h320/` | Correct preprocessing (672x320) |
| Train | 19,000 images | |
| Val | 2,275 images | |
| Test | 17,785 images | |

### Models

| Model | Path |
|-------|------|
| YOLO26s weights | `weights/yolo26s.pt` |
| YOLO26m weights | `weights/yolo26m.pt` |
| Best checkpoint | `runs/finetune/*/weights/best.pt` |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/convert_manifest_to_ultralytics.py` | Data conversion with Y-axis squash |
| `scripts/train_yolo26.py` | Fine-tuning |
| `scripts/val_yolo26.py` | Validation |

---

## Current Status

**Preprocessing Fix Complete** - See [PREPROCESSING.md](PREPROCESSING.md)

- Generated correct Y-axis squash dataset (672x320)
- Next: Retrain with correct preprocessing
- Target: Close Val-Test gap from -51%/-61% to <10%

---

## Key Insight: Y-Axis Squash

The critical fix is using the correct preprocessing method:

```
Original (672x~420)
    | cv2.resize(img, (672, 320))  # Keep width, compress height only
Final (672x320)
```

This matches YOLOv9 v12's preprocessing approach.

See [PREPROCESSING.md](PREPROCESSING.md) for full details.
