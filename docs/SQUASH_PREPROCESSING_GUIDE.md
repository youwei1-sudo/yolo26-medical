# YOLO26 Y-Axis Squash Preprocessing Guide

## Overview

本指南说明如何将 YOLO26 的预处理与 YOLOv9 v12 对齐，以解决 Val-Test 性能差距问题。

## 问题背景

YOLO26 模型在验证集和测试集之间存在显著的性能差距：

| Model | Val mAP@0.5 | Test mAP@0.5 | Drop |
|-------|-------------|--------------|------|
| YOLO26s | 75.9% | 24.7% | -51.2% |
| YOLO26m | 78.1% | 16.5% | -61.6% |

### 根因：预处理方法不匹配

## 正确的预处理方法：Y-Axis Squash

### Pipeline

```
原图 (672×~420)
    ↓ Y-axis squash: cv2.resize(img, (672, 320))
最终 (672×320)
    ↓ 无 padding
```

### 关键特点

1. **宽度不变**: 保持原始宽度 672
2. **只压缩高度**: 从 ~420 压缩到 320
3. **无 padding**: 直接 resize，不需要 padding
4. **Labels 保持有效**: 归一化坐标在 squash 后仍然有效

### YOLOv9 v12 原始实现

代码位置: `/home/ubuntu/ImgAssistClone/imgAssist/modelGenerationTool/utils/dataloaders.py:103-134`

```python
def squash_image(img, labels, hyp):
    """
    Squash image Y-axis before letterboxing.
    """
    if not hyp.get('squash_enabled', False):
        return img, labels

    h0, w0 = img.shape[:2]
    target_h = hyp.get('squash_target_height', 188)  # 我们改为 320

    # Keep original width, only squash height
    squashed = cv2.resize(img, (w0, target_h), interpolation=cv2.INTER_LINEAR)

    # Labels are in NORMALIZED format (0-1) relative to image dimensions
    # After squashing, normalized coordinates remain VALID because:
    # - A point at y=0.5 (middle) of original is still at y=0.5 (middle) of squashed
    # - The squash is a uniform scaling that preserves relative positions
    # NO LABEL MODIFICATION NEEDED for normalized coords

    return squashed, labels
```

## YOLO26 实现

### 数据集生成

```bash
python scripts/convert_manifest_to_ultralytics.py \
    --yaxis-squash \
    --target-height 320
```

### YAxisSquashPreprocessor 类

代码位置: `scripts/convert_manifest_to_ultralytics.py:60-109`

```python
class YAxisSquashPreprocessor:
    """
    Y-Axis Squash preprocessor - keeps width, only compresses height.

    Pipeline (for 672x420 original → 672x320 target):
    1. Keep original width (672)
    2. Squash height to target (320)
    3. No padding needed
    """

    def __init__(self, target_height: int = 320):
        self.target_height = target_height

    def process(self, img, bboxes):
        h0, w0 = img.shape[:2]

        # Y-axis squash: keep width, only change height
        img_squashed = cv2.resize(img, (w0, self.target_height),
                                   interpolation=cv2.INTER_LINEAR)

        # Labels remain UNCHANGED - normalized coordinates are still valid
        return img_squashed, bboxes
```

## 为什么 Labels 不需要修改？

归一化坐标 (0-1) 在 Y-axis squash 后仍然有效：

```
原图 (672×420):
- 某个点在 x=336, y=210 (图像中心)
- 归一化坐标: x=0.5, y=0.5

Squash 后 (672×320):
- 该点变为 x=336, y=160 (依然是图像中心)
- 归一化坐标: x=336/672=0.5, y=160/320=0.5 ✓

结论: 归一化坐标不变！
```

## 之前的错误理解

### 错误 1: 使用 v12.1 的 letterbox 方法

```
❌ 错误 Pipeline:
原图 (672×420) → scale(0.76) → (512×320) → pad → (672×320)
```

问题：
- 等比缩放会把宽度从 672 缩小到 ~512
- 然后需要 padding 回 672
- 浪费了原始图像信息，增加了无用的 padding 区域

### 错误 2: 混淆 v12 和 v12.1

| 版本 | squash_enabled | imgsz | 方法 |
|------|----------------|-------|------|
| v12 | **true** | 640 | Y-axis squash to 188, then letterbox to 640×640 |
| v12.1 | false | [320, 672] | Rectangle letterbox (scale + pad) |

我们需要的是 **v12 的 Y-axis squash 方法**，但：
- 目标高度改为 320 (不是 188)
- 最终输出 672×320 (不需要后续 letterbox 到 640×640)

## 生成的数据集

**输出目录:** `data/yaxis_squash_h320/`

| Split | Images | 尺寸 |
|-------|--------|------|
| Train | 19,000 | 672×320 |
| Val | 2,275 | 672×320 |
| Test | 17,785 | 672×320 |
| **Total** | **39,060** | |

## 训练配置

```bash
python scripts/train_yolo26.py \
    --model yolo26s \
    --squash-data data/yaxis_squash_h320 \
    --imgsz 320 672 \
    --mosaic 0.0 \
    --mixup 0.0
```

或使用 Ultralytics 直接训练：

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

## 评估

```bash
python scripts/val_yolo26.py \
    --weights runs/finetune/yolo26s_yaxis_squash/weights/best.pt \
    --data data/yaxis_squash_h320/data_2class.yaml \
    --imgsz 320 672 \
    --split test \
    --eval-mode patch \
    --conf 0.001
```

## 文件参考

| 文件 | 说明 |
|------|------|
| `scripts/convert_manifest_to_ultralytics.py` | Y-axis squash 数据生成脚本 |
| `data/yaxis_squash_h320/` | 生成的 672×320 数据集 |
| `docs/Report.md` | 完整的问题分析和解决方案 |

## YOLOv9 代码参考

- squash_image(): `/imgAssist/modelGenerationTool/utils/dataloaders.py:103-134`
- letterbox_random_vpad(): `/imgAssist/modelGenerationTool/utils/dataloaders.py:137-193`
- LoadImagesAndLabels.__getitem__(): `/imgAssist/modelGenerationTool/utils/dataloaders.py:850-918`
