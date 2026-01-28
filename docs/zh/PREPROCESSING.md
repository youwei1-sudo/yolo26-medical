# YOLO26 预处理：Y轴压缩修复

**更新日期:** 2026-01-28

本文档说明解决 Val-Test 性能差距的关键预处理修复。

---

## 问题概述

YOLO26 出现严重的 Val-Test 性能下降：

| 模型 | Val mAP@0.5 | Test mAP@0.5 | 差距 |
|------|-------------|--------------|------|
| YOLO26s | 75.9% | 24.7% | -51.2% |
| YOLO26m | 78.1% | 16.5% | -61.6% |

**根本原因：** 与 YOLOv9 基线的预处理方法不匹配。

---

## 解决方案：Y轴压缩

### 正确的处理流程

```
原图 (672x~420)
    | Y轴压缩: cv2.resize(img, (672, 320))
最终 (672x320)
    | 无需 padding
```

**关键特点：**
- **宽度不变：** 保持原始 672
- **高度压缩：** 压缩到 320
- **无 padding：** 直接 resize
- **标签不变：** 归一化坐标仍然有效

### 实现代码

```python
class YAxisSquashPreprocessor:
    def __init__(self, target_height: int = 320):
        self.target_height = target_height

    def process(self, img, bboxes):
        h0, w0 = img.shape[:2]
        # Y轴压缩：保持宽度，只改变高度
        img_squashed = cv2.resize(img, (w0, self.target_height),
                                   interpolation=cv2.INTER_LINEAR)
        # 标签不变 - 归一化坐标仍然有效
        return img_squashed, bboxes
```

**代码位置：** `scripts/convert_manifest_to_ultralytics.py:60-109`

---

## 为什么标签不需要修改？

归一化坐标 (0-1) 在 Y轴压缩后仍然有效：

```
原图 (672x420):
- 某点位于 x=336, y=210 (图像中心)
- 归一化坐标: x=0.5, y=0.5

压缩后 (672x320):
- 该点位于 x=336, y=160 (仍是图像中心)
- 归一化坐标: x=336/672=0.5, y=160/320=0.5

结论：归一化坐标不变！
```

---

## 实验历史

| 版本 | imgsz | 预处理方法 | 问题 |
|------|-------|-----------|------|
| v1 | 768x768 | 标准 Ultralytics resize | 正方形，无压缩 |
| v2 | 640x640 | 标准 Ultralytics resize | 正方形，无压缩 |
| v3 | 640x640 | 错误的压缩 (0.457 ratio) | 使用了错误的 188 目标高度 |
| **v4 (正确)** | **672x320** | **Y轴压缩** | 宽度不变，高度压缩到 320 |

### v1-v2: 正方形 Resize (错误)
- Ultralytics 默认正方形 resize (768x768 或 640x640)
- 丢失原始宽高比信息
- Val-Test 差距: -51% 到 -61%

### v3: 错误的压缩比例 (错误)
- 使用了 `target_height_ratio=0.457` (来自 188/411)
- 这是基于 YOLOv9 v12 的 188px 目标高度，而不是 320

### v4: 正确的 Y轴压缩
- YOLOv9 v12 方法，但 target_height=320
- 保持宽度 672，压缩高度到 320
- 数据集: `data/yaxis_squash_h320/`

---

## 之前的错误理解

### 错误 1: 混淆 v12 和 v12.1

| 版本 | squash_enabled | imgsz | 方法 |
|------|----------------|-------|------|
| v12 | **true** | 640 | Y轴压缩到 188，然后 letterbox 到 640x640 |
| v12.1 | false | [320, 672] | 矩形 letterbox (缩放 + padding) |

我们需要的是 **v12 的 Y轴压缩方法**，但 target_height=320。

### 错误 2: 使用了标准 Letterbox

**错误流程 (之前尝试的):**
```
原图 (672x420) -> scale(0.76) -> (512x320) -> pad -> (672x320)
```

问题：将宽度从 672 缩放到 ~512，然后再 padding 回来。浪费了原始图像信息。

**正确流程:**
```
原图 (672x420) -> cv2.resize(672, 320) -> (672x320)
```

保持宽度不变，只压缩高度。

---

## YOLOv9 参考实现

来自 `/imgAssist/modelGenerationTool/utils/dataloaders.py:103-134`:

```python
def squash_image(img, labels, hyp):
    """在 letterboxing 之前压缩图像 Y轴。"""
    if not hyp.get('squash_enabled', False):
        return img, labels

    h0, w0 = img.shape[:2]
    target_h = hyp.get('squash_target_height', 188)  # 我们用 320

    # 保持原始宽度，只压缩高度
    squashed = cv2.resize(img, (w0, target_h), interpolation=cv2.INTER_LINEAR)

    # 标签是归一化格式 (0-1)，相对于图像尺寸
    # 压缩后，归一化坐标仍然有效
    return squashed, labels
```

---

## 数据集生成

### 命令

```bash
python scripts/convert_manifest_to_ultralytics.py \
    --yaxis-squash \
    --target-height 320
```

### 输出

**目录:** `data/yaxis_squash_h320/`

| 分割 | 图像数 | 尺寸 |
|------|--------|------|
| Train | 19,000 | 672x320 |
| Val | 2,275 | 672x320 |
| Test | 17,785 | 672x320 |
| **总计** | **39,060** | |

---

## 使用修复后的预处理进行训练

### 配置

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

### 评估

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

## 文件参考

| 文件 | 说明 |
|------|------|
| `scripts/convert_manifest_to_ultralytics.py` | Y轴压缩数据生成 |
| `data/yaxis_squash_h320/` | 生成的 672x320 数据集 |
| YOLOv9 参考 | `/imgAssist/modelGenerationTool/utils/dataloaders.py:103-134` |

---

## 下一步

1. 使用 `yaxis_squash_h320` 数据集训练 YOLO26
2. 使用 `imgsz: [320, 672]`
3. 对比 Val 和 Test 性能，验证修复效果
4. 目标: Val-Test 差距 < 10%
