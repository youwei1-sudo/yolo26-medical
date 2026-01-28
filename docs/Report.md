# YOLO26 Test Performance Fix Plan

## 问题概述

YOLO26 模型在验证集上表现良好，但在测试集上性能严重下降：

| Model | Val mAP@0.5 | Test mAP@0.5 | Drop |
|-------|-------------|--------------|------|
| YOLO26s | 75.9% | 24.7% | -51.2% |
| YOLO26m | 78.1% | 16.5% | -61.6% |

---

## 根因分析

### 1. 数据预处理差异 (主要原因)

**YOLOv9 v12 Pipeline (正确做法):**
```
原图 (672×~420) → Y-axis Squash → (672×320)
```
- 宽度保持不变 (672)
- 只压缩高度到目标值 (320)
- 使用 `squash_image()` 函数 (`dataloaders.py:103-134`)
- 无 padding，无等比缩放

**YOLO26 Pipeline (之前错误):**
```
原图 → 直接 Resize → 768×768 (正方形)
```
- 使用 Ultralytics 标准 dataloader
- 无 Y-axis squash preprocessing
- 从 txt 文件列表加载

### 2. 评估方法差异

| 方面 | YOLOv9 | YOLO26 |
|------|--------|--------|
| 评估级别 | Patch-level (图像有无检测) | Box-level (mAP IoU) |
| Conf 阈值 | 0.001 (高敏感度) | 0.25 (标准) |
| 主要指标 | Sensitivity 98%+ | mAP@0.5 |

### 3. 可能的过拟合因素

- YOLO26m 更大 (21.7M vs 9.9M params)
- 启用了 mosaic=0.2 增强
- 图像尺寸 768 vs 640
- 不同的 augmentation 设置

---

## 解决方案

### 正确的 Y-Axis Squash 预处理

**Pipeline:**
```
原图 (672×~420)
    ↓ Y-axis squash: 宽度不变，只压缩高度
最终 (672×320)
    ↓ 无 padding
```

**关键参数:**
- 输入尺寸: 672×~420 (WxH)
- 目标尺寸: 672×320 (WxH)
- 宽度: 保持 672 不变
- 高度: 压缩到 320

### 数据集生成

```bash
python scripts/convert_manifest_to_ultralytics.py \
    --yaxis-squash \
    --target-height 320
```

**输出目录:** `data/yaxis_squash_h320/`

| Split | Images |
|-------|--------|
| Train | 19,000 |
| Val | 2,275 |
| Test | 17,785 |
| **Total** | **39,060** |

---

## 之前的错误理解 vs 正确理解

### 错误 1: 误用 v12.1 的配置

**错误理解:**
- v12.1 用 `imgsz: [320, 672]` + `rectangle_input: true`
- 认为是标准 letterbox (等比缩放 + padding)
- Pipeline: 672×420 → scale(0.76) → 512×320 → pad → 672×320

**问题:**
- v12.1 实际配置是 `squash_enabled: false`
- 标准 letterbox 会把宽度从 672 缩小到 ~512，然后 padding 回 672
- 这浪费了原始图像信息

### 错误 2: 混淆 v12 和 v12.1

| 版本 | squash_enabled | 方法 |
|------|----------------|------|
| v12 | true | Y-axis squash (672×420 → 672×188) + padding to 640×640 |
| v12.1 | false | Rectangle letterbox (672×420 → 512×320 → padding to 672×320) |

### 正确理解

我们需要的是 **v12 的 Y-axis squash 方法**，但目标高度改为 320:

```
原图 (672×~420)
    ↓ cv2.resize(img, (672, 320))  # 宽度不变，只改高度
最终 (672×320)
```

**关键代码** (`dataloaders.py:103-134`):
```python
def squash_image(img, labels, hyp):
    h0, w0 = img.shape[:2]
    target_h = hyp.get('squash_target_height', 188)  # 我们用 320

    # Keep original width, only squash height
    squashed = cv2.resize(img, (w0, target_h), interpolation=cv2.INTER_LINEAR)

    # Labels remain valid - normalized coords still work
    return squashed, labels
```

---

## 实验历史 (imgsz 变化)

| 版本 | 实验 | imgsz | 预处理方法 | 问题 |
|------|------|-------|-----------|------|
| v1 | yolo26m_v2_optimized | **768×768** | 标准 Ultralytics resize | 正方形，无 squash |
| v2 | yolo26s_2class_v12 | **640×640** | 标准 Ultralytics resize | 正方形，无 squash |
| v3 | yolo26s_v3_squash | **640×640** | 错误的 squash (0.457 ratio) | 使用了错误的 188 target height |
| v4 | **当前 (正确)** | **672×320** | Y-axis squash | 宽度不变，只压缩高度到 320 |

### 详细说明

**v1 (768×768):**
- 使用 Ultralytics 默认的正方形 resize
- 图像被拉伸/缩放到正方形
- 丢失了原始 aspect ratio 信息

**v2 (640×640):**
- 改为 640×640 以匹配 YOLOv9
- 仍然是正方形，无 squash
- Val-Test gap: -51.2%

**v3 (640×640 + wrong squash):**
- 尝试添加 squash 预处理
- 错误地使用了 `target_height_ratio=0.457` (来自错误的 188/411)
- 生成了 squashed_640 数据集 (已删除)

**v4 (672×320 + correct Y-axis squash):**
- 正确理解 YOLOv9 v12 的 squash 方法
- 宽度保持 672 不变
- 高度压缩到 320
- 生成了 yaxis_squash_h320 数据集 ✓

---

## 文件更新记录

| 文件 | 状态 | 说明 |
|------|------|------|
| `scripts/convert_manifest_to_ultralytics.py` | ✅ 已更新 | 添加 `--yaxis-squash` 选项 |
| `docs/SQUASH_PREPROCESSING_GUIDE.md` | ✅ 已更新 | 修正为 Y-axis squash 方法 |
| `data/yaxis_squash_h320/` | ✅ 已生成 | 39,060 张 672×320 图像 |

---

## 代码位置

- YOLOv9 squash 实现: `/home/ubuntu/ImgAssistClone/imgAssist/modelGenerationTool/utils/dataloaders.py:103-134`
- YOLO26 Y-axis squash: `/home/ubuntu/ImgAssistClone/experiments/yolo26/scripts/convert_manifest_to_ultralytics.py:60-109`

---

## 下一步

1. 使用新生成的 `yaxis_squash_h320` 数据集训练 YOLO26
2. 训练时使用 `imgsz: [320, 672]`
3. 对比 Val 和 Test 性能，验证预处理修复效果
