# YOLO26 数据流程与代码详解

从原始数据到推理的完整流程，附具体代码位置。

---

## 1. 数据要求

### 输入格式: JSON Manifest

原始数据使用 ImgAssist 的 JSON manifest 格式。

**Manifest 结构:**
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

**Manifest 位置:**
```
/datasets/ImgAssist_Data_10142025/SEL_OTIS_2class_PatientSplit_Manifests_v12/
├── train_manifest.json
├── validation_manifest.json
└── test_manifest.json
```

### 输出格式: Ultralytics YOLO

**转换脚本:** `scripts/convert_manifest_to_ultralytics.py`

**关键函数:**
| 函数 | 行号 | 用途 |
|------|------|------|
| `load_manifest()` | 41-44 | 加载 JSON manifest |
| `create_label_content()` | 47-57 | 转换为 YOLO 格式 |
| `YAxisSquashPreprocessor` | 60-109 | Y轴压缩预处理 |
| `process_manifest()` | 130-263 | 处理整个 manifest |

**输出结构:**
```
data/yaxis_squash_h320/
├── train.txt              # 图像路径列表
├── val.txt
├── test.txt
├── data_2class.yaml       # 数据集配置
├── images/{train,val,test}/*.png   # 预处理后的图像 (672x320)
└── labels/{train,val,test}/*.txt   # YOLO 格式标注
```

**YOLO 标注格式:** `class_id x_center y_center width height` (归一化 0-1)

---

## 2. 预处理

### 方案 A: Y轴压缩 (推荐)

**代码:** `scripts/convert_manifest_to_ultralytics.py:60-109`

```python
class YAxisSquashPreprocessor:
    def process(self, img, bboxes):
        h0, w0 = img.shape[:2]
        # 保持宽度，只压缩高度
        img_squashed = cv2.resize(img, (w0, self.target_height))
        # 标签不变 - 归一化坐标仍然有效
        return img_squashed, bboxes
```

**处理流程:**
```
原图 (672x~420) → cv2.resize(672, 320) → 最终 (672x320)
```

**命令:**
```bash
python scripts/convert_manifest_to_ultralytics.py --yaxis-squash --target-height 320
```

### 方案 B: 矩形 Letterbox

**代码:** `scripts/squash_transform.py:30-149`

```python
class RectangleTransform:
    def __call__(self, img, labels):
        # 等比缩放到目标尺寸
        scale = min(target_w/w, target_h/h)
        new_w, new_h = int(w * scale), int(h * scale)

        # 填充到目标尺寸
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2

        # 标签坐标变换
        x_new = (x * new_w + pad_left) / target_width
        y_new = (y * new_h + pad_top) / target_height
```

---

## 3. 数据增强

### 配置文件: `configs/hyp/hyp_yolo26_medical.yaml`

**医学影像策略 (保守设置):**

```yaml
# 几何变换 - 第19-27行
degrees: 0.0          # 禁用旋转 (OCT固定方向)
translate: 0.014      # 平移 1.4%
scale: 0.42           # 缩放 42%
shear: 0.0            # 禁用剪切
flipud: 0.0           # 禁用垂直翻转 (解剖位置重要)
fliplr: 0.113         # 水平翻转 11.3%

# 颜色变换 - 第15-17行
hsv_h: 0.033          # 色调
hsv_s: 0.098          # 饱和度
hsv_v: 0.038          # 亮度

# 高级增强 - 第28-30行
mosaic: 0.0           # 禁用 (保留空间关系)
mixup: 0.091
copy_paste: 0.24
```

**为什么保守?**
- OCT 图像方向固定 → 不旋转
- 垂直位置有临床意义 → 不垂直翻转
- Mosaic 破坏空间上下文 → 禁用

### 训练时的增强

**代码:** `scripts/train_yolo26.py:178-191`

```python
train_args = {
    # 增强参数传递给 Ultralytics
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

## 4. 训练

### 脚本: `scripts/train_yolo26.py`

**关键代码位置:**
| 功能 | 行号 | 用途 |
|------|------|------|
| 参数解析 | 264-369 | CLI 参数 |
| 模型加载 | 141-155 | 加载 YOLO26 权重 |
| 训练配置 | 157-214 | 构建 train_args 字典 |
| 执行训练 | 248 | `model.train(**train_args)` |
| 结果输出 | 250-259 | 打印和保存结果 |

### 训练参数

```python
train_args = {
    'epochs': 200,
    'imgsz': [320, 672],      # 高度, 宽度
    'batch': 36,
    'lr0': 0.00173,
    'lrf': 0.021,
    'momentum': 0.895,
    'weight_decay': 0.000446,
    'warmup_epochs': 4.8,
    'box': 7.2,               # 框损失权重 (优先定位)
    'cls': 0.31,              # 分类权重
    'patience': 15,           # 早停
    'amp': True,              # 混合精度
}
```

### 训练命令

```bash
# 使用脚本
python scripts/train_yolo26.py \
    --model yolo26s \
    --data data/yaxis_squash_h320/data_2class.yaml \
    --imgsz 320 672 \
    --epochs 200 \
    --batch 36 \
    --mosaic 0.0

# 或使用 shell 脚本
./run_training_yolo26.sh yolo26s
```

### 输出

```
runs/finetune/{name}/
├── weights/
│   ├── best.pt          # 最佳验证 mAP
│   └── last.pt          # 最后一个 epoch
├── results.csv          # 每个 epoch 的指标
├── args.yaml            # 训练参数
└── *.png                # 训练曲线
```

---

## 5. 推理 / 验证

### 脚本: `scripts/val_yolo26.py`

### 两种评估模式

| 模式 | 代码行号 | 用途 |
|------|----------|------|
| 框级 (Box-Level) | 385-446 | 标准检测指标 (mAP) |
| 补丁级 (Patch-Level) | 240-351 | 医学影像 (敏感度/特异度) |

### 框级验证

```bash
python scripts/val_yolo26.py \
    --weights runs/finetune/yolo26s_yaxis_squash/weights/best.pt \
    --data data/yaxis_squash_h320/data_2class.yaml \
    --split test \
    --conf 0.25
```

**输出指标:**
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall

### 补丁级验证 (医学关键指标)

```bash
python scripts/val_yolo26.py \
    --weights runs/finetune/yolo26s_yaxis_squash/weights/best.pt \
    --data data/yaxis_squash_h320/data_2class.yaml \
    --split test \
    --eval-mode patch \
    --conf 0.001 \
    --labels-dir data/yaxis_squash_h320/labels/test
```

**补丁级逻辑 (`val_yolo26.py:134-212`):**
```python
# 每张图像作为一个样本
for img_path in test_images:
    has_detection = any(pred.conf >= conf for pred in predictions)
    is_positive = has_ground_truth_bbox(img_path)

    if is_positive and has_detection: TP += 1
    elif is_positive and not has_detection: FN += 1  # 漏诊!
    elif not is_positive and has_detection: FP += 1
    else: TN += 1

sensitivity = TP / (TP + FN)  # 关键: 不漏诊
specificity = TN / (TN + FP)
f2_score = 5 * P * R / (4 * P + R)  # 召回率权重 4 倍
```

**输出指标:**
- **敏感度 (Sensitivity)**: TP/(TP+FN) - 最重要，不漏诊
- **特异度 (Specificity)**: TN/(TN+FP)
- **F2 Score**: 召回率权重是精确率的 4 倍

---

## 6. 完整工作流

```bash
# 1. 数据准备 (Y轴压缩)
python scripts/convert_manifest_to_ultralytics.py \
    --yaxis-squash \
    --target-height 320

# 2. 训练
python scripts/train_yolo26.py \
    --model yolo26s \
    --data data/yaxis_squash_h320/data_2class.yaml \
    --imgsz 320 672 \
    --epochs 200 \
    --mosaic 0.0

# 3. 框级验证
python scripts/val_yolo26.py \
    --weights runs/finetune/*/weights/best.pt \
    --split test \
    --conf 0.25

# 4. 补丁级评估 (医学指标)
python scripts/val_yolo26.py \
    --weights runs/finetune/*/weights/best.pt \
    --eval-mode patch \
    --conf 0.001

# 5. 与 YOLOv9 对比
python scripts/compare_yolov9_yolo26.py \
    --yolov9-weights /path/to/yolov9.pt \
    --yolo26-weights runs/finetune/*/weights/best.pt \
    --eval-mode patch
```

---

## 代码位置速查表

| 文件 | 关键行号 | 用途 |
|------|----------|------|
| `convert_manifest_to_ultralytics.py` | 60-109 | YAxisSquashPreprocessor 类 |
| `convert_manifest_to_ultralytics.py` | 130-263 | process_manifest() 函数 |
| `squash_transform.py` | 30-149 | RectangleTransform 类 |
| `train_yolo26.py` | 141-155 | 模型加载 |
| `train_yolo26.py` | 157-214 | 训练配置构建 |
| `train_yolo26.py` | 248 | 执行训练 |
| `val_yolo26.py` | 134-212 | 补丁级指标计算 |
| `val_yolo26.py` | 240-351 | 补丁级评估逻辑 |
| `val_yolo26.py` | 385-446 | 框级评估逻辑 |
| `hyp_yolo26_medical.yaml` | 15-30 | 增强配置 |
