# YOLO26 训练结果

YOLO26 在 ImgAssist OCT 癌症检测数据集上的微调结果。

**注意:** 这些结果来自预处理修复之前的实验。使用 Y轴压缩预处理的新实验待进行。

---

## 1. 最佳模型: YOLO26m v2 Optimized

**位置:** `runs/finetune/yolo26m_v2_optimized/`

### 配置

| 参数 | 值 |
|------|-----|
| 模型 | yolo26m |
| 预训练权重 | yolo26m.pt (COCO) |
| 数据集 | 2类 (Suspicious) v12 |
| 图像尺寸 | 768x768 |
| Batch Size | 24 |
| Epochs | 130 (早停) |
| 优化器 | SGD |
| 学习率 | lr0=0.002, lrf=0.01 |
| Patience | 30 |

### 超参数

```yaml
# 优化器
optimizer: SGD
lr0: 0.002
lrf: 0.01
momentum: 0.9
weight_decay: 0.0005
warmup_epochs: 5.0

# 损失权重
box: 7.5
cls: 0.5
dfl: 1.5

# 数据增强
mosaic: 0.2
mixup: 0.1
copy_paste: 0.3
degrees: 0.0      # 无旋转 (OCT 固定方向)
flipud: 0.0       # 无垂直翻转
fliplr: 0.5
scale: 0.5
```

---

## 2. 验证集结果

### 最终指标 (Epoch 130)

| 指标 | 值 |
|------|-----|
| **mAP@0.5** | **0.781** |
| **mAP@0.5:0.95** | **0.422** |
| Precision | 0.761 |
| Recall | 0.693 |
| Box Loss | 1.121 |
| Cls Loss | 0.804 |

### 训练过程

| Epoch | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|---------|--------------|-----------|--------|
| 10 | 0.473 | 0.204 | 0.457 | 0.526 |
| 30 | 0.739 | 0.382 | 0.744 | 0.634 |
| 50 | 0.768 | 0.411 | 0.759 | 0.679 |
| 70 | 0.773 | 0.420 | 0.755 | 0.685 |
| 90 | 0.774 | 0.422 | 0.760 | 0.683 |
| 110 | 0.782 | 0.425 | 0.767 | 0.690 |
| 130 | 0.781 | 0.422 | 0.761 | 0.693 |

---

## 3. 测试集结果 (有问题)

**这些结果显示了预处理问题:**

| 模型 | Val mAP@0.5 | Test mAP@0.5 | 差距 |
|------|-------------|--------------|------|
| YOLO26s | 75.9% | 24.7% | -51.2% |
| YOLO26m | 78.1% | 16.5% | -61.6% |

**根本原因:** 预处理不正确。详见 [PREPROCESSING.md](PREPROCESSING.md)。

---

## 4. YOLO26s 结果 (早期实验)

| 参数 | 值 |
|------|-----|
| 模型 | yolo26s |
| Epochs | 43 |
| 图像尺寸 | 640 |
| Batch Size | 36 |

| 指标 | 值 |
|------|-----|
| mAP@0.5 | 0.773 |
| mAP@0.5:0.95 | 0.372 |
| Precision | 0.769 |
| Recall | 0.672 |

---

## 5. YOLO26m vs YOLO26s 对比

| 指标 | YOLO26s | YOLO26m | 提升 |
|------|---------|---------|------|
| mAP@0.5 | 0.773 | 0.781 | +1.0% |
| mAP@0.5:0.95 | 0.372 | 0.422 | +13.4% |
| Precision | 0.769 | 0.761 | -1.0% |
| Recall | 0.672 | 0.693 | +3.1% |
| 参数量 | 10M | 22M | +120% |

YOLO26m 提供更好的 mAP@0.5:0.95 (+13.4%)，代价是 2 倍参数量。

---

## 6. 检查点

| 文件 | 大小 | 说明 |
|------|------|------|
| `best.pt` | 44 MB | 最佳验证 mAP |
| `last.pt` | 44 MB | 最后一个 epoch |
| `epoch*.pt` | 88 MB | 定期检查点 |

**最佳模型:**
```
runs/finetune/yolo26m_v2_optimized/weights/best.pt
```

---

## 7. 输出文件

```
runs/finetune/yolo26m_v2_optimized/
├── args.yaml                    # 训练参数
├── results.csv                  # 每个 epoch 的指标
├── results.png                  # 训练曲线图
├── confusion_matrix.png
├── BoxF1_curve.png
├── BoxPR_curve.png
├── labels.jpg                   # 标签分布
├── train_batch*.jpg
├── val_batch*_labels.jpg
├── val_batch*_pred.jpg
└── weights/
    ├── best.pt
    └── last.pt
```

---

## 8. 下一步

1. **使用正确预处理重新训练** - 使用 `data/yaxis_squash_h320/` 数据集
2. **使用矩形输入** - `imgsz: [320, 672]`
3. **验证 Val-Test 差距** - 目标 < 10%
4. **导出部署** - ONNX/TensorRT

---

## WandB 跟踪

项目: `ImgAssist_YOLO26`
