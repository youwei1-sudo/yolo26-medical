# YOLO26 vs YOLOv9 评估

ImgAssist OCT 癌症检测的模型对比。

**状态:** 需要在预处理修复后重新评估。详见 [PREPROCESSING.md](PREPROCESSING.md)。

---

## 当前结果 (修复前)

### 验证集

| 指标 | YOLOv9-s | YOLO26s | YOLO26m |
|------|----------|---------|---------|
| **mAP@0.5** | 0.766 | 0.759 | **0.781** |
| **mAP@0.5:0.95** | 0.326 | 0.394 | **0.422** |
| Precision | 0.747 | 0.769 | 0.761 |
| Recall | 0.689 | 0.672 | **0.693** |

### 测试集 (有问题)

| 指标 | YOLOv9-s | YOLO26s | YOLO26m |
|------|----------|---------|---------|
| **mAP@0.5** | **0.607** | 0.247 | 0.165 |
| Precision | **0.658** | 0.279 | 0.205 |
| Recall | **0.559** | 0.407 | 0.396 |

**YOLO26 由于预处理不匹配，显示严重的 Val-Test 差距。**

---

## 配置对比

### 训练配置

| 参数 | YOLOv9-s | YOLO26m |
|------|----------|---------|
| 架构 | GELAN | C3k2 + C2PSA |
| 图像尺寸 | 320x672 (矩形) | 768x768 (正方形) |
| Batch Size | 32 | 24 |
| Epochs | 109 | 130 |
| NMS | 后处理 | 原生端到端 |

### 数据增强

| 增强方法 | YOLOv9-s | YOLO26m |
|----------|----------|---------|
| Mosaic | 0.0 | 0.2 |
| Mixup | 0.0 | 0.1 |
| Copy-Paste | 0.0 | 0.3 |
| Flip LR | 0.113 | 0.5 |

---

## 架构差异

| 特性 | YOLOv9-s | YOLO26m |
|------|----------|---------|
| Backbone | GELAN (RepNCSPELAN4) | C3k2 + C2PSA |
| Head | 双分支 (aux + main) | 单一精简 |
| NMS | 后处理 | 原生端到端 |
| DFL | reg_max=16 (softmax) | reg_max=1 (直接回归) |
| 参数量 | ~9.6M | ~21.9M |
| GFLOPs | ~26 | ~75 |

---

## YOLOv9 测试集指标 (参考)

来自 FP 分析，conf=0.001:

| 指标 | 值 |
|------|-----|
| 总 Patches | 17,785 |
| 阳性 Patches | 4,399 |
| 阴性 Patches | 13,386 |
| TP | 4,318 |
| FN | 81 |
| FP | 13,275 |
| TN | 111 |
| **敏感度** | **98.16%** |
| 特异度 | 0.83% |

注意：非常低的阈值 (0.001) 最大化了敏感度。

---

## 部署考虑

### 导出复杂度

| 格式 | YOLOv9-s | YOLO26m |
|------|----------|---------|
| ONNX | 需要 NMS 节点 | 原生，无需 NMS |
| TensorRT | 需要插件 | 直接导出 |
| INT8 | DFL 可能有问题 | 直接回归，干净 |

### 延迟预期

| 场景 | YOLOv9-s | YOLO26m |
|------|----------|---------|
| GPU (A100) | ~5ms + NMS | ~7ms |
| CPU | ~80ms + NMS | ~220ms |

YOLO26 有确定性延迟 (无 NMS)，但模型更大。

---

## 需要重新评估

预处理修复后，重新运行：

```bash
python scripts/compare_yolov9_yolo26.py \
    --yolov9-weights <yolov9_best.pt> \
    --yolo26-weights <yolo26_best.pt> \
    --data data/yaxis_squash_h320/data_2class.yaml \
    --split test \
    --conf 0.001 \
    --eval-mode patch
```

**预期结果:**
- YOLO26 Val-Test 差距 < 10%
- 敏感度与 YOLOv9 相当或更好

---

## 模型选择建议

| 场景 | 推荐 |
|------|------|
| 最高精度 | YOLO26m |
| 延迟敏感 | YOLOv9-s 或 YOLO26s |
| 边缘部署 | YOLO26 (无 NMS) |
| INT8 量化 | YOLO26 (无 DFL) |

---

## 文件位置

### YOLOv9-s
- 权重: `/lambda/nfs/ImgAssistClone/imgAssist/modelGenerationTool/runs/train/PatientSplit_v12.1_unstretch_320x672_SGD/weights/best.pt`

### YOLO26m
- 权重: `runs/finetune/yolo26m_v2_optimized/weights/best.pt`
