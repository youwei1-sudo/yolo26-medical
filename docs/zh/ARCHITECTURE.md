# YOLO26 架构

YOLO26 (2025年9月发布) - Ultralytics 的部署优先 YOLO 变体。

---

## 核心理念：部署优先

YOLO26 解决了"导出差距" - 模型在 GPU 上运行快，但导出到边缘设备后变慢的问题。

核心创新：
- **无 NMS：** 确定性延迟
- **移除 DFL：** 干净的 INT8 量化
- **MuSGD 优化器：** 快速收敛

---

## 1. 关键架构变化

### 1.1 原生端到端无 NMS

传统 YOLO 使用非极大值抑制 (NMS) 进行过滤，导致延迟波动。

**YOLO26 解决方案：**
- 使用一对一标签分配
- Head 直接输出稀疏、确定性的预测
- 无需后处理

```yaml
# yolo26.yaml
end2end: True  # 无 NMS 模式
```

**影响：** CPU 延迟降低约 43%。

### 1.2 移除 DFL (直接回归)

之前版本 (v8-v13) 使用分布焦点损失 (DFL)，需要 Softmax 积分操作，对边缘硬件不友好。

**YOLO26 解决方案：**
- 回归到直接回归
- 大大简化计算图
- INT8 量化几乎无损

```yaml
# yolo26.yaml
reg_max: 1  # 直接回归 (v8/v9 用 16 做 DFL)
```

---

## 2. 模型变体

| 模型 | 参数量 | GFLOPs | 层数 | 用途 |
|------|--------|--------|------|------|
| yolo26n | 2.6M | 6.1 | 260 | 移动端，快速推理 |
| yolo26s | 10.0M | 22.8 | 260 | 平衡 |
| yolo26m | 21.9M | 75.4 | 280 | 更高精度 |
| yolo26l | 26.3M | 93.8 | 392 | 高精度 |
| yolo26x | 59.0M | 209.5 | 392 | 最高精度 |

---

## 3. Backbone 架构

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]       # P1/2
  - [-1, 1, Conv, [128, 3, 2]]      # P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]      # P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]      # P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]     # P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]]
  - [-1, 2, C2PSA, [1024]]          # PSA 注意力
```

### 关键模块

| 模块 | 说明 |
|------|------|
| **C3k2** | 可配置 kernel size 的 CSP 类块 |
| **C2PSA** | 极化自注意力模块 |
| **SPPF** | 空间金字塔池化 - 快速版 |

---

## 4. 训练特性

### 4.1 MuSGD 优化器

结合 SGD 和 Muon 优化器 (Momentum-Unified)。

- 来源：受 Moonshot AI 的 Kimi K2 训练启发
- 对梯度矩阵进行正交化
- 无需 Warm-up 即可超快收敛

```python
# 用法
from ultralytics.optim import MuSGD
# 可用优化器: Adam, AdamW, SGD, MuSGD 等
```

### 4.2 STAL (小目标感知标签分配)

解决小目标的梯度消失问题：
- 动态调整 IoU 阈值
- 小目标获得更低的阈值

### 4.3 ProgLoss (渐进损失平衡)

动态损失权重平衡：
- **早期阶段：** 聚焦分类 (语义学习)
- **后期阶段：** 聚焦回归 (几何微调)

---

## 5. 多任务支持

YOLO26 原生支持五种任务：

| 任务 | 配置文件 |
|------|----------|
| 检测 | yolo26.yaml |
| 分割 | yolo26-seg.yaml |
| 姿态 | yolo26-pose.yaml |
| OBB | yolo26-obb.yaml |
| 分类 | yolo26-cls.yaml |

---

## 6. 性能 (COCO)

| 模型 | mAP (50-95) | CPU (ms) | T4 TensorRT (ms) |
|------|-------------|----------|------------------|
| YOLOv8-n | 37.3% | 80.4 | 0.99 |
| YOLO11-n | 39.5% | 56.1 | 1.5 |
| **YOLO26-n** | **40.1%** | **38.9** | 1.7 |
| **YOLO26-m** | **53.1%** | 220.0 | 4.7 |
| **YOLO26-x** | **56.9%** | 525.8 | 11.8 |

---

## 7. 部署

### 导出格式

| 格式 | 文件 | 元数据 |
|------|------|--------|
| PyTorch | yolo26n.pt | 是 |
| ONNX | yolo26n.onnx | 是 |
| TensorRT | yolo26n.engine | 是 |
| CoreML | yolo26n.mlpackage | 是 |
| TFLite | yolo26n.tflite | 是 |
| OpenVINO | yolo26n_openvino_model/ | 是 |

### 导出优势

- ONNX 无需 NMS 节点
- TensorRT 直接导出 (无需插件)
- 干净的 INT8 (无 DFL 问题)
- 原生 CoreML 支持

---

## 8. 代码验证

| 特性 | 状态 |
|------|------|
| 无 NMS (end2end: True) | 已验证 |
| 移除 DFL (reg_max: 1) | 已验证 |
| C3k2 模块 | 已验证 |
| C2PSA 注意力 | 已验证 |
| MuSGD 优化器 | 已验证 |
| 多任务配置 | 已验证 |
| 模型参数量 | 已验证 |

**Ultralytics 版本:** 8.4.7

---

## 参考

- [Ultralytics YOLO26 文档](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
