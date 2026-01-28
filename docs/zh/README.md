# YOLO26 医学影像实验文档

YOLO26 在 ImgAssist OCT 癌症检测数据集上的微调实验。

---

## 文档索引

| 文档 | 说明 |
|------|------|
| [PREPROCESSING.md](PREPROCESSING.md) | **关键修复**: Y轴压缩预处理，解决 Val-Test 性能差距 |
| [TRAINING.md](TRAINING.md) | 训练配置和结果 |
| [EVALUATION.md](EVALUATION.md) | YOLOv9 vs YOLO26 对比 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | YOLO26 技术架构 |

---

## 快速链接

### 数据集

| 数据集 | 路径 | 说明 |
|--------|------|------|
| **yaxis_squash_h320** | `data/yaxis_squash_h320/` | 正确预处理 (672x320) |
| Train | 19,000 张 | |
| Val | 2,275 张 | |
| Test | 17,785 张 | |

### 模型

| 模型 | 路径 |
|------|------|
| YOLO26s 权重 | `weights/yolo26s.pt` |
| YOLO26m 权重 | `weights/yolo26m.pt` |
| 最佳检查点 | `runs/finetune/*/weights/best.pt` |

### 脚本

| 脚本 | 用途 |
|------|------|
| `scripts/convert_manifest_to_ultralytics.py` | 数据转换 (含 Y轴压缩) |
| `scripts/train_yolo26.py` | 微调训练 |
| `scripts/val_yolo26.py` | 验证评估 |

---

## 当前状态

**预处理修复已完成** - 详见 [PREPROCESSING.md](PREPROCESSING.md)

- 已生成正确的 Y轴压缩数据集 (672x320)
- 下一步：使用正确预处理重新训练
- 目标：将 Val-Test 差距从 -51%/-61% 缩小到 <10%

---

## 核心要点：Y轴压缩

关键修复是使用正确的预处理方法：

```
原图 (672x~420)
    | cv2.resize(img, (672, 320))  # 保持宽度，只压缩高度
最终 (672x320)
```

这与 YOLOv9 v12 的预处理方法一致。

详见 [PREPROCESSING.md](PREPROCESSING.md)。
