#!/bin/bash
# YOLO26m Optimized Training Script v2
# Larger model + optimized hyperparameters for better recall

cd /home/ubuntu/ImgAssistClone/experiments/yolo26

python scripts/train_yolo26.py \
  --model yolo26m \
  --data configs/data/data_2class.yaml \
  --epochs 300 \
  --imgsz 768 \
  --batch 24 \
  --device 0 \
  --workers 48 \
  --patience 30 \
  --save-period 10 \
  --lr0 0.002 \
  --lrf 0.01 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --warmup-epochs 5 \
  --translate 0.1 \
  --scale 0.5 \
  --hsv-h 0.015 \
  --hsv-s 0.7 \
  --hsv-v 0.4 \
  --fliplr 0.5 \
  --mosaic 0.2 \
  --mixup 0.1 \
  --copy-paste 0.3 \
  --box 7.5 \
  --cls 0.5 \
  --label-smoothing 0.1 \
  --close-mosaic 20 \
  --cos-lr \
  --project runs/finetune \
  --name yolo26m_v2_optimized \
  --wandb-project ImgAssist_YOLO26
