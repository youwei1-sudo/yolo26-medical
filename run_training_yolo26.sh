#!/bin/bash
# YOLO26 Training Launch Script
#
# Usage:
#   ./run_training_yolo26.sh                    # Default: yolo26s with WandB
#   ./run_training_yolo26.sh yolo26n            # Specific variant
#   ./run_training_yolo26.sh yolo26s --epochs 100
#   ./run_training_yolo26.sh yolo26s --no-wandb # Disable WandB logging
#
# WandB Integration:
#   - Training metrics are logged to WandB by default
#   - Project: ImgAssist_YOLO26
#   - Disable with --no-wandb flag
#   - Set WANDB_API_KEY environment variable if not logged in
#
# Prerequisites:
#   1. Run convert_manifest_to_ultralytics.py first:
#      python scripts/convert_manifest_to_ultralytics.py
#
#   2. Verify data files exist:
#      ls -la data/train.txt data/val.txt data/test.txt
#      ls -la data/labels/
#
#   3. (Optional) Login to WandB:
#      wandb login

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${1:-yolo26s}"
shift 2>/dev/null || true

# Default paths
DATA_CONFIG="configs/data/data_2class.yaml"
PROJECT="runs/finetune"
WANDB_PROJECT="ImgAssist_YOLO26"

# Check if data exists
if [ ! -f "$DATA_CONFIG" ]; then
    echo "Error: Data config not found: $DATA_CONFIG"
    echo "Run: python scripts/convert_manifest_to_ultralytics.py"
    exit 1
fi

if [ ! -d "data/labels" ]; then
    echo "Error: Labels directory not found"
    echo "Run: python scripts/convert_manifest_to_ultralytics.py"
    exit 1
fi

echo "=============================================="
echo "YOLO26 Training Pipeline"
echo "=============================================="
echo "Model:       $MODEL"
echo "Data config: $DATA_CONFIG"
echo "Project:     $PROJECT"
echo "WandB:       $WANDB_PROJECT"
echo "Extra args:  $@"
echo "=============================================="

# Activate conda environment if exists
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    # Try to activate common environment names
    conda activate imgassist 2>/dev/null || \
    conda activate yolo 2>/dev/null || \
    conda activate base 2>/dev/null || true
fi

# Run training with WandB enabled by default
python scripts/train_yolo26.py \
    --model "$MODEL" \
    --data "$DATA_CONFIG" \
    --project "$PROJECT" \
    --name "${MODEL}_2class_v12" \
    --wandb-project "$WANDB_PROJECT" \
    --epochs 200 \
    --batch 36 \
    --device 0 \
    --workers 48 \
    --patience 15 \
    "$@"

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Weights saved to: $PROJECT/${MODEL}_2class_v12/weights/"
echo "WandB dashboard: https://wandb.ai/<your-entity>/$WANDB_PROJECT"
echo ""
echo "Next steps:"
echo "  1. Run validation on test set:"
echo "     python scripts/val_yolo26.py --weights $PROJECT/${MODEL}_2class_v12/weights/best.pt --split test"
echo ""
echo "  2. Compare with YOLOv9:"
echo "     python scripts/val_yolo26.py --compare --weights $PROJECT/${MODEL}_2class_v12/weights/best.pt"
echo ""
echo "  3. Compute patch-level metrics:"
echo "     python scripts/compute_patch_metrics.py --pred-dir runs/val/val_test/labels --manifest <test_manifest>"
