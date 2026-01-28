#!/usr/bin/env python3
"""
YOLO26 Fine-tuning Training Script

Fine-tune pretrained YOLO26 models on medical imaging dataset.
Hyperparameters are transferred from optimized YOLOv9 config.

WandB Integration:
    - Automatically logs training metrics, losses, and validation results
    - Logs model checkpoints as artifacts
    - Tracks hyperparameters and system info

Squash Preprocessing:
    - Optional YOLOv9-style squash preprocessing via --squash flag
    - Can use pre-squashed data directory via --squash-data
    - Aligns preprocessing pipeline with YOLOv9 for consistent results

Usage:
    python train_yolo26.py --model yolo26s --epochs 200
    python train_yolo26.py --model yolo26s --wandb-project ImgAssist_YOLO26
    python train_yolo26.py --model yolo26s --no-wandb  # Disable WandB
    python train_yolo26.py --model yolo26s --squash-data data/squashed_640  # Use pre-squashed data
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

# Try to import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be local only")


def setup_wandb(args) -> bool:
    """
    Initialize WandB for experiment tracking.

    Returns:
        bool: True if WandB is enabled and initialized
    """
    if not WANDB_AVAILABLE or args.no_wandb:
        if args.no_wandb:
            print("WandB logging disabled by --no-wandb flag")
        return False

    try:
        # Login to WandB (uses cached credentials or WANDB_API_KEY env var)
        wandb.login()

        # Set environment variables for Ultralytics WandB integration
        os.environ['WANDB_PROJECT'] = args.wandb_project
        os.environ['WANDB_NAME'] = args.wandb_run_name or args.name or f"{args.model}_2class_v12"

        if args.wandb_entity:
            os.environ['WANDB_ENTITY'] = args.wandb_entity

        # Enable WandB in Ultralytics settings
        os.environ['WANDB_DISABLED'] = 'false'

        print(f"\nWandB Configuration:")
        print(f"  Project: {args.wandb_project}")
        print(f"  Run name: {os.environ['WANDB_NAME']}")
        if args.wandb_entity:
            print(f"  Entity: {args.wandb_entity}")

        return True

    except Exception as e:
        print(f"Warning: Failed to initialize WandB: {e}")
        print("Continuing without WandB logging...")
        return False


def log_final_results_to_wandb(results, args):
    """Log final training results to WandB."""
    if not WANDB_AVAILABLE or args.no_wandb:
        return

    try:
        # Check if there's an active WandB run
        if wandb.run is not None:
            # Log final metrics
            final_metrics = {
                'final/mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
                'final/mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'final/precision': results.results_dict.get('metrics/precision(B)', 0),
                'final/recall': results.results_dict.get('metrics/recall(B)', 0),
            }
            wandb.log(final_metrics)

            # Log best model as artifact
            best_weights = Path(results.save_dir) / 'weights' / 'best.pt'
            if best_weights.exists():
                artifact = wandb.Artifact(
                    name=f"{args.model}_best",
                    type='model',
                    description=f"Best YOLO26 {args.model} checkpoint"
                )
                artifact.add_file(str(best_weights))
                wandb.log_artifact(artifact)

            print("Final results logged to WandB")
    except Exception as e:
        print(f"Warning: Failed to log final results to WandB: {e}")


def get_data_config(args) -> str:
    """
    Get the appropriate data config path.

    If --squash-data is specified, create a data config for the squashed dataset.
    Otherwise, use the default data config.
    """
    if args.squash_data:
        squash_data_dir = Path(args.squash_data)
        if not squash_data_dir.is_absolute():
            squash_data_dir = PROJECT_ROOT / squash_data_dir

        data_yaml = squash_data_dir / 'data_2class.yaml'
        if data_yaml.exists():
            print(f"\nUsing squashed data config: {data_yaml}")
            return str(data_yaml)
        else:
            print(f"Warning: Squashed data config not found at {data_yaml}")
            print("Falling back to default data config")

    return args.data


def train_yolo26(args):
    """Run YOLO26 fine-tuning."""
    from ultralytics import YOLO

    # Get data config (handles squash data if specified)
    data_config = get_data_config(args)

    # Load pretrained model
    print(f"\nLoading pretrained {args.model}...")
    # Check weights folder first, then current directory
    weights_path = PROJECT_ROOT / 'weights' / f"{args.model}.pt"
    if weights_path.exists():
        model = YOLO(str(weights_path))
    else:
        model = YOLO(f"{args.model}.pt")

    # Prepare training arguments
    train_args = {
        'data': data_config,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'patience': args.patience,
        'save_period': args.save_period,

        # Optimizer settings (from hyp.v12-unstretch.yaml)
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'warmup_momentum': args.warmup_momentum,
        'warmup_bias_lr': args.warmup_bias_lr,

        # Augmentation - Medical imaging specific (conservative)
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,  # No rotation for OCT fixed orientation
        'translate': args.translate,
        'scale': args.scale,
        'shear': args.shear,  # No shear for medical
        'perspective': args.perspective,
        'flipud': args.flipud,  # No vertical flip for OCT
        'fliplr': args.fliplr,
        'mosaic': args.mosaic,  # Disabled for medical
        'mixup': args.mixup,
        'copy_paste': args.copy_paste,

        # Loss weights
        'box': args.box,
        'cls': args.cls,

        # Output
        'project': args.project,
        'name': args.name or f"{args.model}_2class_v12",
        'exist_ok': args.exist_ok,

        # Other settings
        'seed': args.seed,
        'deterministic': args.deterministic,
        'single_cls': args.single_cls,
        'rect': args.rect,
        'cos_lr': args.cos_lr,
        'close_mosaic': args.close_mosaic,
        'amp': args.amp,

        # Validation
        'val': not args.noval,
        'plots': not args.noplots,
    }

    # Add label smoothing if specified
    if args.label_smoothing > 0:
        train_args['label_smoothing'] = args.label_smoothing

    # Print configuration
    print(f"\n{'='*60}")
    print(f"YOLO26 Training Configuration")
    print(f"{'='*60}")
    print(f"Model:      {args.model}")
    print(f"Data:       {data_config}")
    if args.squash_data:
        print(f"  (Squash preprocessing: ENABLED)")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device:     {args.device}")
    print(f"Workers:    {args.workers}")
    print(f"\nOptimizer:  {args.optimizer}")
    print(f"  lr0:      {args.lr0}")
    print(f"  lrf:      {args.lrf}")
    print(f"  momentum: {args.momentum}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"\nAugmentation:")
    print(f"  mosaic:   {args.mosaic}")
    print(f"  mixup:    {args.mixup}")
    print(f"  copy_paste: {args.copy_paste}")
    print(f"\nLoss weights:")
    print(f"  box:      {args.box}")
    print(f"  cls:      {args.cls}")
    print(f"{'='*60}\n")

    # Start training
    results = model.train(**train_args)

    # Print final results
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best mAP@0.5:     {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Best mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"\nWeights saved to: {results.save_dir}/weights/")

    # Log final results to WandB
    log_final_results_to_wandb(results, args)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLO26 on medical imaging dataset'
    )

    # Model and data
    parser.add_argument('--model', type=str, default='yolo26s',
                        choices=['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x'],
                        help='Model variant')
    parser.add_argument('--data', type=str,
                        default=str(PROJECT_ROOT / 'configs/data/data_2class.yaml'),
                        help='Path to data config YAML')

    # Training settings
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=36, help='Batch size')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--workers', type=int, default=48, help='Dataloader workers')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=5, help='Save checkpoint every N epochs')

    # Optimizer (from hyp.v12-unstretch.yaml)
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer')
    parser.add_argument('--lr0', type=float, default=0.00173, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.021, help='Final LR = lr0 * lrf')
    parser.add_argument('--momentum', type=float, default=0.895, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.000446, help='Weight decay')
    parser.add_argument('--warmup-epochs', type=float, default=4.8, help='Warmup epochs')
    parser.add_argument('--warmup-momentum', type=float, default=0.8, help='Warmup momentum')
    parser.add_argument('--warmup-bias-lr', type=float, default=0.1, help='Warmup bias LR')

    # Augmentation (medical-specific - conservative)
    parser.add_argument('--hsv-h', type=float, default=0.033, help='HSV-Hue augmentation')
    parser.add_argument('--hsv-s', type=float, default=0.098, help='HSV-Saturation augmentation')
    parser.add_argument('--hsv-v', type=float, default=0.038, help='HSV-Value augmentation')
    parser.add_argument('--degrees', type=float, default=0.0, help='Rotation augmentation (disabled for OCT)')
    parser.add_argument('--translate', type=float, default=0.014, help='Translation augmentation')
    parser.add_argument('--scale', type=float, default=0.42, help='Scale augmentation')
    parser.add_argument('--shear', type=float, default=0.0, help='Shear augmentation (disabled for medical)')
    parser.add_argument('--perspective', type=float, default=0.0, help='Perspective augmentation')
    parser.add_argument('--flipud', type=float, default=0.0, help='Vertical flip (disabled for OCT)')
    parser.add_argument('--fliplr', type=float, default=0.113, help='Horizontal flip')
    parser.add_argument('--mosaic', type=float, default=0.0, help='Mosaic augmentation (disabled for medical)')
    parser.add_argument('--mixup', type=float, default=0.091, help='Mixup augmentation')
    parser.add_argument('--copy-paste', type=float, default=0.24, help='Copy-paste augmentation')

    # Loss weights
    parser.add_argument('--box', type=float, default=7.2, help='Box loss weight')
    parser.add_argument('--cls', type=float, default=0.31, help='Class loss weight')
    parser.add_argument('--label-smoothing', type=float, default=0.189, help='Label smoothing')

    # Output
    parser.add_argument('--project', type=str,
                        default=str(PROJECT_ROOT / 'runs/finetune'),
                        help='Project directory')
    parser.add_argument('--name', type=str, default=None, help='Run name')
    parser.add_argument('--exist-ok', action='store_true', help='Overwrite existing run')

    # Other settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Deterministic training')
    parser.add_argument('--single-cls', action='store_true', help='Single class mode')
    parser.add_argument('--rect', action='store_true', help='Rectangular training')
    parser.add_argument('--cos-lr', action='store_true', help='Cosine LR scheduler')
    parser.add_argument('--close-mosaic', type=int, default=10, help='Close mosaic N epochs before end')
    parser.add_argument('--amp', action='store_true', default=True, help='Automatic mixed precision')
    parser.add_argument('--noval', action='store_true', help='Skip validation')
    parser.add_argument('--noplots', action='store_true', help='Skip plotting')

    # WandB settings
    parser.add_argument('--wandb-project', type=str, default='ImgAssist_YOLO26',
                        help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='WandB entity (team/username)')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='WandB run name (defaults to --name)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB logging')

    # Squash preprocessing settings
    parser.add_argument('--squash-data', type=str, default=None,
                        help='Path to pre-squashed data directory (e.g., data/squashed_640)')

    args = parser.parse_args()

    # Verify data config exists
    if not Path(args.data).exists():
        print(f"Error: Data config not found: {args.data}")
        print("Run convert_manifest_to_ultralytics.py first to generate data files.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"YOLO26 Fine-tuning")
    print(f"{'='*60}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup WandB
    wandb_enabled = setup_wandb(args)
    if wandb_enabled:
        print("WandB logging enabled")
    else:
        print("WandB logging disabled")

    # Run training
    train_yolo26(args)


if __name__ == '__main__':
    main()
