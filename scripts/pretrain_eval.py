#!/usr/bin/env python3
"""
YOLO26 Pretrain Evaluation Script

Evaluate pretrained YOLO26 models on our medical dataset without fine-tuning.
This establishes baseline performance for comparison with fine-tuned models.

WandB Integration:
    - Logs evaluation metrics to WandB
    - Creates comparison tables for multiple model variants
    - Tracks pretrained model baselines

Usage:
    python pretrain_eval.py --variant yolo26s
    python pretrain_eval.py --variant all  # Test all variants
    python pretrain_eval.py --variant yolo26s --wandb-project ImgAssist_YOLO26
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_wandb(args, run_name: str = None) -> bool:
    """Initialize WandB for pretrain evaluation."""
    if not WANDB_AVAILABLE or args.no_wandb:
        return False

    try:
        wandb.login()

        # Initialize WandB run
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name or f"pretrain_eval_{args.variant}",
            config={
                'variant': args.variant,
                'imgsz': args.imgsz,
                'batch': args.batch,
                'conf': args.conf,
                'data': args.data,
                'eval_type': 'pretrain'
            },
            tags=['pretrain', 'evaluation', args.variant],
            reinit=True
        )
        return True
    except Exception as e:
        print(f"Warning: Failed to initialize WandB: {e}")
        return False


def log_results_to_wandb(variant: str, results, wandb_enabled: bool):
    """Log evaluation results to WandB."""
    if not wandb_enabled or not WANDB_AVAILABLE:
        return

    try:
        metrics = {
            f'{variant}/mAP50': results.box.map50,
            f'{variant}/mAP50-95': results.box.map,
            f'{variant}/precision': results.box.mp,
            f'{variant}/recall': results.box.mr,
        }
        wandb.log(metrics)
    except Exception as e:
        print(f"Warning: Failed to log to WandB: {e}")


def log_comparison_table_to_wandb(all_results: dict, wandb_enabled: bool):
    """Log comparison table to WandB."""
    if not wandb_enabled or not WANDB_AVAILABLE:
        return

    try:
        # Create comparison table
        columns = ['Model', 'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
        data = []
        for variant, results in all_results.items():
            data.append([
                variant,
                results.box.map50,
                results.box.map,
                results.box.mp,
                results.box.mr
            ])

        table = wandb.Table(columns=columns, data=data)
        wandb.log({'pretrain_comparison': table})

        # Log summary metrics
        summary = {
            'best_mAP50_model': max(all_results.keys(), key=lambda k: all_results[k].box.map50),
            'best_mAP50': max(r.box.map50 for r in all_results.values()),
            'best_recall_model': max(all_results.keys(), key=lambda k: all_results[k].box.mr),
            'best_recall': max(r.box.mr for r in all_results.values()),
        }
        wandb.log(summary)
    except Exception as e:
        print(f"Warning: Failed to log comparison table to WandB: {e}")


def run_pretrain_eval(
    variant: str,
    data_config: str,
    imgsz: int = 640,
    batch_size: int = 16,
    conf_thres: float = 0.001,
    device: int = 0,
    project: str = "runs/pretrain_eval",
    workers: int = 8,
    wandb_enabled: bool = False
):
    """
    Run pretrained YOLO26 model evaluation.

    Args:
        variant: Model variant ('yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x')
        data_config: Path to data YAML config
        imgsz: Inference image size
        batch_size: Batch size for validation
        conf_thres: Confidence threshold (low for high recall)
        device: GPU device ID
        project: Output project directory
        workers: Number of dataloader workers
        wandb_enabled: Whether WandB logging is enabled
    """
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"Evaluating pretrained {variant}")
    print(f"{'='*60}")

    # Load pretrained model
    model = YOLO(f"{variant}.pt")

    # Run validation
    results = model.val(
        data=data_config,
        imgsz=imgsz,
        batch=batch_size,
        conf=conf_thres,
        device=device,
        workers=workers,
        save_json=True,
        save_txt=True,
        project=project,
        name=variant,
        exist_ok=True
    )

    # Print results summary
    print(f"\n{variant} Results:")
    print(f"  mAP@0.5:     {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  Precision:   {results.box.mp:.4f}")
    print(f"  Recall:      {results.box.mr:.4f}")

    # Log to WandB
    log_results_to_wandb(variant, results, wandb_enabled)

    return results


def run_all_variants(args, wandb_enabled: bool = False):
    """Run evaluation on all YOLO26 variants."""
    # Focus on s (similar to yolov9-s) and n (for latency comparison)
    variants = ['yolo26n', 'yolo26s']

    if args.all_sizes:
        variants = ['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x']

    all_results = {}

    for variant in variants:
        try:
            results = run_pretrain_eval(
                variant=variant,
                data_config=args.data,
                imgsz=args.imgsz,
                batch_size=args.batch,
                conf_thres=args.conf,
                device=args.device,
                project=args.project,
                workers=args.workers,
                wandb_enabled=wandb_enabled
            )
            all_results[variant] = results
        except Exception as e:
            print(f"Error evaluating {variant}: {e}")
            continue

    # Print comparison table
    print("\n" + "="*70)
    print("YOLO26 Pretrain Evaluation Comparison")
    print("="*70)
    print(f"{'Model':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<14} {'Precision':<12} {'Recall':<12}")
    print("-"*70)

    for variant, results in all_results.items():
        print(f"{variant:<12} {results.box.map50:<12.4f} {results.box.map:<14.4f} "
              f"{results.box.mp:<12.4f} {results.box.mr:<12.4f}")

    # Log comparison table to WandB
    log_comparison_table_to_wandb(all_results, wandb_enabled)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pretrained YOLO26 models on medical dataset'
    )
    parser.add_argument(
        '--variant',
        type=str,
        default='yolo26s',
        choices=['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x', 'all'],
        help='Model variant to evaluate'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=str(PROJECT_ROOT / 'configs/data/data_2class.yaml'),
        help='Path to data config YAML'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Inference image size'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.001,
        help='Confidence threshold (low for high recall baseline)'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU device ID'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--project',
        type=str,
        default=str(PROJECT_ROOT / 'runs/pretrain_eval'),
        help='Output project directory'
    )
    parser.add_argument(
        '--all-sizes',
        action='store_true',
        help='Test all model sizes (n, s, m, l, x)'
    )

    # WandB settings
    parser.add_argument('--wandb-project', type=str, default='ImgAssist_YOLO26',
                        help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='WandB entity (team/username)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB logging')

    args = parser.parse_args()

    # Verify data config exists
    if not Path(args.data).exists():
        print(f"Error: Data config not found: {args.data}")
        print("Run convert_manifest_to_ultralytics.py first to generate data files.")
        sys.exit(1)

    print(f"YOLO26 Pretrain Evaluation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data config: {args.data}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Confidence threshold: {args.conf}")

    # Setup WandB
    wandb_enabled = setup_wandb(args)
    if wandb_enabled:
        print(f"WandB logging enabled (project: {args.wandb_project})")
    else:
        print("WandB logging disabled")

    if args.variant == 'all':
        run_all_variants(args, wandb_enabled)
    else:
        run_pretrain_eval(
            variant=args.variant,
            data_config=args.data,
            imgsz=args.imgsz,
            batch_size=args.batch,
            conf_thres=args.conf,
            device=args.device,
            project=args.project,
            workers=args.workers,
            wandb_enabled=wandb_enabled
        )

    # Finish WandB run
    if wandb_enabled and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
