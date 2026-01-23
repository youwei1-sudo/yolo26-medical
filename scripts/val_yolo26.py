#!/usr/bin/env python3
"""
YOLO26 Validation Script

Run validation on fine-tuned YOLO26 models and compute metrics.
Supports validation on train/val/test splits and comparison with YOLOv9.

WandB Integration:
    - Logs validation metrics to WandB
    - Creates comparison tables for model evaluation
    - Tracks benchmark results

Usage:
    python val_yolo26.py --weights runs/finetune/yolo26s_2class_v12/weights/best.pt
    python val_yolo26.py --weights best.pt --split test --conf 0.25
    python val_yolo26.py --compare  # Compare with YOLOv9 baseline
    python val_yolo26.py --wandb-project ImgAssist_YOLO26
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json

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
    """Initialize WandB for validation."""
    if not WANDB_AVAILABLE or args.no_wandb:
        return False

    try:
        wandb.login()

        mode = 'compare' if args.compare else ('benchmark' if args.benchmark else 'validation')

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name or f"val_{args.split}_{Path(args.weights).stem}",
            config={
                'weights': args.weights,
                'split': args.split,
                'imgsz': args.imgsz,
                'conf': args.conf,
                'iou': args.iou,
                'mode': mode
            },
            tags=['validation', args.split, mode],
            reinit=True
        )
        return True
    except Exception as e:
        print(f"Warning: Failed to initialize WandB: {e}")
        return False


def log_validation_results_to_wandb(results, model_name: str, wandb_enabled: bool):
    """Log validation results to WandB."""
    if not wandb_enabled or not WANDB_AVAILABLE:
        return

    try:
        metrics = {
            f'{model_name}/mAP50': results.box.map50,
            f'{model_name}/mAP50-95': results.box.map,
            f'{model_name}/precision': results.box.mp,
            f'{model_name}/recall': results.box.mr,
        }
        wandb.log(metrics)

        # Log summary
        wandb.run.summary[f'{model_name}_mAP50'] = results.box.map50
        wandb.run.summary[f'{model_name}_recall'] = results.box.mr
    except Exception as e:
        print(f"Warning: Failed to log to WandB: {e}")


def log_comparison_to_wandb(results: dict, wandb_enabled: bool):
    """Log model comparison results to WandB."""
    if not wandb_enabled or not WANDB_AVAILABLE:
        return

    try:
        # Create comparison table
        columns = ['Model', 'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
        data = []
        for model_name, res in results.items():
            if res is not None:
                data.append([
                    model_name,
                    res.box.map50,
                    res.box.map,
                    res.box.mp,
                    res.box.mr
                ])

        if data:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({'model_comparison': table})

            # Log delta if both models available
            if results.get('yolo26') and results.get('yolov9'):
                delta = {
                    'delta/mAP50': results['yolo26'].box.map50 - results['yolov9'].box.map50,
                    'delta/mAP50-95': results['yolo26'].box.map - results['yolov9'].box.map,
                    'delta/recall': results['yolo26'].box.mr - results['yolov9'].box.mr,
                    'delta/precision': results['yolo26'].box.mp - results['yolov9'].box.mp,
                }
                wandb.log(delta)
    except Exception as e:
        print(f"Warning: Failed to log comparison to WandB: {e}")


def validate_model(
    weights: str,
    data_config: str,
    split: str = 'val',
    imgsz: int = 640,
    batch_size: int = 16,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
    device: int = 0,
    workers: int = 8,
    project: str = None,
    name: str = None,
    save_json: bool = True,
    save_txt: bool = True,
    verbose: bool = True
):
    """
    Run validation on a YOLO26 model.

    Args:
        weights: Path to model weights
        data_config: Path to data YAML config
        split: Dataset split ('train', 'val', 'test')
        imgsz: Inference image size
        batch_size: Batch size
        conf_thres: Confidence threshold
        iou_thres: NMS IoU threshold
        device: GPU device
        workers: Dataloader workers
        project: Output project directory
        name: Run name
        save_json: Save predictions as JSON
        save_txt: Save predictions as txt files

    Returns:
        Validation results
    """
    from ultralytics import YOLO

    print(f"\nLoading model: {weights}")
    model = YOLO(weights)

    print(f"Running validation on {split} split...")

    results = model.val(
        data=data_config,
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        workers=workers,
        save_json=save_json,
        save_txt=save_txt,
        project=project or str(PROJECT_ROOT / 'runs/val'),
        name=name or f"val_{split}",
        exist_ok=True,
        verbose=verbose
    )

    return results


def print_results(results, model_name: str = "Model"):
    """Print validation results in formatted table."""
    print(f"\n{'='*60}")
    print(f"{model_name} Validation Results")
    print(f"{'='*60}")
    print(f"  mAP@0.5:     {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  Precision:   {results.box.mp:.4f}")
    print(f"  Recall:      {results.box.mr:.4f}")

    # Per-class results if available
    if hasattr(results.box, 'ap50') and len(results.box.ap50) > 0:
        print(f"\nPer-class AP@0.5:")
        names = results.names
        for i, ap in enumerate(results.box.ap50):
            class_name = names.get(i, f"Class {i}") if isinstance(names, dict) else names[i]
            print(f"  {class_name}: {ap:.4f}")

    print(f"{'='*60}")


def compare_models(
    yolo26_weights: str,
    yolov9_weights: str,
    data_config: str,
    split: str = 'test',
    conf_thres: float = 0.25,
    device: int = 0,
    wandb_enabled: bool = False
):
    """
    Compare YOLO26 with YOLOv9 baseline.

    Args:
        yolo26_weights: Path to YOLO26 weights
        yolov9_weights: Path to YOLOv9 weights
        data_config: Path to data config
        split: Dataset split
        conf_thres: Confidence threshold
        device: GPU device
        wandb_enabled: Whether WandB logging is enabled
    """
    from ultralytics import YOLO

    print(f"\n{'='*70}")
    print("YOLO26 vs YOLOv9 Comparison")
    print(f"{'='*70}")
    print(f"Split: {split}")
    print(f"Confidence threshold: {conf_thres}")
    print()

    results = {}

    # Validate YOLO26
    print("Evaluating YOLO26...")
    yolo26_model = YOLO(yolo26_weights)
    results['yolo26'] = yolo26_model.val(
        data=data_config,
        split=split,
        conf=conf_thres,
        device=device,
        verbose=False
    )

    # Validate YOLOv9
    print("Evaluating YOLOv9...")
    try:
        # YOLOv9 might need different loading
        yolov9_model = YOLO(yolov9_weights)
        results['yolov9'] = yolov9_model.val(
            data=data_config,
            split=split,
            conf=conf_thres,
            device=device,
            verbose=False
        )
    except Exception as e:
        print(f"Warning: Could not load YOLOv9 model: {e}")
        results['yolov9'] = None

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Model':<15} {'mAP@0.5':<12} {'mAP@0.5:0.95':<14} {'Precision':<12} {'Recall':<12}")
    print(f"{'-'*70}")

    for model_name, res in results.items():
        if res is not None:
            print(f"{model_name:<15} {res.box.map50:<12.4f} {res.box.map:<14.4f} "
                  f"{res.box.mp:<12.4f} {res.box.mr:<12.4f}")
        else:
            print(f"{model_name:<15} {'N/A':<12} {'N/A':<14} {'N/A':<12} {'N/A':<12}")

    print(f"{'='*70}")

    # Print delta if both models available
    if results['yolo26'] is not None and results['yolov9'] is not None:
        print(f"\nDelta (YOLO26 - YOLOv9):")
        print(f"  mAP@0.5:     {results['yolo26'].box.map50 - results['yolov9'].box.map50:+.4f}")
        print(f"  mAP@0.5:0.95: {results['yolo26'].box.map - results['yolov9'].box.map:+.4f}")
        print(f"  Recall:      {results['yolo26'].box.mr - results['yolov9'].box.mr:+.4f}")

    # Log comparison to WandB
    log_comparison_to_wandb(results, wandb_enabled)

    return results


def benchmark_model(weights: str, imgsz: int = 640, device: int = 0):
    """
    Benchmark model inference speed and export to ONNX/TensorRT.

    Args:
        weights: Path to model weights
        imgsz: Image size for benchmark
        device: GPU device
    """
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print("Model Benchmarking")
    print(f"{'='*60}")

    model = YOLO(weights)

    # GPU benchmark
    print("\nGPU Benchmark:")
    model.benchmark(imgsz=imgsz, device=device)

    # Export options
    print("\nExporting to ONNX...")
    try:
        onnx_path = model.export(format="onnx", imgsz=imgsz)
        print(f"  ONNX exported to: {onnx_path}")
    except Exception as e:
        print(f"  ONNX export failed: {e}")

    print("\nExporting to TensorRT...")
    try:
        trt_path = model.export(format="engine", imgsz=imgsz, device=device)
        print(f"  TensorRT exported to: {trt_path}")
    except Exception as e:
        print(f"  TensorRT export failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate YOLO26 models and compare with YOLOv9'
    )

    parser.add_argument('--weights', type=str,
                        default=str(PROJECT_ROOT / 'runs/finetune/yolo26s_2class_v12/weights/best.pt'),
                        help='Path to model weights')
    parser.add_argument('--data', type=str,
                        default=str(PROJECT_ROOT / 'configs/data/data_2class.yaml'),
                        help='Path to data config')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to validate on')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--project', type=str, default=None, help='Output project')
    parser.add_argument('--name', type=str, default=None, help='Run name')

    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Compare YOLO26 with YOLOv9 baseline')
    parser.add_argument('--yolov9-weights', type=str,
                        default='/home/ubuntu/ImgAssistClone/imgAssist/modelGenerationTool/runs/train/PatientSplit_v12_unstretch_SGD_s/weights/best.pt',
                        help='Path to YOLOv9 weights for comparison')

    # Benchmark mode
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference benchmark and export')

    # WandB settings
    parser.add_argument('--wandb-project', type=str, default='ImgAssist_YOLO26',
                        help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='WandB entity (team/username)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB logging')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("YOLO26 Validation")
    print(f"{'='*60}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup WandB
    wandb_enabled = setup_wandb(args)
    if wandb_enabled:
        print(f"WandB logging enabled (project: {args.wandb_project})")
    else:
        print("WandB logging disabled")

    if args.compare:
        # Comparison mode
        compare_models(
            yolo26_weights=args.weights,
            yolov9_weights=args.yolov9_weights,
            data_config=args.data,
            split=args.split,
            conf_thres=args.conf,
            device=args.device,
            wandb_enabled=wandb_enabled
        )
    elif args.benchmark:
        # Benchmark mode
        benchmark_model(
            weights=args.weights,
            imgsz=args.imgsz,
            device=args.device
        )
    else:
        # Standard validation
        results = validate_model(
            weights=args.weights,
            data_config=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch_size=args.batch,
            conf_thres=args.conf,
            iou_thres=args.iou,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name
        )

        print_results(results, model_name=Path(args.weights).stem)

        # Log to WandB
        log_validation_results_to_wandb(results, Path(args.weights).stem, wandb_enabled)

    # Finish WandB run
    if wandb_enabled and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
