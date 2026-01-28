#!/usr/bin/env python3
"""
YOLO26 Validation Script

Run validation on fine-tuned YOLO26 models and compute metrics.
Supports validation on train/val/test splits and comparison with YOLOv9.

Evaluation Modes:
    - box: Standard box-level metrics (mAP, precision, recall)
    - patch: Patch-level metrics (sensitivity, specificity, F2) - matches YOLOv9 evaluation

WandB Integration:
    - Logs validation metrics to WandB
    - Creates comparison tables for model evaluation
    - Tracks benchmark results

Usage:
    python val_yolo26.py --weights runs/finetune/yolo26s_2class_v12/weights/best.pt
    python val_yolo26.py --weights best.pt --split test --conf 0.25
    python val_yolo26.py --weights best.pt --split test --eval-mode patch --conf 0.001
    python val_yolo26.py --compare  # Compare with YOLOv9 baseline
    python val_yolo26.py --wandb-project ImgAssist_YOLO26
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Any, Optional

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


def compute_patch_metrics_from_predictions(
    predictions: Dict[str, List],
    labels: Dict[str, List],
    conf_thres: float = 0.001
) -> Dict[str, Any]:
    """
    Compute patch-level binary classification metrics.

    A patch is:
    - TP if it has GT boxes AND has predictions (above conf_thres)
    - TN if it has no GT boxes AND has no predictions
    - FP if it has no GT boxes BUT has predictions
    - FN if it has GT boxes BUT has no predictions

    Args:
        predictions: Dict mapping image_name to list of predictions with 'conf'
        labels: Dict mapping image_name to list of GT boxes
        conf_thres: Confidence threshold for predictions

    Returns:
        Dict with all computed metrics
    """
    # Get all image names from labels
    all_images = set(labels.keys())

    tp, tn, fp, fn = 0, 0, 0, 0

    for img_name in all_images:
        gt_boxes = labels.get(img_name, [])
        pred_boxes = predictions.get(img_name, [])

        # Filter predictions by confidence
        if pred_boxes:
            pred_boxes = [p for p in pred_boxes if p.get('conf', 1.0) >= conf_thres]

        has_gt = len(gt_boxes) > 0
        has_pred = len(pred_boxes) > 0

        if has_gt and has_pred:
            tp += 1
        elif not has_gt and not has_pred:
            tn += 1
        elif not has_gt and has_pred:
            fp += 1
        else:  # has_gt and not has_pred
            fn += 1

    # Compute metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall = sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        'total_patches': total,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'recall': recall,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'f1_score': f1,
        'f2_score': f2,
        'positive_rate': (tp + fn) / total if total > 0 else 0,
        'negative_rate': (tn + fp) / total if total > 0 else 0,
        'conf_threshold': conf_thres
    }


def load_labels_from_txt_dir(labels_dir: str) -> Dict[str, List]:
    """Load ground truth labels from directory of YOLO-format txt files."""
    labels_dir = Path(labels_dir)
    labels_by_image = {}

    for txt_file in labels_dir.glob('*.txt'):
        img_name = txt_file.stem
        labels = []

        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    labels.append({
                        'class': cls,
                        'bbox': [x, y, w, h]
                    })

        labels_by_image[img_name] = labels

    return labels_by_image


def run_patch_level_evaluation(
    weights: str,
    data_config: str,
    labels_dir: str,
    split: str = 'test',
    imgsz: int = 640,
    conf_thres: float = 0.001,
    device: int = 0,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run patch-level evaluation matching YOLOv9's evaluation methodology.

    Args:
        weights: Path to model weights
        data_config: Path to data YAML config
        labels_dir: Directory containing ground truth labels
        split: Dataset split
        imgsz: Image size
        conf_thres: Confidence threshold (default 0.001 for high sensitivity)
        device: GPU device
        save_dir: Directory to save predictions

    Returns:
        Dict with patch-level metrics
    """
    from ultralytics import YOLO
    import yaml

    print(f"\nLoading model: {weights}")
    model = YOLO(weights)

    # Load data config to get image list
    with open(data_config, 'r') as f:
        data_cfg = yaml.safe_load(f)

    data_path = Path(data_cfg.get('path', ''))
    split_file = data_path / f"{split}.txt"

    if not split_file.exists():
        print(f"Warning: Split file not found: {split_file}")
        # Try to construct from data config path
        split_file = Path(data_config).parent / f"{split}.txt"

    # Load image list
    image_paths = []
    if split_file.exists():
        with open(split_file, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        print(f"Warning: Could not find image list file for split '{split}'")
        return {}

    print(f"Running inference on {len(image_paths)} images...")
    print(f"Confidence threshold: {conf_thres}")

    # Run inference and collect predictions
    predictions = {}
    batch_size = 16

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        results = model.predict(
            source=batch_paths,
            imgsz=imgsz,
            conf=conf_thres,
            device=device,
            verbose=False
        )

        for path, result in zip(batch_paths, results):
            img_name = Path(path).stem
            preds = []

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    preds.append({
                        'conf': float(box.conf[0]),
                        'cls': int(box.cls[0]),
                        'xyxy': box.xyxy[0].tolist()
                    })

            predictions[img_name] = preds

        if (i + batch_size) % 1000 == 0 or i + batch_size >= len(image_paths):
            print(f"  Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")

    # Load ground truth labels
    print(f"\nLoading ground truth labels from: {labels_dir}")
    labels = load_labels_from_txt_dir(labels_dir)
    print(f"  Loaded labels for {len(labels)} images")

    # Compute patch-level metrics
    print(f"\nComputing patch-level metrics...")
    metrics = compute_patch_metrics_from_predictions(predictions, labels, conf_thres)

    # Save predictions if requested
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        pred_file = save_path / 'predictions.json'
        with open(pred_file, 'w') as f:
            json.dump(predictions, f)
        print(f"\nPredictions saved to: {pred_file}")

        metrics_file = save_path / 'patch_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")

    return metrics


def print_patch_metrics(metrics: Dict[str, Any], title: str = "Patch-Level Metrics"):
    """Print formatted patch-level metrics report."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Pos      Neg")
    print(f"Actual Pos   {metrics['tp']:>5}    {metrics['fn']:>5}   (TP, FN)")
    print(f"Actual Neg   {metrics['fp']:>5}    {metrics['tn']:>5}   (FP, TN)")

    print(f"\nPatch Statistics:")
    print(f"  Total patches:     {metrics['total_patches']}")
    print(f"  Positive patches:  {metrics['tp'] + metrics['fn']} ({metrics['positive_rate']*100:.1f}%)")
    print(f"  Negative patches:  {metrics['tn'] + metrics['fp']} ({metrics['negative_rate']*100:.1f}%)")

    print(f"\nCore Metrics (conf={metrics.get('conf_threshold', 'N/A')}):")
    print(f"  Sensitivity (Recall): {metrics['sensitivity']:.4f}  [TP/(TP+FN)] - Don't miss positives")
    print(f"  Specificity:          {metrics['specificity']:.4f}  [TN/(TN+FP)]")
    print(f"  Precision (PPV):      {metrics['precision']:.4f}  [TP/(TP+FP)]")
    print(f"  NPV:                  {metrics['npv']:.4f}  [TN/(TN+FN)]")

    print(f"\nCombined Metrics:")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  F1 Score:   {metrics['f1_score']:.4f}")
    print(f"  F2 Score:   {metrics['f2_score']:.4f}  [Weights recall 4x > precision]")

    print(f"\n{'='*60}")


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

    # Evaluation mode
    parser.add_argument('--eval-mode', type=str, default='box',
                        choices=['box', 'patch'],
                        help='Evaluation mode: box (standard mAP) or patch (patch-level sensitivity/specificity)')
    parser.add_argument('--labels-dir', type=str, default=None,
                        help='Labels directory for patch-level evaluation (defaults to data/labels/{split})')

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
    elif args.eval_mode == 'patch':
        # Patch-level evaluation mode (YOLOv9-style)
        print(f"\nEvaluation Mode: PATCH-LEVEL")
        print(f"  This matches YOLOv9's evaluation methodology")
        print(f"  Confidence threshold: {args.conf}")

        # Determine labels directory
        labels_dir = args.labels_dir
        if labels_dir is None:
            labels_dir = str(PROJECT_ROOT / 'data' / 'labels' / args.split)

        # Run patch-level evaluation
        save_dir = args.project or str(PROJECT_ROOT / 'runs' / 'val')
        save_dir = Path(save_dir) / (args.name or f"patch_eval_{args.split}")

        metrics = run_patch_level_evaluation(
            weights=args.weights,
            data_config=args.data,
            labels_dir=labels_dir,
            split=args.split,
            imgsz=args.imgsz,
            conf_thres=args.conf,
            device=args.device,
            save_dir=str(save_dir)
        )

        if metrics:
            print_patch_metrics(metrics, title=f"Patch-Level Metrics ({args.split})")

            # Log to WandB
            if wandb_enabled and WANDB_AVAILABLE:
                try:
                    wandb.log({
                        'patch/sensitivity': metrics['sensitivity'],
                        'patch/specificity': metrics['specificity'],
                        'patch/precision': metrics['precision'],
                        'patch/f1_score': metrics['f1_score'],
                        'patch/f2_score': metrics['f2_score'],
                        'patch/accuracy': metrics['accuracy'],
                        'patch/tp': metrics['tp'],
                        'patch/tn': metrics['tn'],
                        'patch/fp': metrics['fp'],
                        'patch/fn': metrics['fn'],
                    })
                except Exception as e:
                    print(f"Warning: Failed to log patch metrics to WandB: {e}")
    else:
        # Standard box-level validation
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
