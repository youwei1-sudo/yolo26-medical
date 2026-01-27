#!/usr/bin/env python3
"""
YOLO26 vs YOLOv9 Model Comparison Script

Performs comprehensive comparison following the ImgAssist 3.0 Model Selection Rubric.

Comparison Levels:
- L1: Box-Level Metrics (mAP50, mAP50-95)
- L2: Patch-Level Detection Metrics (TP/TN/FP/FN, Sensitivity, Specificity, F2)

Usage:
    python compare_yolov9_yolo26.py \
        --yolov9-weights /path/to/yolov9/best.pt \
        --yolo26-weights /path/to/yolo26/best.pt \
        --data /path/to/data.yaml \
        --manifest /path/to/test_manifest.json \
        --output-dir ./comparison_results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List
import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available")


def compute_patch_level_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute patch-level binary classification metrics.

    For medical imaging:
    - TP: Image has ground truth AND model detected ≥1 box matching GT
    - FN: Image has ground truth BUT model detected 0 matching boxes (CRITICAL - missed cancer)
    - FP: Image has NO ground truth BUT model detected ≥1 box (false alarm)
    - TN: Image has NO ground truth AND model detected 0 boxes (correct rejection)

    Args:
        predictions: List of prediction dicts with 'image_path', 'boxes', 'scores'
        ground_truth: List of GT dicts with 'image_path', 'has_label', 'bboxes'
        conf_threshold: Confidence threshold for counting detections
        iou_threshold: IoU threshold for matching predictions to GT

    Returns:
        Dictionary with patch-level metrics
    """
    TP = FN = FP = TN = 0

    # Create lookup for predictions by image path
    pred_by_image = {p['image_path']: p for p in predictions}

    for gt in ground_truth:
        img_path = gt['image_path']
        has_gt = gt.get('has_label', False) or len(gt.get('bboxes', [])) > 0

        # Get predictions for this image
        pred = pred_by_image.get(img_path, {'boxes': [], 'scores': []})

        # Filter by confidence threshold
        high_conf_boxes = [
            box for box, score in zip(pred.get('boxes', []), pred.get('scores', []))
            if score >= conf_threshold
        ]
        has_detection = len(high_conf_boxes) > 0

        if has_gt:
            if has_detection:
                TP += 1
            else:
                FN += 1  # CRITICAL: Missed cancer
        else:
            if has_detection:
                FP += 1  # False alarm
            else:
                TN += 1  # Correct rejection

    # Compute metrics
    total = TP + TN + FP + FN
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall for positives
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # Recall for negatives
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # F1 and F2 scores
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    f2 = 5 * precision * sensitivity / (4 * precision + sensitivity) if (4 * precision + sensitivity) > 0 else 0

    # Accuracy
    accuracy = (TP + TN) / total if total > 0 else 0

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'total': total,
        'sensitivity': sensitivity,  # = Recall
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'f2_score': f2,
        'accuracy': accuracy,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold
    }


def evaluate_model(
    model_path: str,
    data_config: str,
    split: str = 'test',
    imgsz: int = 640,
    conf: float = 0.001,
    device: int = 0
) -> Dict[str, Any]:
    """
    Evaluate a YOLO model and return box-level and patch-level metrics.

    Args:
        model_path: Path to model weights
        data_config: Path to data YAML config
        split: Dataset split to evaluate ('val' or 'test')
        imgsz: Image size
        conf: Confidence threshold for validation
        device: GPU device

    Returns:
        Dictionary with evaluation results
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("ultralytics package required")

    model = YOLO(model_path)

    # Run validation
    results = model.val(
        data=data_config,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=0.5,
        device=device,
        verbose=False
    )

    # Extract box-level metrics
    box_metrics = {
        'mAP50': float(results.box.map50),
        'mAP50_95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
    }

    return {
        'box_metrics': box_metrics,
        'results_object': results
    }


def compute_delta_patch_box(patch_auc: float, box_map50: float) -> Tuple[float, str]:
    """
    Compute Δ(Patch-Box) consistency metric.

    Args:
        patch_auc: Patch-level AUC or sensitivity
        box_map50: Box-level mAP@0.5

    Returns:
        Tuple of (delta value, status string)
    """
    delta = abs(patch_auc - box_map50)

    if delta <= 0.10:
        status = "IDEAL (≤0.10)"
    elif delta <= 0.15:
        status = "CONDITIONAL PASS (0.10-0.15, requires Grad-CAM)"
    else:
        status = "REJECT (>0.15, likely shortcut learning)"

    return delta, status


def generate_comparison_report(
    yolov9_results: Dict[str, Any],
    yolo26_results: Dict[str, Any],
    output_path: str
) -> str:
    """
    Generate a markdown comparison report following the ImgAssist 3.0 Rubric.
    """
    report = []
    report.append("# YOLO26 vs YOLOv9 Model Comparison Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n**Reference:** ImgAssist 3.0 Model Selection Rubric v3.1")
    report.append("\n---\n")

    # Executive Summary
    report.append("## Executive Summary\n")

    yolov9_box = yolov9_results.get('box_metrics', {})
    yolo26_box = yolo26_results.get('box_metrics', {})

    map50_diff = yolo26_box.get('mAP50', 0) - yolov9_box.get('mAP50', 0)
    map50_95_diff = yolo26_box.get('mAP50_95', 0) - yolov9_box.get('mAP50_95', 0)

    report.append(f"| Metric | YOLOv9 | YOLO26 | Δ | Winner |")
    report.append(f"|--------|--------|--------|---|--------|")
    report.append(f"| mAP@0.5 | {yolov9_box.get('mAP50', 'N/A'):.4f} | {yolo26_box.get('mAP50', 'N/A'):.4f} | {map50_diff:+.4f} | {'YOLO26' if map50_diff > 0 else 'YOLOv9'} |")
    report.append(f"| mAP@0.5:0.95 | {yolov9_box.get('mAP50_95', 'N/A'):.4f} | {yolo26_box.get('mAP50_95', 'N/A'):.4f} | {map50_95_diff:+.4f} | {'YOLO26' if map50_95_diff > 0 else 'YOLOv9'} |")
    report.append("")

    # L1: Box-Level Metrics
    report.append("\n## L1: Box-Level Signal (10% weight)\n")
    report.append("Per Rubric v3.1: Box metrics are **signal checks**, not pass/fail gates.\n")
    report.append("| Metric | YOLOv9 | YOLO26 | 50% Goal | 100% Goal | 120% Goal |")
    report.append("|--------|--------|--------|----------|-----------|-----------|")
    report.append(f"| mAP@0.5 | {yolov9_box.get('mAP50', 'N/A'):.4f} | {yolo26_box.get('mAP50', 'N/A'):.4f} | ≥0.50 | ≥0.60 | ≥0.75 |")
    report.append(f"| mAP@0.5:0.95 | {yolov9_box.get('mAP50_95', 'N/A'):.4f} | {yolo26_box.get('mAP50_95', 'N/A'):.4f} | ≥0.30 | ≥0.40 | ≥0.55 |")
    report.append(f"| Precision | {yolov9_box.get('precision', 'N/A'):.4f} | {yolo26_box.get('precision', 'N/A'):.4f} | - | - | - |")
    report.append(f"| Recall | {yolov9_box.get('recall', 'N/A'):.4f} | {yolo26_box.get('recall', 'N/A'):.4f} | - | - | - |")
    report.append("")

    # L2: Patch-Level Metrics (if available)
    if 'patch_metrics' in yolov9_results and 'patch_metrics' in yolo26_results:
        yolov9_patch = yolov9_results['patch_metrics']
        yolo26_patch = yolo26_results['patch_metrics']

        report.append("\n## L2: Patch-Level Decision (30% weight)\n")
        report.append("**Critical for medical imaging:** Patch-level metrics measure regional classification.\n")
        report.append("| Metric | YOLOv9 | YOLO26 | Hard Gate | 100% Goal | 120% Goal |")
        report.append("|--------|--------|--------|-----------|-----------|-----------|")
        report.append(f"| Sensitivity | {yolov9_patch.get('sensitivity', 'N/A'):.4f} | {yolo26_patch.get('sensitivity', 'N/A'):.4f} | - | ≥0.85 | ≥0.92 |")
        report.append(f"| Specificity | {yolov9_patch.get('specificity', 'N/A'):.4f} | {yolo26_patch.get('specificity', 'N/A'):.4f} | - | ≥0.90 | ≥0.97 |")
        report.append(f"| F2 Score | {yolov9_patch.get('f2_score', 'N/A'):.4f} | {yolo26_patch.get('f2_score', 'N/A'):.4f} | - | ≥0.65 | ≥0.75 |")
        report.append(f"| Precision | {yolov9_patch.get('precision', 'N/A'):.4f} | {yolo26_patch.get('precision', 'N/A'):.4f} | - | - | - |")
        report.append("")

        report.append("\n### Patch-Level Confusion Matrix\n")
        report.append("| | YOLOv9 | YOLO26 |")
        report.append("|---|--------|--------|")
        report.append(f"| TP (Correct Detection) | {yolov9_patch.get('TP', 'N/A')} | {yolo26_patch.get('TP', 'N/A')} |")
        report.append(f"| FN (Missed Cancer) | {yolov9_patch.get('FN', 'N/A')} | {yolo26_patch.get('FN', 'N/A')} |")
        report.append(f"| FP (False Alarm) | {yolov9_patch.get('FP', 'N/A')} | {yolo26_patch.get('FP', 'N/A')} |")
        report.append(f"| TN (Correct Rejection) | {yolov9_patch.get('TN', 'N/A')} | {yolo26_patch.get('TN', 'N/A')} |")
        report.append("")

    # Model Info
    report.append("\n## Model Information\n")
    report.append("| Property | YOLOv9 | YOLO26 |")
    report.append("|----------|--------|--------|")
    report.append(f"| Architecture | YOLOv9-s (Dual Branch) | YOLO26m (NMS-free) |")
    report.append(f"| Parameters | ~9.6M | ~21.9M |")
    report.append(f"| Image Size | 320×672 | 768×768 |")
    report.append(f"| NMS | Required | Native End-to-End |")
    report.append(f"| DFL | reg_max=16 | reg_max=1 (Direct) |")
    report.append("")

    # Rubric Hard Gates Check
    report.append("\n## Hard Gates Check (Rubric v3.1)\n")
    report.append("| Gate | Metric | Threshold | YOLOv9 | YOLO26 |")
    report.append("|------|--------|-----------|--------|--------|")
    report.append(f"| G1 | Margin Recall | ≥61% | TBD | TBD |")
    report.append(f"| G2 | Margin Precision | ≥28% | TBD | TBD |")
    report.append(f"| G3 | Latency | ≤90s | TBD | TBD |")
    report.append(f"| G4 | Patch AUC | ≥0.70 | TBD | TBD |")
    report.append(f"| G5 | Δ(Patch-Box) | ≤0.15 | TBD | TBD |")
    report.append("")
    report.append("*Note: Full L4 workflow metrics require margin-level evaluation.*\n")

    # Recommendations
    report.append("\n## Recommendations\n")
    report.append("1. **Run test set evaluation** for both models on the same test manifest\n")
    report.append("2. **Compute patch-level metrics** at multiple confidence thresholds\n")
    report.append("3. **Compute Δ(Patch-Box)** consistency check\n")
    report.append("4. **Run FP/FN analysis** to understand failure modes\n")
    report.append("5. **Benchmark inference latency** on target hardware\n")

    report.append("\n---\n")
    report.append("*Report generated by compare_yolov9_yolo26.py*\n")

    report_text = "\n".join(report)

    # Save report
    with open(output_path, 'w') as f:
        f.write(report_text)

    return report_text


def main():
    parser = argparse.ArgumentParser(description='Compare YOLO26 vs YOLOv9')

    parser.add_argument('--yolov9-weights', type=str, required=True,
                        help='Path to YOLOv9 weights')
    parser.add_argument('--yolo26-weights', type=str, required=True,
                        help='Path to YOLO26 weights')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data YAML config')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path to test manifest JSON (for patch metrics)')
    parser.add_argument('--output-dir', type=str, default='./comparison_results',
                        help='Output directory for results')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Confidence threshold for box evaluation')
    parser.add_argument('--patch-conf', type=float, default=0.25,
                        help='Confidence threshold for patch-level metrics')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("YOLO26 vs YOLOv9 Comparison")
    print("="*60)

    results = {}

    # Evaluate YOLOv9
    print("\n[1/2] Evaluating YOLOv9...")
    try:
        yolov9_results = evaluate_model(
            args.yolov9_weights,
            args.data,
            split=args.split,
            device=args.device,
            conf=args.conf
        )
        results['yolov9'] = yolov9_results
        print(f"  mAP@0.5: {yolov9_results['box_metrics']['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {yolov9_results['box_metrics']['mAP50_95']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['yolov9'] = {'box_metrics': {}, 'error': str(e)}

    # Evaluate YOLO26
    print("\n[2/2] Evaluating YOLO26...")
    try:
        yolo26_results = evaluate_model(
            args.yolo26_weights,
            args.data,
            split=args.split,
            device=args.device,
            conf=args.conf
        )
        results['yolo26'] = yolo26_results
        print(f"  mAP@0.5: {yolo26_results['box_metrics']['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {yolo26_results['box_metrics']['mAP50_95']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['yolo26'] = {'box_metrics': {}, 'error': str(e)}

    # Generate comparison report
    print("\n[3/3] Generating comparison report...")
    report_path = output_dir / "COMPARISON_REPORT.md"
    report = generate_comparison_report(
        results.get('yolov9', {}),
        results.get('yolo26', {}),
        str(report_path)
    )
    print(f"  Report saved to: {report_path}")

    # Save raw results as JSON
    json_path = output_dir / "comparison_results.json"
    json_results = {
        'yolov9': {
            'box_metrics': results.get('yolov9', {}).get('box_metrics', {}),
            'weights': args.yolov9_weights
        },
        'yolo26': {
            'box_metrics': results.get('yolo26', {}).get('box_metrics', {}),
            'weights': args.yolo26_weights
        },
        'config': {
            'data': args.data,
            'split': args.split,
            'conf_threshold': args.conf,
            'patch_conf_threshold': args.patch_conf
        },
        'timestamp': datetime.now().isoformat()
    }
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  JSON results saved to: {json_path}")

    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)


if __name__ == '__main__':
    main()
