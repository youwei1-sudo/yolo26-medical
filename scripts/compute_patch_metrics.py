#!/usr/bin/env python3
"""
Compute Patch-Level Metrics for Medical Image Detection

This script computes medical-specific metrics that treat each image patch
as a binary classification (positive/negative) rather than evaluating
individual bounding boxes.

Metrics computed:
- TP: Patches with GT boxes that have at least one detection
- TN: Patches without GT boxes and without detections
- FP: Patches without GT boxes but with detections
- FN: Patches with GT boxes but without detections
- Sensitivity (Recall): TP / (TP + FN) - critical for cancer detection
- Specificity: TN / (TN + FP)
- Precision: TP / (TP + FP)
- F1 Score: 2 * P * R / (P + R)
- F2 Score: 5 * P * R / (4 * P + R) - weights recall 4x more than precision

Usage:
    python compute_patch_metrics.py --predictions runs/val/exp/predictions.json --labels data/labels/test
    python compute_patch_metrics.py --pred-dir runs/val/exp/labels --labels data/labels/test
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


def load_predictions_from_json(json_path: str) -> Dict[str, List[Dict]]:
    """
    Load predictions from COCO-format JSON file.

    Returns:
        Dict mapping image_id to list of predictions
    """
    with open(json_path, 'r') as f:
        predictions = json.load(f)

    # Group predictions by image_id
    pred_by_image = {}
    for pred in predictions:
        img_id = str(pred['image_id'])
        if img_id not in pred_by_image:
            pred_by_image[img_id] = []
        pred_by_image[img_id].append(pred)

    return pred_by_image


def load_predictions_from_txt_dir(pred_dir: str, conf_thres: float = 0.25) -> Dict[str, List[Dict]]:
    """
    Load predictions from directory of YOLO-format txt files.

    Each txt file has format: class x_center y_center width height confidence

    Returns:
        Dict mapping image_name (stem) to list of predictions
    """
    pred_dir = Path(pred_dir)
    pred_by_image = {}

    for txt_file in pred_dir.glob('*.txt'):
        img_name = txt_file.stem
        predictions = []

        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0

                    if conf >= conf_thres:
                        predictions.append({
                            'class': cls,
                            'bbox': [x, y, w, h],
                            'confidence': conf
                        })

        pred_by_image[img_name] = predictions

    return pred_by_image


def load_labels_from_txt_dir(labels_dir: str) -> Dict[str, List[Dict]]:
    """
    Load ground truth labels from directory of YOLO-format txt files.

    Returns:
        Dict mapping image_name (stem) to list of GT boxes
    """
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


def load_labels_from_manifest(manifest_path: str) -> Dict[str, List[Dict]]:
    """
    Load ground truth labels from JSON manifest file.

    Returns:
        Dict mapping image_name (stem) to list of GT boxes
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    labels_by_image = {}

    for patient_id, patient_data in manifest.get('items', {}).items():
        for img_info in patient_data.get('images', []):
            img_path = img_info.get('image_path', '')
            img_name = Path(img_path).stem

            bboxes = img_info.get('bboxes', [])
            labels = []
            for bbox in bboxes:
                labels.append({
                    'class': bbox['class_id'],
                    'bbox': [
                        bbox['x_center'],
                        bbox['y_center'],
                        bbox['width'],
                        bbox['height']
                    ]
                })

            labels_by_image[img_name] = labels

    return labels_by_image


def compute_patch_metrics(
    predictions: Dict[str, List],
    labels: Dict[str, List],
    conf_thres: float = 0.25
) -> Dict[str, float]:
    """
    Compute patch-level binary classification metrics.

    A patch is:
    - TP if it has GT boxes AND has predictions (above conf_thres)
    - TN if it has no GT boxes AND has no predictions
    - FP if it has no GT boxes BUT has predictions
    - FN if it has GT boxes BUT has no predictions

    Args:
        predictions: Dict mapping image_name to list of predictions
        labels: Dict mapping image_name to list of GT boxes
        conf_thres: Confidence threshold for predictions

    Returns:
        Dict with all computed metrics
    """
    # Get all image names from both predictions and labels
    all_images = set(labels.keys())

    tp, tn, fp, fn = 0, 0, 0, 0

    for img_name in all_images:
        gt_boxes = labels.get(img_name, [])
        pred_boxes = predictions.get(img_name, [])

        # Filter predictions by confidence if they have confidence values
        if pred_boxes:
            pred_boxes = [
                p for p in pred_boxes
                if p.get('confidence', p.get('score', 1.0)) >= conf_thres
            ]

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

    # Sensitivity = Recall = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall = sensitivity

    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # F1 Score = 2 * P * R / (P + R)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # F2 Score = 5 * P * R / (4 * P + R) - weights recall 4x more than precision
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

    # Negative Predictive Value = TN / (TN + FN)
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
        'negative_rate': (tn + fp) / total if total > 0 else 0
    }


def print_metrics_report(metrics: Dict[str, float], title: str = "Patch-Level Metrics"):
    """Print formatted metrics report."""
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

    print(f"\nCore Metrics:")
    print(f"  Sensitivity (Recall): {metrics['sensitivity']:.4f}  [TP/(TP+FN)] - Don't miss positives")
    print(f"  Specificity:          {metrics['specificity']:.4f}  [TN/(TN+FP)]")
    print(f"  Precision (PPV):      {metrics['precision']:.4f}  [TP/(TP+FP)]")
    print(f"  NPV:                  {metrics['npv']:.4f}  [TN/(TN+FN)]")

    print(f"\nCombined Metrics:")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  F1 Score:   {metrics['f1_score']:.4f}")
    print(f"  F2 Score:   {metrics['f2_score']:.4f}  [Weights recall 4x > precision]")

    print(f"\n{'='*60}")


def save_metrics_json(metrics: Dict[str, float], output_path: str):
    """Save metrics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute patch-level metrics for medical image detection'
    )

    # Prediction input (one of these required)
    pred_group = parser.add_mutually_exclusive_group(required=True)
    pred_group.add_argument(
        '--predictions',
        type=str,
        help='Path to COCO-format predictions JSON file'
    )
    pred_group.add_argument(
        '--pred-dir',
        type=str,
        help='Directory containing YOLO-format prediction txt files'
    )

    # Label input (one of these required)
    label_group = parser.add_mutually_exclusive_group(required=True)
    label_group.add_argument(
        '--labels',
        type=str,
        help='Directory containing YOLO-format label txt files'
    )
    label_group.add_argument(
        '--manifest',
        type=str,
        help='Path to JSON manifest file with GT labels'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for predictions'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for metrics'
    )
    parser.add_argument(
        '--title',
        type=str,
        default='Patch-Level Metrics',
        help='Title for metrics report'
    )

    args = parser.parse_args()

    # Load predictions
    print("Loading predictions...")
    if args.predictions:
        predictions = load_predictions_from_json(args.predictions)
    else:
        predictions = load_predictions_from_txt_dir(args.pred_dir, args.conf)
    print(f"  Loaded predictions for {len(predictions)} images")

    # Load labels
    print("Loading ground truth labels...")
    if args.labels:
        labels = load_labels_from_txt_dir(args.labels)
    else:
        labels = load_labels_from_manifest(args.manifest)
    print(f"  Loaded labels for {len(labels)} images")

    # Compute metrics
    print(f"\nComputing metrics (conf threshold: {args.conf})...")
    metrics = compute_patch_metrics(predictions, labels, args.conf)

    # Print report
    print_metrics_report(metrics, args.title)

    # Save to JSON if requested
    if args.output:
        save_metrics_json(metrics, args.output)


if __name__ == '__main__':
    main()
