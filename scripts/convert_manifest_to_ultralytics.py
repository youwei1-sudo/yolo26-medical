#!/usr/bin/env python3
"""
Convert JSON manifest format to Ultralytics YOLO format.

This script reads the existing JSON manifests and creates:
1. train.txt, val.txt, test.txt with image paths
2. Symlinks for labels in a standardized location that Ultralytics expects
3. data.yaml configuration file

The Ultralytics YOLO format expects labels in a 'labels' folder parallel to 'images':
  /path/to/images/image.png
  /path/to/labels/image.txt

Since our existing labels are in non-standard paths, we create symlinks.

Rectangle Preprocessing Option (--rectangle):
    When enabled, applies YOLOv9 v12.1 style rectangle letterbox preprocessing:
    - Scale image to fit target height (320) while maintaining aspect ratio
    - Pad horizontally to target width (672)
    - Transform bounding box coordinates accordingly

    Pipeline: Original (672x~420) → Scale (0.76) → (~512x320) → Pad → (672x320)

    This matches YOLOv9 v12.1 configuration:
        imgsz: [320, 672]
        rectangle_input: true
        squash_enabled: false
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def load_manifest(manifest_path: str) -> Dict:
    """Load a JSON manifest file."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def create_label_content(bboxes: List[Dict]) -> str:
    """Create YOLO format label content from bbox list."""
    lines = []
    for bbox in bboxes:
        class_id = bbox['class_id']
        x_center = bbox['x_center']
        y_center = bbox['y_center']
        width = bbox['width']
        height = bbox['height']
        lines.append(f"{class_id} {x_center} {y_center} {width} {height}")
    return '\n'.join(lines)


class YAxisSquashPreprocessor:
    """
    Y-Axis Squash preprocessor - keeps width, only compresses height.

    Pipeline (for 672x420 original → 672x320 target):
    1. Keep original width (672)
    2. Squash height to target (320)
    3. No padding needed

    This is based on YOLOv9 v12's squash approach but with target_height=320.

    Key insight: Normalized bbox coordinates remain VALID after Y-axis squash because:
    - x coordinates: unchanged (width unchanged)
    - y coordinates: still valid because squash is uniform
      (a point at y=0.5 in original is still at y=0.5 after squash)
    """

    def __init__(
        self,
        target_height: int = 320,
    ):
        self.target_height = target_height

    def process(
        self,
        img: np.ndarray,
        bboxes: List[Dict]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Apply Y-axis squash preprocessing to image and bboxes.

        Args:
            img: Input image (H, W, C) in BGR format
            bboxes: List of bbox dicts with class_id, x_center, y_center, width, height

        Returns:
            Tuple of (processed_image, processed_bboxes)
        """
        h0, w0 = img.shape[:2]

        # Y-axis squash: keep width, only change height
        # cv2.resize takes (width, height)
        img_squashed = cv2.resize(img, (w0, self.target_height), interpolation=cv2.INTER_LINEAR)

        # Labels remain UNCHANGED - normalized coordinates are still valid
        # because squash is a uniform vertical compression
        # x_center, width: unchanged (width unchanged)
        # y_center, height: still valid in normalized form (0-1 range)

        return img_squashed, bboxes


# Backward compatibility alias
RectanglePreprocessor = YAxisSquashPreprocessor


# Keep old class for backward compatibility (deprecated)
class SquashPreprocessor(RectanglePreprocessor):
    """DEPRECATED: Use RectanglePreprocessor instead."""

    def __init__(self, target_height_ratio: float = 0.457, imgsz: int = 640, pad_value: int = 114):
        import warnings
        warnings.warn(
            "SquashPreprocessor is deprecated. Use RectanglePreprocessor with "
            "target_height=320, target_width=672 instead.",
            DeprecationWarning
        )
        super().__init__(target_height=320, target_width=672, pad_value=pad_value)


def process_manifest(
    manifest_path: str,
    output_dir: Path,
    split_name: str,
    create_labels: bool = True,
    create_symlinks: bool = True,
    preprocessor: Optional['RectanglePreprocessor'] = None
) -> List[str]:
    """
    Process a manifest file and create image list and labels.

    Args:
        manifest_path: Path to JSON manifest
        output_dir: Base output directory
        split_name: 'train', 'val', or 'test'
        create_labels: Whether to create label files
        create_symlinks: Whether to create symlinks in expected Ultralytics locations
        preprocessor: Optional RectanglePreprocessor for rectangle letterbox preprocessing

    Returns:
        List of image paths
    """
    manifest = load_manifest(manifest_path)

    images_dir = output_dir / 'images' / split_name
    labels_dir = output_dir / 'labels' / split_name

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    skipped = 0
    created = 0
    symlinks_created = 0
    processed_images = 0

    items = manifest.get('items', {})

    for patient_id, patient_data in items.items():
        for img_info in patient_data.get('images', []):
            image_path = img_info.get('image_path')

            if not image_path or not os.path.exists(image_path):
                skipped += 1
                continue

            # Get image filename (without extension) for label file
            img_name = Path(image_path).stem

            # Get bboxes
            bboxes = img_info.get('bboxes', [])

            # Apply squash preprocessing if enabled
            if preprocessor is not None:
                # Read image
                img = cv2.imread(image_path)
                if img is None:
                    skipped += 1
                    continue

                # Apply squash preprocessing
                img_processed, bboxes_processed = preprocessor.process(img, bboxes)

                # Save processed image
                output_img_path = images_dir / f"{img_name}.png"
                cv2.imwrite(str(output_img_path), img_processed)
                image_paths.append(str(output_img_path))
                processed_images += 1

                # Create label file with processed bboxes
                label_file = labels_dir / f"{img_name}.txt"
                if bboxes_processed:
                    label_content = create_label_content(bboxes_processed)
                    with open(label_file, 'w') as f:
                        f.write(label_content)
                    created += 1
                else:
                    label_file.touch()
                    created += 1

                if processed_images % 1000 == 0:
                    print(f"    Processed {processed_images} images...")

            else:
                # No squash preprocessing - use original image path
                image_paths.append(image_path)

                label_file = labels_dir / f"{img_name}.txt"

                # Create label file from bbox data in manifest
                if bboxes:
                    label_content = create_label_content(bboxes)
                    with open(label_file, 'w') as f:
                        f.write(label_content)
                    created += 1
                else:
                    # For negative images (no bboxes), create empty label file
                    # Ultralytics treats missing labels as background
                    label_file.touch()
                    created += 1

                # Create symlink in expected Ultralytics location
                # Ultralytics looks for labels by replacing /images/ with /labels/ in path
                if create_symlinks and '/images/' in image_path:
                    expected_label_path = image_path.replace('/images/', '/labels/')
                    expected_label_path = Path(expected_label_path).with_suffix('.txt')

                    # Create parent directory if needed
                    expected_label_path.parent.mkdir(parents=True, exist_ok=True)

                    # Create symlink if it doesn't exist
                    if not expected_label_path.exists():
                        try:
                            expected_label_path.symlink_to(label_file.resolve())
                            symlinks_created += 1
                        except FileExistsError:
                            pass  # Symlink already exists
                        except Exception as e:
                            # If symlink fails, copy the file instead
                            import shutil
                            try:
                                shutil.copy2(label_file, expected_label_path)
                                symlinks_created += 1
                            except Exception:
                                pass

    if preprocessor is not None:
        print(f"  {split_name}: {len(image_paths)} images processed with rectangle letterbox, "
              f"{created} labels created, {skipped} skipped")
    else:
        print(f"  {split_name}: {len(image_paths)} images, {created} labels created, "
              f"{symlinks_created} symlinks, {skipped} skipped")

    return image_paths


def create_image_list_file(image_paths: List[str], output_path: Path):
    """Create a text file with image paths (one per line)."""
    with open(output_path, 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")


def create_data_yaml(
    output_dir: Path,
    nc: int,
    names: List[str],
    train_path: str = 'train.txt',
    val_path: str = 'val.txt',
    test_path: str = 'test.txt'
):
    """Create Ultralytics data.yaml configuration file."""
    yaml_content = f"""# YOLO26 Medical Dataset Configuration
# Auto-generated from manifest files
#
# Clinical context:
# - Binary classification: Suspicious vs Negative
# - "Don't miss any suspicious region" is primary goal

path: {output_dir}
train: {train_path}
val: {val_path}
test: {test_path}

nc: {nc}
names: {names}
"""

    yaml_path = output_dir / 'data_2class.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON manifests to Ultralytics YOLO format'
    )
    parser.add_argument(
        '--manifest-dir',
        type=str,
        default='/home/ubuntu/ImgAssistClone/datasets/ImgAssist_Data_10142025/SEL_OTIS_2class_PatientSplit_Manifests_v12',
        help='Directory containing manifest JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/ubuntu/ImgAssistClone/experiments/yolo26/data',
        help='Output directory for Ultralytics format data'
    )
    parser.add_argument(
        '--nc',
        type=int,
        default=1,
        help='Number of classes'
    )
    parser.add_argument(
        '--names',
        type=str,
        nargs='+',
        default=['Suspicious'],
        help='Class names'
    )

    # Y-axis squash preprocessing options (YOLOv9 v12 style)
    parser.add_argument(
        '--yaxis-squash',
        action='store_true',
        help='Enable Y-axis squash preprocessing (keep width, compress height only)'
    )
    parser.add_argument(
        '--target-height',
        type=int,
        default=320,
        help='Target height for Y-axis squash (default: 320)'
    )

    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)

    # If Y-axis squash is enabled, modify output directory
    if args.yaxis_squash:
        output_dir = output_dir / f'yaxis_squash_h{args.target_height}'
        print(f"\nY-Axis Squash preprocessing ENABLED")
        print(f"  Keep original width, squash height to {args.target_height}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Define manifest files
    manifests = {
        'train': manifest_dir / 'train_manifest.json',
        'val': manifest_dir / 'validation_manifest.json',
        'test': manifest_dir / 'test_manifest.json'
    }

    # Verify manifests exist
    for split, path in manifests.items():
        if not path.exists():
            print(f"Error: {split} manifest not found: {path}")
            sys.exit(1)

    print("\nConverting manifests to Ultralytics format...")
    print(f"Output directory: {output_dir}")
    print()

    # Create Y-axis squash preprocessor if enabled
    preprocessor = None
    if args.yaxis_squash:
        preprocessor = YAxisSquashPreprocessor(
            target_height=args.target_height
        )

    # Process each split
    all_image_paths = {}
    for split, manifest_path in manifests.items():
        print(f"Processing {split} manifest...")
        image_paths = process_manifest(
            str(manifest_path),
            output_dir,
            split,
            preprocessor=preprocessor
        )
        all_image_paths[split] = image_paths

        # Create image list file
        list_file = output_dir / f"{split}.txt"
        create_image_list_file(image_paths, list_file)
        print(f"  Created: {list_file}")

    print()

    # Create data.yaml
    create_data_yaml(
        output_dir,
        nc=args.nc,
        names=args.names
    )

    # Print summary
    print("\n" + "="*50)
    print("Conversion Summary")
    print("="*50)
    if args.yaxis_squash:
        print(f"  Mode: Y-AXIS SQUASH PREPROCESSING")
        print(f"  Pipeline: Original (672x~420) → Y-squash → (672x{args.target_height})")
        print(f"  Width: unchanged (672)")
        print(f"  Height: squashed to {args.target_height}")
    else:
        print(f"  Mode: Standard (symlink-based)")
    print()
    for split, paths in all_image_paths.items():
        print(f"  {split}: {len(paths)} images")
    print(f"\nTotal: {sum(len(p) for p in all_image_paths.values())} images")
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    print(f"  - {output_dir}/train.txt")
    print(f"  - {output_dir}/val.txt")
    print(f"  - {output_dir}/test.txt")
    print(f"  - {output_dir}/data_2class.yaml")
    print(f"  - {output_dir}/labels/{{train,val,test}}/*.txt")
    if args.yaxis_squash:
        print(f"  - {output_dir}/images/{{train,val,test}}/*.png  (Y-axis squashed)")


if __name__ == '__main__':
    main()
