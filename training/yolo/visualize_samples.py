"""
Visualize synthetic training data with bounding boxes.

Draws YOLO format bounding boxes on sample images to verify correctness.
"""

import cv2
import numpy as np
from pathlib import Path
import random


def draw_yolo_boxes(image: np.ndarray, label_file: Path, color=(0, 255, 0), thickness=2):
    """
    Draw YOLO format bounding boxes on image.

    Args:
        image: Image to draw on
        label_file: Path to YOLO format label file
        color: Box color (B, G, R)
        thickness: Box line thickness
    """
    h, w = image.shape[:2]

    if not label_file.exists():
        return image

    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, x_center, y_center, width, height = map(float, parts)

            # Convert normalized to pixel coordinates
            x_center_px = int(x_center * w)
            y_center_px = int(y_center * h)
            width_px = int(width * w)
            height_px = int(height * h)

            # Calculate top-left and bottom-right
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # Draw center point
            cv2.circle(image, (x_center_px, y_center_px), 3, (0, 0, 255), -1)

            # Add label
            label = f"card {width_px}x{height_px}"
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image


def visualize_dataset(dataset_dir: Path, num_samples: int = 20, output_dir: Path = None):
    """
    Visualize random samples from dataset.

    Args:
        dataset_dir: Root directory of YOLO dataset
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations (if None, display only)
    """
    train_images = list((dataset_dir / "images" / "train").glob("*.jpg"))
    val_images = list((dataset_dir / "images" / "val").glob("*.jpg"))

    print(f"Found {len(train_images)} training images")
    print(f"Found {len(val_images)} validation images")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Sample from training set
    print(f"\nVisualizing {num_samples} training samples...")
    for i, img_path in enumerate(random.sample(train_images, min(num_samples, len(train_images)))):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load {img_path}")
            continue

        # Get corresponding label file
        label_path = dataset_dir / "labels" / "train" / (img_path.stem + ".txt")

        # Draw boxes
        image = draw_yolo_boxes(image, label_path)

        if output_dir:
            output_path = output_dir / f"train_vis_{i:03d}.jpg"
            cv2.imwrite(str(output_path), image)
            print(f"  Saved: {output_path.name}")
        else:
            cv2.imshow(f"Training Sample {i}", image)
            cv2.waitKey(500)

    # Sample from validation set
    print(f"\nVisualizing {num_samples} validation samples...")
    for i, img_path in enumerate(random.sample(val_images, min(num_samples, len(val_images)))):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load {img_path}")
            continue

        # Get corresponding label file
        label_path = dataset_dir / "labels" / "val" / (img_path.stem + ".txt")

        # Draw boxes
        image = draw_yolo_boxes(image, label_path)

        if output_dir:
            output_path = output_dir / f"val_vis_{i:03d}.jpg"
            cv2.imwrite(str(output_path), image)
            print(f"  Saved: {output_path.name}")
        else:
            cv2.imshow(f"Validation Sample {i}", image)
            cv2.waitKey(500)

    if not output_dir:
        cv2.destroyAllWindows()


def main():
    project_root = Path(__file__).parent.parent.parent
    dataset_dir = project_root / "training" / "data" / "yolo_dataset"
    output_dir = dataset_dir / "visualizations"

    print("=" * 60)
    print("YOLO Dataset Visualization")
    print("=" * 60)

    visualize_dataset(dataset_dir, num_samples=20, output_dir=output_dir)

    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
