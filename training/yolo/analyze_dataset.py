"""
Analyze YOLO dataset statistics.

Provides insights into the generated synthetic data:
- Distribution of card counts per image
- Bounding box size distribution
- Label statistics
"""

from pathlib import Path
from collections import Counter
import numpy as np


def analyze_dataset(dataset_dir: Path):
    """Analyze dataset statistics."""
    print("=" * 60)
    print("YOLO Dataset Analysis")
    print("=" * 60)

    # Count images
    train_images = list((dataset_dir / "images" / "train").glob("*.jpg"))
    val_images = list((dataset_dir / "images" / "val").glob("*.jpg"))
    train_labels = list((dataset_dir / "labels" / "train").glob("*.txt"))
    val_labels = list((dataset_dir / "labels" / "val").glob("*.txt"))

    print(f"\nDataset Size:")
    print(f"  Training images:   {len(train_images):,}")
    print(f"  Training labels:   {len(train_labels):,}")
    print(f"  Validation images: {len(val_images):,}")
    print(f"  Validation labels: {len(val_labels):,}")

    # Analyze training labels
    print(f"\nAnalyzing Training Set...")
    train_stats = analyze_labels(dataset_dir / "labels" / "train")

    # Analyze validation labels
    print(f"\nAnalyzing Validation Set...")
    val_stats = analyze_labels(dataset_dir / "labels" / "val")

    # Combined stats
    print(f"\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images:      {len(train_images) + len(val_images):,}")
    print(f"Total card instances: {train_stats['total_cards'] + val_stats['total_cards']:,}")
    print(f"Avg cards/image:   {(train_stats['total_cards'] + val_stats['total_cards']) / (len(train_labels) + len(val_labels)):.2f}")


def analyze_labels(label_dir: Path):
    """Analyze label files in directory."""
    label_files = list(label_dir.glob("*.txt"))

    total_cards = 0
    cards_per_image = []
    box_widths = []
    box_heights = []
    box_areas = []

    for label_file in label_files:
        with open(label_file) as f:
            lines = f.readlines()
            num_cards = len(lines)
            cards_per_image.append(num_cards)
            total_cards += num_cards

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, x, y, w, h = map(float, parts)
                    box_widths.append(w)
                    box_heights.append(h)
                    box_areas.append(w * h)

    # Calculate statistics
    cards_per_image = np.array(cards_per_image)
    box_widths = np.array(box_widths)
    box_heights = np.array(box_heights)
    box_areas = np.array(box_areas)

    print(f"  Total labels: {len(label_files):,}")
    print(f"  Total cards:  {total_cards:,}")
    print(f"\nCards per image:")
    print(f"  Min:  {cards_per_image.min()}")
    print(f"  Max:  {cards_per_image.max()}")
    print(f"  Mean: {cards_per_image.mean():.2f}")
    print(f"  Median: {np.median(cards_per_image):.1f}")

    # Distribution
    card_dist = Counter(cards_per_image)
    print(f"\nCard count distribution:")
    for count in sorted(card_dist.keys()):
        pct = (card_dist[count] / len(label_files)) * 100
        bar = "#" * int(pct / 2)
        print(f"  {count} cards: {card_dist[count]:5d} ({pct:5.1f}%) {bar}")

    if len(box_widths) > 0:
        print(f"\nBounding box sizes (normalized):")
        print(f"  Width:  min={box_widths.min():.3f}, max={box_widths.max():.3f}, mean={box_widths.mean():.3f}")
        print(f"  Height: min={box_heights.min():.3f}, max={box_heights.max():.3f}, mean={box_heights.mean():.3f}")
        print(f"  Area:   min={box_areas.min():.4f}, max={box_areas.max():.4f}, mean={box_areas.mean():.4f}")

    return {
        'total_cards': total_cards,
        'cards_per_image': cards_per_image,
        'box_widths': box_widths,
        'box_heights': box_heights,
        'box_areas': box_areas
    }


def main():
    project_root = Path(__file__).parent.parent.parent
    dataset_dir = project_root / "training" / "data" / "yolo_dataset"

    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return

    analyze_dataset(dataset_dir)


if __name__ == "__main__":
    main()
