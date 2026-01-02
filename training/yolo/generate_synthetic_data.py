"""
Synthetic Data Generation for YOLO Training

Generates 10,000+ synthetic training images by compositing MTG card images
onto background images with realistic augmentations.

Output: YOLO format (normalized xywh bounding boxes)
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Optional
import time

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CARD_DIR = PROJECT_ROOT / "training" / "data" / "reference_images"
BACKGROUND_DIR = PROJECT_ROOT / "training" / "data" / "backgrounds"
OUTPUT_DIR = PROJECT_ROOT / "training" / "data" / "yolo_dataset"

# Output size
OUTPUT_SIZE = (640, 640)

# MTG card aspect ratio (standard card is 63mm x 88mm)
CARD_ASPECT_RATIO = 63.0 / 88.0  # ~0.716


def load_card_paths() -> List[Path]:
    """Load all card image paths."""
    paths = list(CARD_DIR.glob("*.jpg"))
    print(f"  Card images: {len(paths):,}")
    return paths


def load_background_paths() -> List[Path]:
    """Load all background image paths from manifest."""
    manifest_path = BACKGROUND_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    paths = []
    for category in manifest["categories"].values():
        for filename in category["files"]:
            paths.append(BACKGROUND_DIR / filename)

    print(f"  Background images: {len(paths):,}")
    return paths


def get_card_augmentation():
    """Augmentations applied to individual cards before compositing."""
    return A.Compose([
        # Geometric transforms
        A.Perspective(scale=(0.02, 0.08), p=0.5),
        A.Affine(
            scale=(0.95, 1.05),
            shear={"x": (-5, 5), "y": (-5, 5)},
            p=0.3,
            mode=cv2.BORDER_CONSTANT,
            cval=0
        ),

        # Color/lighting
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5
        ),
        A.ColorJitter(p=0.3),

        # Quality degradation
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.3),

        # Shadow/highlight simulation
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 2),
            p=0.3
        ),
    ])


def apply_sleeve_overlay(card_img: np.ndarray, sleeve_type: str = "matte") -> np.ndarray:
    """Simulate card sleeve with semi-transparent overlay."""
    if sleeve_type == "matte":
        # Slight desaturation and brightness reduction
        overlay = np.ones_like(card_img, dtype=np.uint8) * 245
        alpha = 0.08
        result = cv2.addWeighted(card_img, 1 - alpha, overlay, alpha, 0)
        # Slight blur for matte effect
        result = cv2.GaussianBlur(result, (3, 3), 0.5)
        return result

    elif sleeve_type == "glossy":
        # Add specular highlight
        h, w = card_img.shape[:2]
        overlay = np.zeros_like(card_img)

        # Create diagonal highlight stripe
        center_x, center_y = w // 2, h // 3
        cv2.ellipse(overlay, (center_x, center_y), (w // 3, h // 6),
                   45, 0, 360, (255, 255, 255), -1)

        alpha = 0.15
        result = cv2.addWeighted(card_img, 1 - alpha, overlay, alpha, 0)
        return result

    return card_img


def rotate_image(image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image by angle (degrees) and return rotated image with valid mask.

    Returns:
        (rotated_image, alpha_mask) - mask is 255 for valid pixels, 0 for padding
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding box
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust translation
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Rotate image
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # Create alpha mask
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask_rotated = cv2.warpAffine(
        mask, M, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated, mask_rotated


def place_card_on_background(
    background: np.ndarray,
    card: np.ndarray,
    position: Tuple[int, int],
    scale: float = 1.0,
    rotation: float = 0.0
) -> Tuple[np.ndarray, Optional[List[float]]]:
    """
    Place a card on background and return bounding box.

    Args:
        background: Background image
        card: Card image to place
        position: (x, y) top-left position
        scale: Scale factor for card
        rotation: Rotation angle in degrees

    Returns:
        (modified_background, [class_id, x_center, y_center, width, height] normalized)
        Returns None for bbox if card is too small or mostly out of frame
    """
    bg_h, bg_w = background.shape[:2]
    card_h, card_w = card.shape[:2]

    # Scale card
    new_w = int(card_w * scale)
    new_h = int(card_h * scale)
    card_scaled = cv2.resize(card, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Rotate card if needed
    if abs(rotation) > 0.5:
        card_rotated, mask = rotate_image(card_scaled, rotation)
    else:
        card_rotated = card_scaled
        mask = np.ones((card_scaled.shape[0], card_scaled.shape[1]), dtype=np.uint8) * 255

    card_h, card_w = card_rotated.shape[:2]

    # Get placement position
    x, y = position

    # Calculate visible region
    x1_bg = max(0, x)
    y1_bg = max(0, y)
    x2_bg = min(bg_w, x + card_w)
    y2_bg = min(bg_h, y + card_h)

    # Card region to copy
    x1_card = x1_bg - x
    y1_card = y1_bg - y
    x2_card = x1_card + (x2_bg - x1_bg)
    y2_card = y1_card + (y2_bg - y1_bg)

    # Check if enough is visible
    visible_area = (x2_bg - x1_bg) * (y2_bg - y1_bg)
    total_area = card_w * card_h
    if visible_area < total_area * 0.3:  # Less than 30% visible
        return background, None

    if x2_bg <= x1_bg or y2_bg <= y1_bg:
        return background, None

    # Composite using mask
    card_region = card_rotated[y1_card:y2_card, x1_card:x2_card]
    mask_region = mask[y1_card:y2_card, x1_card:x2_card]

    if card_region.size == 0 or mask_region.size == 0:
        return background, None

    # Blend card onto background using mask
    mask_3ch = cv2.cvtColor(mask_region, cv2.COLOR_GRAY2BGR) / 255.0
    bg_region = background[y1_bg:y2_bg, x1_bg:x2_bg]

    blended = (card_region * mask_3ch + bg_region * (1 - mask_3ch)).astype(np.uint8)
    background[y1_bg:y2_bg, x1_bg:x2_bg] = blended

    # Calculate YOLO bbox (normalized)
    # Use the actual card bounds (including rotation)
    center_x = (x + card_w / 2) / bg_w
    center_y = (y + card_h / 2) / bg_h
    width = card_w / bg_w
    height = card_h / bg_h

    # Clip to valid range
    center_x = np.clip(center_x, 0, 1)
    center_y = np.clip(center_y, 0, 1)
    width = min(width, 1.0)
    height = min(height, 1.0)

    return background, [0, center_x, center_y, width, height]


def generate_scene(
    background_path: Path,
    card_paths: List[Path],
    card_aug: A.Compose,
    num_cards: Optional[int] = None,
    output_size: Tuple[int, int] = OUTPUT_SIZE
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Generate a synthetic scene with cards on background.

    Returns:
        (image, list of YOLO labels)
    """
    # Load background
    bg = cv2.imread(str(background_path))
    if bg is None:
        raise ValueError(f"Failed to load background: {background_path}")

    # Resize background to output size
    bg = cv2.resize(bg, output_size, interpolation=cv2.INTER_AREA)

    # Apply background augmentation
    bg_aug = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),
    ])
    bg = bg_aug(image=bg)["image"]

    # Random number of cards (weighted towards 1-3 cards)
    if num_cards is None:
        num_cards = random.choices(
            [1, 2, 3, 4, 5, 6],
            weights=[30, 30, 20, 10, 7, 3],
            k=1
        )[0]

    # Prevent too many cards if we don't have enough unique cards
    num_cards = min(num_cards, len(card_paths))

    # Select random cards
    selected_cards = random.sample(card_paths, num_cards)

    labels = []

    for card_path in selected_cards:
        # Load card
        card = cv2.imread(str(card_path))
        if card is None:
            continue

        # Apply card augmentation
        augmented = card_aug(image=card)
        card = augmented["image"]

        # Maybe apply sleeve (30% chance)
        if random.random() < 0.3:
            sleeve_type = random.choice(["matte", "glossy"])
            card = apply_sleeve_overlay(card, sleeve_type)

        # Random scale (card occupies 10-60% of image width)
        card_h, card_w = card.shape[:2]
        target_width = random.uniform(0.15, 0.50) * output_size[0]
        scale = target_width / card_w

        # Random position (can be partially off-screen)
        scaled_w = int(card_w * scale)
        scaled_h = int(card_h * scale)

        # Allow some off-screen placement
        x = random.randint(-scaled_w // 3, output_size[0] - scaled_w // 3)
        y = random.randint(-scaled_h // 3, output_size[1] - scaled_h // 3)

        # Random rotation (-30 to +30 degrees)
        rotation = random.uniform(-30, 30)

        # Place card
        bg, label = place_card_on_background(bg, card, (x, y), scale, rotation)
        if label is not None:
            labels.append(label)

    # Apply scene-level augmentation
    scene_aug = A.Compose([
        A.GaussNoise(var_limit=(5, 30), p=0.3),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
    ])
    bg = scene_aug(image=bg)["image"]

    return bg, labels


def generate_dataset(
    output_dir: Path,
    card_paths: List[Path],
    bg_paths: List[Path],
    num_train: int = 10000,
    num_val: int = 2000,
    seed: int = 42
):
    """Generate complete YOLO dataset."""
    random.seed(seed)
    np.random.seed(seed)

    # Setup directories
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)

    # Create card augmentation pipeline (reuse for efficiency)
    card_aug = get_card_augmentation()

    # Statistics
    total_cards_train = 0
    total_cards_val = 0

    # Generate training set
    print(f"\nGenerating {num_train:,} training images...")
    start_time = time.time()

    for i in tqdm(range(num_train), desc="Training", unit="img"):
        bg_path = random.choice(bg_paths)

        try:
            image, labels = generate_scene(bg_path, card_paths, card_aug)
        except Exception as e:
            print(f"\nWarning: Failed to generate image {i}: {e}")
            continue

        # Save image
        img_name = f"img_{i:05d}.jpg"
        cv2.imwrite(str(output_dir / "images" / "train" / img_name), image)

        # Save labels
        label_name = f"img_{i:05d}.txt"
        with open(output_dir / "labels" / "train" / label_name, "w") as f:
            for label in labels:
                f.write(" ".join(map(str, label)) + "\n")

        total_cards_train += len(labels)

        # Save first 10 as samples
        if i < 10:
            cv2.imwrite(str(output_dir / "samples" / f"train_sample_{i:03d}.jpg"), image)

    train_time = time.time() - start_time

    # Generate validation set
    print(f"\nGenerating {num_val:,} validation images...")
    start_time = time.time()

    for i in tqdm(range(num_val), desc="Validation", unit="img"):
        bg_path = random.choice(bg_paths)

        try:
            image, labels = generate_scene(bg_path, card_paths, card_aug)
        except Exception as e:
            print(f"\nWarning: Failed to generate image {num_train + i}: {e}")
            continue

        # Save image
        img_name = f"img_{num_train + i:05d}.jpg"
        cv2.imwrite(str(output_dir / "images" / "val" / img_name), image)

        # Save labels
        label_name = f"img_{num_train + i:05d}.txt"
        with open(output_dir / "labels" / "val" / label_name, "w") as f:
            for label in labels:
                f.write(" ".join(map(str, label)) + "\n")

        total_cards_val += len(labels)

        # Save first 10 as samples
        if i < 10:
            cv2.imwrite(str(output_dir / "samples" / f"val_sample_{i:03d}.jpg"), image)

    val_time = time.time() - start_time

    # Create data.yaml
    data_yaml_content = f"""# YOLO Dataset Configuration
# Auto-generated by generate_synthetic_data.py

path: {output_dir.absolute().as_posix()}
train: images/train
val: images/val

nc: 1
names: ['card']
"""

    with open(output_dir / "data.yaml", "w") as f:
        f.write(data_yaml_content)

    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"\nStatistics:")
    print(f"  Total images: {num_train + num_val:,}")
    print(f"  Training images: {num_train:,}")
    print(f"  Validation images: {num_val:,}")
    print(f"  Total card instances (train): {total_cards_train:,}")
    print(f"  Total card instances (val): {total_cards_val:,}")
    print(f"  Avg cards per image (train): {total_cards_train / num_train:.2f}")
    print(f"  Avg cards per image (val): {total_cards_val / num_val:.2f}")
    print(f"\nTiming:")
    print(f"  Training generation: {train_time / 60:.1f} minutes ({train_time / num_train:.2f}s per image)")
    print(f"  Validation generation: {val_time / 60:.1f} minutes ({val_time / num_val:.2f}s per image)")
    print(f"\nOutput directory: {output_dir}")
    print(f"  images/train: {num_train:,} images")
    print(f"  images/val: {num_val:,} images")
    print(f"  labels/train: {num_train:,} labels")
    print(f"  labels/val: {num_val:,} labels")
    print(f"  samples/: 20 sample images")
    print(f"  data.yaml: Created")
    print("\n" + "=" * 60)


def validate_dataset(output_dir: Path):
    """Validate generated dataset."""
    print("\nValidating dataset...")

    train_images = list((output_dir / "images" / "train").glob("*.jpg"))
    train_labels = list((output_dir / "labels" / "train").glob("*.txt"))
    val_images = list((output_dir / "images" / "val").glob("*.jpg"))
    val_labels = list((output_dir / "labels" / "val").glob("*.txt"))

    print(f"  Training images: {len(train_images):,}")
    print(f"  Training labels: {len(train_labels):,}")
    print(f"  Validation images: {len(val_images):,}")
    print(f"  Validation labels: {len(val_labels):,}")

    assert len(train_images) == len(train_labels), "Mismatch in training set"
    assert len(val_images) == len(val_labels), "Mismatch in validation set"

    # Check label format (sample)
    print(f"  Checking label format (sampling 100 files)...")
    errors = []
    for label_file in random.sample(train_labels, min(100, len(train_labels))):
        with open(label_file) as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    errors.append(f"{label_file.name} line {line_num}: Expected 5 values, got {len(parts)}")
                    continue

                try:
                    cls, x, y, w, h = map(float, parts)
                    if cls != 0:
                        errors.append(f"{label_file.name} line {line_num}: Class should be 0, got {cls}")
                    if not (0 <= x <= 1):
                        errors.append(f"{label_file.name} line {line_num}: x out of range [0,1]: {x}")
                    if not (0 <= y <= 1):
                        errors.append(f"{label_file.name} line {line_num}: y out of range [0,1]: {y}")
                    if not (0 < w <= 1):
                        errors.append(f"{label_file.name} line {line_num}: w out of range (0,1]: {w}")
                    if not (0 < h <= 1):
                        errors.append(f"{label_file.name} line {line_num}: h out of range (0,1]: {h}")
                except ValueError as e:
                    errors.append(f"{label_file.name} line {line_num}: Invalid number format: {e}")

    if errors:
        print(f"\n  Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10
            print(f"    - {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
        return False

    print("  Label format: OK")
    print("\nValidation passed!")
    return True


def main():
    print("=" * 60)
    print("Synthetic Data Generation for YOLO Training")
    print("=" * 60)

    print("\nLoading resources...")
    card_paths = load_card_paths()
    bg_paths = load_background_paths()

    if len(card_paths) == 0:
        print(f"Error: No card images found in {CARD_DIR}")
        return

    if len(bg_paths) == 0:
        print(f"Error: No background images found in {BACKGROUND_DIR}")
        return

    # Generate dataset
    generate_dataset(
        output_dir=OUTPUT_DIR,
        card_paths=card_paths,
        bg_paths=bg_paths,
        num_train=10000,
        num_val=2000,
        seed=42
    )

    # Validate
    validate_dataset(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Next step: python training/yolo/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
