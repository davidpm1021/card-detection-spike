# Task 002: Synthetic Data Generation

**Phase**: 2 of 5
**Assigned to**: @spike-worker
**Status**: Blocked by Task 001
**Priority**: High
**Depends on**: Task 001 (Background Collection)

---

## Objective

Generate 10,000+ synthetic training images by compositing MTG card images onto background images with realistic augmentations. Output in YOLO format for training.

---

## Deliverables

### Files to Create

| File | Description |
|------|-------------|
| `training/yolo/generate_synthetic.py` | Main generation script |
| `training/yolo/augmentations.py` | Augmentation utilities |
| `training/data/yolo_dataset/` | Generated dataset |

### Directory Structure

```
training/data/yolo_dataset/
    images/
        train/
            img_00001.jpg
            img_00002.jpg
            ...
        val/
            img_09001.jpg
            ...
    labels/
        train/
            img_00001.txt
            img_00002.txt
            ...
        val/
            img_09001.txt
            ...
    data.yaml
```

### Success Criteria

- [ ] 10,000+ training images generated
- [ ] 1,000+ validation images generated
- [ ] YOLO format labels (normalized xywh)
- [ ] Realistic augmentations applied
- [ ] Cards visible and properly labeled
- [ ] No label errors (boxes match visible cards)

---

## Technical Specification

### YOLO Label Format

Each `.txt` file contains one line per card:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values normalized to [0, 1]:
- `class_id`: Always 0 (we only have one class: "card")
- `x_center`: Center X / image_width
- `y_center`: Center Y / image_height
- `width`: Box width / image_width
- `height`: Box height / image_height

**Example** (image is 640x640, card at center, 200x280 pixels):
```
0 0.5 0.5 0.3125 0.4375
```

### data.yaml

```yaml
path: C:/Users/Dave/Cursor/card-detection-spike/training/data/yolo_dataset
train: images/train
val: images/val
nc: 1
names: ['card']
```

---

## Image Generation Pipeline

### Step 1: Load Resources

```python
from pathlib import Path
import json
import cv2
import numpy as np
from typing import List, Tuple

CARD_DIR = Path("training/data/reference_images")
BACKGROUND_DIR = Path("training/data/backgrounds")

def load_card_paths() -> List[Path]:
    """Load all card image paths."""
    return list(CARD_DIR.glob("*.jpg"))

def load_background_paths() -> List[Path]:
    """Load all background image paths."""
    with open(BACKGROUND_DIR / "manifest.json") as f:
        manifest = json.load(f)
    paths = []
    for category in manifest["categories"].values():
        paths.extend([BACKGROUND_DIR / p for p in category["files"]])
    return paths
```

### Step 2: Card Augmentation

```python
import albumentations as A

def get_card_augmentation():
    """Augmentations applied to individual cards before compositing."""
    return A.Compose([
        # Geometric
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.Perspective(scale=(0.02, 0.08), p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            shear=(-10, 10),
            p=0.3
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
        A.RandomShadow(p=0.3),
    ])
```

### Step 3: Sleeve Overlay (Optional)

```python
def apply_sleeve_overlay(card_img: np.ndarray, sleeve_type: str = "matte") -> np.ndarray:
    """Simulate card sleeve with semi-transparent overlay."""
    if sleeve_type == "matte":
        # Slight desaturation and brightness reduction
        overlay = np.ones_like(card_img) * [245, 245, 245]  # Slight white tint
        alpha = 0.08
    elif sleeve_type == "glossy":
        # Add specular highlight
        overlay = np.zeros_like(card_img)
        # Add diagonal highlight
        h, w = card_img.shape[:2]
        for i in range(h):
            for j in range(w):
                if abs(i - j * h / w) < h * 0.15:
                    overlay[i, j] = [255, 255, 255]
        alpha = 0.15
    else:
        return card_img

    result = cv2.addWeighted(card_img, 1 - alpha, overlay.astype(np.uint8), alpha, 0)
    return result
```

### Step 4: Card Placement

```python
def place_card_on_background(
    background: np.ndarray,
    card: np.ndarray,
    position: Tuple[int, int],
    scale: float = 1.0,
    rotation: float = 0.0
) -> Tuple[np.ndarray, List[float]]:
    """
    Place a card on background and return bounding box.

    Returns:
        (modified_background, [x_center, y_center, width, height] normalized)
    """
    bg_h, bg_w = background.shape[:2]
    card_h, card_w = card.shape[:2]

    # Scale card
    new_w = int(card_w * scale)
    new_h = int(card_h * scale)
    card = cv2.resize(card, (new_w, new_h))

    # Rotate card
    if rotation != 0:
        M = cv2.getRotationMatrix2D((new_w/2, new_h/2), rotation, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        rot_w = int(new_h * sin + new_w * cos)
        rot_h = int(new_h * cos + new_w * sin)
        M[0, 2] += (rot_w - new_w) / 2
        M[1, 2] += (rot_h - new_h) / 2
        card = cv2.warpAffine(card, M, (rot_w, rot_h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
        new_w, new_h = rot_w, rot_h

    # Get placement position (can be partially off-screen)
    x, y = position

    # Calculate visible region
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + new_w)
    y2 = min(bg_h, y + new_h)

    # Card region to copy
    card_x1 = x1 - x
    card_y1 = y1 - y
    card_x2 = card_x1 + (x2 - x1)
    card_y2 = card_y1 + (y2 - y1)

    # Skip if too little visible
    visible_area = (x2 - x1) * (y2 - y1)
    total_area = new_w * new_h
    if visible_area < total_area * 0.3:  # Less than 30% visible
        return background, None

    # Composite (simple copy for now, could use alpha blending)
    background[y1:y2, x1:x2] = card[card_y1:card_y2, card_x1:card_x2]

    # Calculate YOLO bbox (use actual card bounds, not visible region)
    center_x = (x + new_w / 2) / bg_w
    center_y = (y + new_h / 2) / bg_h
    width = new_w / bg_w
    height = new_h / bg_h

    # Clip to image bounds for label
    center_x = np.clip(center_x, 0, 1)
    center_y = np.clip(center_y, 0, 1)

    return background, [0, center_x, center_y, width, height]  # class 0
```

### Step 5: Scene Generation

```python
def generate_scene(
    background_path: Path,
    card_paths: List[Path],
    num_cards: int = None,
    output_size: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Generate a synthetic scene with cards on background.

    Returns:
        (image, list of YOLO labels)
    """
    # Load background
    bg = cv2.imread(str(background_path))
    bg = cv2.resize(bg, output_size)

    # Random number of cards
    if num_cards is None:
        num_cards = np.random.choice([1, 1, 2, 2, 2, 3, 3, 4, 5, 6],
                                      p=[0.15, 0.15, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.03, 0.02])

    # Select random cards
    selected_cards = np.random.choice(card_paths, num_cards, replace=False)

    labels = []
    card_aug = get_card_augmentation()

    for card_path in selected_cards:
        # Load card
        card = cv2.imread(str(card_path))
        if card is None:
            continue

        # Apply augmentation
        augmented = card_aug(image=card)
        card = augmented["image"]

        # Maybe apply sleeve
        if np.random.random() < 0.3:
            sleeve_type = np.random.choice(["matte", "glossy"])
            card = apply_sleeve_overlay(card, sleeve_type)

        # Random scale (0.15 to 0.5 of image size for card width)
        scale = np.random.uniform(0.15, 0.45) * output_size[0] / card.shape[1]

        # Random position (can be partially off-screen)
        max_x = int(output_size[0] * 1.2)
        max_y = int(output_size[1] * 1.2)
        x = np.random.randint(-int(card.shape[1] * scale * 0.3), max_x)
        y = np.random.randint(-int(card.shape[0] * scale * 0.3), max_y)

        # Random rotation
        rotation = np.random.uniform(-30, 30)

        # Place card
        bg, label = place_card_on_background(bg, card, (x, y), scale, rotation)
        if label is not None:
            labels.append(label)

    # Apply scene-level augmentation
    scene_aug = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(5, 30), p=0.3),
        A.ImageCompression(quality_lower=70, quality_upper=95, p=0.3),
    ])
    bg = scene_aug(image=bg)["image"]

    return bg, labels
```

### Step 6: Dataset Generation

```python
from tqdm import tqdm
import random

def generate_dataset(
    output_dir: Path,
    num_train: int = 10000,
    num_val: int = 1000
):
    """Generate complete YOLO dataset."""
    # Setup directories
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Load resources
    card_paths = load_card_paths()
    bg_paths = load_background_paths()

    print(f"Loaded {len(card_paths)} card images")
    print(f"Loaded {len(bg_paths)} background images")

    # Generate training set
    print(f"\nGenerating {num_train} training images...")
    for i in tqdm(range(num_train)):
        bg_path = random.choice(bg_paths)
        image, labels = generate_scene(bg_path, card_paths)

        # Save image
        img_name = f"img_{i:05d}.jpg"
        cv2.imwrite(str(output_dir / "images" / "train" / img_name), image)

        # Save labels
        label_name = f"img_{i:05d}.txt"
        with open(output_dir / "labels" / "train" / label_name, "w") as f:
            for label in labels:
                f.write(" ".join(map(str, label)) + "\n")

    # Generate validation set
    print(f"\nGenerating {num_val} validation images...")
    for i in tqdm(range(num_val)):
        bg_path = random.choice(bg_paths)
        image, labels = generate_scene(bg_path, card_paths)

        # Save image
        img_name = f"img_{num_train + i:05d}.jpg"
        cv2.imwrite(str(output_dir / "images" / "val" / img_name), image)

        # Save labels
        label_name = f"img_{num_train + i:05d}.txt"
        with open(output_dir / "labels" / "val" / label_name, "w") as f:
            for label in labels:
                f.write(" ".join(map(str, label)) + "\n")

    # Create data.yaml
    data_yaml = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["card"]
    }

    import yaml
    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    print(f"\nDataset saved to {output_dir}")
```

---

## Augmentation Summary

| Augmentation | Probability | Range | Purpose |
|--------------|-------------|-------|---------|
| Rotation | 80% | -30 to +30 deg | Simulate hand placement |
| Perspective | 50% | 2-8% | Viewing angle variation |
| Scale | 100% | 0.15x - 0.45x | Distance from camera |
| Brightness | 70% | +/- 30% | Lighting conditions |
| Contrast | 70% | +/- 30% | Lighting conditions |
| Hue shift | 50% | +/- 10 | Color temperature |
| Gaussian blur | 30% | 3-7px | Focus issues |
| Motion blur | 20% | 3-7px | Camera/hand movement |
| Noise | 30% | 10-50 var | Sensor noise |
| Shadow | 30% | random | Natural shadows |
| Sleeve overlay | 30% | matte/glossy | Sleeved cards |
| JPEG compression | 30% | 70-95 quality | Compression artifacts |

---

## Validation Checks

After generation, run these checks:

```python
def validate_dataset(output_dir: Path):
    """Validate generated dataset."""
    train_images = list((output_dir / "images" / "train").glob("*.jpg"))
    train_labels = list((output_dir / "labels" / "train").glob("*.txt"))
    val_images = list((output_dir / "images" / "val").glob("*.jpg"))
    val_labels = list((output_dir / "labels" / "val").glob("*.txt"))

    print(f"Training images: {len(train_images)}")
    print(f"Training labels: {len(train_labels)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Validation labels: {len(val_labels)}")

    assert len(train_images) == len(train_labels), "Mismatch in training set"
    assert len(val_images) == len(val_labels), "Mismatch in validation set"

    # Check label format
    for label_file in train_labels[:100]:  # Sample check
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                assert len(parts) == 5, f"Invalid label format in {label_file}"
                cls, x, y, w, h = map(float, parts)
                assert cls == 0, "Class should be 0"
                assert 0 <= x <= 1, f"x out of range: {x}"
                assert 0 <= y <= 1, f"y out of range: {y}"
                assert 0 < w <= 1, f"w out of range: {w}"
                assert 0 < h <= 1, f"h out of range: {h}"

    print("Validation passed!")
```

---

## Console Output Expected

```
========================================
Synthetic Data Generation
========================================

Loading resources...
  Card images: 32,062
  Background images: 1,053

Generating 10,000 training images...
[====================] 100% | 10000/10000 | 45:32

Generating 1,000 validation images...
[====================] 100% | 1000/1000 | 4:33

Statistics:
  Total images: 11,000
  Cards per image (avg): 2.3
  Total card instances: 25,300

Saving samples to training/data/yolo_dataset/samples/...
  sample_001.jpg (2 cards)
  sample_002.jpg (1 card)
  sample_003.jpg (4 cards)
  ...

Validating dataset...
  Training images: 10,000
  Training labels: 10,000
  Validation images: 1,000
  Validation labels: 1,000
  Label format: OK

========================================
Dataset Generation Complete!
========================================

Output: training/data/yolo_dataset/
  images/train: 10,000 images
  images/val: 1,000 images
  labels/train: 10,000 labels
  labels/val: 1,000 labels
  data.yaml: Created

Next step: python training/yolo/train.py
```

---

## Dependencies

```
opencv-python>=4.8.0
albumentations>=1.3.0
numpy>=1.24.0
tqdm>=4.65.0
PyYAML>=6.0
```

---

## Time Estimate

- Script development: 1-2 hours
- Data generation (10K images): ~45 minutes
- Validation: 10 minutes
- **Total: 2-3 hours**

---

## Notes for Worker

1. **Memory management**: Process in batches if memory becomes an issue
2. **Speed optimization**: Use multiprocessing for parallel generation
3. **Quality check**: Visually inspect sample images before full generation
4. **Card visibility**: Ensure cards are not completely occluded by overlaps
5. **Label accuracy**: Double-check that bounding boxes match visible cards

---

## Next Task

After completing this task, proceed to:
- **Task 003: YOLO Training** (Phase 3)
