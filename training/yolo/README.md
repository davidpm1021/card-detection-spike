# YOLO Training Pipeline

This directory contains scripts for training a YOLO object detection model for MTG card detection.

## Quick Start

```bash
# 1. Generate synthetic training data (if not done)
python training/yolo/generate_synthetic_data.py

# 2. Train YOLO model
python training/yolo/train.py --epochs 10 --device cpu

# 3. Collect test images from webcam
python training/yolo/collect_images.py --camera 0

# 4. Label test images
python training/yolo/validate.py --create-ground-truth

# 5. Run validation
python training/yolo/validate.py
```

## Files

### Data Generation
- `download_backgrounds.py` - Download background images from DTD dataset
- `generate_synthetic_data.py` - Generate synthetic training data by compositing cards onto backgrounds
- `visualize_samples.py` - Visualize generated data with bounding boxes
- `analyze_dataset.py` - Analyze dataset statistics

### Training
- `train.py` - Train YOLOv8 model on synthetic data

### Validation
- `collect_images.py` - Capture test images from webcam
- `validate.py` - Run validation comparing YOLO vs contour detection

## Dataset

### Structure
```
training/data/yolo_dataset/
├── images/
│   ├── train/       # 10,000 training images
│   └── val/         # 2,000 validation images
├── labels/
│   ├── train/       # Training labels (YOLO format)
│   └── val/         # Validation labels
├── samples/         # Sample images for visual inspection
├── visualizations/  # Annotated samples with bounding boxes
└── data.yaml        # YOLO dataset configuration
```

### Label Format

YOLO format (normalized coordinates):
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Always 0 (single class: "card")
- All coordinates normalized to [0, 1]

Example:
```
0 0.5 0.5 0.3125 0.4375
```

## Usage

### 1. Generate Synthetic Data

```bash
python training/yolo/generate_synthetic_data.py
```

Generates:
- 10,000 training images
- 2,000 validation images
- Average 2-3 cards per image
- Total ~30,000 card instances

Time: ~15-20 minutes

### 2. Visualize Samples

```bash
python training/yolo/visualize_samples.py
```

Creates annotated images showing bounding boxes for visual verification.

### 3. Train YOLO Model (Coming Soon)

```bash
python training/yolo/train.py
```

## Augmentations

### Card Augmentations
- Rotation: -30° to +30°
- Perspective transform: 2-8%
- Scale: 0.15x - 0.50x of image width
- Brightness/contrast: ±30%
- Hue/saturation: ±10-20
- Gaussian blur: 3-7px (30%)
- Motion blur: 3-7px (20%)
- Noise: 10-50 variance (30%)
- Shadow effects (30%)
- Sleeve overlay: matte/glossy (30%)

### Scene Augmentations
- Gaussian noise: 5-30 variance
- JPEG compression: 70-100 quality
- Multiple cards per image (1-6 cards)
- Partial off-screen placement allowed

## Statistics

After generation:
- Training images: 10,000
- Validation images: 2,000
- Total card instances: ~28,000
- Average cards per image: ~2.3
- Image size: 640x640 pixels
- File size: ~2-3 GB total

## Requirements

```
opencv-python>=4.8.0
albumentations>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
```

## Output Files

After training completes:
```
training/yolo/
├── runs/detect/train/
│   ├── weights/
│   │   ├── best.pt      # Best model weights
│   │   └── best.onnx    # ONNX export for inference
│   ├── results.csv      # Training metrics
│   └── *.png            # Training plots
├── test_images/         # Captured webcam test images
└── results/
    ├── validation_results.csv    # Per-image metrics
    ├── validation_summary.md     # Final report
    └── failure_cases/            # Annotated failure images
```

## Integration

After training, YOLO detection is automatically available in `spike/inference.py`:

```bash
# Auto-uses YOLO if trained model exists
python spike/inference.py

# Force YOLO detection
python spike/inference.py --yolo

# Force contour detection (fallback)
python spike/inference.py --no-yolo
```

## Success Criteria

| Metric | Target | Minimum |
|--------|--------|---------|
| mAP@0.5 | >= 0.92 | >= 0.90 |
| Detection rate (real images) | >= 95% | >= 90% |
| Inference time (CPU) | < 30ms | < 50ms |
| FPS (detection + ID) | >= 20 | >= 15 |
