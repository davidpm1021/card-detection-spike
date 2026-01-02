#!/usr/bin/env python3
"""
YOLO Training Script for MTG Card Detection

Trains YOLOv8 on synthetic card data for robust detection.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for card detection')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm'],
                        help='YOLO model size: n=nano(3.2MB), s=small(11MB), m=medium(26MB)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device: 0 for GPU, cpu for CPU')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    args = parser.parse_args()

    # Check for ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed")
        print("Run: pip install ultralytics")
        sys.exit(1)

    # Paths
    script_dir = Path(__file__).parent
    data_yaml = script_dir.parent / 'data' / 'yolo_dataset' / 'data.yaml'
    output_dir = script_dir / 'runs'

    if not data_yaml.exists():
        print(f"ERROR: Dataset not found at {data_yaml}")
        print("Run Task 002 (generate_synthetic_data.py) first")
        sys.exit(1)

    # Model name
    model_name = f'yolov8{args.model}.pt'

    print("=" * 60)
    print("YOLO Card Detection Training")
    print("=" * 60)
    print(f"Model:      {model_name}")
    print(f"Dataset:    {data_yaml}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device:     {args.device}")
    print(f"Output:     {output_dir}")
    print("=" * 60)

    # Load model
    if args.resume:
        # Resume from last checkpoint
        last_weights = output_dir / 'detect' / 'train' / 'weights' / 'last.pt'
        if last_weights.exists():
            print(f"Resuming from {last_weights}")
            model = YOLO(str(last_weights))
        else:
            print("No checkpoint found, starting fresh")
            model = YOLO(model_name)
    else:
        model = YOLO(model_name)

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        project=str(output_dir / 'detect'),
        name='train',
        exist_ok=True,
        # Augmentation settings for cards
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        perspective=0.001,
        flipud=0.0,  # No vertical flip for cards
        fliplr=0.0,  # No horizontal flip (cards have orientation)
        mosaic=0.5,
        mixup=0.1,
        # Logging
        verbose=True,
        plots=True,
    )

    # Print final metrics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Get best weights path
    best_weights = output_dir / 'detect' / 'train' / 'weights' / 'best.pt'

    if best_weights.exists():
        print(f"Best weights: {best_weights}")

        # Validate on validation set
        print("\nValidating on validation set...")
        best_model = YOLO(str(best_weights))
        val_results = best_model.val(data=str(data_yaml))

        # Extract metrics
        map50 = val_results.box.map50
        map50_95 = val_results.box.map
        precision = val_results.box.mp
        recall = val_results.box.mr

        print(f"\nValidation Metrics:")
        print(f"  mAP@0.5:    {map50:.4f}")
        print(f"  mAP@0.5:95: {map50_95:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")

        # Check against targets
        print("\n" + "-" * 40)
        if map50 >= 0.90:
            print(f"[PASS] mAP@0.5 >= 0.90: {map50:.4f}")
        else:
            print(f"[WARN] mAP@0.5 < 0.90: {map50:.4f}")

        # Export to ONNX
        print("\nExporting to ONNX...")
        onnx_path = best_model.export(format='onnx', imgsz=args.imgsz, simplify=True)
        print(f"ONNX model: {onnx_path}")

        print("\n" + "=" * 60)
        print("Next Steps:")
        print("1. Review training plots in runs/detect/train/")
        print("2. Test with: python -c \"from ultralytics import YOLO; m=YOLO('training/yolo/runs/detect/train/weights/best.pt'); m.predict('path/to/image.jpg', show=True)\"")
        print("3. Proceed to Task 004: Integration")
        print("=" * 60)
    else:
        print("ERROR: Best weights not found after training")
        sys.exit(1)


if __name__ == '__main__':
    main()
