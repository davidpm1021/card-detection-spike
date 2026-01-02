# Task 003: YOLO Training

**Phase**: 3 of 5
**Assigned to**: @spike-worker
**Status**: Blocked by Task 002
**Priority**: High
**Depends on**: Task 002 (Synthetic Data Generation)

---

## Objective

Train a YOLOv8 model on the synthetic dataset to detect MTG cards. Start with YOLOv8-nano for speed, upgrade to YOLOv8-small if accuracy is insufficient.

---

## Deliverables

### Files to Create

| File | Description |
|------|-------------|
| `training/yolo/train.py` | Training script |
| `training/yolo/runs/detect/train/weights/best.pt` | Best model checkpoint |
| `training/yolo/runs/detect/train/weights/best.onnx` | ONNX export for CPU inference |
| `training/yolo/runs/detect/train/results.csv` | Training metrics |

### Success Criteria

- [ ] mAP@0.5 >= 0.90 on validation set
- [ ] mAP@0.5:0.95 >= 0.75
- [ ] Training completes without errors
- [ ] Model exported to ONNX format
- [ ] Inference time < 50ms on CPU (test on 640x640 image)

---

## Technical Specification

### Model Selection

| Model | Parameters | Size | Speed (CPU) | Expected mAP |
|-------|------------|------|-------------|--------------|
| YOLOv8n | 3.2M | 6.3MB | ~30ms | 0.88-0.92 |
| YOLOv8s | 11.2M | 22.5MB | ~50ms | 0.91-0.94 |
| YOLOv8m | 25.9M | 52MB | ~100ms | 0.93-0.96 |

**Recommendation**: Start with YOLOv8n. Only upgrade if mAP@0.5 < 0.90.

### Training Configuration

```python
# train.py
from ultralytics import YOLO
from pathlib import Path

def train_yolo(
    data_yaml: Path,
    model_size: str = "n",  # n=nano, s=small, m=medium
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0",  # GPU ID or "cpu"
    patience: int = 20,  # Early stopping
):
    """Train YOLOv8 for card detection."""

    # Load pretrained model
    model = YOLO(f"yolov8{model_size}.pt")

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        patience=patience,

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,  # Final LR = lr0 * lrf
        weight_decay=0.0005,
        warmup_epochs=3,

        # Augmentation (Ultralytics handles these)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        shear=5,
        perspective=0.001,
        flipud=0.0,  # No vertical flip
        fliplr=0.0,  # No horizontal flip (cards have orientation)
        mosaic=0.5,
        mixup=0.1,

        # Output
        project="training/yolo/runs/detect",
        name="train",
        exist_ok=True,
        save=True,
        plots=True,
    )

    return results
```

---

## Implementation

### Full Training Script

```python
#!/usr/bin/env python3
"""
Train YOLOv8 for MTG card detection.

Usage:
    python train.py                    # Train YOLOv8-nano
    python train.py --model s          # Train YOLOv8-small
    python train.py --epochs 50        # Custom epochs
    python train.py --device cpu       # Train on CPU (slow)
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO
import torch


def check_gpu():
    """Check GPU availability."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        return "0"
    else:
        print("No GPU detected, using CPU (training will be slow)")
        return "cpu"


def export_onnx(model_path: Path, imgsz: int = 640):
    """Export trained model to ONNX format."""
    print(f"\nExporting model to ONNX...")
    model = YOLO(model_path)
    model.export(format="onnx", imgsz=imgsz, simplify=True, opset=12)
    onnx_path = model_path.with_suffix(".onnx")
    print(f"ONNX model saved to: {onnx_path}")
    return onnx_path


def test_inference_speed(model_path: Path, imgsz: int = 640, num_runs: int = 50):
    """Test inference speed on CPU."""
    import numpy as np

    print(f"\nTesting inference speed on CPU...")
    model = YOLO(model_path)

    # Create dummy image
    dummy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        model.predict(dummy_img, verbose=False, device="cpu")

    # Measure
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model.predict(dummy_img, verbose=False, device="cpu")
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times) * 1000  # ms
    print(f"Average inference time (CPU): {avg_time:.1f}ms")
    print(f"FPS: {1000 / avg_time:.1f}")

    return avg_time


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for card detection")
    parser.add_argument("--model", type=str, default="n",
                        choices=["n", "s", "m"],
                        help="Model size: n=nano, s=small, m=medium")
    parser.add_argument("--data", type=Path,
                        default=Path("training/data/yolo_dataset/data.yaml"),
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 0, 1, cpu, or auto")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv8 Card Detection Training")
    print("=" * 60)

    # Check data exists
    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        print("Run generate_synthetic.py first to create the dataset.")
        return

    # Setup device
    if args.device == "auto":
        device = check_gpu()
    else:
        device = args.device

    # Adjust batch size for CPU
    if device == "cpu" and args.batch > 4:
        print(f"Reducing batch size to 4 for CPU training")
        args.batch = 4

    # Load pretrained model
    model_name = f"yolov8{args.model}.pt"
    print(f"\nLoading pretrained model: {model_name}")
    model = YOLO(model_name)

    # Train
    print(f"\nStarting training...")
    print(f"  Model: YOLOv8-{args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {device}")
    print(f"  Data: {args.data}")
    print()

    start_time = time.time()

    results = model.train(
        data=str(args.data.absolute()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        patience=args.patience,

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        shear=5,
        perspective=0.001,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.5,
        mixup=0.1,

        # Output
        project=str(Path("training/yolo/runs/detect").absolute()),
        name="train",
        exist_ok=True,
        save=True,
        plots=True,
    )

    training_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nTraining time: {training_time / 60:.1f} minutes")

    # Get metrics from results
    metrics = results.results_dict
    if metrics:
        print(f"\nFinal Metrics:")
        print(f"  mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"  Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")

    # Find best model
    best_model = Path("training/yolo/runs/detect/train/weights/best.pt")
    if best_model.exists():
        print(f"\nBest model saved to: {best_model}")

        # Export to ONNX
        onnx_path = export_onnx(best_model, args.imgsz)

        # Test inference speed
        avg_time = test_inference_speed(best_model, args.imgsz)

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Model: YOLOv8-{args.model}")
        print(f"  mAP@0.5: {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  CPU Inference: {avg_time:.1f}ms")

        # Check success criteria
        mAP50 = metrics.get('metrics/mAP50(B)', 0)
        if mAP50 >= 0.90:
            print(f"\n  SUCCESS: mAP@0.5 >= 0.90")
        elif mAP50 >= 0.85:
            print(f"\n  WARNING: mAP@0.5 between 0.85-0.90")
            print(f"  Consider: --model s for better accuracy")
        else:
            print(f"\n  FAIL: mAP@0.5 < 0.85")
            print(f"  Actions: ")
            print(f"    1. Try --model s or --model m")
            print(f"    2. Increase --epochs to 150")
            print(f"    3. Review synthetic data quality")
    else:
        print("Warning: Best model not found!")


if __name__ == "__main__":
    main()
```

---

## Expected Training Output

```
========================================
YOLOv8 Card Detection Training
========================================

GPU detected: NVIDIA GeForce RTX 3060 (12.0 GB)

Loading pretrained model: yolov8n.pt

Starting training...
  Model: YOLOv8-n
  Epochs: 100
  Batch size: 16
  Image size: 640
  Device: 0
  Data: training/data/yolo_dataset/data.yaml

Epoch 1/100:
  box_loss: 1.234, cls_loss: 0.456, dfl_loss: 0.789
  mAP@0.5: 0.123, mAP@0.5:0.95: 0.056

Epoch 10/100:
  box_loss: 0.456, cls_loss: 0.123, dfl_loss: 0.345
  mAP@0.5: 0.678, mAP@0.5:0.95: 0.456

...

Epoch 100/100:
  box_loss: 0.123, cls_loss: 0.045, dfl_loss: 0.234
  mAP@0.5: 0.923, mAP@0.5:0.95: 0.789

========================================
Training Complete!
========================================

Training time: 45.2 minutes

Final Metrics:
  mAP@0.5: 0.9234
  mAP@0.5:0.95: 0.7891
  Precision: 0.9456
  Recall: 0.9123

Best model saved to: training/yolo/runs/detect/train/weights/best.pt

Exporting model to ONNX...
ONNX model saved to: training/yolo/runs/detect/train/weights/best.onnx

Testing inference speed on CPU...
Average inference time (CPU): 28.5ms
FPS: 35.1

========================================
Summary
========================================
  Model: YOLOv8-n
  mAP@0.5: 0.9234
  CPU Inference: 28.5ms

  SUCCESS: mAP@0.5 >= 0.90
```

---

## If Training Fails

### Scenario 1: mAP@0.5 < 0.85

1. **Increase model size**: Try `--model s`
2. **Increase epochs**: Try `--epochs 150`
3. **Check data quality**: Visually inspect synthetic images
4. **Reduce augmentation**: May be too aggressive

### Scenario 2: Training unstable (loss oscillates)

1. **Reduce learning rate**: Set `lr0=0.0005`
2. **Increase warmup**: Set `warmup_epochs=5`
3. **Reduce batch size**: Try `--batch 8`

### Scenario 3: GPU out of memory

1. **Reduce batch size**: `--batch 8` or `--batch 4`
2. **Reduce image size**: `--imgsz 480`
3. **Use smaller model**: `--model n`

### Scenario 4: Training too slow on CPU

1. **Reduce dataset**: Temporarily use 2000 training images
2. **Reduce epochs**: `--epochs 50`
3. **Use cloud GPU**: Google Colab, Kaggle, etc.

---

## Files Produced

After successful training:

```
training/yolo/runs/detect/train/
    weights/
        best.pt          # Best model checkpoint
        best.onnx        # ONNX export
        last.pt          # Last epoch checkpoint
    results.csv          # Training metrics
    confusion_matrix.png # Confusion matrix (trivial for 1 class)
    F1_curve.png         # F1 vs confidence
    P_curve.png          # Precision vs confidence
    R_curve.png          # Recall vs confidence
    PR_curve.png         # Precision-Recall curve
    train_batch*.jpg     # Sample training batches
    val_batch*.jpg       # Sample validation batches
```

---

## Dependencies

```
ultralytics>=8.0.0
torch>=2.0.0
onnx>=1.14.0
```

Install:
```bash
pip install ultralytics
```

---

## Time Estimate

| Hardware | Time (100 epochs) |
|----------|-------------------|
| RTX 3060 | 30-45 minutes |
| RTX 4090 | 15-20 minutes |
| CPU only | 6-10 hours |

---

## Notes for Worker

1. **Start with nano**: YOLOv8n is usually sufficient for single-class detection
2. **Monitor training**: Watch for overfitting (val loss increasing while train decreases)
3. **Early stopping**: Patience=20 will stop if no improvement for 20 epochs
4. **ONNX is critical**: CPU inference requires ONNX export
5. **Test on CPU**: Even if training on GPU, test inference on CPU

---

## Next Task

After completing this task, proceed to:
- **Task 004: Integration** (Phase 4)
