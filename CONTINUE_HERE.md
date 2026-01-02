# Continue Here - YOLO Training on GPU

**Last Updated:** 2026-01-02
**Status:** Ready for GPU training

---

## What's Been Done

1. ✅ Spike conclusion reached: FaceNet embeddings work for identification (32K cards indexed)
2. ✅ Detection needs improvement: Contour-based detection is fragile on complex backgrounds
3. ✅ YOLO training pipeline created:
   - `training/yolo/download_backgrounds.py` - Downloads 5K+ background images
   - `training/yolo/generate_synthetic_data.py` - Creates 10K training images
   - `training/yolo/train.py` - Trains YOLOv8 model
4. ✅ Integration code ready:
   - `spike/yolo_detector.py` - YOLO detector class
   - `spike/inference.py` - Auto-uses YOLO when model exists
5. ✅ Validation scripts ready:
   - `training/yolo/collect_images.py` - Capture test images from webcam
   - `training/yolo/validate.py` - Compare YOLO vs contour detection

---

## What Needs To Be Done

### Step 1: Generate Training Data (~20-25 min total)

```bash
# Activate venv
venv\Scripts\activate

# Download background images (~5 min)
python training/yolo/download_backgrounds.py

# Generate synthetic training data (~15-20 min)
python training/yolo/generate_synthetic_data.py
```

This creates:
- `training/data/backgrounds/` - 5K+ background textures
- `training/data/yolo_dataset/` - 10K train + 2K val images with YOLO labels

### Step 2: Train YOLO Model (~10-30 min with GPU)

```bash
# Train with GPU (recommended settings)
python training/yolo/train.py --epochs 50 --batch 32 --device 0

# Or for more VRAM (8GB+):
python training/yolo/train.py --epochs 50 --batch 64 --device 0

# Or CPU fallback (slow, ~6 hours):
python training/yolo/train.py --epochs 10 --batch 16 --device cpu
```

**Success criteria:**
- mAP@0.5 >= 0.90 on validation set
- Model exports to ONNX automatically

Output: `training/yolo/runs/detect/train/weights/best.pt`

### Step 3: Validate on Real Images

```bash
# Collect 50+ test images from webcam
python training/yolo/collect_images.py --camera 0

# Label ground truth (count cards in each image)
python training/yolo/validate.py --create-ground-truth

# Run validation
python training/yolo/validate.py
```

**Target metrics:**
- Detection rate >= 95%
- Inference time < 50ms
- FPS >= 15 with identification

### Step 4: Test Live Inference

```bash
# Test with webcam (auto-uses YOLO if model exists)
python spike/inference.py --camera 0

# Force YOLO detection
python spike/inference.py --camera 0 --yolo

# Compare with contour detection
python spike/inference.py --camera 0 --no-yolo
```

---

## Key Files

```
training/
├── yolo/
│   ├── train.py              # Training script
│   ├── validate.py           # Validation script
│   ├── collect_images.py     # Test image capture
│   └── README.md             # Full documentation
├── data/
│   ├── backgrounds/          # Background images (generate)
│   ├── yolo_dataset/         # Training data (generate)
│   └── reference_images/     # Card images for synthetic data
spike/
├── inference.py              # Main inference pipeline
└── yolo_detector.py          # YOLO detector class
docs/
├── YOLO_DETECTION_PLAN.md    # Detailed plan
└── tasks/                    # Task breakdowns
```

---

## Success Criteria (from CLAUDE.md)

### Detection Targets
| Metric | Pass | Fail |
|--------|------|------|
| Detection rate (good lighting) | ≥95% | <85% |
| Detection rate (typical room) | ≥85% | <70% |
| Processing FPS | ≥15 | <10 |

### Final Deliverable
After validation, update `SPIKE_CONCLUSION.md` with:
- YOLO detection metrics
- Comparison vs contour detection
- Final Go/No-Go recommendation

---

## Troubleshooting

**CUDA not found:**
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Out of memory:**
- Reduce batch size: `--batch 16` or `--batch 8`
- Use smaller model: `--model n` (nano, default)

**Training too slow:**
- Verify GPU is being used (should show GPU memory in output)
- Check `--device 0` is set correctly
