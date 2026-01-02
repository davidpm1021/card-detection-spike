# YOLO Card Detection Training Plan

**Created:** 2026-01-02
**Status:** Planning
**Goal:** Replace fragile contour-based detection with robust YOLOv8 object detector

---

## Executive Summary

The current contour-based card detection in `spike/inference.py` fails on complex backgrounds. We will train a YOLOv8-nano or YOLOv8-small model on synthetic data (card images composited onto random backgrounds) to achieve robust detection in any scene.

### Why YOLO?

- Single-shot detection (fast)
- Works well with synthetic training data
- YOLOv8-nano runs at 100+ FPS on modern CPUs
- Proven approach for card/document detection
- Ultralytics provides excellent training infrastructure

### Resources Available

- 32,062 reference card images in `training/data/reference_images/`
- Working identification pipeline (FaceNet embeddings + FAISS)
- Python environment with PyTorch already configured

---

## Phase Overview

| Phase | Description | Duration | Owner |
|-------|-------------|----------|-------|
| 1 | Background Collection | 0.5 day | spike-worker |
| 2 | Synthetic Data Generation | 1 day | spike-worker |
| 3 | YOLO Training | 0.5-1 day | spike-worker |
| 4 | Integration | 0.5 day | spike-worker |
| 5 | Validation | 0.5 day | spike-worker |

**Total estimated time:** 3-4 days

---

## Phase 1: Background Collection

### Task: @spike-worker - Collect Background Images

**Phase**: 1
**Files to Create**: `training/yolo/download_backgrounds.py`
**Output Directory**: `training/data/backgrounds/`

**Acceptance Criteria**:
- [ ] Download 1,000+ diverse background images
- [ ] Include: wood textures, playmats, cluttered desks, hands, fabric
- [ ] Images at least 640x640 pixels
- [ ] No MTG cards in backgrounds (important!)
- [ ] Script is reusable for future runs

**Implementation Details**:

Option A - Use existing datasets:
- DTD (Describable Textures Dataset) - textures
- COCO images (filtered for indoor scenes)
- Unsplash API for free high-res images

Option B - Generate procedural backgrounds:
- Solid colors with noise
- Gradient patterns
- Wood grain patterns (procedural)

**Recommended sources** (prioritized by ease):
1. **DTD textures**: https://www.robots.ox.ac.uk/~vgg/data/dtd/ (5,640 images, categorized)
2. **Unsplash**: API for free stock photos (playmat-like, desk surfaces)
3. **Procedural generation**: Perlin noise, gradients, solid colors

**Background categories needed**:
- `wood/` - Various wood textures (gaming tables)
- `fabric/` - Playmats, tablecloths
- `clutter/` - Messy desks with papers, other cards
- `hands/` - Skin tones, hands holding things
- `solid/` - Solid colors (green, black, white, brown)
- `pattern/` - Random patterns, abstract

**Output Required**:
- Console output showing download progress
- `backgrounds_manifest.json` listing all images and categories
- Total count logged

---

## Phase 2: Synthetic Data Generation

### Task: @spike-worker - Build Synthetic Dataset

**Phase**: 2
**Files to Create**:
- `training/yolo/generate_synthetic.py`
- `training/yolo/augmentations.py`

**Output Directory**: `training/data/yolo_dataset/`

**Acceptance Criteria**:
- [ ] Generate 10,000+ synthetic training images
- [ ] Each image has 1-6 cards placed on background
- [ ] YOLO format annotations (normalized xywh)
- [ ] Train/val split (90/10)
- [ ] Augmentations realistic for webcam conditions

**YOLO Dataset Structure**:
```
training/data/yolo_dataset/
    images/
        train/
            img_00001.jpg
            img_00002.jpg
            ...
        val/
            img_10001.jpg
            ...
    labels/
        train/
            img_00001.txt  # YOLO format: class x_center y_center width height
            ...
        val/
            ...
    data.yaml  # Dataset configuration
```

**YOLO Annotation Format** (one line per card):
```
0 0.5 0.5 0.2 0.28  # class=0 (card), normalized center_x, center_y, width, height
```

**Augmentations to Apply**:

**Card-level augmentations** (before compositing):
- Rotation: -30 to +30 degrees
- Scale: 0.3x to 1.5x relative to background
- Perspective warp: Simulate viewing angles
- Brightness: 0.7 to 1.3
- Contrast: 0.8 to 1.2
- Blur: Gaussian, motion blur (simulate focus/movement)
- Color jitter: Hue shift, saturation
- Sleeve overlay: Optional semi-transparent overlay

**Scene-level augmentations** (after compositing):
- Overall brightness/contrast
- Color temperature shifts (warm/cool lighting)
- Gaussian noise
- JPEG compression artifacts
- Random shadows

**Placement rules**:
- Cards can overlap (up to 50%)
- Cards can be partially off-screen (up to 30%)
- Minimum visible area: 70% of card
- Random z-ordering for overlapping cards

**Output Required**:
- `generation_stats.csv` with counts per category
- Sample images saved to `samples/` directory
- Console output with progress and statistics

---

## Phase 3: YOLO Training

### Task: @spike-worker - Train YOLOv8 Detector

**Phase**: 3
**Files to Create**:
- `training/yolo/train.py`
- `training/yolo/config.yaml`

**Output Directory**: `training/yolo/runs/`

**Acceptance Criteria**:
- [ ] Train YOLOv8-nano or YOLOv8-small
- [ ] mAP@0.5 >= 0.90 on validation set
- [ ] Inference time < 50ms on CPU
- [ ] Model exported to ONNX format
- [ ] Training logs and metrics saved

**Training Configuration**:
```yaml
# config.yaml
model: yolov8n.pt  # Start with nano, upgrade to small if needed
epochs: 100
imgsz: 640
batch: 16  # Adjust based on GPU memory
device: 0  # GPU, or 'cpu'
workers: 4
patience: 20  # Early stopping
optimizer: AdamW
lr0: 0.001
weight_decay: 0.0005

# Augmentation (handled by Ultralytics)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 15
translate: 0.1
scale: 0.5
shear: 5
perspective: 0.001
flipud: 0.0  # No vertical flip for cards
fliplr: 0.0  # No horizontal flip (cards have orientation)
mosaic: 0.5
mixup: 0.1
```

**data.yaml**:
```yaml
path: ../data/yolo_dataset
train: images/train
val: images/val
nc: 1
names: ['card']
```

**Training script requirements**:
- Use Ultralytics YOLO API
- Log training to CSV
- Save best model and last model
- Export best model to ONNX
- Print final metrics summary

**Model Selection Criteria**:
| Model | Size | mAP (typical) | CPU Inference | Recommendation |
|-------|------|---------------|---------------|----------------|
| YOLOv8-nano | 3.2MB | 0.88 | ~30ms | Start here |
| YOLOv8-small | 11.2MB | 0.91 | ~50ms | If nano insufficient |
| YOLOv8-medium | 25.9MB | 0.93 | ~100ms | Probably overkill |

**Output Required**:
- `training/yolo/runs/detect/train/weights/best.pt`
- `training/yolo/runs/detect/train/weights/best.onnx`
- `training/yolo/runs/detect/train/results.csv`
- Console output with final mAP, precision, recall

---

## Phase 4: Integration

### Task: @spike-worker - Replace Contour Detection with YOLO

**Phase**: 4
**Files to Modify**:
- `spike/inference.py` (CardDetector class)

**Files to Create**:
- `spike/yolo_detector.py` (new YOLO-based detector)

**Acceptance Criteria**:
- [ ] YOLODetector class with same interface as CardDetector
- [ ] `detect(frame)` returns list of (contour, corners) tuples
- [ ] Works with existing CardIdentifier and WebcamInference
- [ ] Fallback to ONNX Runtime (no PyTorch required at inference)
- [ ] FPS >= 15 with detection + identification

**Interface Specification**:
```python
class YOLODetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """Load YOLO model (ONNX or .pt)"""
        pass

    def detect(self, frame: np.ndarray, fast_mode: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect cards in frame.

        Returns:
            List of (contour, corners) tuples matching CardDetector interface
            - contour: np.ndarray of shape (N, 1, 2) - card contour points
            - corners: np.ndarray of shape (4, 2) - ordered corner points
        """
        pass
```

**Integration approach**:
1. Create `YOLODetector` class in new file
2. Modify `WebcamInference.__init__` to accept detector parameter
3. Default to `YOLODetector` if model exists, else fall back to `CardDetector`
4. Keep `CardDetector` as backup for users without YOLO model

**Output Required**:
- Working inference.py with YOLO detection
- Console output showing which detector is in use
- FPS measurement comparison (contour vs YOLO)

---

## Phase 5: Validation

### Task: @spike-worker - Validate Detection Performance

**Phase**: 5
**Files to Create**:
- `training/yolo/validate.py`
- `training/yolo/test_images/` (real webcam captures)

**Acceptance Criteria**:
- [ ] Capture 50+ real webcam test images
- [ ] Variety of backgrounds: playmat, desk, hand-held, cluttered
- [ ] Variety of lighting: bright, dim, mixed
- [ ] Measure detection rate on real images
- [ ] Document failure cases with screenshots

**Test Protocol**:
1. Capture images with 1, 2, 3+ cards visible
2. Include edge cases: partially visible, overlapping, rotated
3. Include challenging backgrounds: cluttered desk, patterned surface
4. Run YOLO detection on all images
5. Manually verify detections
6. Calculate precision/recall

**Success Metrics**:
| Metric | Target | Minimum |
|--------|--------|---------|
| Detection rate (good lighting) | >= 98% | >= 95% |
| Detection rate (typical room) | >= 95% | >= 90% |
| False positive rate | < 2% | < 5% |
| FPS (detection only) | >= 30 | >= 20 |
| FPS (detection + identification) | >= 15 | >= 10 |

**Output Required**:
- `validation_results.csv` with per-image metrics
- `validation_summary.md` with aggregate statistics
- `failure_cases/` directory with annotated failure screenshots
- Go/No-Go recommendation based on metrics

---

## Dependencies to Add

Add to `training/requirements.txt`:
```
ultralytics>=8.0.0
albumentations>=1.3.0
Pillow>=9.0.0
```

---

## File Structure After Completion

```
training/
    yolo/
        download_backgrounds.py
        generate_synthetic.py
        augmentations.py
        train.py
        validate.py
        config.yaml
        runs/
            detect/
                train/
                    weights/
                        best.pt
                        best.onnx
                    results.csv
    data/
        backgrounds/
            wood/
            fabric/
            clutter/
            ...
            backgrounds_manifest.json
        yolo_dataset/
            images/
                train/
                val/
            labels/
                train/
                val/
            data.yaml
        reference_images/  # Existing 32K cards
spike/
    inference.py  # Modified
    yolo_detector.py  # New
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Synthetic data doesn't transfer to real images | Include diverse backgrounds, aggressive augmentation |
| YOLO too slow on CPU | Start with nano, only upgrade if accuracy insufficient |
| Cards not detected when rotated | Include heavy rotation augmentation in training |
| Overlapping cards cause issues | Train with overlapping cards, NMS handles duplicates |
| Sleeved cards look different | Add sleeve overlay augmentation in synthetic data |

---

## Rollback Plan

If YOLO detection fails to meet targets:
1. Keep contour-based detection as default
2. YOLO available as optional upgrade
3. Document: "For best results, use solid-color playmat"

---

## Next Steps

1. @spike-worker: Start with Phase 1 (Background Collection)
2. Report progress daily
3. Orchestrator reviews after each phase before proceeding
4. Final validation determines Go/No-Go for YOLO integration
