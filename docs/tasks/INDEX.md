# YOLO Detection Training - Task Index

**Project:** Card Detection Spike
**Goal:** Train YOLOv8 for robust MTG card detection

---

## Task Overview

| Task | Phase | Status | Dependencies | Est. Time |
|------|-------|--------|--------------|-----------|
| [TASK_001_BACKGROUND_COLLECTION](TASK_001_BACKGROUND_COLLECTION.md) | 1 | Ready | None | 1 hour |
| [TASK_002_SYNTHETIC_DATA_GENERATION](TASK_002_SYNTHETIC_DATA_GENERATION.md) | 2 | Blocked | Task 001 | 2-3 hours |
| [TASK_003_YOLO_TRAINING](TASK_003_YOLO_TRAINING.md) | 3 | Blocked | Task 002 | 30-60 min (GPU) |
| [TASK_004_INTEGRATION](TASK_004_INTEGRATION.md) | 4 | Blocked | Task 003 | 2-3 hours |
| [TASK_005_VALIDATION](TASK_005_VALIDATION.md) | 5 | Blocked | Task 004 | 1-1.5 hours |

**Total Estimated Time:** 3-4 days

---

## Execution Order

```
Phase 1: Background Collection
    |
    v
Phase 2: Synthetic Data Generation (10K+ images)
    |
    v
Phase 3: YOLO Training (mAP@0.5 >= 0.90)
    |
    v
Phase 4: Integration (YOLODetector class)
    |
    v
Phase 5: Validation (Real webcam tests)
    |
    v
Go/No-Go Decision
```

---

## Quick Start for Workers

### Phase 1: Background Collection

```bash
cd C:\Users\Dave\Cursor\card-detection-spike
python training/yolo/download_backgrounds.py
```

Expected output: `training/data/backgrounds/` with 1000+ images

### Phase 2: Synthetic Data Generation

```bash
python training/yolo/generate_synthetic.py --num-train 10000 --num-val 1000
```

Expected output: `training/data/yolo_dataset/` with YOLO format data

### Phase 3: YOLO Training

```bash
python training/yolo/train.py --model n --epochs 100
```

Expected output: `training/yolo/runs/detect/train/weights/best.pt`

### Phase 4: Integration

Modify `spike/inference.py` to use `spike/yolo_detector.py`

```bash
python spike/inference.py --camera 0  # Test with YOLO
python spike/inference.py --camera 0 --no-yolo  # Test without YOLO
```

### Phase 5: Validation

```bash
python training/yolo/collect_images.py --camera 0
python training/yolo/validate.py
```

Expected output: `training/yolo/results/validation_summary.md`

---

## Success Criteria

| Metric | Target | Minimum |
|--------|--------|---------|
| mAP@0.5 (training) | >= 0.92 | >= 0.90 |
| Detection rate (real images) | >= 95% | >= 90% |
| Inference time (CPU) | < 30ms | < 50ms |
| FPS (detection + ID) | >= 20 | >= 15 |
| False positive rate | < 2% | < 5% |

---

## Key Files Created

After all tasks complete:

```
training/
    yolo/
        download_backgrounds.py
        generate_synthetic.py
        augmentations.py
        train.py
        collect_images.py
        validate.py
        runs/
            detect/
                train/
                    weights/
                        best.pt
                        best.onnx
        results/
            validation_results.csv
            validation_summary.md
            failure_cases/
    data/
        backgrounds/
            manifest.json
            wood/
            fabric/
            ...
        yolo_dataset/
            data.yaml
            images/
                train/
                val/
            labels/
                train/
                val/
spike/
    yolo_detector.py
    inference.py (modified)
```

---

## Reporting

After each task, worker should report:
1. Task completion status
2. Key metrics achieved
3. Any blockers or issues
4. Time taken

Example:
```
TASK 001 COMPLETE
- Downloaded 1,053 background images
- Categories: wood(152), fabric(168), solid(100), desk(145), pattern(132), nature(98), misc(258)
- Time: 45 minutes
- Blockers: None
```

---

## Rollback Plan

If YOLO detection fails to meet targets:

1. Keep contour detection as default in `inference.py`
2. YOLO available via `--yolo` flag only
3. Document limitations in `SPIKE_CONCLUSION.md`
4. Recommend playmat usage for reliable detection
