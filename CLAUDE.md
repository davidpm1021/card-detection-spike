# Card Detection Spike - Project Conventions

## Project Type

This is a RESEARCH SPIKE, not production code. The goal is to answer questions, not build features.

## Time Box

2 weeks maximum. Decision point: Day 14.

## Primary Question

Can we reliably detect and identify MTG cards from a webcam feed?

## Pass/Fail Criteria

### Detection (Finding Cards)

| Metric                         | Pass | Fail |
| ------------------------------ | ---- | ---- |
| Detection rate (good lighting) | ≥95% | <85% |
| Detection rate (typical room)  | ≥85% | <70% |
| Sleeved cards                  | ≥90% | <75% |

### Identification (Which Card)

| Metric                   | Pass   | Fail    |
| ------------------------ | ------ | ------- |
| Accuracy (good lighting) | ≥85%   | <70%    |
| Accuracy (typical room)  | ≥70%   | <55%    |
| Time to identify         | ≤500ms | >1000ms |

### Performance

| Metric         | Pass   | Fail |
| -------------- | ------ | ---- |
| Processing FPS | ≥15    | <10  |
| Memory usage   | ≤500MB | >1GB |

## Tech Stack

- Python 3.11+
- OpenCV 4.9+
- NumPy
- Requests (for Scryfall API)

## Code Style

- Single-file scripts are acceptable (this is a spike)
- Document findings, not code elegance
- Console output for debugging
- CSV logging for quantitative results

## What NOT To Build

- No UI frameworks
- No web servers
- No databases
- No Angular/React/Electron
- No "clean architecture"

## Daily Workflow

1. Update DAILY_LOG.md with goals
2. Do the work
3. Document results quantitatively
4. Plan tomorrow

## Definition of Done (for spike)

- [ ] Detection rate measured across 60 test images
- [ ] Identification accuracy measured
- [ ] Performance benchmarked
- [ ] Failure cases documented with screenshots
- [ ] Go/No-Go recommendation written with data

## Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run
python spike/spike.py
```

---

## Production Deployment Architecture (Post-Spike)

### Use Case

Remote MTG play over Discord video. Each player:
1. Has OBS combining face cam + card cam → virtual webcam → Discord
2. Sees friends' playmats via Discord video call
3. Wants card names overlaid on friends' video streams

### Architecture Decision: Desktop Overlay Application

```
Friend's OBS → Discord Video → Your Discord Window
                                      ↓
                              Screen/Window Capture
                                      ↓
                              YOLO Detection
                                      ↓
                              Card Identification (ONNX)
                                      ↓
                              Transparent Overlay Window
```

### Why Not Web App?

- Need to capture Discord window (browser can't do this)
- Need transparent overlay on desktop
- Local processing avoids bandwidth issues while streaming

### Packaging Strategy

- **PyInstaller** to create standalone .exe (Windows) / .app (Mac)
- Bundle: Python + PyTorch/ONNX Runtime + OpenCV + YOLO model + embedding model + FAISS index
- Expected size: 300-500MB
- User experience: "Download, unzip, run"

### Critical: ONNX for Cross-Platform Consistency

**Problem discovered:** PyTorch produces different embeddings on different builds:
- PyTorch 2.6.0+cu124 (CUDA) vs PyTorch 2.6.0+cpu produce DIFFERENT embeddings
- FAISS index built with CUDA embeddings won't match CPU inference

**Solution:** Export embedding model to ONNX format
- ONNX Runtime produces identical results across all platforms
- Smaller runtime dependency than full PyTorch
- Faster inference

### Distribution Plan

1. Export models to ONNX (embedding model + optionally YOLO)
2. Generate FAISS index using ONNX embeddings
3. Package with PyInstaller for Windows
4. Package for Mac (requires Mac for building)
5. Distribute via GitHub Releases or shared drive
