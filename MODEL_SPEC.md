# Card Recognition Model - Technical Specification

## Goal
Train a model that identifies MTG cards from webcam images, distributable to run locally on any computer.

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Webcam Frame   │────▶│  Card Detection  │────▶│ Cropped Card    │
└─────────────────┘     │  (OpenCV - done) │     │ (450x630)       │
                        └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              ▼
                        │  Embedding Model │◀─────────────┘
                        │  (MobileNetV3)   │
                        └────────┬─────────┘
                                 │
                                 ▼ 512-dim vector
                        ┌──────────────────┐
                        │  Vector Search   │
                        │  (FAISS index)   │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  Card Name +     │
                        │  Confidence      │
                        └──────────────────┘
```

---

## Model Choice: MobileNetV3-Large

**Why MobileNetV3:**
- Designed for mobile/edge deployment
- ~5.4M parameters (vs ResNet50's 25M)
- Fast CPU inference (~50ms per image)
- Good accuracy/speed tradeoff

**Alternative considered:**
- EfficientNet-B0: Slightly more accurate, slightly slower
- ResNet18: Heavier but more proven

---

## Training Approach: Metric Learning

### The Problem with Classification
- 35,000+ classes = huge final layer
- New cards require retraining
- Doesn't generalize to unseen printings

### Solution: Learn Embeddings
Train the model to produce **similar vectors for the same card** and **different vectors for different cards**.

### Loss Function: ArcFace (Additive Angular Margin)
- Better than triplet loss (more stable training)
- Forces larger angular margin between classes
- State-of-the-art for face recognition, works great for cards

```python
# Pseudocode
embedding = model(card_image)  # 512-dim vector
logits = arcface_head(embedding, card_id)
loss = cross_entropy(logits, card_id)
```

### Data Augmentation
To make model robust to webcam conditions:
- Random brightness/contrast
- Random rotation (±15°)
- Random perspective warp
- Gaussian blur (simulates focus issues)
- Color jitter
- Random noise

---

## Training Data

### Source: Scryfall Bulk Data
- ~80,000 unique card images available
- Multiple printings per card (different art)
- High quality source images

### Training Strategy
1. Download all card images from Scryfall
2. Group by card NAME (not by printing)
3. Train model to recognize NAME regardless of printing
4. This handles the "same card, different art" problem

### Data Split
- 90% training
- 5% validation
- 5% test (held out cards never seen during training)

---

## Inference Pipeline

### Pre-computed Index
```
scryfall_embeddings.faiss  (~100MB)
card_metadata.json         (~10MB)
```

For each unique card name:
- Compute embedding from Scryfall reference image
- Store in FAISS index for fast similarity search

### Runtime Flow
1. Detect card in frame (existing OpenCV code)
2. Crop and resize to 224x224
3. Run through model → 512-dim embedding (~50ms)
4. Search FAISS index for nearest neighbor (~1ms)
5. Return card name + confidence (cosine similarity)

---

## Distribution Package

### Files to Distribute
```
mtg_card_recognition/
├── model.onnx              # ~20MB (quantized)
├── embeddings.faiss        # ~100MB
├── card_names.json         # ~2MB
├── detector.py             # Card detection
├── recognizer.py           # Embedding + matching
└── requirements.txt        # Dependencies
```

**Total size: ~125MB**

### Installation
```bash
pip install mtg-card-recognition
# or
git clone ... && pip install -r requirements.txt
```

### Dependencies
- Python 3.9+
- OpenCV
- ONNX Runtime (CPU inference)
- FAISS-cpu
- NumPy

No PyTorch/TensorFlow needed at runtime (ONNX only).

---

## Expected Performance

| Metric | Target | Basis |
|--------|--------|-------|
| Accuracy (same printing) | >99% | Similar to reeshof's 98.8% |
| Accuracy (different printing) | >90% | Trained on card names |
| Inference time (CPU) | <100ms | MobileNetV3 benchmarks |
| Model size | <25MB | Quantized ONNX |
| Total package | <150MB | With embeddings |

---

## Training Requirements

### Hardware
- GPU recommended (RTX 3060 or better)
- ~16GB RAM
- ~50GB disk for training data

### Time Estimate
- Data download: ~2 hours
- Training: ~4-8 hours (GPU)
- Embedding generation: ~2 hours

### Software
- PyTorch 2.0+
- pytorch-metric-learning
- albumentations (augmentation)

---

## Implementation Phases

### Phase 1: Data Pipeline (Day 1)
- [ ] Download Scryfall bulk images
- [ ] Create card name → images mapping
- [ ] Build PyTorch Dataset with augmentation

### Phase 2: Model Training (Day 2-3)
- [ ] Implement MobileNetV3 + ArcFace head
- [ ] Train on Scryfall images
- [ ] Validate on held-out cards

### Phase 3: Embedding Index (Day 3)
- [ ] Generate embeddings for all cards
- [ ] Build FAISS index
- [ ] Test retrieval accuracy

### Phase 4: Integration (Day 4)
- [ ] Export model to ONNX
- [ ] Integrate with existing detection code
- [ ] End-to-end webcam testing

### Phase 5: Packaging (Day 5)
- [ ] Create pip-installable package
- [ ] Write installation instructions
- [ ] Test on fresh machine

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Model too slow on CPU | Use quantization, smaller backbone |
| Poor accuracy on webcam images | Heavy augmentation during training |
| Different printings not recognized | Train on card names, not printings |
| Package too large | Quantize model, compress embeddings |

---

## Success Criteria

1. **Accuracy**: >90% on real webcam captures
2. **Speed**: <200ms per card identification
3. **Portability**: Works on Windows/Mac/Linux
4. **Size**: <200MB total package
5. **Ease of use**: `pip install` + 3 lines of code
