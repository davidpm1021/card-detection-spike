# Card Detection Fix Plan

## Executive Summary

Current accuracy: **43%** Top-1
Target accuracy: **85%** Top-1

Root cause: Training pipeline bug (0% validation accuracy) + insufficient training + weak backbone.

Solution: Use **CLIP** as backbone + **proper metric learning** + **more training epochs**.

---

## Phase 1: Fix Training Pipeline (Day 1-2)

### Problem
`training/history.json` shows validation accuracy stuck at 0% for all 10 epochs.
This means the model never learned to generalize - it memorized training data.

### Actions

1. **Diagnose validation issue**
   ```python
   # Check if validation labels match training labels
   # Check if validation transform is correct (no augmentation)
   # Check if validation split has same card names as training
   ```

2. **Fix data split**
   - Ensure validation has DIFFERENT card images, not different cards
   - Same card, different augmentation = valid test of generalization

3. **Add proper validation metrics**
   - Top-1 accuracy
   - Top-5 accuracy
   - Mean Average Precision (mAP)

---

## Phase 2: Upgrade to CLIP Backbone (Day 2-3)

### Why CLIP?
- Pretrained on 400M image-text pairs (vs ImageNet's 1M)
- Already understands visual similarity
- Width.ai achieved **92.44%** with fine-tuned CLIP
- Easy to fine-tune with `transformers` library

### Implementation

```python
from transformers import CLIPModel, CLIPProcessor

# Load pretrained CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Extract image encoder
image_encoder = model.vision_model

# Add ArcFace head for metric learning
class CardIdentifier(nn.Module):
    def __init__(self, num_classes=32232):
        super().__init__()
        self.backbone = image_encoder  # CLIP vision encoder
        self.embedding = nn.Linear(768, 512)  # Project to 512-dim
        self.arcface = ArcFaceHead(512, num_classes)
```

### Expected Improvement
- CLIP embeddings are already good at visual similarity
- Fine-tuning should reach 80%+ quickly

---

## Phase 3: Proper Metric Learning (Day 3-4)

### Current Problem
ArcFace loss with CrossEntropy = classification task
But we need: **retrieval task** (find most similar card)

### Better Approach: Contrastive Learning

```python
# Option 1: Triplet Loss
# Anchor = card image
# Positive = same card, different augmentation
# Negative = different card

triplet_loss = nn.TripletMarginLoss(margin=0.3)

# Option 2: NT-Xent (SimCLR style)
# Each card is its own class
# Augmented views of same card should be close

# Option 3: ArcFace (keep current, but fix training)
# Works well if training actually converges
```

### Key: Multiple Augmented Views

```python
# For each card, generate 10-20 augmented views
augmentations = [
    # Webcam simulation
    A.RandomBrightnessContrast(p=0.8),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    A.RandomRotate90(p=0.5),
    A.Perspective(scale=(0.05, 0.15), p=0.5),
    # Lighting simulation
    A.RandomShadow(p=0.3),
    A.RandomSunFlare(p=0.2),
]
```

---

## Phase 4: Extended Training (Day 4-5)

### Current: 10 epochs
### Target: 50-100 epochs with early stopping

```python
# Training config
config = {
    'epochs': 100,
    'batch_size': 64,
    'lr': 1e-4,  # Lower LR for fine-tuning
    'warmup_epochs': 5,
    'early_stopping_patience': 10,
    'scheduler': 'cosine_annealing',
}

# Checkpointing
# Save best model by validation Top-1 accuracy
# Not by loss (can overfit while loss decreases)
```

---

## Phase 5: Multi-Card Detection (Day 5-6)

### Current YOLO Setup
Already supports multiple cards. Just need to:

1. **Process all detected cards** (not just first one)
2. **Track cards across frames** with IoU matching
3. **Independent identification** for each card

### Implementation

```python
def process_frame(frame):
    # Detect all cards
    detections = yolo(frame)

    results = []
    for box in detections:
        card_crop = extract_card(frame, box)
        embedding = model.embed(card_crop)
        matches = faiss_index.search(embedding, k=5)
        results.append({
            'box': box,
            'matches': matches,
            'confidence': matches[0].distance
        })

    return results
```

---

## Phase 6: Build New FAISS Index (Day 6)

### Current Index Issues
- Built from MobileNetV3 embeddings
- Different model = incompatible embeddings

### Process

1. Load all 32,232 reference images
2. Generate CLIP embeddings for each
3. Normalize embeddings (L2)
4. Build FAISS IndexFlatIP (inner product = cosine similarity)
5. Save index + mapping

```python
# Generate embeddings
embeddings = []
for img_path in reference_images:
    img = load_image(img_path)
    emb = model.embed(img)
    embeddings.append(emb)

# Build index
embeddings = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(512)
index.add(embeddings)
faiss.write_index(index, 'card_embeddings_clip.faiss')
```

---

## Phase 7: Testing & Validation (Day 7)

### Test Protocol

1. **Static image test** (60 webcam crops)
   - Target: 85% Top-1, 95% Top-5

2. **Live webcam test**
   - 20 different cards
   - Various lighting conditions
   - Sleeved and unsleeved

3. **Multi-card test**
   - 2-4 cards simultaneously
   - Overlapping allowed

### Metrics to Track

| Metric | Target | Current |
|--------|--------|---------|
| Top-1 Accuracy | 85% | 43% |
| Top-5 Accuracy | 95% | ~80% |
| Processing FPS | 15+ | 7-10 |
| False Positive Rate | <5% | Unknown |

---

## Alternative: Use Existing Solution

If the above takes too long, consider:

### Option A: MTG-card-scanner
- Clone https://github.com/reeshof/MTG-card-scanner
- Already achieves 98.8% accuracy
- MIT license
- Would need to update card database

### Option B: Scryfall API Image Search
- Scryfall has image search built-in
- Upload crop, get matches
- Rate limited but works

### Option C: Azure Computer Vision
- MTGScan uses this successfully
- Pay-per-use API
- High accuracy OCR

---

## Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Diagnose training bug | Root cause identified |
| 2 | Fix training + CLIP backbone | Working training loop |
| 3 | Metric learning implementation | Contrastive/ArcFace training |
| 4 | Extended training | Model checkpoint |
| 5 | Multi-card support | Updated inference pipeline |
| 6 | New FAISS index | CLIP-based index |
| 7 | Testing & validation | Accuracy report |

---

## Success Criteria

- [ ] Top-1 accuracy ≥ 85% on 60-image test set
- [ ] Top-5 accuracy ≥ 95%
- [ ] Processing speed ≥ 15 FPS
- [ ] Multi-card support (2-4 cards)
- [ ] No false positives with high confidence

---

## Appendix: Research Sources

- [MTG-card-scanner](https://github.com/reeshof/MTG-card-scanner) - 98.8% accuracy with FaceNet
- [Width.ai CLIP fine-tuning](https://www.width.ai/post/92-44-product-similarity-through-fine-tuning-clip-model-custom-pipeline-for-image-similarity) - 92.44% Top-1 on products
- [Delver Lens](https://www.delverlab.com/) - Production app, 90-96% accuracy
- [MTGScan](https://github.com/fortierq/mtgscan) - Azure OCR approach
- [CenterNet](https://arxiv.org/abs/1904.07850) - Alternative to YOLO for detection
