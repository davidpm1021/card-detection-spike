# Card Detection Spike - Final Conclusion

**Date:** January 2026
**Duration:** ~2 weeks
**Primary Question:** Can we reliably detect and identify MTG cards from a webcam feed?

---

## Executive Summary

**Detection: CONDITIONAL PASS** - Works on simple backgrounds, fragile on complex scenes.
**Identification: CONDITIONAL PASS** - Top-5 accuracy is good, top-1 needs improvement.

**Recommendation:** Viable with constraints. Requires either (1) controlled background (playmat), or (2) trained object detector for robust detection.

---

## Hardware Tested

- **Camera:** Standard USB webcam
- **Max Resolution:** 1280x720 (hardware limit)
- **Processing:** Local CPU (no GPU)

---

## Detection Results (Finding Cards in Frame)

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Detection rate (good lighting) | ≥95% | ~95% | **PASS** |
| Detection rate (typical room) | ≥85% | ~90% | **PASS** |
| Sleeved cards | ≥90% | ~90% | **PASS** |
| Processing FPS | ≥15 | 7-10 | **WARN** |

**Method:** Contour-based detection with edge detection (Canny + morphological ops)

**Conclusion:** Card detection is reliable and production-ready.

---

## Identification Results (Which Card Is It?)

### Approaches Tested

#### 1. Template Matching (Mana Symbols)
| Metric | Result |
|--------|--------|
| Accuracy on spike test images | 100% |
| Accuracy on live webcam | ~60% confidence, wrong IDs |
| Speed | ~9ms per card |
| Card coverage | 5 basic lands only (0.01% of cards) |

**Why it failed:** Templates were created from the same images being tested (circular validation). Real webcam conditions (lighting, angle, blur) produce different results.

#### 2. Color-Based Matching
| Metric | Result |
|--------|--------|
| Accuracy on spike test images | 100% (4/4) |
| Accuracy on live webcam | Wrong identifications |
| Card coverage | 5 basic lands only |

**Why it failed:** Color analysis is too sensitive to lighting conditions. A Swamp was identified as Island.

#### 3. Perceptual Hashing (pHash)
| Metric | Result |
|--------|--------|
| Accuracy on simulated tests | 79.2% (RGB-aware, 48-bit) |
| Accuracy on real webcam | 0% |
| Speed | <1ms per comparison |

**Why it failed:** MTG cards have multiple printings with different artwork. The same card name can look completely different. pHash matches artwork, not card identity.

#### 4. CNN Embeddings (ResNet50 + FAISS)
| Metric | Result |
|--------|--------|
| Accuracy | 20% |
| Speed | ~100ms per identification |

**Why it failed:** Same as pHash - different artworks produce different embeddings. The model matches visual similarity, not card identity.

#### 5. OCR (EasyOCR + Fuzzy Matching)
| Metric | Result |
|--------|--------|
| Accuracy on clean static images | ~70-80% |
| Accuracy on live webcam | Unreliable ("Monk" instead of card names) |
| Speed | ~450ms per identification |
| Card coverage | All 35,803 cards in database |

**Why it partially works:** Card names are the only consistent element across printings. Text doesn't change with artwork.

**Why it fails on webcam:** Image quality too low, motion blur, OCR struggles with stylized MTG fonts.

#### 6. FaceNet-Style Embeddings (MobileNetV3 + ArcFace + FAISS) ✓ NEW
| Metric | Result |
|--------|--------|
| Training accuracy | 99.4% (396 cards, 1000 images) |
| Self-retrieval accuracy | 94.4% |
| Top-1 webcam accuracy | ~30-50% |
| Top-5 webcam accuracy | ~80-100% |
| Speed | ~90ms identification, ~330ms detection |
| Card coverage | 32,062 cards indexed |

**Why it works:**
- Learned embeddings capture visual similarity across artwork variations
- FAISS index enables fast nearest-neighbor search
- ArcFace loss produces discriminative embeddings

**Current limitations:**
- Detection relies on text-box expansion (fragile)
- Complex backgrounds cause detection failures
- Top-1 confidence moderate (~0.5-0.7)

**What's needed for production:**
- Trained object detector (YOLO/CenterNet) for robust detection
- OR controlled background (playmat/solid surface)
- Fine-tuning on more webcam training data

---

## Key Insight

> **The fundamental problem:** MTG cards have thousands of different printings with varying artwork, frames, and designs. The ONLY consistent identifier across all printings is the **card name text**.

This means:
- Any image-based approach (templates, hashing, embeddings) will fail when cards have multiple artworks
- OCR is the only viable approach for full card coverage
- OCR requires higher image quality than a typical webcam provides

---

## Performance Summary

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Processing FPS | ≥15 | 7-10 | **WARN** |
| Memory usage | ≤500MB | ~300MB | **PASS** |
| Identification accuracy | ≥85% | <20% | **FAIL** |
| Time to identify | ≤500ms | 450ms (OCR) | **PASS** |

---

## What Would Be Needed for Success

### Option A: Higher Quality Capture
- Dedicated camera with macro lens
- Controlled lighting setup
- Fixed card position/holder
- Still image capture (not video stream)
- Cloud OCR (Azure, Google Vision) for accuracy

**Trade-off:** No longer "casual webcam" use case

### Option B: Limited Scope
- Only identify basic lands (5 cards)
- Use template matching with carefully crafted templates
- Accept that non-basic cards won't be identified

**Trade-off:** Severely limited utility

### Option C: Different Architecture
- Capture still frame on user action (button press)
- Send to cloud OCR API
- Accept 1-2 second latency
- Display result after processing

**Trade-off:** Not real-time, requires internet

### Option D: Train Custom Model
- Collect thousands of webcam card images
- Train CNN classifier on card names (not artwork)
- Significant data collection and training effort

**Trade-off:** Months of work, may still fail

---

## Files Produced

```
spike/
├── results/
│   ├── detection_results.csv      # Detection accuracy data
│   ├── detection_summary.csv      # Per-image detection stats
│   ├── phash_results.csv          # pHash matching results
│   ├── identification_log.csv     # OCR identification attempts
│   └── IDENTIFICATION_SUMMARY.md  # Detailed findings
├── mana_templates/                # Template images for basic lands
├── reference_cards/               # Test card images
└── test_images/                   # Captured webcam test images

production/
├── camera.py          # Threaded camera capture
├── detector.py        # Contour-based card detection
├── perspective.py     # Perspective correction + CLAHE
├── identifier.py      # Hybrid identification orchestrator
├── mana_matcher.py    # Template-based matching (failed)
├── color_matcher.py   # Color-based matching (failed)
├── ocr_identifier.py  # EasyOCR wrapper
├── card_database.py   # Scryfall database (35,803 cards)
├── cache.py           # Result caching/debouncing
└── ui.py              # OpenCV display overlay
```

---

## Recommendation

**CONDITIONAL GO** - Proceed with constraints.

The FaceNet-style embedding approach shows promise:
- 32K cards indexed and searchable
- Top-5 accuracy sufficient for "did you mean?" suggestions
- ~420ms total latency (acceptable)

**For production, choose one path:**

### Path A: Controlled Environment (Easier)
- Require solid-color playmat or surface
- Detection will be reliable with contour-based approach
- Ship with current codebase + usage instructions

### Path B: Robust Detection (More Work)
- Train YOLO/CenterNet on MTG card images
- Use synthetic data generation (cards on random backgrounds)
- ~1-2 weeks additional development

### Path C: Hybrid Approach (Recommended)
- Use current detection with playmat recommendation
- Show top-5 suggestions with confidence scores
- User confirms correct card
- Gracefully handle detection failures

---

## Lessons Learned

1. **Validate assumptions early** - The "mana symbol matching works 100%" result was from circular testing (templates from same images). Should have tested on separate images immediately.

2. **Artwork variation is the killer** - We underestimated how many different printings exist. This invalidated all image-matching approaches.

3. **Webcam quality matters** - 1280x720 is insufficient for reliable OCR of small, stylized text.

4. **Research before implementation** - We should have researched the artwork variation problem before building the production prototype.

---

## Spike Status: COMPLETE

**Go/No-Go Decision: CONDITIONAL GO**

✓ Identification works (FaceNet embeddings + FAISS)
✓ 32,062 cards indexed
✓ Top-5 accuracy acceptable for suggestions
✓ ~420ms latency meets requirements

⚠ Detection needs controlled background OR trained detector
⚠ Top-1 accuracy needs improvement for auto-identification

**Next Steps if Proceeding:**
1. Test with playmat/solid background to validate detection
2. Collect more test images for accuracy measurement
3. Decide: controlled environment vs. train YOLO detector
