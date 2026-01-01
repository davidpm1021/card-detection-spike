# Card Detection Spike - Final Conclusion

**Date:** January 2026
**Duration:** ~2 weeks
**Primary Question:** Can we reliably detect and identify MTG cards from a webcam feed?

---

## Executive Summary

**Detection: PASS** - Finding cards in frame works reliably.
**Identification: FAIL** - Identifying which card (from 35,000+) is not feasible with tested approaches.

**Recommendation:** Do not proceed with real-time webcam card identification as designed. Consider alternative approaches outlined below.

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

**Do not proceed** with real-time webcam card identification as originally designed.

**If card identification is still required, consider:**

1. **Capture-then-identify workflow** - User takes a photo, system processes it with cloud OCR, returns result in 1-2 seconds. Not real-time but achievable.

2. **Barcode/QR approach** - If you control the cards (e.g., inventory system), add QR codes to sleeves or card holders.

3. **Manual entry with autocomplete** - User types first few letters, fuzzy search suggests matches. Faster than scanning with current tech.

4. **Wait for better tech** - Vision models are improving rapidly. Re-evaluate in 6-12 months.

---

## Lessons Learned

1. **Validate assumptions early** - The "mana symbol matching works 100%" result was from circular testing (templates from same images). Should have tested on separate images immediately.

2. **Artwork variation is the killer** - We underestimated how many different printings exist. This invalidated all image-matching approaches.

3. **Webcam quality matters** - 1280x720 is insufficient for reliable OCR of small, stylized text.

4. **Research before implementation** - We should have researched the artwork variation problem before building the production prototype.

---

## Spike Status: COMPLETE

**Go/No-Go Decision: NO-GO** for real-time webcam identification of all cards.

**Partial GO:** Card detection (finding rectangles) is production-ready if needed for other purposes.
