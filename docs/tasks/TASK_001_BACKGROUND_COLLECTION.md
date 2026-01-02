# Task 001: Background Collection

**Phase**: 1 of 5
**Assigned to**: @spike-worker
**Status**: Ready
**Priority**: High (Blocking Phase 2)

---

## Objective

Download and prepare 1,000+ diverse background images for synthetic training data generation. These backgrounds will be used to composite MTG card images onto, creating realistic training scenes for YOLO detection.

---

## Deliverables

### Files to Create

| File | Description |
|------|-------------|
| `training/yolo/download_backgrounds.py` | Main download script |
| `training/data/backgrounds/` | Directory with categorized images |
| `training/data/backgrounds/manifest.json` | Metadata about all backgrounds |

### Success Criteria

- [ ] 1,000+ background images downloaded
- [ ] Images organized by category
- [ ] All images at least 640x640 pixels
- [ ] No MTG cards in any background images
- [ ] Manifest file with image paths and categories
- [ ] Script can resume interrupted downloads

---

## Technical Specification

### Background Categories (minimum counts)

| Category | Min Count | Description | Priority |
|----------|-----------|-------------|----------|
| wood | 150 | Wood grain textures (tables) | High |
| fabric | 150 | Playmats, tablecloths, felt | High |
| solid | 100 | Solid colors with noise | Medium |
| desk | 150 | Desktop surfaces, may have objects | High |
| pattern | 100 | Abstract patterns, tiles | Medium |
| hands | 100 | Skin tones, holding gestures | Medium |
| nature | 100 | Outdoor backgrounds (grass, leaves) | Low |
| misc | 150 | Other textures and surfaces | Low |

### Recommended Sources

#### 1. DTD Dataset (Primary - High Quality Textures)
```
URL: https://www.robots.ox.ac.uk/~vgg/data/dtd/
Size: ~600MB
Images: 5,640 texture images
Categories: 47 texture classes
```

**Useful DTD categories for our purposes:**
- `banded`, `braided`, `woven` -> fabric
- `matted`, `meshed`, `waffled` -> fabric
- `grained`, `grooved`, `lined` -> wood
- `dotted`, `grid`, `honeycombed` -> pattern
- `cracked`, `weathered` -> desk

#### 2. Unsplash API (Secondary - Real Scenes)
```
API: https://api.unsplash.com
Rate limit: 50 requests/hour (free tier)
Search terms: "wooden table", "desk surface", "fabric texture", "playmat"
```

#### 3. Procedural Generation (Supplementary)
Generate programmatically:
- Solid colors with Perlin noise
- Gradient backgrounds
- Simple geometric patterns

### Output Format

```
training/data/backgrounds/
    wood/
        wood_001.jpg
        wood_002.jpg
        ...
    fabric/
        fabric_001.jpg
        ...
    solid/
        solid_001.jpg
        ...
    desk/
        desk_001.jpg
        ...
    pattern/
        pattern_001.jpg
        ...
    hands/
        hands_001.jpg
        ...
    nature/
        nature_001.jpg
        ...
    misc/
        misc_001.jpg
        ...
    manifest.json
```

### manifest.json Format

```json
{
    "total_count": 1234,
    "categories": {
        "wood": {
            "count": 150,
            "files": ["wood/wood_001.jpg", "wood/wood_002.jpg", ...]
        },
        "fabric": {
            "count": 150,
            "files": [...]
        }
    },
    "sources": {
        "dtd": 600,
        "unsplash": 400,
        "procedural": 234
    },
    "stats": {
        "min_width": 640,
        "min_height": 640,
        "avg_width": 1024,
        "avg_height": 1024
    }
}
```

---

## Implementation Guide

### Step 1: Download DTD Dataset

```python
import requests
import zipfile
from pathlib import Path

DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"

def download_dtd(output_dir: Path):
    """Download and extract DTD dataset."""
    # Download tarball
    # Extract to temp directory
    # Copy relevant categories to output_dir
    # Resize images if needed (min 640x640)
    pass
```

### Step 2: Generate Procedural Backgrounds

```python
import numpy as np
from PIL import Image
from noise import pnoise2  # pip install noise

def generate_solid_with_noise(width=640, height=640, base_color=(139, 90, 43)):
    """Generate solid color with Perlin noise texture."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Base color
    img[:, :] = base_color

    # Add Perlin noise
    scale = 50.0
    for y in range(height):
        for x in range(width):
            noise_val = pnoise2(x/scale, y/scale, octaves=4)
            noise_val = (noise_val + 1) / 2 * 30  # Scale to 0-30
            img[y, x] = np.clip(img[y, x] + noise_val, 0, 255)

    return Image.fromarray(img)
```

### Step 3: Fetch from Unsplash (Optional)

```python
import requests

UNSPLASH_ACCESS_KEY = "your_key_here"  # User must provide

def search_unsplash(query: str, count: int = 30):
    """Search Unsplash for images."""
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    params = {"query": query, "per_page": count}

    response = requests.get(url, headers=headers, params=params)
    return response.json()["results"]
```

### Step 4: Validate and Resize

```python
from PIL import Image

MIN_SIZE = 640

def validate_and_resize(image_path: Path, output_path: Path) -> bool:
    """Ensure image meets minimum size requirements."""
    img = Image.open(image_path)

    if img.width < MIN_SIZE or img.height < MIN_SIZE:
        # Skip if too small to upscale reasonably
        if img.width < MIN_SIZE // 2 or img.height < MIN_SIZE // 2:
            return False
        # Resize smaller dimension to MIN_SIZE, maintain aspect
        scale = MIN_SIZE / min(img.width, img.height)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    img.save(output_path, "JPEG", quality=90)
    return True
```

---

## Validation

After running, verify:

```python
# Check counts
manifest = json.load(open("training/data/backgrounds/manifest.json"))
assert manifest["total_count"] >= 1000

# Check all files exist
for category, info in manifest["categories"].items():
    for filepath in info["files"]:
        assert Path(f"training/data/backgrounds/{filepath}").exists()

# Check minimum dimensions
from PIL import Image
for filepath in all_files:
    img = Image.open(filepath)
    assert img.width >= 640 and img.height >= 640
```

---

## Console Output Expected

```
========================================
Background Image Collection
========================================

Phase 1: Downloading DTD Dataset
  Downloading dtd-r1.0.1.tar.gz... 600MB [====================] 100%
  Extracting...
  Processing textures...
    banded: 120 images
    braided: 111 images
    grained: 120 images
    ...
  Total DTD images: 623

Phase 2: Generating Procedural Backgrounds
  Generating solid colors: 50 images
  Generating gradients: 30 images
  Generating noise patterns: 50 images
  Total procedural: 130

Phase 3: Organizing by Category
  wood: 152 images
  fabric: 168 images
  solid: 100 images
  desk: 145 images
  pattern: 132 images
  hands: 0 images (skipped - requires manual collection or Unsplash)
  nature: 98 images
  misc: 158 images

========================================
Collection Complete!
========================================
Total backgrounds: 953
Manifest saved to: training/data/backgrounds/manifest.json

Note: hands category has 0 images. Consider:
  1. Manual collection from stock photo sites
  2. Skip this category (cards are usually on surfaces, not in hands)
```

---

## Dependencies

```python
# Add to requirements if not present
requests>=2.28.0
Pillow>=9.0.0
numpy>=1.24.0
tqdm>=4.65.0
noise>=1.2.0  # For Perlin noise (optional)
```

---

## Time Estimate

- DTD download and processing: 30 minutes
- Procedural generation: 15 minutes
- Validation and manifest: 15 minutes
- **Total: ~1 hour**

---

## Notes for Worker

1. **No Unsplash key required** - DTD + procedural should be sufficient
2. **Hands category is optional** - Can skip if no easy source available
3. **Quality over quantity** - Better to have 800 good images than 1200 with junk
4. **Check for duplicates** - DTD may have near-duplicates across categories

---

## Next Task

After completing this task, proceed to:
- **Task 002: Synthetic Data Generation** (Phase 2)
