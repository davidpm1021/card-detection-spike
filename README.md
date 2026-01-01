# MTG Card Detection

Machine learning system for detecting and identifying Magic: The Gathering cards from webcam images.

## Overview

This project uses a FaceNet-style embedding approach:
- **MobileNetV3-Large** backbone (3.67M params, runs on CPU)
- **ArcFace loss** for discriminative embeddings
- **FAISS index** for fast similarity search
- Heavy augmentation to simulate webcam conditions

## Quick Start (New Machine)

### Prerequisites

- Python 3.11+ installed
- Git installed
- ~2GB free disk space

### Step-by-Step Setup

Open a terminal (Command Prompt or PowerShell on Windows, Terminal on Mac/Linux) and run:

```bash
# 1. Clone the repository
git clone https://github.com/davidpm1021/card-detection-spike.git
cd card-detection-spike

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows (Command Prompt):
venv\Scripts\activate.bat
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r training/requirements.txt

# 5. Download training images from Scryfall (~5 minutes)
cd training
python download_data.py --max-images 1000

# 6. Train the model (~30-60 min on CPU, ~10 min on GPU)
python train.py --epochs 30 --batch-size 32

# 7. Generate the FAISS search index
python generate_embeddings.py
```

### Verify It Worked

After step 7, you should see output like:
```
Train self-retrieval: 85-90%
Files saved to output/
```

You'll have these files in `training/`:
- `checkpoints/best_model.pt` - Trained model weights
- `output/card_embeddings.faiss` - Search index
- `output/label_mapping.json` - Card name mappings

## Project Structure

```
card-detection-spike/
├── training/
│   ├── download_data.py     # Scryfall data downloader
│   ├── dataset.py           # PyTorch dataset with augmentation
│   ├── model.py             # MobileNetV3 + ArcFace model
│   ├── train.py             # Training loop
│   ├── generate_embeddings.py  # FAISS index builder
│   └── requirements.txt     # Dependencies
├── MODEL_SPEC.md            # Architecture design
├── SPIKE_CONCLUSION.md      # Research findings
└── CLAUDE.md               # Project conventions
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- 4GB RAM minimum
- GPU optional (CPU works fine)

## Current Status

- Detection: Working (contour-based, ~95% accuracy)
- Identification: Proof of concept (88% self-retrieval on 396 cards)
- Next step: Scale to full 35K+ card database

## Architecture

See [MODEL_SPEC.md](MODEL_SPEC.md) for detailed architecture.

The model produces 512-dimensional embeddings. Cards are identified by finding the nearest neighbor in a FAISS index of reference embeddings.
