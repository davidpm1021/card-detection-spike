"""Diagnostic script - run on BOTH PC and laptop to compare."""
import os
import sys

print("=" * 60)
print("CARD DETECTION DIAGNOSTIC")
print("=" * 60)

# 1. Check Python and key dependencies
print("\n[1] ENVIRONMENT")
print(f"Python: {sys.version}")
print(f"Platform: {sys.platform}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("PyTorch: NOT INSTALLED")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError:
    print("OpenCV: NOT INSTALLED")

try:
    import faiss
    print(f"FAISS: {faiss.__version__ if hasattr(faiss, '__version__') else 'installed (no version)'}")
except ImportError:
    print("FAISS: NOT INSTALLED")

try:
    import albumentations
    print(f"Albumentations: {albumentations.__version__}")
except ImportError:
    print("Albumentations: NOT INSTALLED")

try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
except ImportError:
    print("NumPy: NOT INSTALLED")

# 2. Check critical files
print("\n[2] CRITICAL FILES")
files_to_check = [
    'training/yolo/runs/detect/train/weights/best.pt',
    'training/checkpoints/final_model.pt',
    'training/output/card_embeddings_full.faiss',
    'training/output/label_mapping_full.json',
]

for f in files_to_check:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"✓ {f}: {size:,} bytes")
    else:
        print(f"✗ {f}: MISSING!")

# 3. Check FAISS index
print("\n[3] FAISS INDEX CHECK")
try:
    import faiss
    index_path = 'training/output/card_embeddings_full.faiss'
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"Index vectors: {index.ntotal}")
        print(f"Index dimension: {index.d}")
    else:
        print("Index file missing!")
except Exception as e:
    print(f"Error loading index: {e}")

# 4. Check label mapping
print("\n[4] LABEL MAPPING CHECK")
try:
    import json
    mapping_path = 'training/output/label_mapping_full.json'
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        if "card_names" in mapping:
            print(f"Card names: {len(mapping['card_names'])}")
            print(f"First 3: {mapping['card_names'][:3]}")
        else:
            print(f"Keys in mapping: {list(mapping.keys())}")
    else:
        print("Mapping file missing!")
except Exception as e:
    print(f"Error loading mapping: {e}")

# 5. Test embedding model
print("\n[5] EMBEDDING MODEL TEST")
try:
    sys.path.insert(0, 'training')
    from model import CardEmbeddingModel
    import torch
    import numpy as np

    checkpoint_path = 'training/checkpoints/final_model.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        print(f"num_classes: {checkpoint.get('num_classes', 'N/A')}")
        print(f"embedding_dim: {checkpoint.get('embedding_dim', 'N/A')}")

        # Create model and load weights
        model = CardEmbeddingModel(
            num_classes=checkpoint["num_classes"],
            embedding_dim=checkpoint["embedding_dim"],
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Test with random input
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            embedding = model.get_embedding(test_input).numpy()

        print(f"Test embedding shape: {embedding.shape}")
        print(f"Test embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"Test embedding first 5 values: {embedding[0, :5]}")
    else:
        print("Checkpoint file missing!")
except Exception as e:
    print(f"Error testing model: {e}")
    import traceback
    traceback.print_exc()

# 6. Test with actual image if available
print("\n[6] DEBUG IMAGE TEST")
debug_img = 'debug_card_crop.jpg'
if os.path.exists(debug_img):
    try:
        import cv2
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        img = cv2.imread(debug_img)
        print(f"Image shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Image min/max: {img.min()}/{img.max()}")

        # Process through pipeline
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        transformed = transform(image=rgb)
        tensor = transformed["image"].unsqueeze(0)

        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor min/max: {tensor.min():.3f}/{tensor.max():.3f}")

        # Get embedding
        with torch.no_grad():
            embedding = model.get_embedding(tensor).numpy()

        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"Embedding first 5: {embedding[0, :5]}")

        # Search
        faiss.normalize_L2(embedding)
        distances, indices = index.search(embedding, 3)
        print(f"\nTop 3 matches:")
        for i in range(3):
            card_name = mapping['card_names'][indices[0][i]]
            print(f"  {i+1}. {card_name} ({distances[0][i]:.3f})")

    except Exception as e:
        print(f"Error processing debug image: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No debug_card_crop.jpg found")

print("\n" + "=" * 60)
print("Copy this ENTIRE output and share it!")
print("=" * 60)
