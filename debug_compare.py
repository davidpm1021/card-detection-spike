"""Compare live crop vs saved crop processing."""
import cv2
import torch
import numpy as np
import faiss
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.insert(0, 'training')
from model import CardEmbeddingModel

# Load model
checkpoint = torch.load('training/checkpoints/final_model.pt', map_location='cpu', weights_only=False)
embed_model = CardEmbeddingModel(
    num_classes=checkpoint["num_classes"],
    embedding_dim=checkpoint["embedding_dim"],
    pretrained=False,
)
embed_model.load_state_dict(checkpoint["model_state_dict"])
embed_model.eval()

# Load index
index = faiss.read_index('training/output/card_embeddings_full.faiss')
with open('training/output/label_mapping_full.json', 'r') as f:
    card_names = json.load(f)["card_names"]

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def get_embedding_and_match(img_bgr, name):
    """Process image and get top match."""
    print(f"\n=== {name} ===")
    print(f"Input shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")
    print(f"Min/Max: {img_bgr.min()}/{img_bgr.max()}")

    # Convert to RGB
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"RGB shape: {rgb.shape}, min/max: {rgb.min()}/{rgb.max()}")

    # Transform
    transformed = transform(image=rgb)
    tensor = transformed["image"].unsqueeze(0)
    print(f"Tensor shape: {tensor.shape}, min/max: {tensor.min():.3f}/{tensor.max():.3f}")

    # Get embedding
    with torch.no_grad():
        embedding = embed_model.get_embedding(tensor).numpy()
    print(f"Embedding shape: {embedding.shape}, norm before: {np.linalg.norm(embedding):.3f}")

    # Normalize
    faiss.normalize_L2(embedding)
    print(f"Embedding norm after: {np.linalg.norm(embedding):.3f}")

    # Search
    distances, indices = index.search(embedding, 5)
    print(f"Top 5 matches:")
    for i in range(5):
        print(f"  {i+1}. {card_names[indices[0][i]]} ({distances[0][i]:.3f})")

    return embedding

# Test 1: Load saved debug crop
print("\n" + "="*50)
print("TEST 1: Saved debug_card_crop.jpg")
print("="*50)
saved = cv2.imread('debug_card_crop.jpg')
if saved is None:
    print("ERROR: debug_card_crop.jpg not found!")
else:
    emb1 = get_embedding_and_match(saved, "Saved crop")

# Test 2: Simulate live capture (read and immediately use)
print("\n" + "="*50)
print("TEST 2: Simulated live (re-read same file)")
print("="*50)
live_sim = cv2.imread('debug_card_crop.jpg')
emb2 = get_embedding_and_match(live_sim, "Simulated live")

# Compare embeddings
print("\n" + "="*50)
print("EMBEDDING COMPARISON")
print("="*50)
diff = np.abs(emb1 - emb2).max()
print(f"Max embedding difference: {diff:.6f}")
print("(Should be 0.0 if processing is identical)")
