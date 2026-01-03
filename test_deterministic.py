"""Deterministic test - compare exact values across machines."""
import sys
sys.path.insert(0, 'training')

import torch
import numpy as np
from model import CardEmbeddingModel

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load model
checkpoint = torch.load('training/checkpoints/final_model.pt', map_location='cpu', weights_only=False)
model = CardEmbeddingModel(
    num_classes=checkpoint["num_classes"],
    embedding_dim=checkpoint["embedding_dim"],
    pretrained=False,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Create deterministic input (same on all machines)
torch.manual_seed(42)
test_input = torch.randn(1, 3, 224, 224)

print("=" * 50)
print("DETERMINISTIC TEST")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"Input checksum: {test_input.sum().item():.6f}")
print(f"Input first 5: {test_input[0, 0, 0, :5].tolist()}")

with torch.no_grad():
    embedding = model.get_embedding(test_input).numpy()

print(f"\nEmbedding shape: {embedding.shape}")
print(f"Embedding norm: {np.linalg.norm(embedding):.6f}")
print(f"Embedding sum: {embedding.sum():.6f}")
print(f"Embedding first 10: {embedding[0, :10].tolist()}")
print(f"Embedding last 5: {embedding[0, -5:].tolist()}")

# Also test with the actual debug image
print("\n" + "=" * 50)
print("DEBUG IMAGE TEST")
print("=" * 50)
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

img = cv2.imread('debug_card_crop.jpg')
if img is not None:
    print(f"Image shape: {img.shape}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    transformed = transform(image=rgb)
    tensor = transformed["image"].unsqueeze(0)

    print(f"Tensor sum: {tensor.sum().item():.6f}")

    with torch.no_grad():
        emb = model.get_embedding(tensor).numpy()

    print(f"Embedding sum: {emb.sum():.6f}")
    print(f"Embedding first 10: {emb[0, :10].tolist()}")
else:
    print("debug_card_crop.jpg not found")
