"""Test if CPU vs GPU produces different embeddings."""
import sys
sys.path.insert(0, 'training')

import torch
import numpy as np
import cv2
import faiss
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import CardEmbeddingModel

# Load model
checkpoint = torch.load('training/checkpoints/final_model.pt', map_location='cpu', weights_only=False)
model = CardEmbeddingModel(
    num_classes=checkpoint["num_classes"],
    embedding_dim=checkpoint["embedding_dim"],
    pretrained=False,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load index and names
index = faiss.read_index('training/output/card_embeddings_full.faiss')
with open('training/output/label_mapping_full.json', 'r') as f:
    card_names = json.load(f)["card_names"]

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Load image
img = cv2.imread('debug_card_crop.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
transformed = transform(image=rgb)
tensor = transformed["image"].unsqueeze(0)

print("=" * 50)
print("CPU vs GPU Embedding Test")
print("=" * 50)

# Test CPU
model_cpu = model.cpu()
tensor_cpu = tensor.cpu()
with torch.no_grad():
    emb_cpu = model_cpu.get_embedding(tensor_cpu).numpy()
faiss.normalize_L2(emb_cpu)
distances_cpu, indices_cpu = index.search(emb_cpu, 3)
print(f"\nCPU - Top 3:")
for i in range(3):
    print(f"  {card_names[indices_cpu[0][i]]} ({distances_cpu[0][i]:.3f})")
print(f"CPU embedding first 5: {emb_cpu[0, :5]}")

# Test GPU if available
if torch.cuda.is_available():
    model_gpu = model.cuda()
    tensor_gpu = tensor.cuda()
    with torch.no_grad():
        emb_gpu = model_gpu.get_embedding(tensor_gpu).cpu().numpy()
    faiss.normalize_L2(emb_gpu)
    distances_gpu, indices_gpu = index.search(emb_gpu, 3)
    print(f"\nGPU - Top 3:")
    for i in range(3):
        print(f"  {card_names[indices_gpu[0][i]]} ({distances_gpu[0][i]:.3f})")
    print(f"GPU embedding first 5: {emb_gpu[0, :5]}")

    # Compare
    diff = np.abs(emb_cpu - emb_gpu).max()
    print(f"\nMax difference between CPU and GPU embeddings: {diff:.6f}")
else:
    print("\nNo GPU available for comparison")
