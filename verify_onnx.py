"""Verify ONNX model produces consistent embeddings."""
import sys
sys.path.insert(0, 'training')
import numpy as np
import cv2
import torch
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import CardEmbeddingModel

# Load a test image
test_image_path = 'training/data/reference_images'
import os
files = os.listdir(test_image_path)
test_file = [f for f in files if f.endswith('.jpg')][0]
img = cv2.imread(os.path.join(test_image_path, test_file))
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Testing with: {test_file}")

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

transformed = transform(image=rgb)
tensor = transformed["image"].unsqueeze(0)

# PyTorch embedding
print("\nPyTorch inference:")
checkpoint = torch.load('training/checkpoints/final_model.pt', map_location='cpu', weights_only=False)
pytorch_model = CardEmbeddingModel(
    num_classes=checkpoint["num_classes"],
    embedding_dim=checkpoint["embedding_dim"],
    pretrained=False,
)
pytorch_model.load_state_dict(checkpoint["model_state_dict"])
pytorch_model.eval()

with torch.no_grad():
    pytorch_emb = pytorch_model.get_embedding(tensor).numpy()
print(f"  Shape: {pytorch_emb.shape}")
print(f"  First 5: {pytorch_emb[0, :5]}")
print(f"  Norm: {np.linalg.norm(pytorch_emb):.6f}")

# ONNX embedding
print("\nONNX inference:")
session = ort.InferenceSession('training/output/card_embedding_model.onnx')
onnx_emb = session.run(None, {'image': tensor.numpy()})[0]
print(f"  Shape: {onnx_emb.shape}")
print(f"  First 5: {onnx_emb[0, :5]}")
print(f"  Norm: {np.linalg.norm(onnx_emb):.6f}")

# Compare
diff = np.abs(pytorch_emb - onnx_emb)
print(f"\nMax difference: {diff.max():.10f}")
print(f"Mean difference: {diff.mean():.10f}")

if diff.max() < 1e-5:
    print("\nSUCCESS: PyTorch and ONNX produce identical embeddings!")
else:
    print("\nWARNING: Some difference detected between PyTorch and ONNX")
