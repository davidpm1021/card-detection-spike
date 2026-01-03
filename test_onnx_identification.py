"""Test ONNX-based card identification on reference images."""
import os
import json
import numpy as np
import cv2
import faiss
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

print("Loading ONNX embedding model...")
session = ort.InferenceSession('training/output/card_embedding_model.onnx')

print("Loading ONNX-based FAISS index...")
index = faiss.read_index('training/output/card_embeddings_onnx.faiss')
print(f"Index: {index.ntotal} cards")

print("Loading card names...")
with open('training/output/label_mapping_onnx.json', 'r') as f:
    mapping = json.load(f)
card_names = mapping["card_names"]

print("Loading reference metadata...")
with open('training/data/reference_metadata.json', 'r') as f:
    metadata = json.load(f)

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Test on 20 random cards
test_cards = random.sample(list(metadata.items()), 20)
correct = 0

print("\nTesting identification on 20 random reference images:\n")
for card_name, card_info in test_cards:
    img_path = os.path.join('training/data/reference_images', card_info['filename'])
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=rgb)
    tensor = np.expand_dims(transformed["image"].numpy(), axis=0).astype(np.float32)

    # Get embedding
    embedding = session.run(None, {'image': tensor})[0]
    faiss.normalize_L2(embedding)

    # Search
    distances, indices = index.search(embedding, 3)
    predicted = card_names[indices[0][0]]
    similarity = distances[0][0]

    match = "OK" if predicted == card_name else "MISS"
    if predicted == card_name:
        correct += 1

    print(f"[{match}] {card_name[:35]:35s} -> {predicted[:35]:35s} ({similarity:.3f})")

print(f"\nAccuracy: {correct}/20 = {100*correct/20:.0f}%")
print("\nNote: Reference images should match perfectly (100%)")
