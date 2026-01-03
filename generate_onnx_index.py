"""Generate FAISS index using ONNX model for cross-platform consistency."""
import os
import json
import numpy as np
import faiss
import onnxruntime as ort
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

print("Loading ONNX model...")
session = ort.InferenceSession("training/output/card_embedding_model.onnx")

print("Loading reference images...")
with open('training/data/reference_metadata.json', 'r') as f:
    metadata = json.load(f)

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

embeddings = []
card_names = []

print(f"Generating embeddings for {len(metadata)} cards...")
for card_name, card_info in tqdm(metadata.items()):
    img_path = os.path.join('training/data/reference_images', card_info['filename'])
    if not os.path.exists(img_path):
        continue

    # Load and preprocess
    img = cv2.imread(img_path)
    if img is None:
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=rgb)
    tensor = transformed["image"].numpy()  # Already numpy from ToTensorV2
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

    # Get ONNX embedding
    embedding = session.run(None, {'image': tensor})[0]

    embeddings.append(embedding[0])
    card_names.append(card_name)

print(f"Generated {len(embeddings)} embeddings")

# Build FAISS index
embeddings_np = np.array(embeddings).astype(np.float32)
faiss.normalize_L2(embeddings_np)

print("Building FAISS index...")
index = faiss.IndexFlatIP(embeddings_np.shape[1])
index.add(embeddings_np)

# Save
output_dir = 'training/output'
os.makedirs(output_dir, exist_ok=True)

faiss.write_index(index, f'{output_dir}/card_embeddings_onnx.faiss')
with open(f'{output_dir}/label_mapping_onnx.json', 'w') as f:
    json.dump({'card_names': card_names}, f)

print(f"Saved ONNX-based index: {index.ntotal} cards")
print(f"  - {output_dir}/card_embeddings_onnx.faiss")
print(f"  - {output_dir}/label_mapping_onnx.json")
