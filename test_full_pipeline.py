"""Simple test of YOLO detection + card identification."""
import sys
import time
import cv2
import numpy as np
import torch

# Add paths
sys.path.insert(0, 'training')
sys.path.insert(0, 'spike')

print("Loading YOLO model...")
from ultralytics import YOLO
yolo = YOLO('training/yolo/runs/detect/train/weights/best.pt')
print("YOLO loaded!")

print("Loading embedding model...")
from model import CardEmbeddingModel
import json
import faiss
import albumentations as A
from albumentations.pytorch import ToTensorV2

checkpoint = torch.load('training/checkpoints/final_model.pt', map_location='cpu', weights_only=False)
embed_model = CardEmbeddingModel(
    num_classes=checkpoint["num_classes"],
    embedding_dim=checkpoint["embedding_dim"],
    pretrained=False,
)
embed_model.load_state_dict(checkpoint["model_state_dict"])
embed_model.eval()
print("Embedding model loaded!")

print("Loading FAISS index...")
index = faiss.read_index('training/output/card_embeddings.faiss')
print(f"Index loaded ({index.ntotal} cards)")

print("Loading card names...")
with open('training/output/label_mapping.json', 'r') as f:
    mapping = json.load(f)
idx_to_name = mapping["idx_to_name"]
card_names = [""] * (max(int(k) for k in idx_to_name.keys()) + 1)
for idx, name in idx_to_name.items():
    card_names[int(idx)] = name
print(f"Loaded {len(card_names)} card names")

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

print("\nOpening webcam...")
cap = cv2.VideoCapture(1)  # Use external webcam (change to 0 for built-in)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    sys.exit(1)

print("Webcam opened! Press 'q' to quit.")
cv2.namedWindow('Card Detection + ID', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    # YOLO detection
    results = yolo(frame, conf=0.6, verbose=False)[0]

    # Process each detection
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Extract card region
        card_img = frame[y1:y2, x1:x2]
        if card_img.size == 0:
            continue

        # Get embedding
        rgb = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
        transformed = transform(image=rgb)
        tensor = transformed["image"].unsqueeze(0)

        with torch.no_grad():
            embedding = embed_model.get_embedding(tensor).numpy()

        # Normalize and search
        faiss.normalize_L2(embedding)
        distances, indices = index.search(embedding, 1)

        if indices[0][0] < len(card_names):
            card_name = card_names[indices[0][0]]
            similarity = float(distances[0][0])

            # Draw label
            label = f"{card_name[:30]} ({similarity:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS
    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Cards: {len(results.boxes)}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Card Detection + ID', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")
