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
index = faiss.read_index('training/output/card_embeddings_full.faiss')
print(f"Index loaded ({index.ntotal} cards)")

print("Loading card names...")
with open('training/output/label_mapping_full.json', 'r') as f:
    mapping = json.load(f)
card_names = mapping["card_names"]
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

# Set higher resolution (try 1080p, fall back to 720p)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam resolution: {actual_w}x{actual_h}")

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

        # Debug: print crop size
        h, w = card_img.shape[:2]
        print(f"Crop size: {w}x{h}")

        # Get embedding using original crop (not resized)
        rgb = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
        transformed = transform(image=rgb)
        tensor = transformed["image"].unsqueeze(0)

        with torch.no_grad():
            embedding = embed_model.get_embedding(tensor).numpy()

        # Debug: check embedding norm
        norm_before = np.linalg.norm(embedding)

        # Normalize and search
        faiss.normalize_L2(embedding)
        distances, indices = index.search(embedding, 5)  # Get top 5
        print(f"Embedding norm before/after: {norm_before:.3f}/1.000")

        if indices[0][0] < len(card_names):
            card_name = card_names[indices[0][0]]
            similarity = float(distances[0][0])

            # Print top 3 matches for debugging
            print(f"Top 3: {card_names[indices[0][0]]} ({distances[0][0]:.2f}), {card_names[indices[0][1]]} ({distances[0][1]:.2f}), {card_names[indices[0][2]]} ({distances[0][2]:.2f})")

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

    # Single waitKey for all key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and len(results.boxes) > 0:
        # Save the first detected card crop for debugging
        box = results.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        debug_crop = frame[y1:y2, x1:x2]
        cv2.imwrite('debug_card_crop.jpg', debug_crop)
        print(f"Saved debug_card_crop.jpg ({debug_crop.shape})")

cap.release()
cv2.destroyAllWindows()
print("Done!")
