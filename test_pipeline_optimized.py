"""Optimized YOLO detection + card identification with tracking and frame skipping."""
import sys
import time
import cv2
import numpy as np
import faiss
import json
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataclasses import dataclass
from typing import Optional

# --- Configuration ---
EMBEDDING_INTERVAL = 15  # Re-identify tracked cards every N frames
IOU_THRESHOLD = 0.3      # Min overlap to consider same card
CONFIDENCE_THRESHOLD = 0.6

@dataclass
class TrackedCard:
    """A card being tracked across frames."""
    box: tuple  # (x1, y1, x2, y2)
    name: str
    similarity: float
    frames_since_embedding: int = 0

def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute intersection over union of two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def match_detections_to_tracks(detections: list, tracks: list) -> dict:
    """Match current detections to existing tracks by IOU."""
    matches = {}  # detection_idx -> track_idx
    used_tracks = set()

    for det_idx, det_box in enumerate(detections):
        best_iou = IOU_THRESHOLD
        best_track_idx = None

        for track_idx, track in enumerate(tracks):
            if track_idx in used_tracks:
                continue
            iou = compute_iou(det_box, track.box)
            if iou > best_iou:
                best_iou = iou
                best_track_idx = track_idx

        if best_track_idx is not None:
            matches[det_idx] = best_track_idx
            used_tracks.add(best_track_idx)

    return matches

# --- Load models ---
print("Loading YOLO model...")
from ultralytics import YOLO
yolo = YOLO('training/yolo/runs/detect/train/weights/best.pt')
print("YOLO loaded!")

print("Loading ONNX embedding model...")
session = ort.InferenceSession('training/output/card_embedding_model.onnx')
input_name = session.get_inputs()[0].name
print("ONNX model loaded!")

print("Loading ONNX-based FAISS index...")
index = faiss.read_index('training/output/card_embeddings_onnx.faiss')
print(f"Index loaded ({index.ntotal} cards)")

print("Loading card names...")
with open('training/output/label_mapping_onnx.json', 'r') as f:
    mapping = json.load(f)
card_names = mapping["card_names"]
print(f"Loaded {len(card_names)} card names")

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def get_embedding_batch(images: list) -> np.ndarray:
    """Get embeddings for a batch of images."""
    if not images:
        return np.array([])

    tensors = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = transform(image=rgb)
        tensors.append(transformed["image"].numpy())

    batch = np.stack(tensors, axis=0)
    embeddings = session.run(None, {input_name: batch})[0]
    faiss.normalize_L2(embeddings)
    return embeddings

def identify_cards(embeddings: np.ndarray) -> list:
    """Identify cards from embeddings."""
    if len(embeddings) == 0:
        return []

    distances, indices = index.search(embeddings, 1)
    results = []
    for i in range(len(embeddings)):
        if indices[i][0] < len(card_names):
            results.append((card_names[indices[i][0]], float(distances[i][0])))
        else:
            results.append(("Unknown", 0.0))
    return results

# --- Main loop ---
print("\nOpening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam resolution: {actual_w}x{actual_h}")

print("Webcam opened! Press 'q' to quit.")
print(f"Tracking config: IOU={IOU_THRESHOLD}, re-embed every {EMBEDDING_INTERVAL} frames")
cv2.namedWindow('Card Detection (Optimized)', cv2.WINDOW_NORMAL)

tracked_cards: list[TrackedCard] = []
frame_count = 0
embeddings_this_frame = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    frame_count += 1
    embeddings_this_frame = 0

    # YOLO detection (fast, every frame)
    results = yolo(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    # Extract detection boxes
    current_boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        current_boxes.append((x1, y1, x2, y2))

    # Match detections to existing tracks
    matches = match_detections_to_tracks(current_boxes, tracked_cards)

    # Update matched tracks and collect cards needing embedding
    new_tracks = []
    cards_needing_embedding = []  # (detection_idx, box, crop)

    for det_idx, box in enumerate(current_boxes):
        x1, y1, x2, y2 = box
        card_img = frame[y1:y2, x1:x2]
        if card_img.size == 0:
            continue

        if det_idx in matches:
            # Existing track - update position
            track_idx = matches[det_idx]
            track = tracked_cards[track_idx]
            track.box = box
            track.frames_since_embedding += 1

            # Re-embed if stale
            if track.frames_since_embedding >= EMBEDDING_INTERVAL:
                cards_needing_embedding.append((len(new_tracks), box, card_img))
                track.frames_since_embedding = 0

            new_tracks.append(track)
        else:
            # New card - needs embedding
            placeholder = TrackedCard(box=box, name="...", similarity=0.0)
            cards_needing_embedding.append((len(new_tracks), box, card_img))
            new_tracks.append(placeholder)

    # Batch embed all cards that need it
    if cards_needing_embedding:
        crops = [c[2] for c in cards_needing_embedding]
        embeddings = get_embedding_batch(crops)
        identifications = identify_cards(embeddings)
        embeddings_this_frame = len(crops)

        for i, (track_idx, box, _) in enumerate(cards_needing_embedding):
            name, sim = identifications[i]
            new_tracks[track_idx].name = name
            new_tracks[track_idx].similarity = sim
            new_tracks[track_idx].frames_since_embedding = 0

    tracked_cards = new_tracks

    # Draw results
    for track in tracked_cards:
        x1, y1, x2, y2 = track.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{track.name[:30]} ({track.similarity:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Stats
    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Cards: {len(tracked_cards)}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Embeds: {embeddings_this_frame}", (10, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Card Detection (Optimized)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")
