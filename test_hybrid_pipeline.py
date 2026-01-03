"""
Hybrid Card Identification Pipeline

Combines:
1. OCR - Read card title text directly
2. Embedding similarity - Visual matching
3. Multi-frame voting - Require consistency before confirming

Confidence levels:
- HIGH: OCR matches embedding top-1
- MEDIUM: OCR matches embedding top-5
- LOW: Only embedding (no OCR match)
"""
import sys
import time
import cv2
import numpy as np
import faiss
import json
import easyocr
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
from difflib import SequenceMatcher

# --- Configuration ---
CAMERA_INDEX = 1  # 0 = front camera, 1 = rear camera
CONFIDENCE_THRESHOLD = 0.6
VOTE_FRAMES = 5  # Require this many consistent frames
MIN_OCR_SIMILARITY = 0.6  # Fuzzy match threshold for OCR
OCR_INTERVAL = 1.0  # Only run OCR every N seconds (saves CPU)
EMB_CONFIDENCE_FOR_OCR = 0.55  # Only run OCR if embedding below this

# Motion-based re-identification
MOVE_THRESHOLD = 50  # Pixels card must move to trigger re-identification
HIGH_CONFIDENCE = 0.65  # Once we hit this, lock in identification

# --- Load models ---
print("Loading YOLO model...")
from ultralytics import YOLO
yolo = YOLO('training/yolo/runs/detect/train/weights/best.pt')
print("YOLO loaded!")

print("Loading ONNX embedding model...")
session = ort.InferenceSession('training/output/card_embedding_model.onnx')
input_name = session.get_inputs()[0].name
print("ONNX model loaded!")

print("Loading FAISS index...")
index = faiss.read_index('training/output/card_embeddings_onnx.faiss')
print(f"Index loaded ({index.ntotal} cards)")

print("Loading card names...")
with open('training/output/label_mapping_onnx.json', 'r') as f:
    mapping = json.load(f)
card_names = mapping["card_names"]
card_names_lower = [c.lower() for c in card_names]
print(f"Loaded {len(card_names)} card names")

print("Loading EasyOCR (this may take a moment)...")
ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
print("EasyOCR loaded!")

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# --- Helper functions ---
def get_embedding(img):
    """Get embedding from card crop."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=rgb)
    tensor = transformed["image"].numpy()
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
    embedding = session.run(None, {input_name: tensor})[0]
    faiss.normalize_L2(embedding)
    return embedding

def get_embedding_matches(embedding, top_k=5):
    """Get top-k matches from embedding."""
    distances, indices = index.search(embedding, top_k)
    results = []
    for i in range(top_k):
        if indices[0][i] < len(card_names):
            results.append((card_names[indices[0][i]], float(distances[0][i])))
    return results

def extract_title_region(card_img):
    """Extract the title region from the top of the card."""
    h, w = card_img.shape[:2]
    # Title is roughly in the top 12-15% of the card
    title_region = card_img[int(h*0.02):int(h*0.12), int(w*0.08):int(w*0.85)]
    return title_region

def ocr_title(title_region):
    """Use OCR to read the card title."""
    if title_region.size == 0:
        return ""

    # Upscale for better OCR
    scale = 3
    upscaled = cv2.resize(title_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale and enhance
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # Try OCR
    results = ocr_reader.readtext(gray, detail=0, paragraph=True)

    if results:
        return ' '.join(results).strip()
    return ""

def fuzzy_match(ocr_text, card_name):
    """Fuzzy match OCR text to card name."""
    if not ocr_text:
        return 0.0
    return SequenceMatcher(None, ocr_text.lower(), card_name.lower()).ratio()

def find_best_ocr_match(ocr_text, top_k=10):
    """Find best matching card name for OCR text."""
    if not ocr_text or len(ocr_text) < 3:
        return None, 0.0

    ocr_lower = ocr_text.lower()
    best_match = None
    best_score = 0.0

    # First try exact substring match
    for name in card_names:
        if ocr_lower in name.lower() or name.lower() in ocr_lower:
            score = len(min(ocr_lower, name.lower())) / len(max(ocr_lower, name.lower()))
            if score > best_score:
                best_score = score
                best_match = name

    # If no good substring match, try fuzzy
    if best_score < MIN_OCR_SIMILARITY:
        for name in card_names:
            score = fuzzy_match(ocr_text, name)
            if score > best_score:
                best_score = score
                best_match = name

    return best_match, best_score

def combine_results(embedding_matches, ocr_match, ocr_score):
    """
    Combine embedding and OCR results into final prediction.

    Returns: (card_name, confidence, confidence_level)
    """
    emb_top1 = embedding_matches[0] if embedding_matches else (None, 0.0)
    emb_names = [m[0] for m in embedding_matches]

    # Case 1: OCR matches embedding top-1 -> HIGH confidence
    if ocr_match and ocr_match == emb_top1[0]:
        return ocr_match, min(1.0, emb_top1[1] + ocr_score * 0.3), "HIGH"

    # Case 2: OCR matches something in embedding top-5 -> MEDIUM confidence
    if ocr_match and ocr_match in emb_names:
        idx = emb_names.index(ocr_match)
        emb_score = embedding_matches[idx][1]
        return ocr_match, min(1.0, emb_score + ocr_score * 0.2), "MEDIUM"

    # Case 3: OCR has high confidence but doesn't match embedding
    if ocr_match and ocr_score > 0.8:
        return ocr_match, ocr_score * 0.9, "OCR-ONLY"

    # Case 4: Only embedding available
    if emb_top1[0]:
        return emb_top1[0], emb_top1[1], "EMB-ONLY"

    return None, 0.0, "NONE"

# --- Main loop ---
print("\n" + "="*60)
print("HYBRID CARD IDENTIFICATION PIPELINE")
print("="*60)
print(f"Multi-frame voting: {VOTE_FRAMES} frames")
print("Press 'q' to quit")
print("="*60)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"ERROR: Cannot open camera {CAMERA_INDEX}")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam: {actual_w}x{actual_h}")

cv2.namedWindow('Hybrid Pipeline', cv2.WINDOW_NORMAL)

# Voting state
vote_history = []  # List of (card_name, confidence, level)
confirmed_card = None
confirmed_confidence = 0.0
confirmed_level = ""

frame_count = 0
last_ocr_time = 0
cached_ocr_result = ("", 0.0)

# Locked identification state (motion-based)
locked_card = None  # Card name once locked
locked_confidence = 0.0
locked_level = ""
locked_box = None  # (x1, y1, x2, y2) when locked
is_locked = False

def box_center(box):
    """Get center point of a box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def box_moved(box1, box2, threshold):
    """Check if box moved more than threshold pixels."""
    if box1 is None or box2 is None:
        return True
    c1 = box_center(box1)
    c2 = box_center(box2)
    dist = ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) ** 0.5
    return dist > threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    start = time.time()
    display = frame.copy()

    # YOLO detection
    results = yolo(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    if len(results.boxes) > 0:
        box = results.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        current_box = (x1, y1, x2, y2)

        # Draw detection box
        box_color = (0, 255, 0) if not is_locked else (255, 200, 0)  # Yellow when locked
        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

        card_img = frame[y1:y2, x1:x2]
        if card_img.size > 0:
            # Check if we should use locked identification or re-identify
            card_moved = box_moved(locked_box, current_box, MOVE_THRESHOLD)

            if is_locked and not card_moved:
                # Card hasn't moved - use locked identification (FAST PATH)
                confirmed_card = locked_card
                confirmed_confidence = locked_confidence
                confirmed_level = locked_level + " [LOCKED]"
                emb_top1_conf = locked_confidence  # For display

                # Display locked status
                cv2.putText(display, "LOCKED - card stable", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            else:
                # Card moved or not locked - run identification
                if is_locked and card_moved:
                    # Unlock due to movement - reset everything including OCR cache
                    is_locked = False
                    locked_card = None
                    vote_history = []
                    cached_ocr_result = ("", 0.0)  # Reset OCR cache!
                    last_ocr_time = 0  # Force OCR to run again
                    cv2.putText(display, "UNLOCKED - card moved", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

                # Get embedding matches
                embedding = get_embedding(card_img)
                embedding_matches = get_embedding_matches(embedding, top_k=5)

                # OCR (only when embedding uncertain AND enough time passed)
                current_time = time.time()
                emb_top1_conf = embedding_matches[0][1] if embedding_matches else 0

                # Run OCR more aggressively when not confirmed yet
                ocr_interval = 0.5 if confirmed_card is None else OCR_INTERVAL

                # Only run OCR if embedding is uncertain
                if emb_top1_conf < EMB_CONFIDENCE_FOR_OCR and current_time - last_ocr_time > ocr_interval:
                    title_region = extract_title_region(card_img)
                    ocr_text = ocr_title(title_region)
                    ocr_match, ocr_score = find_best_ocr_match(ocr_text)
                    cached_ocr_result = (ocr_match, ocr_score)
                    last_ocr_time = current_time
                else:
                    ocr_match, ocr_score = cached_ocr_result

                # Combine results
                card_name, confidence, level = combine_results(
                    embedding_matches, ocr_match, ocr_score
                )

                # Add to voting history
                if card_name:
                    vote_history.append((card_name, confidence, level))
                    if len(vote_history) > VOTE_FRAMES * 2:
                        vote_history = vote_history[-VOTE_FRAMES * 2:]

                # Check for consensus
                if len(vote_history) >= VOTE_FRAMES:
                    recent_votes = [v[0] for v in vote_history[-VOTE_FRAMES:]]
                    vote_counts = Counter(recent_votes)
                    most_common, count = vote_counts.most_common(1)[0]

                    if count >= VOTE_FRAMES - 1:  # Allow 1 outlier
                        confirmed_card = most_common
                        # Get average confidence for confirmed card
                        conf_votes = [v[1] for v in vote_history[-VOTE_FRAMES:] if v[0] == most_common]
                        confirmed_confidence = sum(conf_votes) / len(conf_votes)
                        confirmed_level = vote_history[-1][2]

                        # Lock if confidence is high enough
                        if confirmed_confidence >= HIGH_CONFIDENCE and not is_locked:
                            is_locked = True
                            locked_card = confirmed_card
                            locked_confidence = confirmed_confidence
                            locked_level = confirmed_level
                            locked_box = current_box

                # Display current prediction
                if embedding_matches:
                    emb_text = f"Emb: {embedding_matches[0][0][:25]} ({embedding_matches[0][1]:.2f})"
                    cv2.putText(display, emb_text, (x1, y2 + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

                if ocr_match:
                    ocr_text_display = f"OCR: {ocr_match[:25]} ({ocr_score:.2f})"
                    cv2.putText(display, ocr_text_display, (x1, y2 + 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

                # Show OCR status
                if not is_locked:
                    if emb_top1_conf < EMB_CONFIDENCE_FOR_OCR:
                        cv2.putText(display, "OCR: ACTIVE", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.putText(display, "IDENTIFYING...", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Display confirmed result
            if confirmed_card:
                level_colors = {
                    "HIGH": (0, 255, 0),
                    "HIGH [LOCKED]": (255, 200, 0),
                    "MEDIUM": (0, 255, 255),
                    "MEDIUM [LOCKED]": (255, 200, 0),
                    "OCR-ONLY": (255, 255, 0),
                    "OCR-ONLY [LOCKED]": (255, 200, 0),
                    "EMB-ONLY": (0, 165, 255),
                    "EMB-ONLY [LOCKED]": (255, 200, 0),
                }
                color = level_colors.get(confirmed_level, (255, 255, 255))

                label = f"{confirmed_card[:35]}"
                cv2.putText(display, label, (x1, y1 - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, f"{confirmed_level} ({confirmed_confidence:.2f})",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        # No detection - reset everything
        vote_history = []
        confirmed_card = None
        is_locked = False
        locked_card = None
        locked_box = None
        cached_ocr_result = ("", 0.0)  # Reset OCR cache
        last_ocr_time = 0

    # FPS display
    fps = 1.0 / (time.time() - start)
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Legend
    cv2.putText(display, "HIGH=OCR+Emb match | MEDIUM=OCR in top5 | OCR-ONLY | EMB-ONLY",
               (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Hybrid Pipeline', display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")
