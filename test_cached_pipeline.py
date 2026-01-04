"""
Cached Card Identification Pipeline

Strategy:
1. Detect card (YOLO - fast, every frame)
2. Compute perceptual hash of card crop
3. Check cache - if seen before, return cached result instantly
4. If new card:
   a. Capture and enhance the image
   b. Run full identification (embedding + OCR)
   c. Cache result with phash key

This means expensive ID only runs ONCE per unique card!
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
from collections import OrderedDict
from difflib import SequenceMatcher

# --- Configuration ---
CAMERA_INDEX = 1  # 0 = front camera, 1 = rear camera
CONFIDENCE_THRESHOLD = 0.6
CACHE_SIZE = 50  # Max cards to remember
PHASH_THRESHOLD = 10  # Max hamming distance to consider same card
STABLE_FRAMES = 10  # Frames card must be stable before identifying
MIN_CACHE_CONFIDENCE = 0.60  # Only cache results above this confidence

# --- Perceptual Hash ---
def compute_phash(img, hash_size=16):
    """Compute perceptual hash of image."""
    # Resize to hash_size x hash_size
    resized = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Compute DCT
    dct = cv2.dct(np.float32(gray))
    dct_low = dct[:8, :8]  # Keep low frequencies

    # Compute median and create hash
    median = np.median(dct_low)
    hash_bits = (dct_low > median).flatten()

    # Convert to integer
    hash_val = 0
    for bit in hash_bits:
        hash_val = (hash_val << 1) | int(bit)
    return hash_val

def hamming_distance(hash1, hash2):
    """Compute hamming distance between two hashes."""
    xor = hash1 ^ hash2
    return bin(xor).count('1')

# --- Image Enhancement ---
def enhance_card_image(img):
    """Clean up card image for better identification."""
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE on luminance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Slight denoise
    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

    return enhanced

# --- LRU Cache ---
class CardCache:
    def __init__(self, max_size=50):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, phash):
        """Find cached result by phash, allowing some distance."""
        for cached_hash, result in self.cache.items():
            if hamming_distance(phash, cached_hash) <= PHASH_THRESHOLD:
                # Move to end (most recent)
                self.cache.move_to_end(cached_hash)
                return result
        return None

    def put(self, phash, result):
        """Cache a result."""
        self.cache[phash] = result
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove oldest

    def clear(self):
        self.cache.clear()

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
print(f"Loaded {len(card_names)} card names")

print("Loading EasyOCR...")
ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
print("EasyOCR loaded!")

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# --- Helper functions ---
def get_embedding(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=rgb)
    tensor = transformed["image"].numpy()
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
    embedding = session.run(None, {input_name: tensor})[0]
    faiss.normalize_L2(embedding)
    return embedding

def get_embedding_matches(embedding, top_k=5):
    distances, indices = index.search(embedding, top_k)
    results = []
    for i in range(top_k):
        if indices[0][i] < len(card_names):
            results.append((card_names[indices[0][i]], float(distances[0][i])))
    return results

def extract_title_region(card_img):
    h, w = card_img.shape[:2]
    title_region = card_img[int(h*0.02):int(h*0.12), int(w*0.08):int(w*0.85)]
    return title_region

def ocr_title(title_region):
    if title_region.size == 0:
        return ""
    scale = 3
    upscaled = cv2.resize(title_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    results = ocr_reader.readtext(gray, detail=0, paragraph=True)
    if results:
        return ' '.join(results).strip()
    return ""

def fuzzy_match(ocr_text, card_name):
    if not ocr_text:
        return 0.0
    return SequenceMatcher(None, ocr_text.lower(), card_name.lower()).ratio()

def find_best_ocr_match(ocr_text):
    if not ocr_text or len(ocr_text) < 3:
        return None, 0.0

    ocr_lower = ocr_text.lower().strip()
    best_match = None
    best_score = 0.0

    # First try substring matching
    for name in card_names:
        name_lower = name.lower()
        if ocr_lower in name_lower or name_lower in ocr_lower:
            # Score based on length similarity (fixed bug: was using min/max on strings!)
            score = min(len(ocr_lower), len(name_lower)) / max(len(ocr_lower), len(name_lower))
            if score > best_score:
                best_score = score
                best_match = name

    # If no good substring match, try fuzzy
    if best_score < 0.6:
        for name in card_names:
            score = fuzzy_match(ocr_text, name)
            if score > best_score:
                best_score = score
                best_match = name

    # Cap score at 1.0
    return best_match, min(best_score, 1.0)

def identify_card_full(card_img):
    """Run full identification pipeline on enhanced image."""
    # Enhance image
    enhanced = enhance_card_image(card_img)

    # Get embedding
    embedding = get_embedding(enhanced)
    emb_matches = get_embedding_matches(embedding, top_k=5)
    emb_top = emb_matches[0] if emb_matches else (None, 0.0)
    emb_names = [m[0] for m in emb_matches]

    # Get OCR
    title_region = extract_title_region(enhanced)
    ocr_text = ocr_title(title_region)
    ocr_match, ocr_score = find_best_ocr_match(ocr_text)

    # Combine results - embedding preferred when confident
    # Case 1: Both agree - highest confidence
    if ocr_match and ocr_match == emb_top[0]:
        combined_conf = min(1.0, emb_top[1] + 0.15)
        return ocr_match, combined_conf, "HIGH", emb_matches, ocr_text, ocr_score

    # Case 2: Embedding is very confident (>0.65) - trust it
    if emb_top[0] and emb_top[1] >= 0.65:
        return emb_top[0], emb_top[1], "EMB-CONF", emb_matches, ocr_text, ocr_score

    # Case 3: OCR matches something in embedding top-5 - trust OCR
    if ocr_match and ocr_match in emb_names and ocr_score >= 0.6:
        idx = emb_names.index(ocr_match)
        combined_conf = min(1.0, emb_matches[idx][1] + ocr_score * 0.2)
        return ocr_match, combined_conf, "OCR+EMB", emb_matches, ocr_text, ocr_score

    # Case 4: Strong OCR (>0.75) and embedding is weak (<0.55) - trust OCR
    if ocr_match and ocr_score >= 0.75 and emb_top[1] < 0.55:
        return ocr_match, ocr_score, "OCR", emb_matches, ocr_text, ocr_score

    # Case 5: Decent embedding (>0.50) - use it
    if emb_top[0] and emb_top[1] >= 0.50:
        return emb_top[0], emb_top[1], "EMB", emb_matches, ocr_text, ocr_score

    # Case 6: Low confidence - return best guess
    if emb_top[1] >= ocr_score:
        return emb_top[0] or "Unknown", emb_top[1], "LOW", emb_matches, ocr_text, ocr_score
    else:
        return ocr_match or "Unknown", ocr_score, "LOW", emb_matches, ocr_text, ocr_score

# --- Main loop ---
print("\n" + "="*60)
print("CACHED CARD IDENTIFICATION PIPELINE")
print("="*60)
print(f"Cache size: {CACHE_SIZE} cards")
print(f"Stable frames needed: {STABLE_FRAMES}")
print("Press 'c' to clear cache, 'q' to quit")
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

cv2.namedWindow('Cached Pipeline', cv2.WINDOW_NORMAL)

# State
cache = CardCache(CACHE_SIZE)
stable_count = 0
last_phash = None
current_result = None
identifying = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    display = frame.copy()

    # YOLO detection
    results = yolo(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    if len(results.boxes) > 0:
        box = results.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        card_img = frame[y1:y2, x1:x2]
        if card_img.size > 0:
            # Compute phash
            phash = compute_phash(card_img)

            # Check if card is stable (same phash)
            if last_phash is not None and hamming_distance(phash, last_phash) <= PHASH_THRESHOLD:
                stable_count += 1
            else:
                stable_count = 1
                current_result = None
            last_phash = phash

            # Check cache first
            cached = cache.get(phash)

            if cached:
                # Cache hit!
                current_result = cached
                box_color = (0, 255, 0)  # Green - cached
                status = f"CACHED ({stable_count} frames)"
            elif stable_count >= STABLE_FRAMES and current_result is None:
                # Card is stable and not identified yet - run full ID
                identifying = True
                cv2.putText(display, "IDENTIFYING...", (x1, y1 - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow('Cached Pipeline', display)
                cv2.waitKey(1)

                # Run full identification
                name, conf, level, emb_matches, ocr_text, ocr_score = identify_card_full(card_img)
                current_result = {
                    'name': name,
                    'confidence': conf,
                    'level': level,
                    'emb_top': emb_matches[0] if emb_matches else None,
                    'ocr_text': ocr_text,
                    'ocr_score': ocr_score
                }

                # Only cache high-confidence results
                if conf >= MIN_CACHE_CONFIDENCE:
                    cache.put(phash, current_result)
                    status = "NEW - cached"
                else:
                    status = f"NEW - low conf ({conf:.2f})"
                identifying = False
                box_color = (255, 200, 0)  # Yellow - just identified
            else:
                # Waiting for stability
                box_color = (200, 200, 200)  # Gray - waiting
                status = f"Stabilizing... ({stable_count}/{STABLE_FRAMES})"

            # Draw box
            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

            # Display result
            if current_result:
                name = current_result['name']
                conf = current_result['confidence']
                level = current_result['level']

                label = f"{name[:35]}"
                cv2.putText(display, label, (x1, y1 - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                cv2.putText(display, f"{level} ({conf:.2f})",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                # Show details below card
                if current_result.get('emb_top'):
                    emb_text = f"Emb: {current_result['emb_top'][0][:20]} ({current_result['emb_top'][1]:.2f})"
                    cv2.putText(display, emb_text, (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                if current_result.get('ocr_text'):
                    ocr_display = f"OCR: '{current_result['ocr_text'][:20]}' ({current_result['ocr_score']:.2f})"
                    cv2.putText(display, ocr_display, (x1, y2 + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

            # Status
            cv2.putText(display, status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    else:
        # No card - reset
        stable_count = 0
        last_phash = None
        current_result = None

    # FPS and cache info
    fps = 1.0 / (time.time() - start)
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display, f"Cache: {len(cache.cache)} cards", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Cached Pipeline', display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        cache.clear()
        print("Cache cleared!")

cap.release()
cv2.destroyAllWindows()
print("Done!")
