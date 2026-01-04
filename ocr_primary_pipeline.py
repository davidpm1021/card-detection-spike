"""
OCR-Primary Card Identification Pipeline

The card name is PRINTED ON THE CARD. OCR should be primary, embeddings secondary.

Key improvements over hybrid_pipeline:
1. OCR ALWAYS runs (not conditional on embedding confidence)
2. Better title region extraction (excludes mana cost symbols)
3. Proper OCR preprocessing (CLAHE, adaptive threshold, morphology)
4. Fast fuzzy matching with RapidFuzz
5. Embedding used only for verification/fallback
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
from rapidfuzz import fuzz, process
from collections import Counter

# --- Configuration ---
CAMERA_INDEX = 1
CONFIDENCE_THRESHOLD = 0.6
VOTE_FRAMES = 3  # Fewer frames needed with better OCR
MIN_OCR_SIMILARITY = 0.65
MOVE_THRESHOLD = 50
OCR_INTERVAL = 0.5  # Only run OCR every 0.5 seconds (it's slow!)

# --- Load models ---
print("Loading YOLO model...")
from ultralytics import YOLO
yolo = YOLO('training/yolo/runs/detect/train/weights/best.pt')
print("YOLO loaded!")

print("Loading ONNX embedding model...")
session = ort.InferenceSession('training/output/card_embedding_model.onnx')
input_name = session.get_inputs()[0].name
print("ONNX model loaded!")

print("Loading FAISS index (augmented)...")
index = faiss.read_index('training/output/card_embeddings_aug10_mean.faiss')
print(f"Index loaded ({index.ntotal} cards)")

print("Loading card names...")
with open('training/output/label_mapping_aug10_mean.json', 'r') as f:
    mapping = json.load(f)
card_names = mapping["card_names"]
card_names_set = set(card_names)
print(f"Loaded {len(card_names)} card names")

print("Loading EasyOCR...")
ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
print("EasyOCR loaded!")

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# --- Improved title extraction ---
def extract_title_region(card_img):
    """
    Extract title region from MTG card.

    MTG card layout:
    - Title bar is 5-10% from top
    - Title text is left-aligned
    - Mana cost is right-aligned (EXCLUDE THIS)
    """
    h, w = card_img.shape[:2]
    # Top 3-11% of height, left 5-72% of width (exclude mana cost on right)
    y1, y2 = int(h * 0.03), int(h * 0.11)
    x1, x2 = int(w * 0.05), int(w * 0.72)
    return card_img[y1:y2, x1:x2]


def preprocess_for_ocr(img):
    """
    Preprocess image for optimal OCR accuracy.

    Steps:
    1. Upscale 4x for better character recognition
    2. Convert to grayscale
    3. Apply CLAHE for contrast enhancement
    4. Adaptive threshold for binarization
    5. Morphological cleanup
    """
    if img.size == 0:
        return None

    # Upscale
    scale = 4
    upscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Grayscale
    if len(upscaled.shape) == 3:
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    else:
        gray = upscaled

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive threshold - works better for varying lighting
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Morphological cleanup - remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned


def ocr_title(card_img, debug=False):
    """
    Extract card title using MULTI-STRATEGY OCR.

    Tries multiple combinations of:
    - Orientations: 0° and 180° (cards may be upside-down)
    - Preprocessing: grayscale, inverted, CLAHE, high contrast

    Returns: (ocr_text, confidence)
    """
    best_text = ""
    best_conf = 0.0

    # Try both orientations
    orientations = [
        card_img,
        cv2.rotate(card_img, cv2.ROTATE_180),
    ]

    for oriented in orientations:
        title_region = extract_title_region(oriented)
        if title_region.size == 0:
            continue

        # Upscale
        scale = 3
        upscaled = cv2.resize(title_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

        # Try multiple preprocessing strategies
        strategies = [
            gray,                                                  # Simple grayscale
            255 - gray,                                            # Inverted (dark title bars)
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray),  # CLAHE
            cv2.convertScaleAbs(gray, alpha=1.5, beta=0),          # High contrast
        ]

        for processed in strategies:
            results = ocr_reader.readtext(processed, detail=1)
            if results:
                texts = []
                total_conf = 0
                for bbox, text, conf in results:
                    if conf > 0.1:  # Lower threshold to catch more text
                        texts.append(text)
                        total_conf += conf

                if texts and total_conf / len(texts) > best_conf:
                    best_text = ' '.join(texts).strip()
                    best_conf = total_conf / len(texts)

    return best_text, best_conf


def find_best_match(ocr_text, embedding_matches=None, top_n=5):
    """
    Find best matching card name using RapidFuzz.

    Fast fuzzy matching with multiple strategies:
    1. Exact match
    2. Prefix match
    3. Token-based matching (handles word order)
    4. Fuzzy ratio

    Returns: (best_match, confidence)
    """
    if not ocr_text or len(ocr_text) < 2:
        return None, 0.0

    ocr_clean = ocr_text.strip().lower()

    # Strategy 1: Exact match
    for name in card_names:
        if name.lower() == ocr_clean:
            return name, 1.0

    # Strategy 2: Use RapidFuzz for fast fuzzy matching
    # process.extract returns [(match, score, index), ...]
    matches = process.extract(
        ocr_clean,
        card_names,
        scorer=fuzz.WRatio,  # Weighted ratio - handles partial matches well
        limit=top_n
    )

    if not matches:
        return None, 0.0

    best_match, best_score, _ = matches[0]

    # If embedding matches are available, boost cards that appear in both
    if embedding_matches:
        emb_names = [m[0] for m in embedding_matches]
        for match, score, _ in matches:
            if match in emb_names:
                # This card appears in both OCR and embedding results
                # Give it a significant boost
                emb_rank = emb_names.index(match)
                boost = 15 - (emb_rank * 3)  # Higher boost for higher embedding rank
                boosted_score = min(100, score + boost)
                if boosted_score > best_score:
                    best_match = match
                    best_score = boosted_score

    # Normalize to 0-1
    return best_match, best_score / 100.0


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


def identify_card_fast(card_img, cached_ocr):
    """
    Fast card identification using CACHED OCR results.

    OCR is slow, so we run it periodically and cache the result.
    Embedding is fast, so we run it every frame.

    Returns: (card_name, confidence, method)
    """
    ocr_text, ocr_conf = cached_ocr

    # Embedding is fast - run every frame
    embedding = get_embedding(card_img)
    emb_matches = get_embedding_matches(embedding, top_k=5)

    # Use cached OCR + fresh embedding
    emb_names = [m[0] for m in emb_matches]
    emb_top1 = emb_matches[0] if emb_matches else (None, 0)

    if ocr_text:
        best_match, match_score = find_best_match(ocr_text, emb_matches)

        if best_match and match_score >= MIN_OCR_SIMILARITY:
            # REQUIRE embedding verification to prevent false positives
            if best_match == emb_top1[0]:
                # Perfect: OCR and embedding agree on #1
                confidence = (match_score + emb_top1[1]) / 2 + 0.1
                return best_match, min(1.0, confidence), "OCR+EMB"
            elif best_match in emb_names:
                # Good: OCR is in embedding top-5
                confidence = match_score * 0.8  # Lower confidence since not #1
                return best_match, min(1.0, confidence), "OCR(verified)"
            else:
                # OCR not verified by embedding - DON'T trust it!
                # This prevents false positives like "Rite of Raging Storm"
                pass  # Fall through to embedding-only

    # Embedding only (OCR failed or not verified)
    if emb_matches and emb_top1[1] >= 0.4:
        return emb_top1[0], emb_top1[1], "EMB-ONLY"

    return None, 0.0, "NONE"


def identify_card(card_img):
    """
    Full card identification (slower - runs OCR).

    Flow:
    1. Run OCR on title region (ALWAYS)
    2. Get embedding matches (for verification)
    3. Combine: OCR has priority, embedding confirms

    Returns: (card_name, confidence, method)
    """
    # Step 1: OCR (primary)
    ocr_text, ocr_conf = ocr_title(card_img)

    # Step 2: Embedding (secondary)
    embedding = get_embedding(card_img)
    emb_matches = get_embedding_matches(embedding, top_k=5)

    # Step 3: Find best match using OCR + embedding verification
    if ocr_text:
        best_match, match_score = find_best_match(ocr_text, emb_matches)

        if best_match and match_score >= MIN_OCR_SIMILARITY:
            # Check if embedding confirms
            emb_names = [m[0] for m in emb_matches]
            emb_top1 = emb_matches[0] if emb_matches else (None, 0)

            if best_match == emb_top1[0]:
                # Perfect: OCR and embedding agree on #1
                confidence = (match_score + emb_top1[1]) / 2 + 0.1
                return best_match, min(1.0, confidence), "OCR+EMB"
            elif best_match in emb_names:
                # Good: OCR match is in embedding top-5
                confidence = match_score + 0.05
                return best_match, min(1.0, confidence), "OCR(emb-verified)"
            else:
                # OCR only - still trust it if score is high
                if match_score >= 0.75:
                    return best_match, match_score, "OCR-HIGH"
                elif match_score >= MIN_OCR_SIMILARITY:
                    return best_match, match_score * 0.9, "OCR-MED"

    # Fallback: embedding only
    if emb_matches and emb_matches[0][1] >= 0.5:
        return emb_matches[0][0], emb_matches[0][1], "EMB-ONLY"

    return None, 0.0, "NONE"


# --- Main loop ---
def main():
    print("\n" + "="*60)
    print("OCR-PRIMARY CARD IDENTIFICATION")
    print("="*60)
    print("Press 'q' to quit, 'd' for debug mode")
    print("="*60)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print(f"Webcam: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    cv2.namedWindow('OCR Primary', cv2.WINDOW_NORMAL)

    vote_history = []
    confirmed_card = None
    confirmed_conf = 0.0
    confirmed_method = ""
    locked = False
    locked_box = None
    debug_mode = False

    # OCR throttling - it's slow so only run periodically
    last_ocr_time = 0
    cached_ocr = ("", 0.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        display = frame.copy()

        # YOLO detection (fast)
        results = yolo(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        if len(results.boxes) > 0:
            box = results.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            current_box = (x1, y1, x2, y2)

            # Draw box
            color = (255, 200, 0) if locked else (0, 255, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            card_img = frame[y1:y2, x1:x2]

            if card_img.size > 0:
                # Check for movement
                if locked and locked_box:
                    cx1, cy1 = (locked_box[0] + locked_box[2]) // 2, (locked_box[1] + locked_box[3]) // 2
                    cx2, cy2 = (x1 + x2) // 2, (y1 + y2) // 2
                    moved = ((cx1 - cx2)**2 + (cy1 - cy2)**2) ** 0.5 > MOVE_THRESHOLD
                    if moved:
                        locked = False
                        vote_history = []
                        confirmed_card = None
                        cached_ocr = ("", 0.0)  # Reset OCR cache on movement
                        last_ocr_time = 0

                if not locked:
                    # Only run OCR periodically (it's slow!)
                    current_time = time.time()
                    if current_time - last_ocr_time >= OCR_INTERVAL:
                        cached_ocr = ocr_title(card_img)
                        last_ocr_time = current_time

                    # Fast identification using cached OCR + embedding
                    card_name, conf, method = identify_card_fast(card_img, cached_ocr)

                    if card_name:
                        vote_history.append((card_name, conf, method))
                        if len(vote_history) > VOTE_FRAMES * 2:
                            vote_history = vote_history[-VOTE_FRAMES * 2:]

                    # Check consensus
                    if len(vote_history) >= VOTE_FRAMES:
                        recent = [v[0] for v in vote_history[-VOTE_FRAMES:]]
                        counts = Counter(recent)
                        most_common, count = counts.most_common(1)[0]

                        if count >= VOTE_FRAMES - 1:
                            confirmed_card = most_common
                            confs = [v[1] for v in vote_history[-VOTE_FRAMES:] if v[0] == most_common]
                            confirmed_conf = sum(confs) / len(confs)
                            confirmed_method = vote_history[-1][2]

                            if confirmed_conf >= 0.7:
                                locked = True
                                locked_box = current_box

                    # Show current OCR
                    if debug_mode:
                        title_region = extract_title_region(card_img)
                        if title_region.size > 0:
                            cv2.imshow('Title Region', cv2.resize(title_region, None, fx=4, fy=4))
                            processed = preprocess_for_ocr(title_region)
                            if processed is not None:
                                cv2.imshow('Processed', processed)

                # Display result
                if confirmed_card:
                    method_colors = {
                        "OCR+EMB": (0, 255, 0),
                        "OCR(emb-verified)": (0, 255, 128),
                        "OCR-HIGH": (0, 255, 255),
                        "OCR-MED": (0, 200, 255),
                        "EMB-ONLY": (0, 165, 255),
                    }
                    color = method_colors.get(confirmed_method, (255, 255, 255))

                    label = f"{confirmed_card[:40]}"
                    cv2.putText(display, label, (x1, y1 - 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    status = f"{confirmed_method} ({confirmed_conf:.2f})"
                    if locked:
                        status += " [LOCKED]"
                    cv2.putText(display, status, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            vote_history = []
            confirmed_card = None
            locked = False
            locked_box = None

        # FPS
        fps = 1.0 / (time.time() - start)
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if locked:
            cv2.putText(display, "LOCKED", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        cv2.imshow('OCR Primary', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            if not debug_mode:
                cv2.destroyWindow('Title Region')
                cv2.destroyWindow('Processed')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
