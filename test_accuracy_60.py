"""
60-Card Accuracy Test for Spike Evaluation

Instructions:
1. Hold up a card to the webcam
2. Press SPACE/ENTER if prediction is CORRECT
3. Press 'x' if prediction is WRONG
4. Press 'q' to quit early and save results

Results are saved to accuracy_test_results.csv
"""
import sys
import time
import csv
import os
from datetime import datetime
import cv2
import numpy as np
import faiss
import json
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataclasses import dataclass

# --- Configuration ---
CAMERA_INDEX = 1  # 0 = front camera, 1 = rear/external camera
CONFIDENCE_THRESHOLD = 0.6
TARGET_CARDS = 60
GLARE_CORRECTION = True  # Toggle glare correction preprocessing

# Auto-capture settings
AUTO_CAPTURE = True  # Auto-capture when prediction stabilizes
STABILITY_FRAMES = 15  # Number of frames with same prediction to trigger capture
STABILITY_CONFIDENCE_MIN = 0.40  # Minimum confidence to consider stable
SAVE_CROPS = True  # Save cropped card images for review

# Test conditions (edit these for your setup)
TEST_CONDITIONS = {
    "webcam_resolution": "1280x960",
    "surface": "wood grain desk",
    "lighting": "typical room",
    "sleeves": "yes",
    "glare_correction": "yes" if GLARE_CORRECTION else "no",
    "auto_capture": "yes" if AUTO_CAPTURE else "no",
    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
}

# Create crops directory
CROPS_DIR = "accuracy_test_crops"
if SAVE_CROPS:
    os.makedirs(CROPS_DIR, exist_ok=True)

@dataclass
class TestResult:
    card_number: int
    actual_name: str
    predicted_name: str
    confidence: float
    correct: bool
    top3_predictions: str

# --- Load models ---
print("Loading models...")
from ultralytics import YOLO
yolo = YOLO('training/yolo/runs/detect/train/weights/best.pt')

session = ort.InferenceSession('training/output/card_embedding_model.onnx')
input_name = session.get_inputs()[0].name

index = faiss.read_index('training/output/card_embeddings_onnx.faiss')

with open('training/output/label_mapping_onnx.json', 'r') as f:
    mapping = json.load(f)
card_names = mapping["card_names"]

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

print(f"Models loaded! Index has {len(card_names)} cards")

# --- Helper functions ---
def correct_glare(img):
    """Apply CLAHE and glare reduction to image."""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel (luminance)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_corrected = clahe.apply(l)

    # Detect and reduce glare (very bright spots)
    # Threshold high values and blend them down
    glare_mask = l_corrected > 240
    if np.any(glare_mask):
        # Reduce intensity of glare spots
        l_corrected[glare_mask] = (l_corrected[glare_mask] * 0.7).astype(np.uint8)

    # Merge back
    lab_corrected = cv2.merge([l_corrected, a, b])
    corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    return corrected

def get_embedding(img, apply_glare_correction=True):
    # Apply glare correction if enabled
    if apply_glare_correction:
        img = correct_glare(img)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=rgb)
    tensor = transformed["image"].numpy()
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
    embedding = session.run(None, {input_name: tensor})[0]
    faiss.normalize_L2(embedding)
    return embedding

def identify_card(embedding):
    distances, indices = index.search(embedding, 5)
    top5 = [(card_names[indices[0][i]], float(distances[0][i])) for i in range(5)]
    return top5

def save_results(results, filename="accuracy_test_results.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write conditions header
        writer.writerow(["# Test Conditions"])
        for key, value in TEST_CONDITIONS.items():
            writer.writerow([f"# {key}", value])
        writer.writerow([])

        # Write results
        writer.writerow(["card_number", "actual_name", "predicted_name", "confidence", "correct", "top3_predictions"])
        for r in results:
            writer.writerow([r.card_number, r.actual_name, r.predicted_name, f"{r.confidence:.3f}", r.correct, r.top3_predictions])

        # Write summary
        writer.writerow([])
        correct = sum(1 for r in results if r.correct)
        total = len(results)
        writer.writerow(["# Summary"])
        writer.writerow(["# Total cards", total])
        writer.writerow(["# Correct", correct])
        writer.writerow(["# Accuracy", f"{100*correct/total:.1f}%" if total > 0 else "N/A"])

    print(f"\nResults saved to {filename}")

# --- Main loop ---
print("\n" + "="*60)
print("60-CARD ACCURACY TEST")
print("="*60)
print(f"Conditions: {TEST_CONDITIONS['lighting']}, {TEST_CONDITIONS['sleeves']} sleeves, {TEST_CONDITIONS['webcam_resolution']}")
print()
print("Controls (press in the VIDEO WINDOW, not terminal):")
if AUTO_CAPTURE:
    print("  AUTO-CAPTURE enabled - will capture when prediction stabilizes")
    print("  Just press X if the auto-captured prediction is WRONG")
else:
    print("  SPACE or ENTER = Prediction is CORRECT")
print("  X = Prediction is WRONG")
print("  G = Toggle glare correction")
print("  Q = Quit and save results")
print(f"\nGlare correction: {'ON' if GLARE_CORRECTION else 'OFF'}")
print(f"Auto-capture: {'ON' if AUTO_CAPTURE else 'OFF'} (stability: {STABILITY_FRAMES} frames)")
print(f"Saving crops: {'ON' if SAVE_CROPS else 'OFF'}")
print("="*60)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"ERROR: Cannot open camera {CAMERA_INDEX}")
    print("Try changing CAMERA_INDEX at top of script (0, 1, 2...)")
    sys.exit(1)
print(f"Using camera index: {CAMERA_INDEX}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
TEST_CONDITIONS["webcam_resolution"] = f"{actual_w}x{actual_h}"
print(f"Webcam: {actual_w}x{actual_h}")

cv2.namedWindow('Accuracy Test', cv2.WINDOW_NORMAL)

results = []
current_prediction = None
current_confidence = 0.0
current_top3 = ""
current_top5 = []
current_crop = None  # Store current card crop
card_count = 0
glare_enabled = GLARE_CORRECTION  # Local copy for toggling

# Stability tracking
stability_history = []  # List of recent predictions
stability_count = 0  # Consecutive frames with same prediction
last_stable_prediction = None

while card_count < TARGET_CARDS:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # Show glare correction status
    glare_status = "GLARE FIX: ON" if glare_enabled else "GLARE FIX: OFF"
    glare_color = (100, 255, 100) if glare_enabled else (100, 100, 255)
    cv2.putText(display, glare_status, (display.shape[1] - 200, display.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, glare_color, 2)

    # YOLO detection
    yolo_results = yolo(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    # Process first detection
    if len(yolo_results.boxes) > 0:
        box = yolo_results.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        # Draw box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Get identification
        card_img = frame[y1:y2, x1:x2]
        if card_img.size > 0:
            current_crop = card_img.copy()  # Save crop for later
            embedding = get_embedding(card_img, apply_glare_correction=glare_enabled)
            current_top5 = identify_card(embedding)

            current_prediction = current_top5[0][0]
            current_confidence = current_top5[0][1]
            current_top3 = f"{current_top5[0][0]} ({current_top5[0][1]:.2f}), {current_top5[1][0]} ({current_top5[1][1]:.2f}), {current_top5[2][0]} ({current_top5[2][1]:.2f})"

            # Track stability
            if current_confidence >= STABILITY_CONFIDENCE_MIN:
                if len(stability_history) > 0 and stability_history[-1] == current_prediction:
                    stability_count += 1
                else:
                    stability_count = 1
                stability_history.append(current_prediction)
                if len(stability_history) > STABILITY_FRAMES:
                    stability_history.pop(0)
            else:
                stability_count = 0
                stability_history = []

            # Display prediction
            label = f"Predicted: {current_prediction[:40]}"
            cv2.putText(display, label, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Confidence: {current_confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show stability progress bar
            if AUTO_CAPTURE:
                stability_pct = min(stability_count / STABILITY_FRAMES, 1.0)
                bar_width = int(200 * stability_pct)
                bar_color = (0, 255, 255) if stability_pct < 1.0 else (0, 255, 0)
                cv2.rectangle(display, (x1, y2 + 10), (x1 + 200, y2 + 30), (100, 100, 100), -1)
                cv2.rectangle(display, (x1, y2 + 10), (x1 + bar_width, y2 + 30), bar_color, -1)
                cv2.putText(display, f"Stable: {stability_count}/{STABILITY_FRAMES}", (x1, y2 + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Status bar at top
    cv2.rectangle(display, (0, 0), (display.shape[1], 100), (0, 0, 0), -1)
    cv2.putText(display, f"Card {card_count + 1}/{TARGET_CARDS}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(display, "SPACE/ENTER=Correct  X=Wrong  Q=Quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

    correct_count = sum(1 for r in results if r.correct)
    if results:
        acc_pct = 100*correct_count/len(results)
        color = (100, 255, 100) if acc_pct >= 70 else (100, 100, 255)
        cv2.putText(display, f"Accuracy: {correct_count}/{len(results)} ({acc_pct:.0f}%)",
                   (display.shape[1] - 350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Accuracy Test', display)

    key = cv2.waitKey(1) & 0xFF

    # Check for auto-capture trigger
    auto_triggered = False
    if AUTO_CAPTURE and stability_count >= STABILITY_FRAMES and current_prediction:
        if last_stable_prediction != current_prediction:  # Don't re-capture same card
            auto_triggered = True
            last_stable_prediction = current_prediction

    if key == ord('q'):
        print("\nQuitting early...")
        break

    elif (key == ord(' ') or key == 13 or auto_triggered) and current_prediction:  # SPACE/ENTER/AUTO = correct
        card_count += 1
        result = TestResult(
            card_number=card_count,
            actual_name=current_prediction,
            predicted_name=current_prediction,
            confidence=current_confidence,
            correct=True,
            top3_predictions=current_top3
        )
        results.append(result)

        # Save crop
        if SAVE_CROPS and current_crop is not None:
            crop_filename = f"{CROPS_DIR}/card_{card_count:02d}_correct_{current_prediction[:30].replace(' ', '_').replace('//', '-')}.jpg"
            cv2.imwrite(crop_filename, current_crop)

        trigger = "AUTO" if auto_triggered else "MANUAL"
        print(f"Card {card_count} [{trigger}]: CORRECT - {current_prediction} ({current_confidence:.2f})")

        # Reset for next card
        current_prediction = None
        stability_count = 0
        stability_history = []

    elif key == ord('x') and current_prediction:  # X = wrong
        card_count += 1
        result = TestResult(
            card_number=card_count,
            actual_name="[wrong]",
            predicted_name=current_prediction,
            confidence=current_confidence,
            correct=False,
            top3_predictions=current_top3
        )
        results.append(result)

        # Save crop for review
        if SAVE_CROPS and current_crop is not None:
            crop_filename = f"{CROPS_DIR}/card_{card_count:02d}_WRONG_{current_prediction[:30].replace(' ', '_').replace('//', '-')}.jpg"
            cv2.imwrite(crop_filename, current_crop)

        print(f"Card {card_count}: WRONG - predicted {current_prediction} ({current_confidence:.2f})")

        # Reset for next card
        current_prediction = None
        stability_count = 0
        stability_history = []
        last_stable_prediction = None  # Allow re-detection of same card if wrong

    elif key == ord('g'):  # G = toggle glare correction
        glare_enabled = not glare_enabled
        print(f"Glare correction: {'ON' if glare_enabled else 'OFF'}")

cap.release()
cv2.destroyAllWindows()

# Save results
if results:
    save_results(results)

    # Print summary
    correct = sum(1 for r in results if r.correct)
    total = len(results)
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total cards tested: {total}")
    print(f"Correct identifications: {correct}")
    print(f"Accuracy: {100*correct/total:.1f}%")
    print()
    print("Pass/Fail Criteria (typical room lighting):")
    print(f"  Target: >= 70% accuracy")
    print(f"  Result: {100*correct/total:.1f}% - {'PASS' if correct/total >= 0.70 else 'FAIL'}")
else:
    print("No results to save.")

print("\nDone!")
