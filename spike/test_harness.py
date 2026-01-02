"""
Test harness for measuring card detection and identification accuracy.

Workflow:
1. Capture mode: Take photos of cards, label them with correct names
2. Test mode: Run detection/identification on labeled images, measure accuracy
3. Report mode: Generate accuracy report with failure cases

Usage:
    python test_harness.py capture    # Capture and label test images
    python test_harness.py test       # Run accuracy test on captured images
    python test_harness.py report     # Generate summary report
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add training directory to path
SCRIPT_DIR = Path(__file__).parent
TRAINING_DIR = SCRIPT_DIR.parent / "training"
sys.path.insert(0, str(TRAINING_DIR))

from inference import CardDetector, CardIdentifier

# Paths
TEST_IMAGES_DIR = SCRIPT_DIR / "test_images"
LABELS_FILE = SCRIPT_DIR / "test_labels.json"
RESULTS_FILE = SCRIPT_DIR / "test_results.csv"
FAILURES_DIR = SCRIPT_DIR / "failure_cases"
REPORT_FILE = SCRIPT_DIR / "ACCURACY_REPORT.md"


def load_labels():
    """Load existing labels."""
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(labels):
    """Save labels to file."""
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


def capture_mode(camera_id=0):
    """Capture test images and label them interactively."""
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()

    print("=" * 60)
    print("CAPTURE MODE")
    print("=" * 60)
    print("Controls:")
    print("  SPACE - Capture image")
    print("  Q     - Quit")
    print("=" * 60)
    print()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    detector = CardDetector()
    image_count = len(list(TEST_IMAGES_DIR.glob("*.jpg")))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect cards for preview
        cards = detector.detect(frame, fast_mode=True)

        # Draw detection preview
        display = frame.copy()
        for contour, corners in cards:
            corners_int = corners.astype(np.int32)
            cv2.polylines(display, [corners_int], True, (0, 255, 0), 2)

        cv2.putText(display, f"Images: {image_count} | SPACE=capture, Q=quit",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Cards detected: {len(cards)}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Capture Test Images", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if not cards:
                print("No card detected! Position card and try again.")
                continue

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_{timestamp}.jpg"
            filepath = TEST_IMAGES_DIR / filename
            cv2.imwrite(str(filepath), frame)

            # Get label from user
            cv2.destroyAllWindows()
            print(f"\nSaved: {filename}")
            print("Enter the EXACT card name (or 'skip' to skip, 'undo' to delete):")
            card_name = input("> ").strip()

            if card_name.lower() == "skip":
                filepath.unlink()
                print("Skipped.")
            elif card_name.lower() == "undo":
                filepath.unlink()
                print("Deleted.")
            else:
                labels[filename] = {
                    "card_name": card_name,
                    "timestamp": timestamp,
                    "num_cards_detected": len(cards),
                }
                save_labels(labels)
                image_count += 1
                print(f"Labeled as: {card_name}")

            print()

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCapture complete. {len(labels)} labeled images.")


def test_mode():
    """Run accuracy test on labeled images."""
    labels = load_labels()
    if not labels:
        print("No labeled images found. Run 'capture' mode first.")
        return

    print("=" * 60)
    print("TEST MODE")
    print("=" * 60)
    print(f"Testing {len(labels)} labeled images...")
    print()

    # Initialize components
    model_path = TRAINING_DIR / "checkpoints" / "final_model.pt"
    index_path = TRAINING_DIR / "output" / "card_embeddings_full.faiss"
    mapping_path = TRAINING_DIR / "output" / "label_mapping_full.json"

    if not all(p.exists() for p in [model_path, index_path, mapping_path]):
        print("Error: Model files not found. Run training first.")
        return

    detector = CardDetector()
    identifier = CardIdentifier(model_path, index_path, mapping_path)

    FAILURES_DIR.mkdir(parents=True, exist_ok=True)

    # Results tracking
    results = []
    detection_success = 0
    identification_correct = 0
    identification_top5 = 0
    total_detect_time = 0
    total_id_time = 0

    for filename, label_info in labels.items():
        filepath = TEST_IMAGES_DIR / filename
        if not filepath.exists():
            print(f"  SKIP: {filename} (file not found)")
            continue

        expected_name = label_info["card_name"]
        image = cv2.imread(str(filepath))

        # Detection
        detect_start = time.time()
        cards = detector.detect(image, fast_mode=False)  # Thorough mode for testing
        detect_time = time.time() - detect_start
        total_detect_time += detect_time

        detected = len(cards) > 0
        if detected:
            detection_success += 1

        # Identification
        predicted_name = None
        confidence = 0
        top5_names = []
        id_time = 0

        if detected:
            # Use first detected card
            contour, corners = cards[0]
            card_image = detector.extract_card(image, corners)

            id_start = time.time()
            matches = identifier.identify(card_image, top_k=5, try_rotations=True)
            id_time = time.time() - id_start
            total_id_time += id_time

            if matches:
                predicted_name = matches[0][0]
                confidence = matches[0][1]
                top5_names = [m[0] for m in matches]

        # Check correctness (case-insensitive, partial match)
        correct = False
        in_top5 = False
        if predicted_name:
            correct = expected_name.lower() in predicted_name.lower() or \
                     predicted_name.lower() in expected_name.lower()
            in_top5 = any(expected_name.lower() in n.lower() or n.lower() in expected_name.lower()
                        for n in top5_names)

        if correct:
            identification_correct += 1
        if in_top5:
            identification_top5 += 1

        # Log result
        result = {
            "filename": filename,
            "expected": expected_name,
            "predicted": predicted_name or "N/A",
            "confidence": f"{confidence:.3f}",
            "detected": detected,
            "correct": correct,
            "in_top5": in_top5,
            "detect_ms": f"{detect_time*1000:.1f}",
            "id_ms": f"{id_time*1000:.1f}",
            "top5": "; ".join(top5_names) if top5_names else "N/A",
        }
        results.append(result)

        # Status indicator
        status = "OK" if correct else ("TOP5" if in_top5 else "FAIL")
        print(f"  [{status}] {expected_name[:30]:30s} -> {(predicted_name or 'N/A')[:30]:30s} ({confidence:.2f})")

        # Save failure case
        if not correct and detected:
            failure_path = FAILURES_DIR / f"fail_{filename}"
            # Draw annotation on image
            annotated = image.copy()
            corners_int = cards[0][1].astype(np.int32)
            cv2.polylines(annotated, [corners_int], True, (0, 0, 255), 3)
            cv2.putText(annotated, f"Expected: {expected_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated, f"Got: {predicted_name} ({confidence:.2f})", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(failure_path), annotated)

    # Save results CSV
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    total = len(results)
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total images tested: {total}")
    print()
    print("DETECTION:")
    print(f"  Success rate: {detection_success}/{total} ({100*detection_success/total:.1f}%)")
    print()
    print("IDENTIFICATION:")
    print(f"  Top-1 accuracy: {identification_correct}/{total} ({100*identification_correct/total:.1f}%)")
    print(f"  Top-5 accuracy: {identification_top5}/{total} ({100*identification_top5/total:.1f}%)")
    print()
    print("PERFORMANCE:")
    print(f"  Avg detection time: {1000*total_detect_time/total:.1f}ms")
    print(f"  Avg identification time: {1000*total_id_time/total:.1f}ms")
    print(f"  Total avg time: {1000*(total_detect_time+total_id_time)/total:.1f}ms")
    print()
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"Failure cases saved to: {FAILURES_DIR}/")
    print("=" * 60)

    return {
        "total": total,
        "detection_rate": detection_success / total if total > 0 else 0,
        "top1_accuracy": identification_correct / total if total > 0 else 0,
        "top5_accuracy": identification_top5 / total if total > 0 else 0,
        "avg_detect_ms": 1000 * total_detect_time / total if total > 0 else 0,
        "avg_id_ms": 1000 * total_id_time / total if total > 0 else 0,
    }


def report_mode():
    """Generate markdown accuracy report."""
    labels = load_labels()
    if not RESULTS_FILE.exists():
        print("No test results found. Run 'test' mode first.")
        return

    # Load results
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = list(reader)

    # Calculate stats
    total = len(results)
    detected = sum(1 for r in results if r["detected"] == "True")
    correct = sum(1 for r in results if r["correct"] == "True")
    top5 = sum(1 for r in results if r["in_top5"] == "True")

    avg_detect = sum(float(r["detect_ms"]) for r in results) / total
    avg_id = sum(float(r["id_ms"]) for r in results if r["detected"] == "True") / max(detected, 1)

    # Count by confidence ranges
    high_conf = sum(1 for r in results if r["detected"] == "True" and float(r["confidence"]) >= 0.7)
    med_conf = sum(1 for r in results if r["detected"] == "True" and 0.5 <= float(r["confidence"]) < 0.7)
    low_conf = sum(1 for r in results if r["detected"] == "True" and float(r["confidence"]) < 0.5)

    # Generate report
    report = f"""# Card Detection Accuracy Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Test images:** {total}

---

## Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Detection rate | {100*detected/total:.1f}% | >=85% | {"PASS" if detected/total >= 0.85 else "FAIL"} |
| Top-1 accuracy | {100*correct/total:.1f}% | >=70% | {"PASS" if correct/total >= 0.70 else "FAIL"} |
| Top-5 accuracy | {100*top5/total:.1f}% | >=85% | {"PASS" if top5/total >= 0.85 else "FAIL"} |
| Avg detect time | {avg_detect:.1f}ms | <100ms | {"PASS" if avg_detect < 100 else "FAIL"} |
| Avg identify time | {avg_id:.1f}ms | <500ms | {"PASS" if avg_id < 500 else "FAIL"} |

---

## Detection Results

- **Detected:** {detected}/{total} ({100*detected/total:.1f}%)
- **Missed:** {total-detected}/{total} ({100*(total-detected)/total:.1f}%)

---

## Identification Results

- **Correct (Top-1):** {correct}/{total} ({100*correct/total:.1f}%)
- **In Top-5:** {top5}/{total} ({100*top5/total:.1f}%)
- **Wrong:** {total-top5}/{total} ({100*(total-top5)/total:.1f}%)

### Confidence Distribution

| Confidence | Count | Correct |
|------------|-------|---------|
| High (>=0.7) | {high_conf} | {sum(1 for r in results if r["detected"]=="True" and float(r["confidence"])>=0.7 and r["correct"]=="True")} |
| Medium (0.5-0.7) | {med_conf} | {sum(1 for r in results if r["detected"]=="True" and 0.5<=float(r["confidence"])<0.7 and r["correct"]=="True")} |
| Low (<0.5) | {low_conf} | {sum(1 for r in results if r["detected"]=="True" and float(r["confidence"])<0.5 and r["correct"]=="True")} |

---

## Failure Cases

"""

    # List failures
    failures = [r for r in results if r["correct"] != "True"]
    if failures:
        report += "| Image | Expected | Predicted | Confidence |\n"
        report += "|-------|----------|-----------|------------|\n"
        for r in failures[:20]:  # Limit to 20
            report += f"| {r['filename']} | {r['expected']} | {r['predicted']} | {r['confidence']} |\n"
        if len(failures) > 20:
            report += f"\n*...and {len(failures)-20} more failures*\n"
    else:
        report += "*No failures!*\n"

    report += f"""
---

## Go/No-Go Recommendation

"""

    # Determine recommendation
    detection_pass = detected/total >= 0.85
    accuracy_pass = correct/total >= 0.70
    speed_pass = avg_id < 500

    if detection_pass and accuracy_pass and speed_pass:
        report += "**RECOMMENDATION: GO**\n\nAll criteria met. Card detection and identification is viable.\n"
    elif detection_pass and top5/total >= 0.85:
        report += "**RECOMMENDATION: CONDITIONAL GO**\n\nDetection works. Top-5 accuracy is acceptable. Consider showing multiple suggestions to user.\n"
    else:
        report += "**RECOMMENDATION: NO-GO**\n\nAccuracy does not meet minimum thresholds.\n"
        if not detection_pass:
            report += f"- Detection rate {100*detected/total:.1f}% below 85% target\n"
        if not accuracy_pass:
            report += f"- Top-1 accuracy {100*correct/total:.1f}% below 70% target\n"
        if not speed_pass:
            report += f"- Identification time {avg_id:.0f}ms exceeds 500ms target\n"

    # Save report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {REPORT_FILE}")


def list_mode():
    """List current test images and labels."""
    labels = load_labels()

    print("=" * 60)
    print("LABELED TEST IMAGES")
    print("=" * 60)

    if not labels:
        print("No labeled images. Run 'capture' mode to add some.")
        return

    for filename, info in labels.items():
        exists = (TEST_IMAGES_DIR / filename).exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {filename}: {info['card_name']}")

    print()
    print(f"Total: {len(labels)} images")


def batch_capture_mode(camera_id=0, count=10, delay=2):
    """Capture multiple images automatically without labeling."""
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BATCH CAPTURE MODE")
    print("=" * 60)
    print(f"Will capture {count} images with {delay}s delay between each.")
    print("Position different cards in front of camera.")
    print("Press Q to stop early.")
    print("=" * 60)
    print()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    detector = CardDetector()
    captured = 0

    while captured < count:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect cards for preview
        cards = detector.detect(frame, fast_mode=True)

        # Draw detection preview
        display = frame.copy()
        for contour, corners in cards:
            corners_int = corners.astype(np.int32)
            cv2.polylines(display, [corners_int], True, (0, 255, 0), 2)

        cv2.putText(display, f"Captured: {captured}/{count} | Cards: {len(cards)} | Q=quit",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Batch Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') or (len(cards) > 0):
            # Auto-capture when card detected or space pressed
            if len(cards) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"unlabeled_{timestamp}.jpg"
                filepath = TEST_IMAGES_DIR / filename
                cv2.imwrite(str(filepath), frame)
                captured += 1
                print(f"Captured: {filename}")

                # Wait before next capture
                time.sleep(delay)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCaptured {captured} images to {TEST_IMAGES_DIR}/")
    print("Run 'python test_harness.py label' to label them.")


def label_mode():
    """Label unlabeled images by viewing them."""
    labels = load_labels()
    unlabeled = list(TEST_IMAGES_DIR.glob("unlabeled_*.jpg"))

    if not unlabeled:
        print("No unlabeled images found.")
        return

    print(f"Found {len(unlabeled)} unlabeled images.")
    print("For each image, enter the card name (or 'skip'/'delete').")
    print()

    for filepath in unlabeled:
        # Show image
        img = cv2.imread(str(filepath))
        cv2.imshow("Label This Card", img)
        cv2.waitKey(500)  # Show briefly

        print(f"\nImage: {filepath.name}")
        card_name = input("Card name (skip/delete): ").strip()

        if card_name.lower() == "delete":
            filepath.unlink()
            print("Deleted.")
        elif card_name.lower() == "skip":
            print("Skipped.")
        else:
            # Rename file and add label
            new_filename = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            new_path = TEST_IMAGES_DIR / new_filename
            filepath.rename(new_path)

            labels[new_filename] = {
                "card_name": card_name,
                "timestamp": datetime.now().isoformat(),
            }
            save_labels(labels)
            print(f"Labeled as: {card_name}")

        cv2.destroyAllWindows()

    print(f"\nLabeling complete. {len(labels)} total labeled images.")


def main():
    parser = argparse.ArgumentParser(description="Card Detection Test Harness")
    parser.add_argument("mode", choices=["capture", "batch", "label", "test", "report", "list"],
                       help="Mode: capture, batch, label, test, report, or list")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID for capture mode")
    parser.add_argument("--count", type=int, default=10, help="Number of images for batch mode")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between captures in batch mode")
    args = parser.parse_args()

    if args.mode == "capture":
        capture_mode(args.camera)
    elif args.mode == "batch":
        batch_capture_mode(args.camera, args.count, args.delay)
    elif args.mode == "label":
        label_mode()
    elif args.mode == "test":
        test_mode()
    elif args.mode == "report":
        report_mode()
    elif args.mode == "list":
        list_mode()


if __name__ == "__main__":
    main()
