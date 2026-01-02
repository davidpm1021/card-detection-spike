#!/usr/bin/env python3
"""
Validate YOLO detection on real webcam test images.

Compares YOLO detection vs contour-based detection and generates
a detailed report with metrics and failure analysis.

Usage:
    python validate.py
    python validate.py --test-dir training/yolo/test_images
    python validate.py --create-ground-truth  # Interactive labeling
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Add spike directory to path for detectors
SCRIPT_DIR = Path(__file__).parent
SPIKE_DIR = SCRIPT_DIR.parent.parent / "spike"
sys.path.insert(0, str(SPIKE_DIR))


class ValidationRunner:
    """Run validation on test images comparing YOLO vs contour detection."""

    def __init__(self, test_dir: Path, output_dir: Path):
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = []

        # Detectors (loaded lazily)
        self._yolo_detector = None
        self._contour_detector = None

    @property
    def yolo_detector(self):
        """Lazy load YOLO detector."""
        if self._yolo_detector is None:
            try:
                from yolo_detector import YOLODetector
                self._yolo_detector = YOLODetector()
                print("YOLO detector loaded")
            except Exception as e:
                print(f"Warning: Could not load YOLO detector: {e}")
                self._yolo_detector = False  # Mark as unavailable
        return self._yolo_detector if self._yolo_detector else None

    @property
    def contour_detector(self):
        """Lazy load contour detector."""
        if self._contour_detector is None:
            from inference import CardDetector
            self._contour_detector = CardDetector()
            print("Contour detector loaded")
        return self._contour_detector

    def load_ground_truth(self) -> Dict[str, dict]:
        """
        Load ground truth card counts from JSON file.

        Format:
        {
            "playmat_20260102_123456.jpg": {"num_cards": 1, "notes": ""},
            ...
        }
        """
        gt_path = self.test_dir / "ground_truth.json"

        if gt_path.exists():
            with open(gt_path, encoding="utf-8") as f:
                return json.load(f)

        return {}

    def create_ground_truth(self):
        """Interactively create ground truth labels for test images."""
        gt_path = self.test_dir / "ground_truth.json"

        # Load existing ground truth
        ground_truth = self.load_ground_truth()

        # Get all images
        images = sorted(self.test_dir.glob("*.jpg"))
        if not images:
            print(f"No images found in {self.test_dir}")
            return

        print("=" * 60)
        print("Ground Truth Labeling")
        print("=" * 60)
        print("For each image, enter the number of cards visible (0-9)")
        print("Press 's' to skip, 'q' to quit and save")
        print("=" * 60)

        labeled = 0
        skipped = 0

        for img_path in images:
            # Skip if already labeled
            if img_path.name in ground_truth:
                continue

            # Load and display image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Show image with filename
            display = img.copy()
            cv2.putText(display, img_path.name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Press 0-9 for card count, s=skip, q=quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Label Ground Truth", display)

            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    # Save and quit
                    cv2.destroyAllWindows()
                    self._save_ground_truth(ground_truth, gt_path)
                    print(f"\nLabeled: {labeled}, Skipped: {skipped}")
                    return

                elif key == ord('s'):
                    skipped += 1
                    print(f"Skipped: {img_path.name}")
                    break

                elif ord('0') <= key <= ord('9'):
                    num_cards = key - ord('0')
                    ground_truth[img_path.name] = {
                        "num_cards": num_cards,
                        "notes": "",
                        "category": img_path.stem.split("_")[0]
                    }
                    labeled += 1
                    print(f"{img_path.name}: {num_cards} cards")
                    break

        cv2.destroyAllWindows()
        self._save_ground_truth(ground_truth, gt_path)
        print(f"\nLabeling complete! Labeled: {labeled}, Skipped: {skipped}")

    def _save_ground_truth(self, ground_truth: dict, gt_path: Path):
        """Save ground truth to JSON file."""
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2)
        print(f"Ground truth saved to {gt_path}")

    def run_validation(self):
        """Run validation on all test images with ground truth."""
        ground_truth = self.load_ground_truth()

        if not ground_truth:
            print("No ground truth found. Run with --create-ground-truth first.")
            return False

        print("\n" + "=" * 60)
        print("Running Validation")
        print("=" * 60)
        print(f"Test images: {len(ground_truth)}")

        yolo_available = self.yolo_detector is not None

        if not yolo_available:
            print("WARNING: YOLO detector not available. Training may not be complete.")
            print("Continuing with contour detection only...")

        for img_path in sorted(self.test_dir.glob("*.jpg")):
            if img_path.name not in ground_truth:
                continue

            gt = ground_truth[img_path.name]
            expected_cards = gt["num_cards"]

            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Parse category from filename or ground truth
            category = gt.get("category", img_path.stem.split("_")[0])

            # Test YOLO detection
            if yolo_available:
                yolo_start = time.perf_counter()
                yolo_detections = self.yolo_detector.detect(img)
                yolo_time = (time.perf_counter() - yolo_start) * 1000
            else:
                yolo_detections = []
                yolo_time = 0

            # Test contour detection
            contour_start = time.perf_counter()
            contour_detections = self.contour_detector.detect(img)
            contour_time = (time.perf_counter() - contour_start) * 1000

            # Calculate metrics
            yolo_correct = len(yolo_detections) == expected_cards
            contour_correct = len(contour_detections) == expected_cards

            # False positives / negatives
            yolo_fp = max(0, len(yolo_detections) - expected_cards)
            yolo_fn = max(0, expected_cards - len(yolo_detections))
            contour_fp = max(0, len(contour_detections) - expected_cards)
            contour_fn = max(0, expected_cards - len(contour_detections))

            result = {
                "filename": img_path.name,
                "category": category,
                "expected_cards": expected_cards,
                "yolo_detected": len(yolo_detections),
                "yolo_correct": yolo_correct,
                "yolo_fp": yolo_fp,
                "yolo_fn": yolo_fn,
                "yolo_time_ms": round(yolo_time, 1),
                "contour_detected": len(contour_detections),
                "contour_correct": contour_correct,
                "contour_fp": contour_fp,
                "contour_fn": contour_fn,
                "contour_time_ms": round(contour_time, 1),
            }

            self.results.append(result)

            # Log result
            if yolo_available:
                y_status = "OK" if yolo_correct else "FAIL"
                c_status = "OK" if contour_correct else "FAIL"
                print(f"{img_path.name}: YOLO={len(yolo_detections)}/{expected_cards} [{y_status}] "
                      f"Contour={len(contour_detections)}/{expected_cards} [{c_status}] "
                      f"({yolo_time:.0f}ms/{contour_time:.0f}ms)")
            else:
                c_status = "OK" if contour_correct else "FAIL"
                print(f"{img_path.name}: Contour={len(contour_detections)}/{expected_cards} [{c_status}] "
                      f"({contour_time:.0f}ms)")

            # Save failure case
            if yolo_available and not yolo_correct:
                self._save_failure_case(img, yolo_detections, contour_detections,
                                        expected_cards, img_path.name)

        return True

    def _save_failure_case(self, img, yolo_dets, contour_dets, expected, filename):
        """Save annotated failure case image."""
        failure_dir = self.output_dir / "failure_cases"
        failure_dir.mkdir(exist_ok=True)

        # Create annotated image
        annotated = img.copy()

        # Draw YOLO detections in green
        for contour, corners in yolo_dets:
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 3)
            cv2.putText(annotated, "YOLO", (int(corners[0][0]), int(corners[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw contour detections in blue (offset slightly to distinguish)
        for contour, corners in contour_dets:
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (255, 0, 0), 2)

        # Add legend
        cv2.rectangle(annotated, (5, 5), (300, 100), (0, 0, 0), -1)
        cv2.putText(annotated, f"Expected: {expected}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated, f"YOLO (green): {len(yolo_dets)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f"Contour (blue): {len(contour_dets)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imwrite(str(failure_dir / f"fail_{filename}"), annotated)

    def generate_report(self):
        """Generate validation report with metrics and recommendations."""
        if not self.results:
            print("No results to report")
            return

        # Save raw results to CSV
        csv_path = self.output_dir / "validation_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        print(f"\nResults saved to {csv_path}")

        # Calculate summary statistics
        total = len(self.results)

        # Check if YOLO was available
        yolo_available = any(r["yolo_time_ms"] > 0 for r in self.results)

        if yolo_available:
            yolo_correct = sum(1 for r in self.results if r["yolo_correct"])
            yolo_times = [r["yolo_time_ms"] for r in self.results]
            yolo_fps = sum(r["yolo_fp"] for r in self.results)
            yolo_fns = sum(r["yolo_fn"] for r in self.results)
        else:
            yolo_correct = 0
            yolo_times = [0]
            yolo_fps = 0
            yolo_fns = 0

        contour_correct = sum(1 for r in self.results if r["contour_correct"])
        contour_times = [r["contour_time_ms"] for r in self.results]

        # Per-category breakdown
        categories = set(r["category"] for r in self.results)
        category_stats = {}
        for cat in categories:
            cat_results = [r for r in self.results if r["category"] == cat]
            if yolo_available:
                cat_yolo = sum(1 for r in cat_results if r["yolo_correct"])
                cat_yolo_rate = cat_yolo / len(cat_results) * 100 if cat_results else 0
            else:
                cat_yolo = 0
                cat_yolo_rate = 0
            cat_contour = sum(1 for r in cat_results if r["contour_correct"])
            cat_contour_rate = cat_contour / len(cat_results) * 100 if cat_results else 0

            category_stats[cat] = {
                "total": len(cat_results),
                "yolo_correct": cat_yolo,
                "yolo_rate": cat_yolo_rate,
                "contour_correct": cat_contour,
                "contour_rate": cat_contour_rate,
            }

        # Generate markdown report
        yolo_rate = yolo_correct / total * 100 if yolo_available else 0
        contour_rate = contour_correct / total * 100
        avg_yolo_time = np.mean(yolo_times) if yolo_available else 0
        avg_contour_time = np.mean(contour_times)

        report = f"""# YOLO Detection Validation Report

**Date:** {time.strftime('%Y-%m-%d %H:%M')}
**Test Images:** {total}
**YOLO Available:** {'Yes' if yolo_available else 'No (training not complete)'}

---

## Overall Results

| Metric | YOLO | Contour | Target |
|--------|------|---------|--------|
| Accuracy | {yolo_rate:.1f}% | {contour_rate:.1f}% | >= 95% |
| Avg Time | {avg_yolo_time:.1f}ms | {avg_contour_time:.1f}ms | < 50ms |
| Min Time | {min(yolo_times):.1f}ms | {min(contour_times):.1f}ms | - |
| Max Time | {max(yolo_times):.1f}ms | {max(contour_times):.1f}ms | - |
| False Positives | {yolo_fps} | - | - |
| False Negatives | {yolo_fns} | - | - |

---

## Results by Category

| Category | Images | YOLO Rate | Contour Rate | Status |
|----------|--------|-----------|--------------|--------|
"""
        for cat, stats in sorted(category_stats.items()):
            status = "PASS" if stats["yolo_rate"] >= 90 else ("WARN" if stats["yolo_rate"] >= 75 else "FAIL")
            report += f"| {cat} | {stats['total']} | {stats['yolo_rate']:.0f}% | {stats['contour_rate']:.0f}% | {status} |\n"

        # Failure analysis
        failures = [r for r in self.results if not r["yolo_correct"]] if yolo_available else []
        if failures:
            report += f"""

---

## Failure Cases ({len(failures)} failures)

| Filename | Expected | YOLO | Contour | Category | Issue |
|----------|----------|------|---------|----------|-------|
"""
            for f in failures:
                if f["yolo_detected"] > f["expected_cards"]:
                    issue = f"FP (+{f['yolo_fp']})"
                elif f["yolo_detected"] < f["expected_cards"]:
                    issue = f"FN (-{f['yolo_fn']})"
                else:
                    issue = "?"
                report += f"| {f['filename']} | {f['expected_cards']} | {f['yolo_detected']} | {f['contour_detected']} | {f['category']} | {issue} |\n"

            report += """

See `failure_cases/` directory for annotated screenshots.
"""

        # Recommendation
        if not yolo_available:
            recommendation = "PENDING"
            details = "YOLO training not complete. Run training and re-validate."
        elif yolo_rate >= 95 and avg_yolo_time < 50:
            recommendation = "GO"
            details = "All targets met. YOLO detection is ready for production."
        elif yolo_rate >= 90 and avg_yolo_time < 100:
            recommendation = "CONDITIONAL GO"
            details = "Performance acceptable. Monitor failure cases in production."
        else:
            recommendation = "NO-GO"
            details = "Targets not met. Review failure cases and retrain."

        report += f"""

---

## Recommendation

**{recommendation}**

{details}

### Metrics vs Targets

| Metric | Achieved | Target | Status |
|--------|----------|--------|--------|
| Detection Rate | {yolo_rate:.1f}% | >= 95% | {"PASS" if yolo_rate >= 95 else ("WARN" if yolo_rate >= 90 else "FAIL")} |
| Avg Inference Time | {avg_yolo_time:.1f}ms | < 50ms | {"PASS" if avg_yolo_time < 50 else "FAIL"} |
| FPS (est.) | {1000/avg_yolo_time:.1f} | >= 20 | {"PASS" if avg_yolo_time < 50 else "FAIL"} |

---

## Improvement vs Contour Detection

| Metric | YOLO | Contour | Improvement |
|--------|------|---------|-------------|
| Accuracy | {yolo_rate:.1f}% | {contour_rate:.1f}% | {yolo_rate - contour_rate:+.1f}% |
| Avg Time | {avg_yolo_time:.1f}ms | {avg_contour_time:.1f}ms | {avg_contour_time - avg_yolo_time:+.1f}ms |

---

## Next Steps

"""
        if recommendation == "GO":
            report += """1. Deploy YOLO detection as default in `inference.py`
2. Monitor production metrics for regression
3. Consider training with more epochs for even better accuracy
"""
        elif recommendation == "CONDITIONAL GO":
            report += """1. Deploy YOLO detection with monitoring
2. Log failure cases in production for analysis
3. Collect more training data from failure categories
4. Retrain after collecting additional failure cases
"""
        elif recommendation == "PENDING":
            report += """1. Wait for YOLO training to complete
2. Re-run validation: `python training/yolo/validate.py`
3. Review results and make Go/No-Go decision
"""
        else:
            report += """1. Analyze failure cases in `failure_cases/` directory
2. Identify patterns (lighting? background? card position?)
3. Generate more synthetic data for failure categories
4. Increase training epochs or adjust augmentation
5. Retrain and re-validate
"""

        # Save report
        report_path = self.output_dir / "validation_summary.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {report_path}")

        # Print summary to console
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        if yolo_available:
            print(f"YOLO Detection Rate: {yolo_rate:.1f}%")
            print(f"YOLO Average Time: {avg_yolo_time:.1f}ms")
        print(f"Contour Detection Rate: {contour_rate:.1f}%")
        print(f"Contour Average Time: {avg_contour_time:.1f}ms")
        print(f"Recommendation: {recommendation}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO detection on test images")
    parser.add_argument("--test-dir", type=Path,
                        default=SCRIPT_DIR / "test_images",
                        help="Directory with test images")
    parser.add_argument("--output-dir", type=Path,
                        default=SCRIPT_DIR / "results",
                        help="Output directory for results")
    parser.add_argument("--create-ground-truth", action="store_true",
                        help="Interactively create ground truth labels")
    args = parser.parse_args()

    runner = ValidationRunner(args.test_dir, args.output_dir)

    if args.create_ground_truth:
        runner.create_ground_truth()
    else:
        if runner.run_validation():
            runner.generate_report()


if __name__ == "__main__":
    main()
