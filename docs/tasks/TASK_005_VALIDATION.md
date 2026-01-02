# Task 005: Validation

**Phase**: 5 of 5
**Assigned to**: @spike-worker
**Status**: Blocked by Task 004
**Priority**: High
**Depends on**: Task 004 (Integration)

---

## Objective

Validate the YOLO-based detection system on real webcam images across various conditions. Document performance, failure cases, and make final Go/No-Go recommendation.

---

## Deliverables

### Files to Create

| File | Description |
|------|-------------|
| `training/yolo/validate.py` | Validation script |
| `training/yolo/test_images/` | Real webcam test images |
| `training/yolo/results/validation_results.csv` | Per-image metrics |
| `training/yolo/results/validation_summary.md` | Summary report |
| `training/yolo/results/failure_cases/` | Annotated failure screenshots |

### Success Criteria

- [ ] 50+ real webcam test images captured
- [ ] Detection rate >= 95% on good lighting
- [ ] Detection rate >= 90% on typical lighting
- [ ] False positive rate < 5%
- [ ] FPS >= 15 (detection + identification)
- [ ] All failure cases documented

---

## Test Protocol

### Phase A: Image Collection (Manual)

Capture diverse test images covering:

| Scenario | Min Images | Description |
|----------|------------|-------------|
| Single card, playmat | 10 | Ideal conditions |
| Single card, wood table | 10 | Common surface |
| Single card, cluttered desk | 5 | Challenging background |
| Multiple cards (2-4) | 10 | Stacked/adjacent |
| Hand-held card | 5 | In motion |
| Dim lighting | 5 | Low light |
| Harsh lighting | 5 | Direct light, shadows |
| **Total** | **50+** | |

### Phase B: Automated Validation

Run detection on all collected images and measure:
- Detection rate (true positives)
- False positive rate
- Processing time
- Confidence scores

### Phase C: Comparison

Compare YOLO vs contour detection on same images:
- Detection rate improvement
- Speed improvement
- Failure case analysis

---

## Implementation

### Image Collection Script

```python
#!/usr/bin/env python3
"""
Collect test images from webcam for validation.

Usage:
    python collect_images.py --camera 0 --output training/yolo/test_images
"""

import argparse
import cv2
from pathlib import Path
from datetime import datetime


def collect_images(camera_id: int, output_dir: Path, resolution: tuple = (1280, 720)):
    """Interactive image collection from webcam."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return

    print("=" * 60)
    print("Test Image Collection")
    print("=" * 60)
    print("Controls:")
    print("  SPACE - Capture image")
    print("  1-7   - Tag image category:")
    print("          1=playmat, 2=wood, 3=clutter, 4=multiple")
    print("          5=handheld, 6=dim, 7=bright")
    print("  Q     - Quit")
    print("=" * 60)

    categories = {
        '1': 'playmat',
        '2': 'wood',
        '3': 'clutter',
        '4': 'multiple',
        '5': 'handheld',
        '6': 'dim',
        '7': 'bright',
    }

    current_category = 'playmat'
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display with info
        display = frame.copy()
        cv2.putText(display, f"Category: {current_category}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Captured: {count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "SPACE to capture, Q to quit", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Collect Test Images", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{current_category}_{timestamp}_{count:03d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            print(f"Saved: {filename}")
            count += 1
        elif chr(key) in categories:
            current_category = categories[chr(key)]
            print(f"Category: {current_category}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal images captured: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("training/yolo/test_images"))
    args = parser.parse_args()

    collect_images(args.camera, args.output)


if __name__ == "__main__":
    main()
```

### Validation Script

```python
#!/usr/bin/env python3
"""
Validate YOLO detection on test images.

Usage:
    python validate.py --test-dir training/yolo/test_images
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Import detectors
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "spike"))

from yolo_detector import YOLODetector
from inference import CardDetector


class ValidationRunner:
    """Run validation on test images."""

    def __init__(self, test_dir: Path, output_dir: Path):
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize detectors
        self.yolo_detector = YOLODetector()
        self.contour_detector = CardDetector()

        # Results storage
        self.results = []

    def load_ground_truth(self) -> Dict[str, int]:
        """
        Load ground truth card counts.

        Ground truth file format (ground_truth.json):
        {
            "playmat_20260102_123456_001.jpg": {"num_cards": 1, "notes": ""},
            ...
        }

        If no ground truth file, prompt user to create one.
        """
        gt_path = self.test_dir / "ground_truth.json"

        if gt_path.exists():
            with open(gt_path) as f:
                return json.load(f)

        # Create ground truth interactively
        print("No ground truth file found. Creating interactively...")
        ground_truth = {}

        for img_path in sorted(self.test_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            cv2.imshow("Count cards (press 0-9)", img)
            print(f"Image: {img_path.name}")
            print("Press 0-9 for number of cards visible, or 's' to skip")

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('s'):
                    break
                elif ord('0') <= key <= ord('9'):
                    num_cards = key - ord('0')
                    ground_truth[img_path.name] = {"num_cards": num_cards, "notes": ""}
                    print(f"  -> {num_cards} cards")
                    break

        cv2.destroyAllWindows()

        # Save ground truth
        with open(gt_path, "w") as f:
            json.dump(ground_truth, f, indent=2)

        print(f"Ground truth saved to {gt_path}")
        return ground_truth

    def run_validation(self):
        """Run validation on all test images."""
        ground_truth = self.load_ground_truth()

        print("\n" + "=" * 60)
        print("Running Validation")
        print("=" * 60)

        for img_path in sorted(self.test_dir.glob("*.jpg")):
            if img_path.name not in ground_truth:
                continue

            gt = ground_truth[img_path.name]
            expected_cards = gt["num_cards"]

            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Parse category from filename
            category = img_path.stem.split("_")[0]

            # Test YOLO detection
            yolo_start = time.perf_counter()
            yolo_detections = self.yolo_detector.detect(img)
            yolo_time = (time.perf_counter() - yolo_start) * 1000

            # Test contour detection
            contour_start = time.perf_counter()
            contour_detections = self.contour_detector.detect(img)
            contour_time = (time.perf_counter() - contour_start) * 1000

            # Calculate metrics
            yolo_correct = len(yolo_detections) == expected_cards
            contour_correct = len(contour_detections) == expected_cards

            result = {
                "filename": img_path.name,
                "category": category,
                "expected_cards": expected_cards,
                "yolo_detected": len(yolo_detections),
                "yolo_correct": yolo_correct,
                "yolo_time_ms": round(yolo_time, 1),
                "contour_detected": len(contour_detections),
                "contour_correct": contour_correct,
                "contour_time_ms": round(contour_time, 1),
            }

            self.results.append(result)

            # Log result
            status = "OK" if yolo_correct else "FAIL"
            print(f"{img_path.name}: YOLO={len(yolo_detections)}/{expected_cards} [{status}] "
                  f"Contour={len(contour_detections)}/{expected_cards} "
                  f"({yolo_time:.0f}ms/{contour_time:.0f}ms)")

            # Save failure case if YOLO failed
            if not yolo_correct:
                self._save_failure_case(img, yolo_detections, contour_detections,
                                        expected_cards, img_path.name)

    def _save_failure_case(self, img, yolo_dets, contour_dets, expected, filename):
        """Save annotated failure case."""
        failure_dir = self.output_dir / "failure_cases"
        failure_dir.mkdir(exist_ok=True)

        # Create annotated image
        annotated = img.copy()

        # Draw YOLO detections in green
        for contour, corners in yolo_dets:
            cv2.polylines(annotated, [contour], True, (0, 255, 0), 2)

        # Draw contour detections in blue
        for contour, corners in contour_dets:
            cv2.polylines(annotated, [contour], True, (255, 0, 0), 2)

        # Add text
        cv2.putText(annotated, f"Expected: {expected}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated, f"YOLO (green): {len(yolo_dets)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Contour (blue): {len(contour_dets)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imwrite(str(failure_dir / f"fail_{filename}"), annotated)

    def generate_report(self):
        """Generate validation report."""
        if not self.results:
            print("No results to report")
            return

        # Save raw results
        csv_path = self.output_dir / "validation_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        print(f"\nResults saved to {csv_path}")

        # Calculate summary statistics
        total = len(self.results)
        yolo_correct = sum(1 for r in self.results if r["yolo_correct"])
        contour_correct = sum(1 for r in self.results if r["contour_correct"])

        yolo_times = [r["yolo_time_ms"] for r in self.results]
        contour_times = [r["contour_time_ms"] for r in self.results]

        # Per-category breakdown
        categories = set(r["category"] for r in self.results)
        category_stats = {}
        for cat in categories:
            cat_results = [r for r in self.results if r["category"] == cat]
            cat_yolo = sum(1 for r in cat_results if r["yolo_correct"])
            category_stats[cat] = {
                "total": len(cat_results),
                "yolo_correct": cat_yolo,
                "yolo_rate": cat_yolo / len(cat_results) * 100 if cat_results else 0,
            }

        # Generate markdown report
        report = f"""# YOLO Detection Validation Report

**Date:** {time.strftime('%Y-%m-%d %H:%M')}
**Test Images:** {total}

---

## Overall Results

| Metric | YOLO | Contour | Target |
|--------|------|---------|--------|
| Accuracy | {yolo_correct/total*100:.1f}% | {contour_correct/total*100:.1f}% | >= 95% |
| Avg Time | {np.mean(yolo_times):.1f}ms | {np.mean(contour_times):.1f}ms | < 50ms |
| Min Time | {min(yolo_times):.1f}ms | {min(contour_times):.1f}ms | - |
| Max Time | {max(yolo_times):.1f}ms | {max(contour_times):.1f}ms | - |

---

## Results by Category

| Category | Images | YOLO Correct | YOLO Rate | Status |
|----------|--------|--------------|-----------|--------|
"""
        for cat, stats in sorted(category_stats.items()):
            status = "PASS" if stats["yolo_rate"] >= 90 else "FAIL"
            report += f"| {cat} | {stats['total']} | {stats['yolo_correct']} | {stats['yolo_rate']:.0f}% | {status} |\n"

        # Add failure analysis
        failures = [r for r in self.results if not r["yolo_correct"]]
        if failures:
            report += f"""

---

## Failure Cases ({len(failures)} failures)

| Filename | Expected | YOLO Detected | Category |
|----------|----------|---------------|----------|
"""
            for f in failures:
                report += f"| {f['filename']} | {f['expected_cards']} | {f['yolo_detected']} | {f['category']} |\n"

            report += f"""

See `failure_cases/` directory for annotated screenshots.
"""

        # Final recommendation
        yolo_rate = yolo_correct / total * 100
        avg_time = np.mean(yolo_times)

        if yolo_rate >= 95 and avg_time < 50:
            recommendation = "GO"
            details = "All targets met. YOLO detection is ready for production."
        elif yolo_rate >= 90 and avg_time < 100:
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
| Detection Rate | {yolo_rate:.1f}% | >= 95% | {"PASS" if yolo_rate >= 95 else "FAIL"} |
| Avg Inference Time | {avg_time:.1f}ms | < 50ms | {"PASS" if avg_time < 50 else "FAIL"} |
| False Positive Rate | - | < 5% | (Manual review needed) |

---

## Next Steps

"""
        if recommendation == "GO":
            report += """1. Deploy YOLO detection as default
2. Remove contour detection fallback after burn-in period
3. Monitor production metrics
"""
        elif recommendation == "CONDITIONAL GO":
            report += """1. Deploy YOLO detection with contour fallback
2. Log failure cases in production
3. Collect more training data from failures
4. Retrain after collecting 100+ failure cases
"""
        else:
            report += """1. Analyze failure cases in `failure_cases/` directory
2. Identify patterns in failures
3. Generate more synthetic data for failure categories
4. Retrain with augmented dataset
5. Re-run validation
"""

        # Save report
        report_path = self.output_dir / "validation_summary.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report saved to {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"YOLO Detection Rate: {yolo_rate:.1f}%")
        print(f"Average Time: {avg_time:.1f}ms")
        print(f"Recommendation: {recommendation}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path,
                        default=Path("training/yolo/test_images"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("training/yolo/results"))
    args = parser.parse_args()

    runner = ValidationRunner(args.test_dir, args.output_dir)
    runner.run_validation()
    runner.generate_report()


if __name__ == "__main__":
    main()
```

---

## Validation Report Format

The generated `validation_summary.md` will look like:

```markdown
# YOLO Detection Validation Report

**Date:** 2026-01-02 15:30
**Test Images:** 54

---

## Overall Results

| Metric | YOLO | Contour | Target |
|--------|------|---------|--------|
| Accuracy | 96.3% | 78.5% | >= 95% |
| Avg Time | 28.5ms | 45.2ms | < 50ms |
| Min Time | 22.1ms | 35.0ms | - |
| Max Time | 45.3ms | 120.5ms | - |

---

## Results by Category

| Category | Images | YOLO Correct | YOLO Rate | Status |
|----------|--------|--------------|-----------|--------|
| playmat | 12 | 12 | 100% | PASS |
| wood | 10 | 10 | 100% | PASS |
| clutter | 8 | 6 | 75% | FAIL |
| multiple | 10 | 9 | 90% | PASS |
| handheld | 5 | 5 | 100% | PASS |
| dim | 5 | 5 | 100% | PASS |
| bright | 4 | 4 | 100% | PASS |

---

## Recommendation

**CONDITIONAL GO**

Performance acceptable. Monitor failure cases in production.
```

---

## Success Metrics Summary

| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| Detection rate (playmat) | >= 98% | >= 95% |
| Detection rate (wood/desk) | >= 95% | >= 90% |
| Detection rate (clutter) | >= 85% | >= 75% |
| False positive rate | < 2% | < 5% |
| Average detection time | < 30ms | < 50ms |
| FPS (detection + ID) | >= 20 | >= 15 |

---

## Time Estimate

- Image collection: 30-45 minutes (manual)
- Ground truth labeling: 15 minutes
- Validation run: 10 minutes
- Report review: 15 minutes
- **Total: 1-1.5 hours**

---

## Notes for Worker

1. **Diverse conditions**: Collect images in various lighting and backgrounds
2. **Include edge cases**: Partially visible, rotated, overlapping cards
3. **Ground truth accuracy**: Be precise when counting cards
4. **Analyze failures**: Look for patterns in failure cases
5. **Document everything**: Notes in ground_truth.json help future analysis

---

## Final Deliverable

After validation, update `docs/YOLO_DETECTION_PLAN.md` with:
- Final metrics achieved
- Go/No-Go recommendation
- Any caveats or known limitations
- Path to production deployment
