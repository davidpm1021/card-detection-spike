# Task 004: Integration

**Phase**: 4 of 5
**Assigned to**: @spike-worker
**Status**: Blocked by Task 003
**Priority**: High
**Depends on**: Task 003 (YOLO Training)

---

## Objective

Replace the fragile contour-based detection in `spike/inference.py` with the trained YOLO model while maintaining backward compatibility.

---

## Deliverables

### Files to Create

| File | Description |
|------|-------------|
| `spike/yolo_detector.py` | New YOLO-based detector class |

### Files to Modify

| File | Changes |
|------|---------|
| `spike/inference.py` | Add detector selection, use YOLODetector by default |

### Success Criteria

- [ ] YOLODetector class matches CardDetector interface
- [ ] `detect(frame)` returns same format as CardDetector
- [ ] Works with existing CardIdentifier
- [ ] WebcamInference works with either detector
- [ ] FPS >= 15 with YOLO detection + identification
- [ ] Fallback to contour detection if YOLO model missing

---

## Technical Specification

### Interface Compatibility

The existing `CardDetector.detect()` returns:
```python
List[Tuple[np.ndarray, np.ndarray]]
# Each tuple: (contour, corners)
# - contour: np.ndarray shape (N, 1, 2) - contour points
# - corners: np.ndarray shape (4, 2) - ordered corner points
```

The new `YOLODetector.detect()` must return the same format.

---

## Implementation

### yolo_detector.py

```python
#!/usr/bin/env python3
"""
YOLO-based MTG card detector.

Provides robust card detection using a trained YOLOv8 model.
Falls back to contour detection if YOLO model is not available.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Try to import ultralytics, fall back gracefully
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Try ONNX Runtime for CPU inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class YOLODetector:
    """
    YOLO-based card detector.

    Provides same interface as CardDetector for drop-in replacement.
    Uses ONNX Runtime for fast CPU inference when available.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        use_onnx: bool = True,
    ):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model (.pt or .onnx)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            use_onnx: Prefer ONNX runtime if available
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.onnx_session = None
        self.use_onnx = use_onnx
        self.input_size = 640

        # Find model path
        if model_path is None:
            model_path = self._find_model()

        if model_path is None:
            print("Warning: No YOLO model found. Detection will fail.")
            return

        model_path = Path(model_path)

        # Load model
        if use_onnx and ONNX_AVAILABLE and model_path.suffix == ".onnx":
            self._load_onnx(model_path)
        elif use_onnx and ONNX_AVAILABLE and model_path.with_suffix(".onnx").exists():
            self._load_onnx(model_path.with_suffix(".onnx"))
        elif YOLO_AVAILABLE:
            self._load_yolo(model_path)
        else:
            print("Warning: Neither ultralytics nor onnxruntime available.")
            print("Install with: pip install ultralytics onnxruntime")

    def _find_model(self) -> Optional[Path]:
        """Find YOLO model in expected locations."""
        script_dir = Path(__file__).parent
        search_paths = [
            script_dir.parent / "training" / "yolo" / "runs" / "detect" / "train" / "weights" / "best.onnx",
            script_dir.parent / "training" / "yolo" / "runs" / "detect" / "train" / "weights" / "best.pt",
            script_dir / "models" / "card_detector.onnx",
            script_dir / "models" / "card_detector.pt",
        ]

        for path in search_paths:
            if path.exists():
                print(f"Found YOLO model: {path}")
                return path

        return None

    def _load_onnx(self, model_path: Path):
        """Load ONNX model for inference."""
        print(f"Loading ONNX model: {model_path}")
        self.onnx_session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_names = [o.name for o in self.onnx_session.get_outputs()]

        # Get input shape
        input_shape = self.onnx_session.get_inputs()[0].shape
        if len(input_shape) == 4:
            self.input_size = input_shape[2]  # Assuming NCHW format

        print(f"ONNX model loaded (input size: {self.input_size})")

    def _load_yolo(self, model_path: Path):
        """Load YOLO model using ultralytics."""
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        print("YOLO model loaded")

    def detect(
        self,
        frame: np.ndarray,
        fast_mode: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect cards in frame.

        Args:
            frame: Input BGR image
            fast_mode: Ignored (always fast with YOLO)

        Returns:
            List of (contour, corners) tuples for detected cards
        """
        if self.onnx_session is not None:
            return self._detect_onnx(frame)
        elif self.model is not None:
            return self._detect_yolo(frame)
        else:
            return []

    def _detect_yolo(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Detect using ultralytics YOLO."""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        cards = []
        for box in results.boxes:
            # Get bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Create corners (clockwise from top-left)
            corners = np.array([
                [x1, y1],  # top-left
                [x2, y1],  # top-right
                [x2, y2],  # bottom-right
                [x1, y2],  # bottom-left
            ], dtype=np.float32)

            # Create contour (same as corners for rectangles)
            contour = corners.reshape(-1, 1, 2).astype(np.int32)

            cards.append((contour, corners))

        return cards

    def _detect_onnx(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Detect using ONNX Runtime."""
        # Preprocess
        img, ratio, (pad_w, pad_h) = self._preprocess(frame)

        # Inference
        outputs = self.onnx_session.run(self.output_names, {self.input_name: img})

        # Postprocess
        boxes = self._postprocess(outputs, ratio, pad_w, pad_h, frame.shape[:2])

        # Convert to card format
        cards = []
        for x1, y1, x2, y2, conf in boxes:
            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ], dtype=np.float32)

            contour = corners.reshape(-1, 1, 2).astype(np.int32)
            cards.append((contour, corners))

        return cards

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for ONNX inference."""
        # Get original size
        h, w = frame.shape[:2]

        # Calculate scaling ratio
        ratio = min(self.input_size / w, self.input_size / h)

        # Resize
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        resized = cv2.resize(frame, (new_w, new_h))

        # Pad to input size
        pad_w = (self.input_size - new_w) // 2
        pad_h = (self.input_size - new_h) // 2

        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Convert to float, normalize, transpose
        img = padded.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, 0)  # Add batch dimension

        return img, ratio, (pad_w, pad_h)

    def _postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float,
        pad_w: int,
        pad_h: int,
        original_shape: Tuple[int, int]
    ) -> List[Tuple[float, float, float, float, float]]:
        """Postprocess ONNX outputs."""
        # YOLOv8 output format: [batch, num_detections, 5 + num_classes]
        # For single class: [batch, num_detections, 6] where cols are [x, y, w, h, conf, class_conf]

        predictions = outputs[0]  # Shape: [1, num_boxes, 5] or similar

        # Handle different output formats
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension

        # YOLOv8 outputs: shape is typically (8400, 5) for 640x640 with 1 class
        # Columns: [x_center, y_center, width, height, class_conf]
        if predictions.shape[1] == 5:
            # Standard format
            pass
        elif predictions.shape[0] == 5:
            # Transposed format
            predictions = predictions.T

        boxes = []
        orig_h, orig_w = original_shape

        for pred in predictions:
            if len(pred) >= 5:
                x_center, y_center, width, height = pred[:4]
                conf = pred[4] if len(pred) == 5 else pred[4] * pred[5]

                if conf < self.conf_threshold:
                    continue

                # Convert from center to corner format
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                # Remove padding and scale back
                x1 = (x1 - pad_w) / ratio
                y1 = (y1 - pad_h) / ratio
                x2 = (x2 - pad_w) / ratio
                y2 = (y2 - pad_h) / ratio

                # Clip to image bounds
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))

                # Filter small boxes
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    boxes.append((x1, y1, x2, y2, conf))

        # Apply NMS
        if boxes:
            boxes = self._nms(boxes)

        return boxes

    def _nms(
        self,
        boxes: List[Tuple[float, float, float, float, float]]
    ) -> List[Tuple[float, float, float, float, float]]:
        """Non-maximum suppression."""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        keep = []

        while boxes:
            best = boxes.pop(0)
            keep.append(best)

            boxes = [
                box for box in boxes
                if self._iou(best[:4], box[:4]) < self.iou_threshold
            ]

        return keep

    def _iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def extract_card(self, frame: np.ndarray, corners: np.ndarray, target_size=(224, 224)):
        """
        Extract and warp card region to standard size.

        Same interface as CardDetector.extract_card().
        """
        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(corners)

        # Card dimensions: 63mm x 88mm
        card_aspect = 63 / 88
        target_h, target_w = target_size

        if target_w / target_h > card_aspect:
            new_w = int(target_h * card_aspect)
            new_h = target_h
        else:
            new_w = target_w
            new_h = int(target_w / card_aspect)

        dst_corners = np.array([
            [0, 0],
            [new_w - 1, 0],
            [new_w - 1, new_h - 1],
            [0, new_h - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(corners, dst_corners)
        warped = cv2.warpPerspective(frame, matrix, (new_w, new_h))
        warped = cv2.resize(warped, target_size)

        return warped

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners: top-left, top-right, bottom-right, bottom-left."""
        corners = corners.reshape(4, 2)

        # Sort by y-coordinate (top to bottom)
        corners = corners[np.argsort(corners[:, 1])]

        # Top two points
        top = corners[:2]
        top = top[np.argsort(top[:, 0])]

        # Bottom two points
        bottom = corners[2:]
        bottom = bottom[np.argsort(bottom[:, 0])]

        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def get_detector(prefer_yolo: bool = True) -> object:
    """
    Get the best available detector.

    Returns YOLODetector if model is available, else CardDetector.
    """
    if prefer_yolo:
        detector = YOLODetector()
        if detector.model is not None or detector.onnx_session is not None:
            return detector

    # Fall back to contour detector
    from inference import CardDetector
    print("Falling back to contour-based detection")
    return CardDetector()
```

---

### Modifications to inference.py

Add these changes to `spike/inference.py`:

#### 1. Add import at top

```python
# Add near top of file, after existing imports
from pathlib import Path

# Try to import YOLO detector
try:
    from yolo_detector import YOLODetector, get_detector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
```

#### 2. Modify WebcamInference.__init__

```python
class WebcamInference:
    """Real-time webcam inference pipeline."""

    def __init__(
        self,
        model_path,
        index_path,
        mapping_path,
        camera_id=0,
        resolution="1080p",
        fast_mode=False,
        use_yolo=True,  # NEW PARAMETER
    ):
        # Initialize detector
        if use_yolo and YOLO_AVAILABLE:
            self.detector = get_detector(prefer_yolo=True)
            self.using_yolo = isinstance(self.detector, YOLODetector)
        else:
            self.detector = CardDetector()
            self.using_yolo = False

        if self.using_yolo:
            print("Using YOLO detection")
        else:
            print("Using contour detection")

        # Rest of existing __init__ code...
        self.identifier = CardIdentifier(model_path, index_path, mapping_path)
        # ... etc
```

#### 3. Add CLI argument

```python
def main():
    parser = argparse.ArgumentParser(description="MTG Card Detection and Identification")
    # ... existing arguments ...

    # Add new argument
    parser.add_argument("--no-yolo", action="store_true",
                       help="Disable YOLO detection, use contour-based instead")

    args = parser.parse_args()

    # ... existing code ...

    if args.image:
        # Test on single image
        test_on_image(args.image, args.model, args.index, args.mapping)
    else:
        # Run webcam inference
        pipeline = WebcamInference(
            args.model, args.index, args.mapping,
            args.camera, args.resolution, fast_mode=args.fast,
            use_yolo=not args.no_yolo,  # NEW
        )
        pipeline.run()
```

---

## Testing

### Test 1: Detector Interface Compatibility

```python
# test_detector.py
import cv2
import numpy as np
from yolo_detector import YOLODetector
from inference import CardDetector

def test_interface_compatibility():
    """Ensure YOLODetector has same interface as CardDetector."""
    # Create dummy image
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test CardDetector
    cd = CardDetector()
    cd_results = cd.detect(frame)

    # Test YOLODetector
    yd = YOLODetector()
    yd_results = yd.detect(frame)

    # Verify return format
    for results in [cd_results, yd_results]:
        for contour, corners in results:
            assert isinstance(contour, np.ndarray)
            assert isinstance(corners, np.ndarray)
            assert corners.shape == (4, 2)

    print("Interface compatibility: PASS")

if __name__ == "__main__":
    test_interface_compatibility()
```

### Test 2: End-to-End Webcam Test

```bash
# Test with YOLO (default)
python spike/inference.py --camera 0

# Test without YOLO (contour fallback)
python spike/inference.py --camera 0 --no-yolo
```

### Test 3: Performance Comparison

```python
# benchmark.py
import time
import cv2
import numpy as np
from yolo_detector import YOLODetector
from inference import CardDetector

def benchmark_detectors(num_frames=100):
    """Compare detection speed."""
    # Load sample frame
    frame = cv2.imread("test_image.jpg")
    if frame is None:
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Benchmark CardDetector
    cd = CardDetector()
    start = time.perf_counter()
    for _ in range(num_frames):
        cd.detect(frame)
    cd_time = (time.perf_counter() - start) / num_frames * 1000

    # Benchmark YOLODetector
    yd = YOLODetector()
    start = time.perf_counter()
    for _ in range(num_frames):
        yd.detect(frame)
    yd_time = (time.perf_counter() - start) / num_frames * 1000

    print(f"CardDetector (contour): {cd_time:.1f}ms per frame")
    print(f"YOLODetector:           {yd_time:.1f}ms per frame")
    print(f"Speedup: {cd_time / yd_time:.1f}x")

if __name__ == "__main__":
    benchmark_detectors()
```

---

## Console Output Expected

```
========================================
MTG Card Detection Running!
========================================

Found YOLO model: training/yolo/runs/detect/train/weights/best.onnx
Loading ONNX model: training/yolo/runs/detect/train/weights/best.onnx
ONNX model loaded (input size: 640)
Using YOLO detection
Loading model from training/checkpoints/final_model.pt...
Model loaded (embedding_dim=512)
Loading FAISS index from training/output/card_embeddings_full.faiss...
Index loaded (32062 cards)

Resolution: 1280x720
- Hold a card in front of the camera
- Press 'q' to quit
- Press 's' to save screenshot
========================================

FPS: 22.5 | Cards: 2 | Detection: 28ms | Identification: 85ms
```

---

## Dependencies

No new dependencies required. Uses existing:
- `ultralytics` (if available)
- `onnxruntime` (if available)
- `opencv-python`
- `numpy`

---

## Time Estimate

- YOLODetector implementation: 1-2 hours
- inference.py modifications: 30 minutes
- Testing: 30 minutes
- **Total: 2-3 hours**

---

## Notes for Worker

1. **Preserve backward compatibility**: Old code should still work
2. **Graceful fallback**: If YOLO unavailable, use contour detector
3. **ONNX preferred**: ONNX is faster than PyTorch for CPU inference
4. **Same interface**: Must match CardDetector's return format exactly
5. **Test both paths**: Test with and without `--no-yolo` flag

---

## Next Task

After completing this task, proceed to:
- **Task 005: Validation** (Phase 5)
