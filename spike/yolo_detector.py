"""
YOLO-based card detector for MTG cards.

Replaces fragile contour-based detection with robust YOLOv8 object detection.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class YOLODetector:
    """
    Detects MTG cards using a trained YOLOv8 model.

    Falls back to ONNX Runtime if available, otherwise uses ultralytics.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to YOLO model (.pt or .onnx)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.use_onnx = False

        # Default model path
        if model_path is None:
            script_dir = Path(__file__).parent
            training_dir = script_dir.parent / "training" / "yolo"

            # Try ONNX first (faster, no PyTorch needed)
            onnx_path = training_dir / "runs" / "detect" / "train" / "weights" / "best.onnx"
            pt_path = training_dir / "runs" / "detect" / "train" / "weights" / "best.pt"

            if onnx_path.exists():
                model_path = str(onnx_path)
            elif pt_path.exists():
                model_path = str(pt_path)
            else:
                raise FileNotFoundError(
                    f"No trained YOLO model found. Expected at:\n"
                    f"  {onnx_path}\n"
                    f"  or {pt_path}\n"
                    f"Run: python training/yolo/train.py"
                )

        self.model_path = Path(model_path)
        self._load_model()

    def _load_model(self):
        """Load the YOLO model."""
        if self.model_path.suffix == ".onnx":
            self._load_onnx()
        else:
            self._load_ultralytics()

    def _load_onnx(self):
        """Load ONNX model using ONNX Runtime."""
        try:
            import onnxruntime as ort

            print(f"Loading YOLO model (ONNX): {self.model_path}")
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 640, 640]
            self.use_onnx = True
            print("YOLO model loaded (ONNX Runtime)")

        except ImportError:
            print("ONNX Runtime not available, falling back to ultralytics")
            # Fall back to .pt model
            pt_path = self.model_path.with_suffix(".pt")
            if pt_path.exists():
                self.model_path = pt_path
                self._load_ultralytics()
            else:
                raise ImportError("Neither onnxruntime nor ultralytics available")

    def _load_ultralytics(self):
        """Load model using ultralytics YOLO."""
        try:
            from ultralytics import YOLO

            print(f"Loading YOLO model (ultralytics): {self.model_path}")
            self.model = YOLO(str(self.model_path))
            self.use_onnx = False
            print("YOLO model loaded (ultralytics)")

        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

    def detect(
        self,
        frame: np.ndarray,
        fast_mode: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Detect cards in frame.

        Args:
            frame: BGR image (numpy array)
            fast_mode: Not used (kept for API compatibility with CardDetector)

        Returns:
            List of (contour, corners) tuples matching CardDetector interface:
            - contour: np.ndarray of shape (N, 1, 2) - card contour points
            - corners: np.ndarray of shape (4, 2) - ordered corner points
        """
        if self.use_onnx:
            return self._detect_onnx(frame)
        else:
            return self._detect_ultralytics(frame)

    def _detect_ultralytics(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Detect using ultralytics YOLO."""
        # Run inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]

        cards = []
        for box in results.boxes:
            # Get bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Create corners (ordered: top-left, top-right, bottom-right, bottom-left)
            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)

            # Create contour (same as corners but in cv2 contour format)
            contour = corners.reshape((-1, 1, 2)).astype(np.int32)

            cards.append((contour, corners))

        return cards

    def _detect_onnx(self, frame: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Detect using ONNX Runtime."""
        import cv2

        # Preprocess
        h, w = frame.shape[:2]
        input_h, input_w = 640, 640

        # Resize maintaining aspect ratio with letterbox
        scale = min(input_w / w, input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        # Create letterbox image
        input_img = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2
        input_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

        # Convert to model input format
        input_img = input_img.transpose(2, 0, 1)  # HWC -> CHW
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, 0)  # Add batch dimension

        # Run inference
        outputs = self.session.run(None, {self.input_name: input_img})[0]

        # Process outputs (YOLOv8 format: [1, 84, 8400] or similar)
        # Transpose to [1, 8400, 84] for easier processing
        outputs = outputs.transpose(0, 2, 1)

        cards = []
        for detection in outputs[0]:
            # YOLOv8 output: [x_center, y_center, width, height, class_scores...]
            x_center, y_center, width, height = detection[:4]
            scores = detection[4:]

            # Get class confidence
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence < self.conf_threshold:
                continue

            # Convert to corner coordinates
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Remove letterbox padding and scale back to original size
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # Clip to image bounds
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            # Create corners
            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)

            contour = corners.reshape((-1, 1, 2)).astype(np.int32)
            cards.append((contour, corners))

        # Apply NMS if multiple detections
        if len(cards) > 1:
            cards = self._apply_nms(cards)

        return cards

    def _apply_nms(
        self,
        cards: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if len(cards) <= 1:
            return cards

        # Get bounding boxes
        boxes = []
        for contour, corners in cards:
            x1, y1 = corners[0]
            x2, y2 = corners[2]
            boxes.append([x1, y1, x2, y2])

        boxes = np.array(boxes)

        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Sort by area (larger first)
        order = areas.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            intersection = w * h
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union

            # Keep boxes with IoU below threshold
            mask = iou <= self.iou_threshold
            order = order[1:][mask]

        return [cards[i] for i in keep]


def test_detector():
    """Test the YOLO detector on a sample image."""
    import cv2

    # Initialize detector
    try:
        detector = YOLODetector()
    except FileNotFoundError as e:
        print(f"Cannot test: {e}")
        return

    # Test on webcam frame or sample image
    print("\nTesting YOLO detector...")

    # Try to get a frame from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        print(f"Frame size: {frame.shape}")

        import time
        start = time.time()
        cards = detector.detect(frame)
        elapsed = time.time() - start

        print(f"Detection time: {elapsed*1000:.1f}ms")
        print(f"Cards detected: {len(cards)}")

        for i, (contour, corners) in enumerate(cards):
            print(f"  Card {i+1}: corners = {corners.tolist()}")
    else:
        print("Could not get webcam frame for testing")


if __name__ == "__main__":
    test_detector()
