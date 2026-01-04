"""
Real-time MTG card detection and identification from webcam.

This script:
1. Captures frames from webcam
2. Detects card regions using contour detection
3. Identifies cards using the trained embedding model + FAISS index
4. Displays results in real-time

Usage:
    python inference.py              # Use default webcam
    python inference.py --camera 1   # Use specific camera
    python inference.py --image path/to/image.jpg  # Test on single image
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Try to import ONNX Runtime
ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    pass

# Add training directory to path for model import
SCRIPT_DIR = Path(__file__).parent
TRAINING_DIR = SCRIPT_DIR.parent / "training"
sys.path.insert(0, str(TRAINING_DIR))

from model import CardEmbeddingModel

# Try to import YOLO detector
YOLO_AVAILABLE = False
try:
    from yolo_detector import YOLODetector
    # Check if trained model exists
    yolo_model_path = TRAINING_DIR / "yolo" / "runs" / "detect" / "train" / "weights" / "best.pt"
    yolo_onnx_path = TRAINING_DIR / "yolo" / "runs" / "detect" / "train" / "weights" / "best.onnx"
    YOLO_AVAILABLE = yolo_model_path.exists() or yolo_onnx_path.exists()
except ImportError:
    pass


class CardDetector:
    """Detects card regions in images using contour detection."""

    def __init__(self, min_area=15000, max_area=400000):
        # Increased min_area to filter out small false positives
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame, fast_mode=True):
        """
        Detect card-like rectangles in the frame.

        Uses CLAHE in LAB color space for robust detection across lighting conditions.

        Args:
            frame: Input BGR image
            fast_mode: If True, use fast single-method detection (for real-time)
                      If False, use all methods (for thorough detection)

        Returns:
            List of (contour, corners) tuples for detected cards
        """
        kernel = np.ones((3, 3), np.uint8)

        # Preprocess with CLAHE in LAB color space (standard approach for card detection)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)

        # Also try on individual RGB channels for colored borders
        b, g, r = cv2.split(frame)

        if fast_mode:
            # Try CLAHE-enhanced luminance first
            blurred = cv2.GaussianBlur(enhanced_l, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            edges = cv2.dilate(edges, kernel, iterations=2)
            cards = self._find_cards_in_edges(edges)

            if not cards:
                # Try adaptive threshold on enhanced image
                adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
                edges = cv2.Canny(adaptive, 30, 100)
                edges = cv2.dilate(edges, kernel, iterations=2)
                cards = self._find_cards_in_edges(edges)

            if not cards:
                # Try blue channel (good for black-bordered cards)
                blurred_b = cv2.GaussianBlur(b, (5, 5), 0)
                edges = cv2.Canny(blurred_b, 30, 100)
                edges = cv2.dilate(edges, kernel, iterations=2)
                cards = self._find_cards_in_edges(edges)

            return cards
        else:
            # Thorough mode: try multiple channels and methods
            cards = []

            # Method 1: CLAHE-enhanced luminance
            blurred = cv2.GaussianBlur(enhanced_l, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            edges = cv2.dilate(edges, kernel, iterations=2)
            cards.extend(self._find_cards_in_edges(edges))

            # Method 2: Adaptive threshold on CLAHE
            adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
            edges = cv2.Canny(adaptive, 30, 100)
            edges = cv2.dilate(edges, kernel, iterations=2)
            cards.extend(self._find_cards_in_edges(edges))

            # Method 3-5: Individual RGB channels
            for channel in [b, g, r]:
                blurred_ch = cv2.GaussianBlur(channel, (5, 5), 0)
                edges = cv2.Canny(blurred_ch, 30, 100)
                edges = cv2.dilate(edges, kernel, iterations=2)
                cards.extend(self._find_cards_in_edges(edges))

            # Method 6: Otsu thresholding
            _, otsu = cv2.threshold(enhanced_l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            edges = cv2.Canny(otsu, 50, 150)
            edges = cv2.dilate(edges, kernel, iterations=2)
            cards.extend(self._find_cards_in_edges(edges))

            return self._remove_duplicates(cards)

    def _get_edge_maps(self, frame, blurred, gray, kernel):
        """Generate multiple edge detection maps for robustness."""
        edge_maps = []

        # Method 1: Standard Canny on blurred image
        edges1 = cv2.Canny(blurred, 50, 150)
        edges1 = cv2.dilate(edges1, kernel, iterations=2)
        edge_maps.append(edges1)

        # Method 2: Adaptive threshold (better for varying lighting/sleeves)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        edges2 = cv2.Canny(adaptive, 50, 150)
        edges2 = cv2.dilate(edges2, kernel, iterations=2)
        edge_maps.append(edges2)

        # Method 3: Lower Canny threshold (catches more edges, good for sleeves)
        edges3 = cv2.Canny(blurred, 30, 100)
        edges3 = cv2.dilate(edges3, kernel, iterations=3)
        edge_maps.append(edges3)

        # Method 4: Bilateral filter + Canny (preserves edges, reduces noise)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        edges4 = cv2.Canny(bilateral, 50, 150)
        edges4 = cv2.dilate(edges4, kernel, iterations=2)
        edge_maps.append(edges4)

        return edge_maps

    def _find_cards_in_edges(self, edges):
        """Find card contours in an edge map."""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cards = []
        card_parts = []  # Collect potential card parts (text box, art box)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            # Try different approximation tolerances
            peri = cv2.arcLength(contour, True)
            for eps in [0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(contour, eps * peri, True)

                # Look for quadrilaterals (4 corners)
                if len(approx) == 4:
                    corners = approx.reshape(4, 2)
                    if self._is_card_shaped(corners):
                        cards.append((contour, corners))
                        break
                    else:
                        # Might be a card part (text box, art box) - save for later
                        rect = cv2.minAreaRect(contour)
                        w, h = rect[1]
                        if w > 0 and h > 0:
                            aspect = min(w, h) / max(w, h)
                            min_dim = min(w, h)
                            # Text box is roughly 0.5-0.7 aspect, decent size
                            if 0.4 < aspect < 0.75 and min_dim > 60:
                                card_parts.append((contour, corners, rect))
                    break

            # Also try minimum area rectangle for rounded corners (sleeves)
            if len(approx) != 4:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                corners = np.array(box, dtype=np.float32)
                if self._is_card_shaped(corners):
                    # Verify the contour roughly fills the rectangle
                    rect_area = rect[1][0] * rect[1][1]
                    if rect_area > 0 and area / rect_area > 0.7:
                        cards.append((contour, corners))

        # If no full cards found but we have card parts, try to combine them
        if not cards and card_parts:
            # Try to find card parts that are vertically aligned (same card)
            if len(card_parts) >= 2:
                combined = self._combine_card_parts(card_parts)
                if combined is not None:
                    cards.append((card_parts[0][0], combined))

            # If still no card, expand the largest part
            if not cards:
                card_parts.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
                contour, corners, rect = card_parts[0]
                expanded = self._expand_to_card(rect)
                if expanded is not None:
                    cards.append((contour, expanded))

        return cards

    def _combine_card_parts(self, card_parts):
        """Combine multiple card parts (text box, art box) into full card."""
        # Get bounding boxes of all parts
        all_corners = []
        for contour, corners, rect in card_parts:
            all_corners.extend(corners.tolist())

        if len(all_corners) < 4:
            return None

        all_corners = np.array(all_corners)

        # Find bounding rectangle of all parts
        x_min, y_min = all_corners.min(axis=0)
        x_max, y_max = all_corners.max(axis=0)

        # Check if this forms a card-like shape
        width = x_max - x_min
        height = y_max - y_min

        if width == 0 or height == 0:
            return None

        aspect = min(width, height) / max(width, height)

        # Card aspect is ~0.716, allow some tolerance
        if 0.6 < aspect < 0.8:
            # Return corners of bounding box
            return np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ], dtype=np.float32)

        return None

    def _expand_to_card(self, rect):
        """Expand a detected card part (text box) to full card dimensions."""
        center, (w, h), angle = rect
        cx, cy = center

        # minAreaRect returns (width, height) where width is along the x-axis
        # For a text box that's wider than tall, we want to expand vertically

        # Text box is roughly 90% of card width and 35% of card height
        # Calculate full card dimensions
        text_box_width = max(w, h)  # The longer dimension is width
        text_box_height = min(w, h)  # The shorter dimension is height

        card_w = text_box_width / 0.88
        card_h = card_w / 0.716  # MTG card aspect ratio (63/88)

        # Text box center is roughly 65% down from top of card
        # So we need to shift UP by (0.65 - 0.5) * card_h = 0.15 * card_h
        shift_amount = card_h * 0.25

        # Shift up in image coordinates (decrease Y)
        # Account for rotation if any
        angle_rad = np.radians(angle)

        # For an upright card (angle ~0 or ~180), shift in Y direction
        # The text box is at bottom, so shift center toward top
        new_cy = cy - shift_amount  # Simple vertical shift for now
        new_cx = cx

        # Create corners manually for an axis-aligned rectangle
        # (ignoring rotation for simplicity - cards are usually upright)
        half_w = card_w / 2
        half_h = card_h / 2

        corners = np.array([
            [new_cx - half_w, new_cy - half_h],  # top-left
            [new_cx + half_w, new_cy - half_h],  # top-right
            [new_cx + half_w, new_cy + half_h],  # bottom-right
            [new_cx - half_w, new_cy + half_h],  # bottom-left
        ], dtype=np.float32)

        return corners

    def _remove_duplicates(self, cards):
        """Remove duplicate detections based on center proximity."""
        if len(cards) <= 1:
            return cards

        unique = []
        centers = []

        for contour, corners in cards:
            center = np.mean(corners, axis=0)

            # Check if this center is close to an existing one
            is_duplicate = False
            for existing_center in centers:
                dist = np.linalg.norm(center - existing_center)
                if dist < 50:  # Within 50 pixels = duplicate
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append((contour, corners))
                centers.append(center)

        return unique

    def _is_card_shaped(self, corners):
        """Check if corners form a card-like rectangle (aspect ratio ~0.716)."""
        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(corners)

        # Calculate width and height
        width = np.linalg.norm(corners[1] - corners[0])
        height = np.linalg.norm(corners[3] - corners[0])

        if width == 0 or height == 0:
            return False

        # Card aspect ratio is approximately 63mm x 88mm = 0.716
        # Tighter tolerance: 0.65-0.78 to avoid matching text boxes (~0.55-0.60)
        aspect = min(width, height) / max(width, height)
        if not (0.65 < aspect < 0.78):
            return False

        # Require minimum size (card should be reasonably large to identify)
        min_dim = min(width, height)
        if min_dim < 80:  # Too small to be useful
            return False

        return True

    def _order_corners(self, corners):
        """Order corners: top-left, top-right, bottom-right, bottom-left."""
        # Sort by y-coordinate (top to bottom)
        corners = corners[np.argsort(corners[:, 1])]

        # Top two points
        top = corners[:2]
        top = top[np.argsort(top[:, 0])]  # Sort by x (left to right)

        # Bottom two points
        bottom = corners[2:]
        bottom = bottom[np.argsort(bottom[:, 0])]  # Sort by x (left to right)

        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

    def extract_card(self, frame, corners, target_size=(224, 224)):
        """Extract and warp card region to standard size."""
        corners = self._order_corners(corners)

        # Target corners for perspective transform
        # Card dimensions: 63mm x 88mm, scaled to target size maintaining aspect
        card_aspect = 63 / 88
        target_h, target_w = target_size

        if target_w / target_h > card_aspect:
            # Width constrained
            new_w = int(target_h * card_aspect)
            new_h = target_h
        else:
            # Height constrained
            new_w = target_w
            new_h = int(target_w / card_aspect)

        dst_corners = np.array([
            [0, 0],
            [new_w - 1, 0],
            [new_w - 1, new_h - 1],
            [0, new_h - 1]
        ], dtype=np.float32)

        # Perspective transform
        matrix = cv2.getPerspectiveTransform(corners, dst_corners)
        warped = cv2.warpPerspective(frame, matrix, (new_w, new_h))

        # Resize to target size
        warped = cv2.resize(warped, target_size)

        return warped


class CardIdentifier:
    """Identifies cards using embedding model and FAISS index."""

    def __init__(self, model_path, index_path, mapping_path, device="cpu"):
        self.device = torch.device(device)

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model = CardEmbeddingModel(
            num_classes=checkpoint["num_classes"],
            embedding_dim=checkpoint["embedding_dim"],
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = checkpoint["embedding_dim"]
        print(f"Model loaded (embedding_dim={self.embedding_dim})")

        # Load FAISS index
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        print(f"Index loaded ({self.index.ntotal} cards)")

        # Load label mapping
        print(f"Loading label mapping from {mapping_path}...")
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        # Support both formats: "card_names" list or "idx_to_name" dict
        if "card_names" in mapping:
            self.card_names = mapping["card_names"]
        elif "idx_to_name" in mapping:
            # Convert idx_to_name dict to list
            idx_to_name = mapping["idx_to_name"]
            max_idx = max(int(k) for k in idx_to_name.keys())
            self.card_names = [""] * (max_idx + 1)
            for idx, name in idx_to_name.items():
                self.card_names[int(idx)] = name
        else:
            raise ValueError("Label mapping must contain 'card_names' or 'idx_to_name'")
        print(f"Loaded {len(self.card_names)} card names")

        # Setup transform
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    @torch.no_grad()
    def identify(self, card_image, top_k=3, try_rotations=True):
        """
        Identify a card from its image.

        Args:
            card_image: BGR image of the card (numpy array)
            top_k: Number of top matches to return
            try_rotations: Try 180-degree rotation and pick best

        Returns:
            List of (card_name, confidence) tuples
        """
        results = self._identify_single(card_image, top_k)

        # Always try 180-degree rotation and use whichever is better
        if try_rotations:
            rotated = cv2.rotate(card_image, cv2.ROTATE_180)
            rotated_results = self._identify_single(rotated, top_k)

            # Use whichever has higher confidence
            if rotated_results and (not results or rotated_results[0][1] > results[0][1]):
                results = rotated_results

        return results

    @torch.no_grad()
    def _identify_single(self, card_image, top_k=3):
        """Identify a single orientation of a card."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB)

        # Apply transform
        transformed = self.transform(image=rgb_image)
        image_tensor = transformed["image"].unsqueeze(0).to(self.device)

        # Get embedding
        embedding = self.model.get_embedding(image_tensor)
        embedding = embedding.cpu().numpy()

        # Normalize for cosine similarity
        faiss.normalize_L2(embedding)

        # Search index
        distances, indices = self.index.search(embedding, top_k)

        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.card_names):
                card_name = self.card_names[idx]
                # Convert distance to confidence (inner product after normalization = cosine similarity)
                confidence = float(dist)
                results.append((card_name, confidence))

        return results


class ONNXCardIdentifier:
    """Identifies cards using ONNX model and FAISS index (cross-platform consistent)."""

    def __init__(self, onnx_path, index_path, mapping_path):
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not installed. Install with: pip install onnxruntime")

        # Load ONNX model
        print(f"Loading ONNX model from {onnx_path}...")
        self.session = ort.InferenceSession(str(onnx_path))
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX model loaded")

        # Load FAISS index
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        print(f"Index loaded ({self.index.ntotal} cards)")

        # Load label mapping
        print(f"Loading label mapping from {mapping_path}...")
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if "card_names" in mapping:
            self.card_names = mapping["card_names"]
        elif "idx_to_name" in mapping:
            idx_to_name = mapping["idx_to_name"]
            max_idx = max(int(k) for k in idx_to_name.keys())
            self.card_names = [""] * (max_idx + 1)
            for idx, name in idx_to_name.items():
                self.card_names[int(idx)] = name
        else:
            raise ValueError("Label mapping must contain 'card_names' or 'idx_to_name'")
        print(f"Loaded {len(self.card_names)} card names")

        # Setup transform
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def identify(self, card_image, top_k=3, try_rotations=True):
        """Identify a card from its image."""
        results = self._identify_single(card_image, top_k)

        if try_rotations:
            rotated = cv2.rotate(card_image, cv2.ROTATE_180)
            rotated_results = self._identify_single(rotated, top_k)
            if rotated_results and (not results or rotated_results[0][1] > results[0][1]):
                results = rotated_results

        return results

    def _identify_single(self, card_image, top_k=3):
        """Identify a single orientation of a card."""
        rgb_image = cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=rgb_image)
        tensor = transformed["image"].numpy()
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

        # Get embedding from ONNX
        embedding = self.session.run(None, {self.input_name: tensor})[0]
        faiss.normalize_L2(embedding)

        # Search index
        distances, indices = self.index.search(embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.card_names):
                card_name = self.card_names[idx]
                confidence = float(dist)
                results.append((card_name, confidence))

        return results


class WebcamInference:
    """Real-time webcam inference pipeline."""

    def __init__(self, model_path, index_path, mapping_path, camera_id=0, resolution="1080p", fast_mode=False, use_yolo=None, use_onnx=True, onnx_path=None):
        # Select detector
        if use_yolo is None:
            use_yolo = YOLO_AVAILABLE  # Auto-detect

        if use_yolo and YOLO_AVAILABLE:
            print("Using YOLO detector (trained model found)")
            self.detector = YOLODetector()
            self.detector_type = "yolo"
        else:
            if use_yolo and not YOLO_AVAILABLE:
                print("YOLO model not found, falling back to contour detection")
            else:
                print("Using contour-based detection")
            self.detector = CardDetector()
            self.detector_type = "contour"

        # Select identifier (ONNX preferred for consistency with augmented index)
        if use_onnx and ONNX_AVAILABLE and onnx_path and Path(onnx_path).exists():
            print("Using ONNX identifier (cross-platform consistent)")
            self.identifier = ONNXCardIdentifier(onnx_path, index_path, mapping_path)
            self.identifier_type = "onnx"
        else:
            if use_onnx and not ONNX_AVAILABLE:
                print("ONNX Runtime not available, falling back to PyTorch")
            elif use_onnx and (not onnx_path or not Path(onnx_path).exists()):
                print("ONNX model not found, falling back to PyTorch")
            self.identifier = CardIdentifier(model_path, index_path, mapping_path)
            self.identifier_type = "pytorch"
        self.camera_id = camera_id
        self.resolution = resolution
        self.fast_mode = fast_mode
        self.try_rotations = not fast_mode  # Skip rotations in fast mode
        self.min_confidence = 0.5  # Filter out low-confidence matches

        # Resolution presets
        self.resolutions = {
            "480p": (640, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "1440p": (2560, 1440),
            "max": (2560, 1920),
        }

        # Performance tracking
        self.frame_times = []
        self.detection_times = []
        self.identification_times = []

        # Caching for performance
        self.cached_results = []  # Last identification results
        self.identify_interval = 3 if fast_mode else 5  # Re-identify every N frames
        self.last_card_centers = []  # Track card positions

    def _cards_are_stable(self, cards):
        """Check if detected cards are in similar positions to last frame."""
        if len(cards) != len(self.last_card_centers):
            return False

        current_centers = [np.mean(corners, axis=0) for _, corners in cards]

        for curr in current_centers:
            matched = False
            for prev in self.last_card_centers:
                if np.linalg.norm(curr - prev) < 30:  # Within 30 pixels
                    matched = True
                    break
            if not matched:
                return False

        return True

    def run(self):
        """Run the webcam inference loop."""
        print(f"\nOpening camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return

        # Set camera resolution
        width, height = self.resolutions.get(self.resolution, (1920, 1080))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Resolution: {actual_w}x{actual_h}")

        print("\n" + "=" * 60)
        print("MTG Card Detection Running!")
        print("=" * 60)
        print("- Hold a card in front of the camera")
        print("- Press 'q' to quit")
        print("- Press 's' to save screenshot")
        print("=" * 60 + "\n")

        frame_count = 0
        frames_since_identify = 0

        while True:
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Detect cards (always - it's fast now)
            detect_start = time.time()
            cards = self.detector.detect(frame, fast_mode=True)
            detect_time = time.time() - detect_start
            self.detection_times.append(detect_time)

            # Check if we need to re-identify
            cards_stable = self._cards_are_stable(cards)
            need_identify = (
                not cards_stable or  # Cards moved
                frames_since_identify >= self.identify_interval or  # Time to refresh
                len(self.cached_results) != len(cards)  # Card count changed
            )

            if cards and need_identify:
                # Identify cards
                results = []
                for contour, corners in cards:
                    card_image = self.detector.extract_card(frame, corners)

                    id_start = time.time()
                    matches = self.identifier.identify(card_image, top_k=3, try_rotations=self.try_rotations)
                    id_time = time.time() - id_start
                    self.identification_times.append(id_time)

                    # Only show if confidence meets threshold
                    if matches and matches[0][1] >= self.min_confidence:
                        results.append((corners, matches))

                self.cached_results = results
                self.last_card_centers = [np.mean(corners, axis=0) for _, corners in cards]
                frames_since_identify = 0
            elif cards and self.cached_results:
                # Use cached results but update corners to current positions
                results = []
                for i, (contour, corners) in enumerate(cards):
                    if i < len(self.cached_results):
                        _, matches = self.cached_results[i]
                        results.append((corners, matches))
                frames_since_identify += 1
            else:
                results = []
                self.cached_results = []
                self.last_card_centers = []
                frames_since_identify = 0

            # Draw results
            display_frame = self.draw_results(frame, results)

            # Calculate FPS
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            fps = 1.0 / frame_time if frame_time > 0 else 0

            # Draw FPS and detector/identifier type
            detector_label = "YOLO" if self.detector_type == "yolo" else "Contour"
            id_label = "ONNX" if self.identifier_type == "onnx" else "PyTorch"
            cv2.putText(display_frame, f"FPS: {fps:.1f} ({detector_label}+{id_label})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw card count
            cv2.putText(display_frame, f"Cards: {len(cards)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show frame
            cv2.imshow("MTG Card Detection", display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Saved {filename}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        # Print performance stats
        self.print_stats()

    def draw_results(self, frame, results):
        """Draw detection and identification results on frame."""
        display = frame.copy()

        for corners, matches in results:
            # Draw card outline
            corners_int = corners.astype(np.int32)
            cv2.polylines(display, [corners_int], True, (0, 255, 0), 3)

            # Draw top match
            if matches:
                card_name, confidence = matches[0]

                # Position text above card
                text_pos = (int(corners[0][0]), int(corners[0][1]) - 10)

                # Draw background for text
                text = f"{card_name} ({confidence:.2f})"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display,
                             (text_pos[0] - 5, text_pos[1] - text_h - 10),
                             (text_pos[0] + text_w + 5, text_pos[1] + 5),
                             (0, 0, 0), -1)

                # Draw text
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                cv2.putText(display, text, text_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return display

    def print_stats(self):
        """Print performance statistics."""
        print("\n" + "=" * 60)
        print("Performance Statistics")
        print("=" * 60)

        if self.frame_times:
            avg_frame = np.mean(self.frame_times[-100:])
            print(f"Average FPS: {1/avg_frame:.1f}")
            print(f"Average frame time: {avg_frame*1000:.1f}ms")

        if self.detection_times:
            avg_detect = np.mean(self.detection_times[-100:])
            print(f"Average detection time: {avg_detect*1000:.1f}ms")

        if self.identification_times:
            avg_id = np.mean(self.identification_times[-100:])
            print(f"Average identification time: {avg_id*1000:.1f}ms")

        print("=" * 60)


def test_on_image(image_path, model_path, index_path, mapping_path):
    """Test pipeline on a single image."""
    print(f"\nTesting on image: {image_path}")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Initialize components
    detector = CardDetector()
    identifier = CardIdentifier(model_path, index_path, mapping_path)

    # Detect cards
    print("\nDetecting cards...")
    cards = detector.detect(image)
    print(f"Found {len(cards)} cards")

    # Identify each card
    for i, (contour, corners) in enumerate(cards):
        print(f"\nCard {i+1}:")

        # Extract card
        card_image = detector.extract_card(image, corners)

        # Identify
        matches = identifier.identify(card_image, top_k=5)

        for rank, (name, conf) in enumerate(matches, 1):
            print(f"  {rank}. {name} (confidence: {conf:.3f})")

        # Draw on image
        corners_int = corners.astype(np.int32)
        cv2.polylines(image, [corners_int], True, (0, 255, 0), 3)

        if matches:
            text = f"{matches[0][0]} ({matches[0][1]:.2f})"
            cv2.putText(image, text, (int(corners[0][0]), int(corners[0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save result
    output_path = Path(image_path).stem + "_detected.jpg"
    cv2.imwrite(output_path, image)
    print(f"\nSaved result to {output_path}")

    # Show result
    cv2.imshow("Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="MTG Card Detection and Identification")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--resolution", type=str, default="1080p",
                       choices=["480p", "720p", "1080p", "1440p", "max"],
                       help="Camera resolution (default: 1080p)")
    parser.add_argument("--fast", action="store_true",
                       help="Fast mode: 720p, skip rotations (higher FPS, slightly lower accuracy)")
    parser.add_argument("--image", type=Path, help="Test on single image instead of webcam")
    parser.add_argument("--model", type=Path,
                       default=TRAINING_DIR / "checkpoints" / "final_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--index", type=Path,
                       default=TRAINING_DIR / "output" / "card_embeddings_aug10_mean.faiss",
                       help="Path to FAISS index (default: augmented index)")
    parser.add_argument("--mapping", type=Path,
                       default=TRAINING_DIR / "output" / "label_mapping_aug10_mean.json",
                       help="Path to label mapping (default: augmented mapping)")
    parser.add_argument("--onnx", type=Path,
                       default=TRAINING_DIR / "output" / "card_embedding_model.onnx",
                       help="Path to ONNX model (for cross-platform consistency)")
    # YOLO detector options
    yolo_group = parser.add_mutually_exclusive_group()
    yolo_group.add_argument("--yolo", action="store_true", dest="use_yolo",
                           help="Force use YOLO detector (requires trained model)")
    yolo_group.add_argument("--no-yolo", action="store_false", dest="use_yolo",
                           help="Force use contour-based detection (default if no YOLO model)")
    parser.set_defaults(use_yolo=None)  # None = auto-detect
    # ONNX identifier options
    onnx_group = parser.add_mutually_exclusive_group()
    onnx_group.add_argument("--use-onnx", action="store_true", dest="use_onnx",
                           help="Use ONNX identifier (default, recommended with augmented index)")
    onnx_group.add_argument("--no-onnx", action="store_false", dest="use_onnx",
                           help="Use PyTorch identifier instead of ONNX")
    parser.set_defaults(use_onnx=True)  # Default to ONNX
    args = parser.parse_args()

    # Apply fast mode settings
    if args.fast:
        args.resolution = "720p"
        print("Fast mode enabled: 720p, skipping rotations")

    # Validate paths
    if not args.model.exists():
        print(f"Error: Model not found at {args.model}")
        print("Run training first: python training/train.py")
        return

    if not args.index.exists():
        print(f"Error: FAISS index not found at {args.index}")
        print("Generate embeddings first: python training/generate_embeddings.py --reference")
        return

    if not args.mapping.exists():
        print(f"Error: Label mapping not found at {args.mapping}")
        return

    if args.image:
        # Test on single image
        test_on_image(args.image, args.model, args.index, args.mapping)
    else:
        # Run webcam inference
        pipeline = WebcamInference(
            args.model, args.index, args.mapping,
            args.camera, args.resolution, fast_mode=args.fast,
            use_yolo=args.use_yolo,
            use_onnx=args.use_onnx, onnx_path=args.onnx
        )
        pipeline.run()


if __name__ == "__main__":
    main()
