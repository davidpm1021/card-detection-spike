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

# Add training directory to path for model import
SCRIPT_DIR = Path(__file__).parent
TRAINING_DIR = SCRIPT_DIR.parent / "training"
sys.path.insert(0, str(TRAINING_DIR))

from model import CardEmbeddingModel


class CardDetector:
    """Detects card regions in images using contour detection."""

    def __init__(self, min_area=5000, max_area=500000):
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame, fast_mode=True):
        """
        Detect card-like rectangles in the frame.

        Args:
            frame: Input BGR image
            fast_mode: If True, use fast single-method detection (for real-time)
                      If False, use all methods (for thorough detection)

        Returns:
            List of (contour, corners) tuples for detected cards
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        kernel = np.ones((3, 3), np.uint8)

        if fast_mode:
            # Fast path: single method
            edges = cv2.Canny(blurred, 30, 100)
            edges = cv2.dilate(edges, kernel, iterations=2)
            cards = self._find_cards_in_edges(edges)

            # If no cards found, try one more method (adaptive threshold)
            if not cards:
                adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
                edges = cv2.Canny(adaptive, 50, 150)
                edges = cv2.dilate(edges, kernel, iterations=2)
                cards = self._find_cards_in_edges(edges)

            return cards
        else:
            # Thorough mode: try all methods
            cards = []
            for edges in self._get_edge_maps(frame, blurred, gray, kernel):
                detected = self._find_cards_in_edges(edges)
                cards.extend(detected)
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

        return cards

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
        # Tighten tolerance: 0.65-0.78 instead of 0.5-0.9
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
        self.card_names = mapping["card_names"]
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


class WebcamInference:
    """Real-time webcam inference pipeline."""

    def __init__(self, model_path, index_path, mapping_path, camera_id=0, resolution="1080p", fast_mode=False):
        self.detector = CardDetector()
        self.identifier = CardIdentifier(model_path, index_path, mapping_path)
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

            # Draw FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
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
                       default=TRAINING_DIR / "output" / "card_embeddings_full.faiss",
                       help="Path to FAISS index")
    parser.add_argument("--mapping", type=Path,
                       default=TRAINING_DIR / "output" / "label_mapping_full.json",
                       help="Path to label mapping")
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
            args.camera, args.resolution, fast_mode=args.fast
        )
        pipeline.run()


if __name__ == "__main__":
    main()
