#!/usr/bin/env python3
"""
Collect test images from webcam for YOLO validation.

Usage:
    python collect_images.py --camera 0
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

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 60)
    print("Test Image Collection for YOLO Validation")
    print("=" * 60)
    print(f"Resolution: {actual_w}x{actual_h}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    print("Controls:")
    print("  SPACE - Capture image")
    print("  1-7   - Change category:")
    print("          1 = playmat (ideal surface)")
    print("          2 = wood (table surface)")
    print("          3 = clutter (busy background)")
    print("          4 = multiple (2+ cards)")
    print("          5 = handheld (in hand)")
    print("          6 = dim (low light)")
    print("          7 = bright (harsh light)")
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

    # Count existing images
    existing = list(output_dir.glob("*.jpg"))
    if existing:
        print(f"Found {len(existing)} existing images in output directory")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from camera")
            break

        # Create display with overlay
        display = frame.copy()

        # Draw semi-transparent background for text
        overlay = display.copy()
        cv2.rectangle(overlay, (5, 5), (400, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        # Draw info text
        cv2.putText(display, f"Category: {current_category}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Captured this session: {count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=capture, 1-7=category, Q=quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw category hints
        hint_y = 120
        for key, cat in categories.items():
            color = (0, 255, 255) if cat == current_category else (150, 150, 150)
            cv2.putText(display, f"{key}:{cat}", (10 + (int(key)-1) * 90, hint_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow("Collect Test Images", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord(' '):
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{current_category}_{timestamp}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            print(f"Saved: {filename}")
            count += 1

            # Flash effect
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (actual_w, actual_h), (255, 255, 255), -1)
            cv2.addWeighted(flash, 0.3, frame, 0.7, 0, flash)
            cv2.imshow("Collect Test Images", flash)
            cv2.waitKey(100)

        elif chr(key) in categories:
            current_category = categories[chr(key)]
            print(f"Category changed to: {current_category}")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"Collection complete!")
    print(f"Images captured this session: {count}")
    print(f"Total images in {output_dir}: {len(list(output_dir.glob('*.jpg')))}")
    print("=" * 60)
    print("\nNext step: Run validation")
    print("  python training/yolo/validate.py")


def main():
    parser = argparse.ArgumentParser(description="Collect test images for YOLO validation")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "test_images",
                        help="Output directory for images")
    parser.add_argument("--resolution", type=str, default="720p",
                        choices=["480p", "720p", "1080p"],
                        help="Camera resolution (default: 720p)")
    args = parser.parse_args()

    resolutions = {
        "480p": (640, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
    }

    collect_images(args.camera, args.output, resolutions[args.resolution])


if __name__ == "__main__":
    main()
