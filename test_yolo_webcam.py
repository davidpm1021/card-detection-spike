"""Simple YOLO webcam test with explicit window display."""
import cv2
from ultralytics import YOLO

# Load model
model = YOLO('training/yolo/runs/detect/train/weights/best.pt')

# Open webcam
cap = cv2.VideoCapture(1)  # Use external webcam (change to 0 for built-in)
if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit(1)

print("Webcam opened. Press 'q' to quit.")
print("Hold a card in front of the camera to test detection.")

cv2.namedWindow('YOLO Card Detection', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run detection (0.6 confidence filters out most false positives)
    results = model(frame, conf=0.6, verbose=False)

    # Draw results on frame
    annotated = results[0].plot()

    # Count detections
    num_cards = len(results[0].boxes)

    # Add text overlay
    cv2.putText(annotated, f"Cards detected: {num_cards}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('YOLO Card Detection', annotated)

    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
