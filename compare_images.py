"""Compare the good vs bad debug images."""
import cv2
import numpy as np

good = cv2.imread('debug_card_crop_GOOD.jpg')
bad = cv2.imread('debug_card_crop.jpg')

print("=== IMAGE COMPARISON ===\n")

print(f"GOOD image: {good.shape}")
print(f"BAD image:  {bad.shape}")

print(f"\nGOOD - Mean brightness: {good.mean():.1f}, Std: {good.std():.1f}")
print(f"BAD  - Mean brightness: {bad.mean():.1f}, Std: {bad.std():.1f}")

# Check for blur using Laplacian variance
good_gray = cv2.cvtColor(good, cv2.COLOR_BGR2GRAY)
bad_gray = cv2.cvtColor(bad, cv2.COLOR_BGR2GRAY)

good_lap = cv2.Laplacian(good_gray, cv2.CV_64F).var()
bad_lap = cv2.Laplacian(bad_gray, cv2.CV_64F).var()

print(f"\nGOOD - Laplacian variance (sharpness): {good_lap:.1f}")
print(f"BAD  - Laplacian variance (sharpness): {bad_lap:.1f}")

if bad_lap < good_lap * 0.5:
    print(">>> BAD image is significantly BLURRIER!")
elif bad_lap > good_lap * 1.5:
    print(">>> BAD image is sharper (unexpected)")

# Color distribution
print(f"\nGOOD - R:{good[:,:,2].mean():.0f} G:{good[:,:,1].mean():.0f} B:{good[:,:,0].mean():.0f}")
print(f"BAD  - R:{bad[:,:,2].mean():.0f} G:{bad[:,:,1].mean():.0f} B:{bad[:,:,0].mean():.0f}")

# Save side-by-side comparison
good_resized = cv2.resize(good, (300, 400))
bad_resized = cv2.resize(bad, (300, 400))
comparison = np.hstack([good_resized, bad_resized])
cv2.putText(comparison, "GOOD (0.54)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(comparison, "BAD (0.17)", (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imwrite('comparison.jpg', comparison)
print("\nSaved comparison.jpg - push it so I can see!")
