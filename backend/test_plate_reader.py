import cv2
from plate_reader import extract_plate_text
import sys

# Get image path from command line or use default
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = input("Enter image path: ")

print(f"Testing plate reader on: {img_path}")

# Load image
img = cv2.imread(img_path)
if img is None:
    print("❌ Could not load image")
    exit(1)

print(f"Image size: {img.shape}")

# Try plate extraction
result = extract_plate_text(img)
print(f"\n✅ Result: {result}")