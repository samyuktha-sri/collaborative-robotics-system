import cv2
import time
import os

# =============================
# SETTINGS
# =============================
CAMERA_INDEX = 1          # change to 1 if needed
SAVE_DIR = r"C:\\Users\\srisa\\OneDrive\\Desktop\\dataset\\raw_images"
INTERVAL = 5              # seconds

# =============================
# SETUP
# =============================
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ Camera opened")
print("Saving one image every 5 seconds")
print("Press ESC to stop")

last_time = time.time()
count = 0

# =============================
# MAIN LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    cv2.imshow("Webcam Capture", frame)

    current_time = time.time()

    # Save image every 5 seconds
    if current_time - last_time >= INTERVAL:
        filename = f"img_{count:04d}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)

        cv2.imwrite(filepath, frame)
        print(f"Saved: {filename}")

        count += 1
        last_time = current_time

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Image capture stopped")
