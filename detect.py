import cv2
import numpy as np
from ultralytics import YOLO

# ==================================================
# WORKSPACE & CAMERA PARAMETERS
# ==================================================

WORKSPACE_SIZE_CM = 40.0          # 40 x 40 cm workspace
CAM_WIDTH_PX = 640
CAM_HEIGHT_PX = 480

# Pixel → cm scale (top-down, no perspective distortion)
SCALE_X = WORKSPACE_SIZE_CM / CAM_WIDTH_PX
SCALE_Y = WORKSPACE_SIZE_CM / CAM_HEIGHT_PX

# ==================================================
# LOAD YOLO MODEL
# ==================================================

model = YOLO("yolov8n.pt")

# ==================================================
# OPEN CAMERA
# ==================================================

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH_PX)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT_PX)

# ==================================================
# STATE VARIABLES
# ==================================================

detected_once = False
stored_pickup_px = None    # store pixel coordinates
stored_pickup_cm = None    # store cm coordinates

# ==================================================
# MAIN LOOP
# ==================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ------------------ DETECT ONLY ONCE ------------------
    if not detected_once:
        results = model(frame, conf=0.6, iou=0.5, verbose=False)

        for r in results:
            # SAFETY CHECK (prevents crash)
            if r.boxes is None or len(r.boxes) == 0:
                continue

            # Take ONLY the first detected object
            box = r.boxes[0]

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Center in pixels
            cx_px = (x1 + x2) / 2.0
            cy_px = (y1 + y2) / 2.0

            # Convert to centimeters
            cx_cm = cx_px * SCALE_X
            cy_cm = cy_px * SCALE_Y

            # Print ONLY ONCE
            print(f"Pick-up Point → X = {cx_cm:.2f} cm , Y = {cy_cm:.2f} cm")

            # Store values for display
            stored_pickup_px = (cx_px, cy_px)
            stored_pickup_cm = (cx_cm, cy_cm)

            detected_once = True
            break   # stop after first valid detection

    # ------------------ DISPLAY PICK-UP POINT ------------------
    if detected_once and stored_pickup_px is not None:
        cx_px, cy_px = stored_pickup_px

        # Draw red center point
        cv2.circle(frame,
                   (int(cx_px), int(cy_px)),
                   6, (0, 0, 255), -1)

        # Label
        cv2.putText(frame,
                    "Pick-up Point",
                    (int(cx_px) + 8, int(cy_px) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    # ------------------ SHOW WINDOW ------------------
    cv2.imshow("YOLO Pick-up Detection", frame)

    # Manual exit only
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==================================================
# CLEANUP
# ==================================================

cap.release()
cv2.destroyAllWindows()