from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(r"C:\\Users\\srisa\\OneDrive\\Desktop\\runs\\detect\\train3\\weights\\best.pt")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Cube / Cuboid Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
