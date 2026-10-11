
from ultralytics import YOLO
import cv2
import numpy as np
import serial
import time
import math

# =====================================================
# 1. CONFIGURATION
# =====================================================

MODEL_PATH = r"C:\Users\srisa\OneDrive\Desktop\runs\detect\train3\weights\best.pt"
HOMOGRAPHY_PATH = r"C:\Users\srisa\OneDrive\Desktop\calibration\homography.npy"

COM_PORT = "COM6"
BAUD_RATE = 115200
CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.50

model = YOLO(MODEL_PATH)
H = np.load(HOMOGRAPHY_PATH)

# =====================================================
# 2. ROBOT GEOMETRY - DEMO VALUES, NOT MEASURED VALUES
# =====================================================

L1 = 6.0
L2 = 12.0
L3 = 10.0

# Base positions in the calibrated camera workspace (cm)
ROBOT1_BASE = (10.0, 20.0)
ROBOT2_BASE = (30.0, 20.0)

ROBOT1_YAW_OFFSET = 0.0
ROBOT2_YAW_OFFSET = 180.0

# Example Cartesian targets in workspace coordinates (cm)
HANDOVER_TARGET = (20.0, 20.0)
DROP_TARGET = (32.78, 18.57)

# Example gripper heights (cm)
PICKUP_Z = 2.0
HANDOVER_Z = 10.0
DROP_Z = 2.0

# Example servo calibration placeholders.
# Replace with experimentally measured offsets and directions.
SERVO_OFFSETS = {
    "R1": (0.0, 0.0, 0.0),
    "R2": (0.0, 0.0, 0.0)
}

SERVO_DIRECTIONS = {
    "R1": (1, 1, 1),
    "R2": (1, 1, 1)
}

# Enable only after checking configuration and firmware.
SEND_TO_ESP32 = True

# =====================================================
# 3. CAMERA PIXELS TO WORKSPACE CM
# =====================================================

def pixel_to_cm(u, v):
    point = np.array([u, v, 1.0], dtype=float)
    result = H @ point

    if abs(result[2]) < 1e-9:
        raise ValueError("Invalid homography result")

    result /= result[2]
    return float(result[0]), float(result[1])


# =====================================================
# 4. WORKSPACE COORDINATES TO ROBOT COORDINATES
# =====================================================

def world_to_robot(X, Y, Z, base, yaw_offset):
    dx = X - base[0]
    dy = Y - base[1]

    yaw = math.radians(yaw_offset)

    x = dx * math.cos(yaw) + dy * math.sin(yaw)
    y = -dx * math.sin(yaw) + dy * math.cos(yaw)

    return x, y, Z


# =====================================================
# 5. INVERSE KINEMATICS
# =====================================================

def inverse_kinematics(X, Y, Z, robot_id):
    if robot_id == "R1":
        base = ROBOT1_BASE
        yaw = ROBOT1_YAW_OFFSET
    else:
        base = ROBOT2_BASE
        yaw = ROBOT2_YAW_OFFSET

    x, y, z = world_to_robot(X, Y, Z, base, yaw)

    theta1 = math.atan2(y, x)
    R = math.sqrt(x*x + y*y)
    Zs = z - L1

    D = (
        R*R + Zs*Zs - L2*L2 - L3*L3
    ) / (2.0 * L2 * L3)

    if D < -1.0 or D > 1.0:
        raise ValueError(
            f"{robot_id}: target ({X:.2f}, {Y:.2f}, "
            f"{Z:.2f}) is unreachable"
        )

    # Elbow-down solution
    theta3 = math.acos(D)

    theta2 = (
        math.atan2(Zs, R)
        - math.atan2(
            L3 * math.sin(theta3),
            L2 + L3 * math.cos(theta3)
        )
    )

    angles = [
        math.degrees(theta1),
        math.degrees(theta2),
        math.degrees(theta3)
    ]

    offsets = SERVO_OFFSETS[robot_id]
    directions = SERVO_DIRECTIONS[robot_id]

    servo_angles = [
        directions[i] * angles[i] + offsets[i]
        for i in range(3)
    ]

    return tuple(round(a, 2) for a in servo_angles)


def solve_target(xy, z, robot_id):
    return inverse_kinematics(
        xy[0], xy[1], z, robot_id
    )


# =====================================================
# 6. OBJECT PICKUP POINTS
# =====================================================

def get_object_points(box, class_id):
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Scenario 1: cube -> center pickup
    if class_id == 0:
        points_px = [
            ((x1 + x2) // 2, (y1 + y2) // 2)
        ]

    # Scenario 2: cuboid -> opposite short-edge centers
    else:
        w = x2 - x1
        h = y2 - y1

        if w >= h:
            # Horizontal cuboid: left and right edge centers
            points_px = [
                (x1, (y1 + y2) // 2),
                (x2, (y1 + y2) // 2)
            ]
        else:
            # Vertical cuboid: top and bottom edge centers
            points_px = [
                ((x1 + x2) // 2, y1),
                ((x1 + x2) // 2, y2)
            ]

    points_cm = [
        pixel_to_cm(u, v) for u, v in points_px
    ]

    return points_px, points_cm, (x1, y1, x2, y2)


# =====================================================
# 7. SCENARIO 1: CUBE
# =====================================================

def calculate_scenario1(pickup_xy):
    # Robot 1: center pickup
    r1_pick = solve_target(
        pickup_xy, PICKUP_Z, "R1"
    )

    # Robot 1: handover pose
    r1_hand = solve_target(
        HANDOVER_TARGET, HANDOVER_Z, "R1"
    )

    # Robot 2: handover pose
    r2_hand = solve_target(
        HANDOVER_TARGET, HANDOVER_Z, "R2"
    )

    # Robot 2: drop pose
    r2_drop = solve_target(
        DROP_TARGET, DROP_Z, "R2"
    )

    return r1_pick, r1_hand, r2_hand, r2_drop


# =====================================================
# 8. SCENARIO 2: CUBOID
# =====================================================

def calculate_scenario2(pickup_points):
    if len(pickup_points) != 2:
        raise ValueError(
            "Cuboid requires two pickup points"
        )

    # Robot 1 picks the first short-edge center
    r1_pick = solve_target(
        pickup_points[0], PICKUP_Z, "R1"
    )

    # Robot 2 picks the opposite short-edge center.
    # Existing protocol uses R2H for this target
    # in Scenario 2. The ESP32 must implement this meaning.
    r2_pick = solve_target(
        pickup_points[1], PICKUP_Z, "R2"
    )

    # Robot 1 coordinated carry/handover target
    r1_hand = solve_target(
        HANDOVER_TARGET, HANDOVER_Z, "R1"
    )

    # Robot 2 drop target
    r2_drop = solve_target(
        DROP_TARGET, DROP_Z, "R2"
    )

    # Existing protocol has only R1P, R1H, R2H, R2D.
    # Scenario 2 mapping:
    # R1P = Robot 1 pickup
    # R1H = Robot 1 handover/carry target
    # R2H = Robot 2 opposite-edge pickup
    # R2D = Robot 2 drop target
    return r1_pick, r1_hand, r2_pick, r2_drop


# =====================================================
# 9. SEND ANGLES TO MAIN ESP32
# =====================================================

def send_to_esp32(ser, scenario_id, angles):
    r1_pick, r1_hand, r2_hand, r2_drop = angles

    message = (
        f"SCN,{scenario_id},"
        f"R1P,{r1_pick[0]},{r1_pick[1]},{r1_pick[2]},"
        f"R1H,{r1_hand[0]},{r1_hand[1]},{r1_hand[2]},"
        f"R2H,{r2_hand[0]},{r2_hand[1]},{r2_hand[2]},"
        f"R2D,{r2_drop[0]},{r2_drop[1]},{r2_drop[2]}\n"
    )

    ser.write(message.encode("utf-8"))
    ser.flush()

    print("\nSerial data sent to ESP32:")
    print(message.strip())


# =====================================================
# 10. MAIN PROGRAM
# =====================================================

def main():
    cap = None
    ser = None
    detection_done = False

    try:
        ser = serial.Serial(
            COM_PORT, BAUD_RATE, timeout=1
        )
        time.sleep(2)

        cap = cv2.VideoCapture(
            CAMERA_ID, cv2.CAP_DSHOW
        )

        if not cap.isOpened():
            raise RuntimeError("Camera could not be opened")

        print("YOLO vision system started.")
        print("Scenario 1 = Cube")
        print("Scenario 2 = Cuboid")
        print("Press Q to quit.")

        while True:
            ret, frame = cap.read()

            if not ret:
                continue

            if not detection_done:
                results = model(frame, verbose=False)

                candidates = []

                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if (
                        class_id in (0, 1)
                        and confidence >= CONFIDENCE_THRESHOLD
                    ):
                        candidates.append(
                            (class_id, confidence, box)
                        )

                if candidates:
                    # Highest-confidence cube/cuboid detection
                    class_id, confidence, box = max(
                        candidates,
                        key=lambda item: item[1]
                    )

                    try:
                        points_px, points_cm, bbox = (
                            get_object_points(box, class_id)
                        )

                        if class_id == 0:
                            scenario_id = 1
                            scenario_name = "SCENARIO 1 - CUBE"

                            angles = calculate_scenario1(
                                points_cm[0]
                            )

                        else:
                            scenario_id = 2
                            scenario_name = "SCENARIO 2 - CUBOID"

                            angles = calculate_scenario2(
                                points_cm
                            )

                        print("\n" + "=" * 42)
                        print(scenario_name)
                        print(f"Confidence: {confidence:.3f}")

                        for i, point in enumerate(points_cm):
                            print(
                                f"Pickup point {i+1}: "
                                f"X={point[0]:.2f} cm, "
                                f"Y={point[1]:.2f} cm"
                            )

                        print("\nRobot 1 pickup:", angles[0])
                        print("Robot 1 handover:", angles[1])
                        print("Robot 2 target:", angles[2])
                        print("Robot 2 drop:", angles[3])

                        # Send only if all required IK calculations pass.
                        if SEND_TO_ESP32:
                            send_to_esp32(
                                ser, scenario_id, angles
                            )

                        detection_done = True

                        x1, y1, x2, y2 = bbox

                        cv2.rectangle(
                            frame, (x1, y1), (x2, y2),
                            (0, 255, 255), 2
                        )

                        for point in points_px:
                            cv2.circle(
                                frame, point, 7,
                                (0, 255, 0), -1
                            )

                        cv2.putText(
                            frame,
                            scenario_name,
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2
                        )

                    except ValueError as error:
                        print("\nIK/configuration error:", error)
                        print("No serial command sent.")

            cv2.imshow("Vision-Based Robot IK", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        if ser is not None and ser.is_open:
            ser.close()

        print("Program terminated.")


if __name__ == "__main__":
    main()
