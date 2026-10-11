# AI-Powered Vision-Based Collaborative Robotic System

## Project Overview

This project develops an AI-powered vision-based robotic system for automated pick-and-place operations in a warehouse-like environment. It integrates YOLOv8 object detection, inverse kinematics, ESP32 controllers, and wireless communication to enable two robotic manipulators to handle objects collaboratively.

An overhead camera monitors a 40 × 40 cm workspace and detects two object categories: cubes and cuboids. Based on the detected object class and location, the system selects the appropriate manipulation scenario and calculates the required joint angles using inverse kinematics.

## Key Features

- Real-time object detection and classification using YOLOv8.
- Camera-to-robot coordinate mapping.
- Inverse kinematics for calculating joint angles.
- Scenario-based task selection.
- ESP-NOW wireless communication between ESP32 controllers.
- Electromagnetic grippers for object manipulation.
- Coordinated pick-and-place operations using two 3-DOF robotic arms.

## System Architecture

The system consists of three main modules:

1. **Vision and Processing:** Captures images, detects objects, maps coordinates, and calculates joint angles.
2. **Wireless Communication:** Uses a main ESP32 and ESP-NOW to transmit commands to the robot controllers.
3. **Robotic Execution:** Two ESP32-controlled robotic arms perform object pickup, transportation, and placement.

## Task Execution Scenarios

**Scenario 1: Cube Handling**

Robot 1 picks up the cube and transfers it to a predefined handover position. Robot 2 receives the object and transports it to the destination.

**Scenario 2: Cuboid Handling**

Both robots coordinate their movements to grasp, lift, transport, and place the cuboid.

## Hardware Requirements

- Overhead camera
- Computer or laptop
- Two 3-DOF robotic arms
- Servo motors
- Three ESP32 microcontrollers
- Electromagnetic grippers
- Suitable power supplies and driver circuits

## Software and Technologies

- Python
- YOLOv8 and Ultralytics
- OpenCV
- NumPy
- Arduino IDE
- ESP32 and ESP-NOW
- Inverse kinematics

## Workflow

1. Capture workspace images using the overhead camera.
2. Detect and classify objects using YOLOv8.
3. Convert pixel coordinates into calibrated robot coordinates.
4. Select the appropriate manipulation scenario.
5. Calculate joint angles using inverse kinematics.
6. Transmit motion commands through ESP-NOW.
7. Execute coordinated pickup and placement operations.

## Applications

- Warehouse automation
- Intelligent material handling
- Collaborative robotics
- Automated sorting and pick-and-place systems

## Future Enhancements

- Improve detection accuracy with a larger dataset.
- Implement collision avoidance and trajectory planning.
- Add feedback sensors for pickup and placement verification.
- Improve motion synchronization and fault recovery.



Add an appropriate open-source license if you intend to permit reuse or redistribution of this project.
