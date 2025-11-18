# Real-Time-Patient-Monitoring-System

Intelligent Patient Activity Tracking using YOLOv8 & Temporal Movement Analysis.

This project is a real-time monitoring application designed to assist hospitals in keeping patients safe. It combines YOLOv8-based person detection, temporal movement analysis, and an interactive Streamlit interface to identify patient activity patterns and automatically trigger alerts when unusual behavior is detected—such as a single patient moving without supervision.The system supports live webcam feeds, uploaded videos, and images, making it suitable for real-world monitoring and offline analysis.

KEY FEATURES : 

Real-Time Detection

1. Detects people in frames using YOLOv8

2. Differentiates between patient movement and idle state

2. Handles multiple individuals (patient + staff)

Temporal Activity Analysis

1. Tracks bounding box movement over time

2. Computes movement scores across frames

3. Determines room states such as:
   
      Empty Room
   
      Single Patient Idle
   
      Single Patient Moving (Warning)
   
      Multiple People (Safe)

Smart Alert Mechanism

1. Triggers alerts only when needed

2. Reduces false warnings

3. Displays color-coded status banners

Multi-Source Input

1. Webcam monitoring (Live)

2. Video file analysis

3. Image analysis

4. Auto-switches to Demo Mode if YOLO is unavailable (helps with CPU-only systems)

TECH STACK : 

Backend / CV Processing

1. Python

2. YOLOv8 (Ultralytics)

3. OpenCV

4. PyTorch

5. NumPy

Frontend

1. Streamlit

Visualization & Tracking

1. Custom temporal movement tracker

2. Bounding box overlays

3. Status annotations on frames
