#!/usr/bin/env python3
"""
Test script for Trash Collector Robot components
"""

import sys
import os
import time

# Add the robot directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"  OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"  OpenCV not available: {e}")
    
    try:
        import numpy
        print(f"  NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"  NumPy not available: {e}")
    
    try:
        from ultralytics import YOLO
        print("  YOLOv8n: Available")
    except ImportError as e:
        print(f"  YOLOv8n not available: {e}")
    
    try:
        import serial
        print("  PySerial: Available")
    except ImportError as e:
        print(f"  PySerial not available: {e}")

def test_camera():
    """Test camera initialization"""
    print("\nTesting camera...")
    
    try:
        import cv2
        import os
        
        # Check for video devices
        video_devices = []
        if os.path.exists('/dev'):
            for i in range(10):
                if os.path.exists(f'/dev/video{i}'):
                    video_devices.append(i)
        
        print(f"  Found video devices: {video_devices}")
        
        # Try to initialize camera
        for idx in video_devices if video_devices else [0]:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"  Camera {idx}: Working (resolution: {frame.shape[1]}x{frame.shape[0]})")
                    cap.release()
                    return True
                cap.release()
        
        print("  No working camera found")
        return False
        
    except Exception as e:
        print(f"  Camera test error: {e}")
        return False

def test_robot_initialization():
    """Test robot class initialization"""
    print("\nTesting robot initialization...")
    
    try:
        from main_ai_robot import TrashCollectorRobot
        robot = TrashCollectorRobot()
        print("  Robot class: OK")
        return robot
    except Exception as e:
        print(f"  Robot initialization error: {e}")
        return None

def main():
    print("Trash Collector Robot Test Suite")
    print("=" * 40)
    
    # Test imports
    test_imports()
    
    # Test camera
    camera_available = test_camera()
    
    # Test robot initialization
    robot = test_robot_initialization()
    
    if robot:
        print("\nTesting robot components...")
        
        # Test camera initialization
        if robot.initialize_camera():
            print(f"  Camera initialization: OK (mode: {robot.camera_mode})")
        else:
            print("  Camera initialization: FAILED")
        
        # Test model loading
        if robot.load_model():
            print("  Model loading: OK")
        else:
            print("  Model loading: FAILED (expected for simulation)")
        
        # Test Arduino initialization
        if robot.initialize_arduino():
            print("  Arduino initialization: OK")
        else:
            print("  Arduino initialization: FAILED (expected without hardware)")
        
        print("\nTest simulation frame generation...")
        test_image = robot.create_test_image()
        if test_image is not None:
            print(f"  Test image: OK (shape: {test_image.shape})")
        else:
            print("  Test image: FAILED")
        
        print("\nTest complete!")
        print("The robot is ready for:")
        print(f"  - Camera: {'Real' if camera_available else 'Simulation'}")
        print("  - YOLOv8n AI Detection")
        print("  - Arduino Control (when connected)")
        print("  - Enhanced Simulation Mode")

if __name__ == "__main__":
    main()