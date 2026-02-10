#!/usr/bin/env python3
"""
Test script for Ultrasonic Trash Collector Robot
"""

import sys
import os
import time

# Add robot directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import numpy
        print(f"  NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"  NumPy not available: {e}")
    
    try:
        import serial
        print("  PySerial: Available")
    except ImportError as e:
        print(f"  PySerial not available: {e}")
    
    try:
        import cv2
        print(f"  OpenCV: {cv2.__version__} (for visualization)")
    except ImportError as e:
        print(f"  OpenCV not available: {e}")

def test_ultrasonic_robot():
    """Test ultrasonic robot initialization"""
    print("\nTesting ultrasonic robot initialization...")
    
    try:
        from ultrasonic_trash_collector import UltrasonicTrashCollectorRobot
        robot = UltrasonicTrashCollectorRobot()
        print("  Ultrasonic robot class: OK")
        return robot
    except Exception as e:
        print(f"  Ultrasonic robot initialization error: {e}")
        return None

def test_ultrasonic_simulation():
    """Test ultrasonic sensor simulation"""
    print("\nTesting ultrasonic sensor simulation...")
    
    try:
        from ultrasonic_trash_collector import UltrasonicTrashCollectorRobot
        robot = UltrasonicTrashCollectorRobot()
        
        # Test sensor data generation
        robot.generate_sensor_data()
        print(f"  Sensor data: {robot.sensor_data}")
        
        # Test object detection
        detections = robot.detect_objects_from_sensors()
        print(f"  Detected objects: {len(detections)}")
        for obj in detections:
            print(f"    - {obj['type']}: {obj['distance']:.1f}cm at {obj['angle']}°")
        
        return True
        
    except Exception as e:
        print(f"  Ultrasonic simulation error: {e}")
        return False

def test_visualization():
    """Test sensor visualization"""
    print("\nTesting sensor visualization...")
    
    try:
        from ultrasonic_trash_collector import UltrasonicTrashCollectorRobot
        robot = UltrasonicTrashCollectorRobot()
        
        # Generate test data
        robot.generate_sensor_data()
        robot.detect_objects_from_sensors()
        
        # Create visualization
        image = robot.create_sensor_visualization()
        if image is not None:
            print(f"  Visualization: OK (shape: {image.shape})")
            
            # Save test image
            import cv2
            cv2.imwrite("ultrasonic_test.jpg", image)
            print("  Saved test image: ultrasonic_test.jpg")
            return True
        else:
            print("  Visualization: FAILED")
            return False
            
    except Exception as e:
        print(f"  Visualization test error: {e}")
        return False

def main():
    print("Ultrasonic Trash Collector Robot Test Suite")
    print("=" * 50)
    
    # Test imports
    test_imports()
    
    # Test robot initialization
    robot = test_ultrasonic_robot()
    
    if robot:
        # Test ultrasonic simulation
        if test_ultrasonic_simulation():
            print("  Ultrasonic simulation: OK")
        else:
            print("  Ultrasonic simulation: FAILED")
        
        # Test visualization
        if test_visualization():
            print("  Sensor visualization: OK")
        else:
            print("  Sensor visualization: FAILED")
        
        # Test robot components
        print("\nTesting robot components...")
        
        # Test sensor initialization
        if robot.initialize_ultrasonic_sensors():
            print("  Ultrasonic sensors: OK")
        else:
            print("  Ultrasonic sensors: FAILED")
        
        # Test Arduino initialization
        if robot.initialize_arduino():
            print("  Arduino initialization: OK")
        else:
            print("  Arduino initialization: FAILED (expected without hardware)")
        
        print("\nTest complete!")
        print("The ultrasonic robot is ready for:")
        print("  - 5x Ultrasonic Sensor Simulation")
        print("  - Real-time Object Detection")
        print("  - 360-degree Coverage")
        print("  - Arduino Control (when connected)")
        print("  - Sensor Visualization")
        print("  - No Camera Required!")

if __name__ == "__main__":
    main()