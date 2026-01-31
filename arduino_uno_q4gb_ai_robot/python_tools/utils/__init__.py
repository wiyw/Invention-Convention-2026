#!/usr/bin/env python3
"""
Shared utilities for Arduino UNO Q4GB AI Robot
Standardized model loading, camera initialization, and utilities
"""

import os
import cv2
import numpy as np

def get_project_root():
    """Get project root directory consistently"""
    current = os.path.abspath(__file__)
    # Go up from python_tools/utils to project root
    return os.path.dirname(os.path.dirname(os.path.dirname(current)))

def load_yolo_model():
    """Standardized YOLO model loading with fallback"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not available")
        return None, None
    
    project_root = get_project_root()
    model_paths = [
        os.path.join(project_root, "models", "yolo26n.pt"),
        os.path.join(project_root, "models", "yolo26n.pt"),
        os.path.join(project_root, "yolo26n.pt"),
        "yolo26n.pt"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = YOLO(path)
                print(f"✅ YOLO26n loaded: {path}")
                return model, path
            except Exception as e:
                print(f"⚠ YOLO load failed: {path} - {e}")
                continue
    
    print("❌ YOLO26n not found, using placeholder detection")
    return None, None

def load_qwen_prompt():
    """Load Qwen prompt template"""
    project_root = get_project_root()
    prompt_paths = [
        os.path.join(project_root, "models", "qwen_prompt.txt"),
        os.path.join(project_root, "yolo26n", "prompt.txt"),
        os.path.join(project_root, "prompt.txt")
    ]
    
    for path in prompt_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return f.read()
            except Exception as e:
                print(f"⚠ Prompt load failed: {path} - {e}")
                continue
    
    return None

def initialize_camera_with_fallback(camera_id=0):
    """Robust camera initialization with backend fallback"""
    # Backend priority: DirectShow -> MSMF -> Auto (Windows optimized)
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    backend_names = ["DirectShow", "MSMF", "Auto"]
    
    for backend_idx, backend in enumerate(backends):
        print(f"Trying {backend_names[backend_idx]} backend...")
        
        for cam_id in range(3):  # Try IDs 0, 1, 2
            try:
                camera = cv2.VideoCapture(cam_id, backend)
                
                if not camera.isOpened():
                    camera.release()
                    continue
                
                # Quick test frame
                ret, test_frame = camera.read()
                if not ret or test_frame is None:
                    camera.release()
                    continue
                
                print(f"✓ Camera {cam_id} with {backend_names[backend_idx]} backend")
                print(f"  Initial frame: {test_frame.shape[1]}x{test_frame.shape[0]}")
                
                # Set basic properties
                set_camera_properties(camera)
                
                # Final test
                ret, final_frame = camera.read()
                if ret and final_frame is not None:
                    print(f"✓ Camera configured: {final_frame.shape[1]}x{final_frame.shape[0]}")
                    return camera, cam_id, backend_names[backend_idx]
                
                camera.release()
                
            except Exception as e:
                print(f"  ⚠ Camera {cam_id} failed: {e}")
                continue
    
    print("❌ No working camera found")
    return None, None, None

def set_camera_properties(camera):
    """Set and validate camera properties"""
    properties = [
        (cv2.CAP_PROP_FRAME_WIDTH, 800, "Width"),
        (cv2.CAP_PROP_FRAME_HEIGHT, 600, "Height"),
        (cv2.CAP_PROP_FPS, 30, "FPS")
    ]
    
    for prop, value, name in properties:
        camera.set(prop, value)
        
        # Verify with tolerance
        actual = camera.get(prop)
        if name == "FPS":
            if actual < 5:
                print(f"  ⚠ {name} too low: {actual:.1f}")
        else:
            if abs(actual - value) > value * 0.5:
                print(f"  ⚠ {name} not set correctly: {actual} (wanted {value})")
    
    # Try MJPG codec for better compatibility
    try:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    except:
        pass

def clamp(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))

def calculate_servo_commands(action, confidence, bbox_norm):
    """Calculate servo PWM values for Arduino control"""
    base_speed = clamp(int(160 * confidence), 50, 220)
    
    if not bbox_norm:
        return base_speed, base_speed, 500  # Default forward
    
    cx = bbox_norm.get('cx', 0.5)
    w = bbox_norm.get('w', 0.1)
    
    # Steering calculation
    steering_offset = int((0.5 - cx) * 2 * 80)  # Range roughly [-80, 80]
    
    left = clamp(base_speed - steering_offset, 0, 255)
    right = clamp(base_speed + steering_offset, 0, 255)
    
    # Duration based on action
    if 'stop' in action:
        duration = 200
    elif 'turn' in action or 'slow' in action:
        duration = 300
    else:
        duration = 500
    
    return left, right, duration

class ModelValidator:
    """Validate model files and paths"""
    
    @staticmethod
    def validate_yolo_model(path):
        """Validate YOLO model file"""
        if not os.path.exists(path):
            return False, "File not found"
        
        file_size = os.path.getsize(path)
        if file_size < 1000000:  # Less than 1MB seems too small
            return False, f"File too small: {file_size} bytes"
        
        # Try to load it
        try:
            from ultralytics import YOLO
            model = YOLO(path)
            return True, f"Valid YOLO model ({file_size//1024}KB)"
        except Exception as e:
            return False, f"Load failed: {e}"
    
    @staticmethod
    def validate_qwen_prompt(path):
        """Validate Qwen prompt file"""
        if not os.path.exists(path):
            return False, "File not found"
        
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            if len(content) < 100:
                return False, "Prompt too short"
            
            if 'CMD L' not in content or 'JSON' not in content:
                return False, "Invalid prompt format"
            
            return True, f"Valid prompt ({len(content)} chars)"
        except Exception as e:
            return False, f"Read failed: {e}"

if __name__ == "__main__":
    """Test utilities"""
    print("=== Testing Arduino AI Utilities ===")
    
    # Test model loading
    print("\n1. Testing YOLO model loading...")
    model, path = load_yolo_model()
    if model:
        print(f"✅ YOLO loaded from: {path}")
    else:
        print("❌ YOLO loading failed")
    
    # Test prompt loading
    print("\n2. Testing Qwen prompt loading...")
    prompt = load_qwen_prompt()
    if prompt:
        print(f"✅ Prompt loaded ({len(prompt)} chars)")
    else:
        print("❌ Prompt loading failed")
    
    # Test camera initialization
    print("\n3. Testing camera initialization...")
    camera, cam_id, backend = initialize_camera_with_fallback()
    if camera:
        print(f"✅ Camera {cam_id} with {backend}")
        camera.release()
    else:
        print("❌ Camera initialization failed")
    
    print("\n=== Utility Tests Complete ===")