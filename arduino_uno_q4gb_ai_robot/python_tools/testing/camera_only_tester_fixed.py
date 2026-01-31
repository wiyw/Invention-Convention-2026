#!/usr/bin/env python3
"""
Fixed Camera-Only Testing Suite for Arduino UNO Q4GB AI Robot
Tests AI system using laptop webcam without ultrasonic sensors
"""

import cv2
import numpy as np
import time
import json
import os
import sys
import argparse
from datetime import datetime
from collections import deque

try:
    from ultralytics import YOLO
except ImportError:
    print("YOLO not available. Please install: py -m pip install ultralytics")
    sys.exit(1)

class CameraOnlyAITester:
    def __init__(self, camera_id=0, mock_ultrasonic=True):
        self.camera_id = camera_id
        self.mock_ultrasonic = mock_ultrasonic
        self.camera = None
        self.model = None
        self.running = False
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.detection_history = deque(maxlen=10)
        self.decision_history = deque(maxlen=20)
        
        # Mock ultrasonic values (since sensors not used)
        self.mock_sensors = {
            'center': 50.0,  # Safe distance
            'left45': 45.0,
            'right45': 45.0,
            'timestamp': 0
        }
        
        # Display resolution (reasonable for testing)
        self.display_size = (800, 600)
        
        print("Camera-Only AI Tester Initialized")
        print(f"Camera ID: {camera_id}")
        print(f"Display Resolution: {self.display_size}")
        print(f"Mock Ultrasonic: {mock_ultrasonic}")
    
    def initialize_camera(self):
        """Initialize laptop webcam with robust backend selection"""
        # Backend priority: DirectShow -> MSMF -> Auto (Windows optimization)
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        backend_names = ["DirectShow", "MSMF", "Auto"]
        
        for backend_idx, backend in enumerate(backends):
            print(f"Trying {backend_names[backend_idx]} backend...")
            
            # Try different camera IDs with this backend
            for cam_id in range(3):
                try:
                    # Create camera with specific backend
                    self.camera = cv2.VideoCapture(cam_id, backend)
                    
                    if not self.camera.isOpened():
                        continue
                    
                    # Quick test - try to read a frame immediately
                    ret, test_frame = self.camera.read()
                    if not ret or test_frame is None:
                        self.camera.release()
                        continue
                    
                    print(f"✓ Camera {cam_id} opened with {backend_names[backend_idx]} backend")
                    print(f"  Initial frame: {test_frame.shape[1]}x{test_frame.shape[0]}")
                    
                    # Store working configuration
                    self.camera_id = cam_id
                    self.backend = backend
                    self.backend_name = backend_names[backend_idx]
                    
                    # Set camera properties with validation
                    self._set_camera_properties()
                    
                    # Final test after property setting
                    ret, final_frame = self.camera.read()
                    if not ret or final_frame is None:
                        print("  ⚠ Camera failed after setting properties, trying next...")
                        self.camera.release()
                        continue
                    
                    print(f"✓ Camera configured: {final_frame.shape[1]}x{final_frame.shape[0]}")
                    return True
                    
                except Exception as e:
                    print(f"  ⚠ Camera {cam_id} with {backend_names[backend_idx]} failed: {e}")
                    if hasattr(self, 'camera') and self.camera is not None:
                        self.camera.release()
                    continue
            
            print("❌ No working camera found after trying all backends and IDs")
            return False
    
    def _set_camera_properties(self):
        """Set camera properties with validation"""
        properties = [
            (cv2.CAP_PROP_FRAME_WIDTH, 800, "Width"),
            (cv2.CAP_PROP_FRAME_HEIGHT, 600, "Height"),
            (cv2.CAP_PROP_FPS, 30, "FPS")
        ]
        
        for prop, value, name in properties:
            # Set property
            self.camera.set(prop, value)
            
            # Verify it was set (allow some tolerance)
            actual = self.camera.get(prop)
            if name == "FPS":
                # FPS can vary, just check it's reasonable
                if actual < 5:
                    print(f"  ⚠ {name} too low: {actual:.1f}, using default")
            else:
                # For resolution, check if it's close
                if abs(actual - value) > value * 0.5:
                    print(f"  ⚠ {name} not set correctly: {actual} (wanted {value})")
        
        # Set MJPG codec for better compatibility (optional)
        try:
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
        except:
            pass  # Ignore if codec setting fails
    
    def initialize_model(self, model_path=None, qwen_model_path=None):
        """Initialize YOLO26n model and Qwen reasoning engine"""
        try:
            # Get project root for standardized paths
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Load YOLO26n model from standardized models directory
            yolo_path = os.path.join(project_root, "models", "yolo26n.pt")
            yolo_fallback_paths = [
                yolo_path,
                "yolo26n.pt"
            ]
            
            self.model = None
            for path in yolo_fallback_paths:
                if os.path.exists(path):
                    try:
                        self.model = YOLO(path)
                        print(f"✅ YOLO26n model loaded: {path}")
                        break
                    except Exception as e:
                        print(f"⚠ YOLO model load failed: {path} - {e}")
                        continue
            
            if self.model is None:
                print("❌ YOLO26n model not found, using placeholder detection")
            else:
                print("✅ YOLO26n loaded successfully")
            
            # Initialize hybrid Qwen reasoning system (simplified)
            self.qwen_mode = "rule_based"
            self.qwen_available = False  # Simplified for stability
            
            return True
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            print("  Using placeholder detection for testing")
            self.model = None
            self.qwen_available = False
            return True
    
    def detect_objects(self, frame):
        """Run object detection using YOLO26n or placeholder"""
        if self.model is not None:
            # Real YOLO26n detection
            results = self.model(frame, verbose=False)
            detections = []
            
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    # Get detection info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_names = self.model.names
                    class_name = class_names.get(cls, f'class_{cls}')
                    
                    # Calculate normalized center and size
                    h, w = frame.shape[:2]
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bbox_w = (x2 - x1) / w
                    bbox_h = (y2 - y1) / h
                    
                    detections.append({
                        'label': class_name,
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'bbox_norm': {
                            'cx': cx,
                            'cy': cy,
                            'w': bbox_w,
                            'h': bbox_h
                        }
                    })
            
            return detections
        else:
            # Placeholder detection (simple edge detection)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for i, contour in enumerate(contours[:3]):  # Max 3 detections
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum size threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate normalized coordinates
                    h_img, w_img = frame.shape[:2]
                    cx = (x + w/2) / w_img
                    cy = (y + h/2) / h_img
                    norm_w = w / w_img
                    norm_h = h / h_img
                    
                    confidence = min(0.8, area / 5000.0)
                    
                    detections.append({
                        'label': f'object_{i}',
                        'confidence': confidence,
                        'bbox': [float(x), float(y), float(x+w), float(y+h)],
                        'bbox_norm': {
                            'cx': cx,
                            'cy': cy,
                            'w': norm_w,
                            'h': norm_h
                        }
                    })
            
            return detections
    
    def make_ai_decision(self, detections, ultrasonic_data=None):
        """Make AI decision using rule-based reasoning (simplified)"""
        if not detections:
            return 'forward', 0.95, 'Clear path ahead - rule based'
        
        # Find best detection (highest confidence)
        best_detection = max(detections, key=lambda d: d['confidence'])
        bbox_norm = best_detection['bbox_norm']
        cx = bbox_norm['cx']
        w = bbox_norm['w']
        confidence = best_detection['confidence']
        label = best_detection['label']
        
        # Safety check - object too close
        if w > 0.6:
            return 'stop', 0.8, f'Object too close: {label}'
        
        # Navigation decision based on center position
        if 0.4 <= cx <= 0.6:  # Centered
            return 'forward', confidence, f'Centered: {label}'
        elif cx < 0.4:  # Object on left
            return 'turn_right', confidence, f'Turn right for: {label}'
        else:  # Object on right
            return 'turn_left', confidence, f'Turn left for: {label}'
    
    def run_test(self):
        """Run camera-only AI test"""
        print("\n" + "="*60)
        print("CAMERA-ONLY AI TEST")
        print("="*60)
        print("Instructions:")
        print("- Hold objects in front of camera")
        print("- Press 'q' to quit, 's' to save screenshot")
        print()
        
        if not self.initialize_camera():
            print("❌ Camera initialization failed")
            return False
        
        if not self.initialize_model():
            print("❌ Model initialization failed")
            return False
        
        print("✅ Starting camera feed...")
        print(f"YOLO Model: {'Loaded' if self.model else 'Not found (placeholder)'}")
        print(f"Qwen Available: {'Yes' if self.qwen_available else 'No'}")
        
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                print("Camera read failed!")
                break
            
            frame_count += 1
            
            # Run object detection
            detections = self.detect_objects(frame)
            action, confidence, reason = self.make_ai_decision(detections)
            
            # Track performance
            self.fps_history.append(1.0 / (time.time() - start_time) * frame_count)
            
            # Draw decision overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Decision text
            decision_color = (0, 255, 0) if action == 'forward' else (0, 165, 255) if action in ['turn_left', 'turn_right'] else (0, 0, 255)
            cv2.putText(frame, f"Decision: {action.upper()}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, decision_color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add performance metrics
            fps = self.fps_history[-1] if self.fps_history else 0
            cv2.putText(frame, f"FPS:{fps:.1f} Objs:{len(detections)}", 
                       (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            
            # Display results
            cv2.imshow('Arduino UNO Q4GB AI - Camera Test', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                filename = f"camera_test_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        
        # Cleanup
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"YOLO Model: {'Loaded' if self.model else 'Not found'}")
        print(f"Total Frames: {frame_count}")
        
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            print(f"Average FPS: {avg_fps:.1f}")
        
        print("Test completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Arduino UNO Q4GB AI Camera Test')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--test', action='store_true', help='Run test')
    
    args = parser.parse_args()
    
    print("Arduino UNO Q4GB AI Robot - Camera-Only Test")
    print("="*60)
    
    # Create and run tester
    tester = CameraOnlyAITester(camera_id=args.camera)
    
    try:
        if args.test or True:  # Default to test
            success = tester.run_test()
            return 0 if success else 1
        else:
            print("Use --test to run the test")
            return 0
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 0
    except Exception as e:
        print(f"Test error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())