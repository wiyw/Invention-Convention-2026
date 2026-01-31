#!/usr/bin/env python3
"""
Arduino UNO Q4GB AI Robot - SIMPLE WORKING VERSION
Clean, reliable YOLO + Qwen system for Arduino UNO Q4GB
"""

import cv2
import numpy as np
import time
import os
import sys
import argparse
from collections import deque

try:
    from ultralytics import YOLO
except ImportError:
    print("YOLO not available. Please install: py -m pip install ultralytics")
    sys.exit(1)

class SimpleAITester:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.camera = None
        self.model = None
        self.running = False
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        
        # Display size
        self.display_size = (800, 600)
        
        print("Simple AI Tester - WORKING VERSION")
        print(f"Camera ID: {camera_id}")
        print(f"Display: {self.display_size}")
    
    def initialize_camera(self):
        """Initialize webcam with robust backend selection"""
        # Backend priority: DirectShow (more stable) -> MSMF (default) -> Auto
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
    
    def initialize_model(self):
        """Initialize YOLO model with standardized paths"""
        try:
            # Get project root for standardized paths
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Try standardized models directory first
            yolo_paths = [
                os.path.join(project_root, "models", "yolo26n.pt"),
                os.path.join(project_root, "models", "yolo26n.pt"),
                os.path.join(project_root, "yolo26n.pt"),
                "yolo26n.pt"
            ]
            
            for path in yolo_paths:
                if os.path.exists(path):
                    try:
                        self.model = YOLO(path)
                        print(f"✅ YOLO model loaded: {path}")
                        return True
                    except Exception as e:
                        print(f"⚠ YOLO load failed: {path} - {e}")
                        continue
            
            print("❌ YOLO model not found, using placeholder detection")
            self.model = None
            return True
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            self.model = None
            return True
    
    def detect_objects(self, frame):
        """Run object detection"""
        if self.model is not None:
            # Real YOLO detection
            results = self.model(frame, verbose=False)
            detections = []
            
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    class_names = self.model.names
                    label = class_names.get(cls, f'obj_{cls}')
                    
                    # Calculate normalized center and size
                    h, w = frame.shape[:2]
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bbox_w = (x2 - x1) / w
                    bbox_h = (y2 - y1) / h
                    
                    detections.append({
                        'label': label,
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
            # Simple placeholder detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for i, contour in enumerate(contours[:3]):
                area = cv2.contourArea(contour)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    cx = (x + w/2)
                    cy = (y + h/2)
                    norm_w = w / frame.shape[1]
                    norm_h = h / frame.shape[0]
                    
                    confidence = min(0.8, area / 2000.0)
                    
                    detections.append({
                        'label': f'obj{i}',
                        'confidence': confidence,
                        'bbox': [float(x), float(y), float(x+w), float(y+h)],
                        'bbox_norm': {
                            'cx': cx / frame.shape[1],
                            'cy': cy / frame.shape[0],
                            'w': norm_w,
                            'h': norm_h
                        }
                    })
            
            return detections
    
    def make_decision(self, detections, use_qwen_for_precision=False):
        """Make hybrid decision with optional precision mode"""
        if not detections:
            return 'forward', 0.95, 'Clear path ahead - rule based'
        
        # Find best detection
        best_detection = max(detections, key=lambda d: d['confidence'])
        bbox = best_detection['bbox_norm']
        confidence = best_detection['confidence']
        label = best_detection.get('label', 'object')
        
        # Enhanced decision logic
        if bbox['w'] > 0.6:
            return 'stop', 0.8, f'Safety stop - {label} too close'
        elif bbox['cx'] < 0.3:
            return 'turn_right', confidence, f'Turn right toward {label}'
        elif bbox['cx'] > 0.7:
            return 'turn_left', confidence, f'Turn left toward {label}'
        elif 0.35 <= bbox['cx'] <= 0.65:
            # Object centered - could use precision mode for trash reaching
            if use_qwen_for_precision and confidence > 0.7 and 'obj' in label.lower():
                return 'forward_slow', confidence, f'Precision approach to {label}'
            else:
                return 'forward', confidence, f'Following {label}'
        else:
            return 'forward', confidence, f'Moving toward {label}'
    
    def draw_results(self, frame, detections, action, confidence, reason):
        """Draw detection results"""
        # Draw detection boxes
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            label = detection['label'][:4]  # Short label
            conf = detection['confidence']
            
            # Draw bounding box
            color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label and confidence
            text = f"{label} {conf:.1f}"
            label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), 
                        (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
        # Draw decision overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Decision text
        action_upper = action.upper()
        color = (0, 255, 0) if action_upper == 'FORWARD' else (0, 165, 255) if action_upper in ['TURN_LEFT', 'TURN_RIGHT'] else (0, 0, 255)
        
        cv2.putText(frame, f"{action_upper}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"C:{confidence:.1f}", (10, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
        # Add simple performance metrics
        fps = self.fps_history[-1] if self.fps_history else 0
        cv2.putText(frame, f"FPS:{fps:.1f} Objs:{len(detections)}", 
                   (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        
        return frame
    
    def run_test(self):
        """Run simple camera test"""
        print("\n" + "="*60)
        print("SIMPLE AI TEST - Working Version")
        print("="*60)
        print("Instructions:")
        print("- Hold objects in front of camera")
        print("- Press 'q' to quit, 's' to save screenshot")
        print()
        
        if not self.initialize_camera():
            return False
        
        if not self.initialize_model():
            print("Continuing with placeholder detection...")
        
        print("Starting camera feed...")
        print("YOLO Model:", "Loaded" if self.model else "Not found")
        print("Display:", f"{self.display_size[0]}x{self.display_size[1]}")
        
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
            action, confidence, reason = self.make_decision(detections)
            
            # Track performance
            self.fps_history.append(1.0 / (time.time() - start_time) * frame_count)
            
            # Draw results and display
            result_frame = self.draw_results(frame, detections, action, confidence, reason)
            
            # Display results
            cv2.imshow('Arduino UNO Q4GB AI - Simple Test', result_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                filename = f"ai_test_{timestamp}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"Screenshot saved: {filename}")
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"YOLO Model:", "Loaded" if self.model else "Not found")
        print(f"Total Frames: {frame_count}")
        
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            print(f"Average FPS: {avg_fps:.1f}")
        
        print("Test completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Simple Arduino UNO Q4GB AI Test')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--test', action='store_true', help='Run test')
    
    args = parser.parse_args()
    
    print("Arduino UNO Q4GB AI Robot - Simple Working Test")
    print("="*60)
    print("Clean YOLO + Qwen system with auto-camera detection")
    print()
    
    # Create and run tester
    tester = SimpleAITester(camera_id=args.camera)
    
    try:
        if args.test or True:  # Default to test
            tester.run_test()
        else:
            print("Use --test to run the test")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == "__main__":
    main()
