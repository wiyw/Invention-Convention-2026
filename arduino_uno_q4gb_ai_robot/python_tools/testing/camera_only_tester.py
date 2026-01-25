#!/usr/bin/env python3
"""
Windows Camera-Only Testing Suite for Arduino UNO Q4GB AI Robot
Tests AI system using laptop built-in webcam without ultrasonic sensors
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
    print("YOLO not available. Please install: pip install ultralytics")
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
        
        # TinyYOLO input size (for AI processing)
        self.input_size = (160, 120)
        # Display resolution (reasonable for testing)
        self.display_size = (800, 600)
        
        print("Camera-Only AI Tester Initialized")
        print(f"Camera ID: {camera_id}")
        print(f"Input Resolution: {self.input_size}")
        print(f"Mock Ultrasonic: {mock_ultrasonic}")
    
    def initialize_camera(self):
        """Initialize laptop webcam"""
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {self.camera_id}")
            
            # Set camera properties (reasonable size for testing)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret or frame is None:
                raise Exception("Camera test failed")
            
            print(f"✅ Camera {self.camera_id} initialized successfully")
            print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def initialize_model(self, model_path=None):
        """Initialize YOLO26n model"""
        try:
            # Try multiple possible model paths
            possible_paths = [
                "yolo26n.pt",  # Project root
                "../yolo26n.pt",  # From testing directory
                "../../yolo26n.pt",  # From deeper directory
                "yolo26n/yolo26n.pt",  # Original location
                "../yolo26n/yolo26n.pt",  # From testing dir
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolo26n.pt"),  # Project root
            ]
            
            model_found = False
            for path in possible_paths:
                if os.path.exists(path):
                    self.model = YOLO(path)
                    print(f"✅ YOLO26n model loaded: {path}")
                    model_found = True
                    break
            
            if not model_found:
                print("❌ YOLO26n model not found in any location")
                print("  Checked paths:")
                for path in possible_paths:
                    print(f"    - {path}")
                print("  Using placeholder detection for testing")
                print("  To fix: Copy yolo26n.pt to project root or run from original directory")
                self.model = None
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("  Using placeholder detection for testing")
            self.model = None
            return True
    
    def preprocess_frame(self, frame):
        """Preprocess camera frame for TinyYOLO input"""
        # Resize to TinyYOLO input size
        resized = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize pixel values (0-1 range)
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized, resized
    
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
    
    def make_ai_decision(self, detections):
        """Make AI decision using simplified TinyQwen logic"""
        if not detections:
            return 'forward', 0.9, 'Clear path ahead'
        
        # Find best detection (highest confidence)
        best_detection = max(detections, key=lambda d: d['confidence'])
        bbox_norm = best_detection['bbox_norm']
        confidence = best_detection['confidence']
        label = best_detection['label']
        
        # Center position analysis
        cx = bbox_norm['cx']
        w = bbox_norm['w']
        
        # Safety check based on object size
        if w > 0.5:
            return 'stop', 0.8, f'Object too close: {label}'
        
        # Navigation decision based on center position
        if 0.4 <= cx <= 0.6:  # Centered
            if confidence > 0.6:
                return 'forward', confidence, f'Following: {label}'
            else:
                return 'forward', 0.7, f'Cautious forward: {label}'
        elif cx < 0.4:  # Object on left
            return 'turn_left', confidence, f'Turn left for: {label}'
        else:  # Object on right
            return 'turn_right', confidence, f'Turn right for: {label}'
    
    def draw_detections(self, frame, detections, decision, confidence):
        """Draw detection boxes and decision on frame"""
        # Draw detection boxes
        for detection in detections:
            bbox = detection['bbox']
            label = detection['label']
            conf = detection['confidence']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label and confidence (very compact)
            text = f"{label[:3]}"  # Very short label
            label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), 
                        (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw decision overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Decision text
        decision_color = (0, 255, 0) if decision == 'forward' else (0, 165, 255) if decision in ['turn_left', 'turn_right'] else (0, 0, 255)
        cv2.putText(frame, f"Decision: {decision.upper()}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, decision_color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run_interactive_test(self):
        """Run interactive camera test with real-time display"""
        print("\n" + "="*60)
        print("CAMERA-ONLY AI TEST - Interactive Mode")
        print("="*60)
        print("Instructions:")
        print("- Hold objects in front of camera to test detection")
        print("- Position objects left/right to test turning")
        print("- Press 'q' to quit, 's' to save screenshot")
        print("- Press 't' to trigger test sequence")
        print()
        
        if not self.initialize_camera():
            return False
        
        if not self.initialize_model():
            print("Continuing with placeholder detection...")
        
        print("Starting camera feed...")
        print("Position objects in front of camera to test AI detection and decisions")
        
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                print("Camera read failed!")
                break
            
            frame_count += 1
            
            # Preprocess frame
            processed_frame, display_frame = self.preprocess_frame(frame)
            
            # Run object detection
            detection_start = time.time()
            detections = self.detect_objects(display_frame)
            detection_time = (time.time() - detection_start) * 1000
            
            # Make AI decision
            decision_start = time.time()
            action, confidence, reason = self.make_ai_decision(detections)
            decision_time = (time.time() - decision_start) * 1000
            
            # Track metrics
            self.fps_history.append(1.0 / (time.time() - start_time) * frame_count)
            self.detection_history.append(len(detections))
            self.decision_history.append(action)
            
            # Draw results
            result_frame = self.draw_detections(display_frame, detections, action, confidence)
            
            # Resize for better display
            result_frame = cv2.resize(result_frame, self.display_size, interpolation=cv2.INTER_LINEAR)
            
            # Add performance metrics
            fps = self.fps_history[-1] if self.fps_history else 0
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, result_frame.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Objects: {len(detections)}", (10, result_frame.shape[0] - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Detect: {detection_time:.0f}ms | Decision: {decision_time:.0f}ms", 
                       (10, result_frame.shape[0] - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Display result with larger window
            cv2.imshow('Arduino UNO Q4GB AI Camera Test - High Resolution', result_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"camera_test_{timestamp}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('t'):
                # Trigger test sequence
                self.run_test_sequence(display_frame)
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        self.print_summary()
        
        return True
    
    def run_test_sequence(self, current_frame):
        """Run automated test sequence"""
        print("\n--- Running Test Sequence ---")
        
        test_scenarios = [
            ("Clear Path", "Hold no objects in camera view", "forward"),
            ("Front Object", "Hold object in center of view", "stop"),
            ("Left Object", "Hold object on left side", "turn_right"),
            ("Right Object", "Hold object on right side", "turn_left"),
            ("Multiple Objects", "Hold 2-3 objects in view", "turn_left/right or stop")
        ]
        
        for scenario_name, instruction, expected_action in test_scenarios:
            print(f"\n{scenario_name}:")
            print(f"  Instruction: {instruction}")
            print(f"  Expected: {expected_action}")
            
            # Wait 3 seconds for user to position objects
            for i in range(3, 0, -1):
                print(f"  Starting in {i}...", end='\r')
                time.sleep(1)
            
            print("\n  Testing now...")
            
            # Test for 5 seconds
            results = []
            start_time = time.time()
            
            while time.time() - start_time < 5:
                ret, frame = self.camera.read()
                if ret:
                    processed_frame, display_frame = self.preprocess_frame(frame)
                    detections = self.detect_objects(display_frame)
                    action, confidence, reason = self.make_ai_decision(detections)
                    results.append(action)
                    
                    # Show live feedback
                    result_frame = self.draw_detections(display_frame, detections, action, confidence)
                    result_frame = cv2.resize(result_frame, self.display_size, interpolation=cv2.INTER_LINEAR)
                    cv2.putText(result_frame, f"TEST: {scenario_name}", (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow('Test Sequence - High Resolution', result_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            cv2.destroyWindow('Test Sequence - High Resolution')
            
            # Analyze results
            if results:
                most_common = max(set(results), key=results.count)
                success_rate = results.count(expected_action) / len(results) * 100
                print(f"  Result: {most_common} ({success_rate:.1f}% correct)")
            else:
                print("  Result: No data collected")
        
        print("\n--- Test Sequence Complete ---")
    
    def run_benchmark(self, duration=30):
        """Run performance benchmark test"""
        print(f"\nRunning {duration}-second benchmark...")
        
        if not self.initialize_camera() or not self.initialize_model():
            return False
        
        self.running = True
        start_time = time.time()
        frame_count = 0
        total_detections = 0
        total_detection_time = 0
        total_decision_time = 0
        
        while self.running and (time.time() - start_time) < duration:
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Process frame
            processed_frame, display_frame = self.preprocess_frame(frame)
            
            # Detection timing
            det_start = time.time()
            detections = self.detect_objects(display_frame)
            total_detection_time += (time.time() - det_start)
            total_detections += len(detections)
            
            # Decision timing
            dec_start = time.time()
            action, confidence, reason = self.make_ai_decision(detections)
            total_decision_time += (time.time() - dec_start)
            
            # Brief display every 100 frames
            if frame_count % 100 == 0:
                print(f"  Frame {frame_count}: {len(detections)} objects, action={action}")
        
        # Calculate metrics
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        avg_detection_time = (total_detection_time / frame_count) * 1000
        avg_decision_time = (total_decision_time / frame_count) * 1000
        avg_detections = total_detections / frame_count
        
        print(f"\nBenchmark Results ({duration}s):")
        print(f"  FPS: {fps:.2f}")
        print(f"  Avg Detection Time: {avg_detection_time:.1f}ms")
        print(f"  Avg Decision Time: {avg_decision_time:.1f}ms")
        print(f"  Avg Detections/Frame: {avg_detections:.2f}")
        
        self.camera.release()
        return True
    
    def print_summary(self):
        """Print testing summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            print(f"Average FPS: {avg_fps:.1f}")
        
        if self.detection_history:
            avg_detections = np.mean(self.detection_history)
            max_detections = np.max(self.detection_history)
            print(f"Average Detections: {avg_detections:.1f}")
            print(f"Maximum Detections: {max_detections}")
        
        if self.decision_history:
            decision_counts = {}
            for decision in self.decision_history:
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
            
            print("Decision Distribution:")
            for decision, count in decision_counts.items():
                percentage = (count / len(self.decision_history)) * 100
                print(f"  {decision}: {count} ({percentage:.1f}%)")

def create_shortcuts():
    """Create desktop shortcuts to essential folders"""
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        shortcuts = [
            ("Arduino AI Robot", project_root),
            ("Camera Testing", os.path.join(project_root, "python_tools", "testing")),
            ("Arduino Firmware", os.path.join(project_root, "arduino_firmware")),
            ("Documentation", os.path.join(project_root, "docs")),
            ("Windows Setup", os.path.join(project_root, "windows_setup"))
        ]
        
        for name, path in shortcuts:
            shortcut_path = os.path.join(desktop, f"{name}.lnk")
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = path
            shortcut.WorkingDirectory = path
            shortcut.IconLocation = "shell32.dll,3"
            shortcut.Save()
            print(f"✅ Created shortcut: {name}")
        
        return True
        
    except ImportError:
        print("Creating shortcuts requires winshell and pywin32")
        print("Install with: pip install winshell pywin32")
        return False
    except Exception as e:
        print(f"Error creating shortcuts: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Arduino UNO Q4GB AI Camera Test (Windows Only)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--no-ultrasonic', action='store_true', help='Disable ultrasonic sensor mock')
    parser.add_argument('--benchmark', type=int, help='Run benchmark for N seconds')
    parser.add_argument('--test', action='store_true', help='Run interactive test')
    parser.add_argument('--create-shortcuts', action='store_true', help='Create desktop shortcuts')
    
    args = parser.parse_args()
    
    print("Arduino UNO Q4GB AI Robot - Windows Camera-Only Testing")
    print("="*70)
    
    if args.create_shortcuts:
        create_shortcuts()
        return
    
    # Create tester instance
    tester = CameraOnlyAITester(
        camera_id=args.camera,
        mock_ultrasonic=not args.no_ultrasonic
    )
    
    try:
        if args.benchmark:
            tester.run_benchmark(duration=args.benchmark)
        elif args.test:
            tester.run_interactive_test()
        else:
            # Default to interactive test
            tester.run_interactive_test()
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()