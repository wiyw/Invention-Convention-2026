#!/usr/bin/env python3
"""
Camera AI Pipeline for Arduino UNO Q4GB Testing
Real-time AI decision making using laptop camera without ultrasonic sensors
"""

import cv2
import numpy as np
import time
import json
import os
import sys
from datetime import datetime

class CameraAIPipeline:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.camera = None
        self.model = None
        self.running = False
        
        # Configuration
        self.input_size = (160, 120)  # For AI processing
        self.display_size = (800, 600)  # Reasonable display size
        self.fps_target = 15
        
        # Mock sensor data (since ultrasonic not used)
        self.mock_sensors = {
            'center': 50.0,
            'left45': 45.0, 
            'right45': 45.0,
            'timestamp': time.time()
        }
        
        print("Camera AI Pipeline Initialized")
        print(f"Camera: {camera_id}")
        print(f"Resolution: {self.input_size}")
    
    def initialize(self):
        """Initialize camera and AI model"""
        # Initialize camera
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            print(f"❌ Cannot open camera {self.camera_id}")
            return False
        
        # Set camera properties (reasonable size for testing)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps_target)
        
        # Test camera
        ret, frame = self.camera.read()
        if not ret:
            print("❌ Camera test failed")
            return False
        
        # Use standardized model paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        possible_paths = [
            os.path.join(project_root, "models", "yolo26n.pt"),  # Standardized location
            os.path.join(project_root, "yolo26n.pt"),  # Fallback to project root
            "yolo26n.pt"  # Current directory fallback
        ]
        
        model_found = False
        for model_path in possible_paths:
            if os.path.exists(model_path):
                try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"✅ YOLO26n model loaded: {model_path}")
                model_found = True
                break
            except Exception as e:
                print(f"⚠️  YOLO loading failed: {e}")
                print("Using placeholder detection")
                self.model = None
        
        if not model_found:
            print("❌ YOLO model not found in any location")
            print("  Checked paths:")
            for path in possible_paths:
                print(f"    - {path}")
            print("  To fix: Copy yolo26n.pt to project root")
            print("  Using placeholder detection for testing")
            self.model = None
        
        print(f"✅ Camera {self.camera_id} initialized")
        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        return True
    
    def process_frame(self, frame):
        """Process single frame through AI pipeline"""
        start_time = time.time()
        
        # 1. Resize for TinyYOLO
        small_frame = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # 2. Object detection
        detections = self.detect_objects(small_frame)
        detection_time = (time.time() - start_time) * 1000
        
        # 3. AI decision making
        decision_start = time.time()
        action, confidence, reason = self.make_decision(detections)
        decision_time = (time.time() - decision_start) * 1000
        
        # 4. Create output
        result = {
            'detections': detections,
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'detection_time_ms': detection_time,
            'decision_time_ms': decision_time,
            'total_time_ms': (time.time() - start_time) * 1000
        }
        
        return result, small_frame
    
    def detect_objects(self, frame):
        """Detect objects using YOLO or placeholder"""
        if self.model is not None:
            # Real YOLO26n detection
            results = self.model(frame, verbose=False)
            detections = []
            
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    class_names = self.model.names
                    label = class_names.get(cls, f'class_{cls}')
                    
                    # Normalize coordinates
                    h, w = frame.shape[:2]
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    norm_w = (x2 - x1) / w
                    norm_h = (y2 - y1) / h
                    
                    detections.append({
                        'label': label,
                        'confidence': float(conf),
                        'bbox_norm': {
                            'cx': cx,
                            'cy': cy,
                            'w': norm_w,
                            'h': norm_h
                        }
                    })
            
            return detections
        else:
            # Simple edge detection placeholder
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for i, contour in enumerate(contours[:3]):
                area = cv2.contourArea(contour)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Normalize coordinates
                    h_img, w_img = frame.shape[:2]
                    cx = (x + w/2) / w_img
                    cy = (y + h/2) / h_img
                    norm_w = w / w_img
                    norm_h = h / h_img
                    
                    confidence = min(0.8, area / 2000.0)
                    
                    detections.append({
                        'label': f'object_{i}',
                        'confidence': confidence,
                        'bbox_norm': {
                            'cx': cx,
                            'cy': cy,
                            'w': norm_w,
                            'h': norm_h
                        }
                    })
            
            return detections
    
    def make_decision(self, detections):
        """Make AI decision from detections"""
        if not detections:
            return 'forward', 0.9, 'Clear path - move forward'
        
        # Find best detection
        best = max(detections, key=lambda d: d['confidence'])
        bbox_norm = best['bbox_norm']
        cx = bbox_norm['cx']
        w = bbox_norm['w']
        confidence = best['confidence']
        label = best['label']
        
        # Safety check - object too close
        if w > 0.6:
            return 'stop', 0.8, f'Object too close: {label}'
        
        # Navigation decision
        if 0.35 <= cx <= 0.65:  # Centered
            if confidence > 0.7:
                return 'forward', confidence, f'Following: {label}'
            else:
                return 'forward', 0.6, f'Cautious forward: {label}'
        elif cx < 0.35:  # Object left
            return 'turn_right', confidence, f'Turn right: {label}'
        else:  # Object right
            return 'turn_left', confidence, f'Turn left: {label}'
    
    def draw_results(self, frame, result):
        """Draw detection results on frame"""
        # Scale frame back to display size
        display_frame = cv2.resize(frame, self.display_size, interpolation=cv2.INTER_LINEAR)
        
        # Draw detections
        for detection in result['detections']:
            bbox_norm = detection['bbox_norm']
            label = detection['label']
            conf = detection['confidence']
            
            # Convert normalized coordinates to display frame
            scale_x, scale_y = self.display_size[0], self.display_size[1]
            x1 = int(bbox_norm['cx'] * scale_x - bbox_norm['w'] * scale_x / 2)
            y1 = int(bbox_norm['cy'] * scale_y - bbox_norm['h'] * scale_y / 2)
            x2 = int(bbox_norm['cx'] * scale_x + bbox_norm['w'] * scale_x / 2)
            y2 = int(bbox_norm['cy'] * scale_y + bbox_norm['h'] * scale_y / 2)
            
            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(scale_x - 1, x2), min(scale_y - 1, y2)
            
            # Draw bounding box
            color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            text = f"{label} {conf:.2f}"
            cv2.putText(display_frame, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw decision overlay
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (scale_x, 80), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
        
        # Decision text
        action = result['action'].upper()
        color = (0, 255, 0) if action == 'FORWARD' else (0, 165, 255) if action in ['TURN_LEFT', 'TURN_RIGHT'] else (0, 0, 255)
        
        cv2.putText(display_frame, f"Action: {action}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, int(0.7 * scale_factor), color, 2)
        cv2.putText(display_frame, f"Confidence: {result['confidence']:.2f}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, int(0.6 * scale_factor), (255, 255, 255), 2)
        
        # Performance metrics
        # Performance metrics
        # Performance metrics (scaled for high resolution)
        scale_factor = self.display_size[1] / 480  # Scale text for higher resolution
        cv2.putText(display_frame, f"Detect: {result['detection_time_ms']:.0f}ms | Decision: {result['decision_time_ms']:.0f}ms", 
                   (10, scale_y - 10), cv2.FONT_HERSHEY_SIMPLEX, int(0.4 * scale_factor), (255, 255, 0), 1)
        
        return display_frame
    
    def run_pipeline(self, show_display=True):
        """Run continuous AI pipeline"""
        print("\nStarting Camera AI Pipeline...")
        print("Press 'q' to quit, 's' to save screenshot")
        print("Camera-only mode (no ultrasonic sensors)")
        
        self.running = True
        frame_count = 0
        start_time = time.time()
        fps_history = []
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                print("Camera read failed!")
                break
            
            frame_count += 1
            
            # Process frame through AI pipeline
            result, processed_frame = self.process_frame(frame)
            
            # Calculate FPS
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
            
            # Display results
            if show_display:
                display_frame = self.draw_results(frame, result)
                cv2.imshow('Arduino UNO Q4GB AI Pipeline - High Resolution', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"ai_pipeline_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Screenshot saved: {filename}")
            
            # Print periodic status
            if frame_count % 100 == 0:
                avg_fps = np.mean(fps_history) if fps_history else 0
                print(f"Frame {frame_count}: {len(result['detections'])} objects, action={result['action']}, FPS={avg_fps:.1f}")
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        self.print_summary(frame_count, start_time, fps_history)
    
    def print_summary(self, frame_count, start_time, fps_history):
        """Print pipeline summary"""
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time
        
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"Total Frames: {frame_count}")
        print(f"Duration: {elapsed_time:.1f} seconds")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Peak FPS: {max(fps_history) if fps_history else 0:.1f}")
        
        print("\nPipeline completed successfully!")

def main():
    print("Arduino UNO Q4GB AI Robot - Camera Pipeline")
    print("="*60)
    print("Camera-only mode (no ultrasonic sensors required)")
    print()
    
    # Initialize pipeline
    pipeline = CameraAIPipeline(camera_id=0)
    
    if not pipeline.initialize():
        print("Failed to initialize camera pipeline")
        return
    
    try:
        # Run pipeline
        pipeline.run_pipeline(show_display=True)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"\nPipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()