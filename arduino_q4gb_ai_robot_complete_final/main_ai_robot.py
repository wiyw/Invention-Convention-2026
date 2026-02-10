#!/usr/bin/env python3
"""
Arduino UNO Q4GB Trash Collector Robot Application
Phase 3: Hardware-Specific Optimization
AI-Powered Trash Detection using YOLOv8n + Camera + Arduino Control
Compatible with Logitech C270 HD Webcam and Enhanced Simulation Mode
"""

import os
import sys
import json
import time
import threading
import numpy as np
from pathlib import Path

class TrashCollectorRobot:
    def __init__(self):
        self.config = {}
        self.framework = None
        self.models = {}
        self.camera = None
        self.arduino = None
        self.running = False
        self.detection_results = []
        self.camera_mode = "unknown"
        self.frame_count = 0
        
        # Setup paths
        self.install_dir = Path.home() / 'arduino_q4gb_ai_robot_phase3'
        self.models_dir = self.install_dir / 'models'
        self.hardware_dir = self.install_dir / 'hardware_detection'
        
        print("Trash Collector Robot Initialized")
        print("  Features: YOLOv8n AI Detection, Logitech C270 Support, Enhanced Simulation")
        
    def load_configuration(self):
        """Load system configuration"""
        print("Loading configuration...")
        
        config_file = self.install_dir / 'config.json'
        if not config_file.exists():
            print("Configuration file not found")
            return False
        
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            
            self.framework = self.config.get('framework', 'onnx')
            print(f"  Framework: {self.framework}")
            print(f"  Hardware optimized: {self.config.get('arduino_uno_q4gb', False)}")
            return True
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def initialize_ai_framework(self):
        """Initialize AI framework based on configuration"""
        print(f"Initializing {self.framework} framework...")
        
        try:
            if self.framework == 'onnx':
                import onnxruntime
                print("  ONNX Runtime loaded")
                self.ort_session = None
                return True
            elif self.framework == 'tflite':
                try:
                    import tflite_runtime
                    print("  TensorFlow Lite Runtime loaded")
                    self.tflite_interpreter = None
                    return True
                except:
                    import tensorflow
                    print("  TensorFlow loaded (fallback)")
                    self.tf_model = None
                    return True
            else:
                print(f"Unknown framework: {self.framework}")
                return False
                
        except ImportError as e:
            print(f"Framework import error: {e}")
            return False
    
    def load_model(self):
        """Load AI model for object detection"""
        print("Loading AI model...")
        
        model_file = None
        
        # Try to find model file
        if self.framework == 'onnx':
            onnx_model = self.models_dir / 'onnx' / 'yolov8n_int8.onnx'
            if onnx_model.exists():
                model_file = str(onnx_model)
            else:
                # Create placeholder model for testing
                print("  Using placeholder model (real model would be downloaded)")
                model_file = "placeholder"
        
        elif self.framework == 'tflite':
            tflite_model = self.models_dir / 'tflite' / 'mobilenetv2_int8.tflite'
            if tflite_model.exists():
                model_file = str(tflite_model)
            else:
                print("  Using placeholder model (real model would be downloaded)")
                model_file = "placeholder"
        
        if model_file:
            self.models['detection'] = {
                'file': model_file,
                'type': 'object_detection',
                'framework': self.framework
            }
            print(f"  Model loaded: {model_file}")
            return True
        else:
            print("No model found")
            return False
    
    def initialize_camera(self):
        """Initialize camera for video capture"""
        print("Initializing camera...")
        
        try:
            import cv2
            import os
            
            # Check for video devices first
            video_devices = []
            if os.path.exists('/dev'):
                for i in range(10):
                    if os.path.exists(f'/dev/video{i}'):
                        video_devices.append(i)
            
            print(f"  Found video devices: {video_devices}")
            
            # Try to initialize camera with available indices
            self.camera = None
            camera_index = None
            
            # Try Logitech C270 specific settings first
            for idx in video_devices if video_devices else [0]:
                print(f"  Trying camera index {idx}...")
                
                # Try with V4L2 backend for Linux
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                
                if cap.isOpened():
                    # Test if we can actually read a frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.camera = cap
                        camera_index = idx
                        
                        # Optimize settings for Logitech C270
                        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.camera.set(cv2.CAP_PROP_FPS, 15)
                        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                        
                        break
                    else:
                        cap.release()
                else:
                    cap.release()
            
            if self.camera is not None and self.camera.isOpened():
                # Get camera properties
                width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.camera.get(cv2.CAP_PROP_FPS)
                backend = self.camera.getBackendName()
                
                print(f"  Camera initialized (Logitech C270)")
                print(f"  Index: {camera_index}, Backend: {backend}")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {fps}")
                self.camera_mode = "real"
                return True
            else:
                print("  No working camera found (using enhanced simulation mode)")
                self.camera = None
                self.camera_mode = "simulation"
                return True  # Continue without camera
                
        except ImportError:
            print("  OpenCV not available (using simulation mode)")
            self.camera = None
            self.camera_mode = "simulation"
            return True
        except Exception as e:
            print(f"  Camera initialization failed: {e}")
            print("  Falling back to enhanced simulation mode")
            self.camera = None
            self.camera_mode = "simulation"
            return True  # Continue without camera
    
    def initialize_arduino(self):
        """Initialize Arduino communication"""
        print("Initializing Arduino communication...")
        
        try:
            import serial
            import serial.tools.list_ports
            
            # List available serial ports
            ports = serial.tools.list_ports.comports()
            
            if ports:
                # Try first available port
                port = ports[0].device
                print(f"  Found Arduino at: {port}")
                
                try:
                    self.arduino = serial.Serial(port, 115200, timeout=1)
                    print(f"  Arduino connected: {port}")
                    return True
                except Exception as e:
                    print(f"  Arduino connection failed: {e}")
                    self.arduino = None
                    return True  # Continue without Arduino
            else:
                print("  No Arduino ports found (simulation mode)")
                self.arduino = None
                return True
                
        except ImportError:
            print("  PySerial not available (simulation mode)")
            self.arduino = None
            return True
        except Exception as e:
            print(f"  Arduino initialization failed: {e}")
            self.arduino = None
            return True  # Continue without Arduino
    
    def create_test_image(self, width=640, height=480):
        """Create enhanced test image for simulation mode with YOLOv8n-compatible objects"""
        import cv2
        import random
        import time
        
        # Create background (indoor/concrete floor look)
        image = np.full((height, width, 3), [120, 120, 120], dtype=np.uint8)
        
        # Add some texture to simulate floor
        noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Simulate trash objects for YOLOv8n detection
        self.frame_count += 1
        time_factor = (self.frame_count % 120) / 120.0  # 4 second cycle at 30fps
        
        # Trash object types that YOLOv8n can detect
        trash_objects = [
            {"type": "bottle", "color": [0, 150, 255], "size": (40, 80), "class_id": 39},  # bottle
            {"type": "cup", "color": [255, 150, 0], "size": (35, 45), "class_id": 41},     # cup
            {"type": "cell phone", "color": [100, 100, 200], "size": (30, 60), "class_id": 67},  # cell phone
            {"type": "book", "color": [139, 69, 19], "size": (60, 80), "class_id": 73},      # book
            {"type": "laptop", "color": [80, 80, 80], "size": (80, 60), "class_id": 63},    # laptop
            {"type": "person", "color": [0, 255, 0], "size": (100, 200), "class_id": 0},     # person
        ]
        
        # Simulate 2-4 moving trash objects
        num_objects = random.randint(2, 4)
        detected_objects = []
        
        for i in range(num_objects):
            obj = random.choice(trash_objects)
            
            # Create movement patterns
            if i == 0:
                # Object moving left to right
                x = int(100 + (width - 200) * time_factor)
                y = height // 2
            elif i == 1:
                # Object moving in circle
                angle = time_factor * 2 * 3.14159
                x = int(width // 2 + 150 * np.cos(angle))
                y = int(height // 2 + 100 * np.sin(angle))
            else:
                # Random movement
                x = int(width * (0.2 + 0.6 * abs(np.sin(time_factor + i))))
                y = int(height * (0.3 + 0.4 * abs(np.cos(time_factor * 1.5 + i))))
            
            # Ensure object stays within bounds
            w, h = obj["size"]
            x = max(w//2, min(width - w//2, x))
            y = max(h//2, min(height - h//2, y))
            
            # Draw object
            cv2.rectangle(image, (x - w//2, y - h//2), (x + w//2, y + h//2), obj["color"], -1)
            cv2.rectangle(image, (x - w//2, y - h//2), (x + w//2, y + h//2), (255, 255, 255), 2)
            
            # Add label
            label = f"{obj['type']}: {random.uniform(0.6, 0.95):.2f}"
            cv2.putText(image, label, (x - w//2, y - h//2 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add to detected objects for YOLOv8n processing
            detected_objects.append({
                'class': obj['type'],
                'class_id': obj['class_id'],
                'confidence': random.uniform(0.6, 0.95),
                'bbox': [x - w//2, y - h//2, w, h]
            })
        
        # Add simulation info overlay
        cv2.putText(image, "SIMULATION MODE - YOLOv8n", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f"Frame: {self.frame_count} Objects: {num_objects}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Store detected objects for processing
        self.simulation_detections = detected_objects
        
        return image
    
    def run_object_detection(self, image):
        """Run object detection using YOLOv8n"""
        try:
            detections = []
            
            if self.camera_mode == "simulation":
                # Use simulated detections from test image generation
                detections = getattr(self, 'simulation_detections', [])
            else:
                # Try to use YOLOv8n for real camera input
                try:
                    from ultralytics import YOLO
                    
                    # Load YOLOv8n model if not already loaded
                    if not hasattr(self, 'yolo_model'):
                        print("  Loading YOLOv8n model...")
                        self.yolo_model = YOLO('yolov8n.pt')
                        print("  YOLOv8n model loaded")
                    
                    # Run inference
                    results = self.yolo_model(image, verbose=False)
                    
                    # Process results
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                # Get box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].cpu().numpy()
                                cls = int(box.cls[0].cpu().numpy())
                                
                                # Get class name
                                class_name = self.yolo_model.names[cls]
                                
                                # Filter for trash-related objects
                                trash_classes = ['bottle', 'cup', 'cell phone', 'book', 'laptop', 'person', 'chair', 'backpack']
                                if class_name in trash_classes and conf > 0.5:
                                    detections.append({
                                        'class': class_name,
                                        'class_id': cls,
                                        'confidence': float(conf),
                                        'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                                    })
                    
                except ImportError:
                    print("  YOLOv8n not available, using simulated detections")
                    # Fallback to simple simulated detections
                    import random
                    num_detections = random.randint(0, 2)
                    for i in range(num_detections):
                        detection = {
                            'class': random.choice(['bottle', 'cup', 'cell phone']),
                            'confidence': random.uniform(0.6, 0.95),
                            'bbox': [random.randint(50, 300), random.randint(50, 200), 
                                   random.randint(30, 80), random.randint(40, 100)]
                        }
                        detections.append(detection)
                except Exception as e:
                    print(f"  YOLOv8n inference error: {e}")
                    # Fallback to basic detections
                    detections = []
            
            self.detection_results = detections
            
            # Print detection info
            if detections:
                trash_count = len(detections)
                print(f"  Detected {trash_count} objects:")
                for det in detections[:3]:  # Show first 3
                    print(f"    - {det['class']}: {det['confidence']:.2f}")
            
            return detections
            
        except Exception as e:
            print(f"  ❌ Detection error: {e}")
            return []
    
    def process_camera_frame(self):
        """Process a single camera frame"""
        import cv2
        
        try:
            if self.camera is None or self.camera_mode == "simulation":
                # Use enhanced test image in simulation mode
                image = self.create_test_image()
            else:
                ret, frame = self.camera.read()
                if not ret:
                    print("  Failed to capture frame, switching to simulation")
                    self.camera_mode = "simulation"
                    image = self.create_test_image()
                else:
                    image = frame
            
            # Run object detection
            detections = self.run_object_detection(image)
            
            # Draw detection results on image
            if detections:
                for detection in detections:
                    bbox = detection['bbox']
                    label = f"{detection['class']}: {detection['confidence']:.2f}"
                    
                    x, y, w, h = bbox
                    
                    # Choose color based on object type
                    if detection['class'] == 'person':
                        color = (0, 0, 255)  # Red for person
                    elif detection['class'] in ['bottle', 'cup']:
                        color = (255, 0, 0)  # Blue for plastic/glass
                    elif detection['class'] in ['book', 'laptop']:
                        color = (0, 255, 0)  # Green for paper/electronics
                    else:
                        color = (0, 255, 255)  # Yellow for other
                    
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, label, (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add mode indicator
            mode_text = f"Mode: {self.camera_mode.upper()}"
            if self.camera_mode == "simulation":
                mode_text += " (YOLOv8n Simulated)"
            else:
                mode_text += " (YOLOv8n Live)"
            
            cv2.putText(image, mode_text, (10, image.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return image
            
        except Exception as e:
            print(f"  Frame processing error: {e}")
            # Fallback to basic test image
            return np.full((480, 640, 3), [100, 100, 100], dtype=np.uint8)
    
    def send_to_arduino(self, detections):
        """Send detection results to Arduino for trash collection"""
        if self.arduino is None:
            return
        
        try:
            # Filter for trash-related objects
            trash_objects = [d for d in detections if d['class'] in ['bottle', 'cup', 'cell phone', 'book', 'laptop']]
            num_trash = len(trash_objects)
            
            if trash_objects:
                # Find nearest trash object
                nearest = min(trash_objects, key=lambda x: (x['bbox'][0] + x['bbox'][2]//2))
                center_x = nearest['bbox'][0] + nearest['bbox'][2]//2
                center_y = nearest['bbox'][1] + nearest['bbox'][3]//2
                
                # Send detailed detection info to Arduino
                message = f"TRASH:{num_trash}:{nearest['class']}:{center_x}:{center_y}\n"
                print(f"  Found {num_trash} trash items, nearest: {nearest['class']} at ({center_x},{center_y})")
            else:
                message = "TRASH:0:NONE:0:0\n"
            
            self.arduino.write(message.encode())
            self.arduino.flush()
            
            # Read response
            response = self.arduino.readline().decode().strip()
            if response:
                print(f"  Arduino: {response}")
            
        except Exception as e:
            print(f"  Arduino communication error: {e}")
    
    def run_ai_loop(self):
        """Main AI processing loop for Trash Collector Robot"""
        print("Starting Trash Collector Robot AI loop...")
        print(f"  Mode: {self.camera_mode.upper()}")
        print("  Press Ctrl+C to stop")
        
        frame_count = 0
        start_time = time.time()
        trash_collected = 0
        
        try:
            while self.running:
                # Process frame
                processed_image = self.process_camera_frame()
                
                if processed_image is not None:
                    frame_count += 1
                    
                    # Send results to Arduino for trash collection
                    self.send_to_arduino(self.detection_results)
                    
                    # Count trash items detected
                    current_trash = len([d for d in self.detection_results if d['class'] in ['bottle', 'cup', 'cell phone', 'book', 'laptop']])
                    if current_trash > 0:
                        trash_collected += current_trash
                    
                    # Print status every 30 frames
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        print(f"  Frames: {frame_count}, FPS: {fps:.1f}, Trash detected: {current_trash}, Total processed: {trash_collected}")
                
                # Adaptive delay based on mode
                if self.camera_mode == "simulation":
                    time.sleep(0.033)  # ~30 FPS for simulation
                else:
                    time.sleep(0.050)  # ~20 FPS for real camera (more stable)
                
        except KeyboardInterrupt:
            print("\n  Trash Collector Robot stopped by user")
        except Exception as e:
            print(f"  AI loop error: {e}")
        
        finally:
            # Calculate final statistics
            if frame_count > 0:
                elapsed_time = time.time() - start_time
                avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"  Final stats: {frame_count} frames, Average FPS: {avg_fps:.1f}")
                print(f"  Total trash items processed: {trash_collected}")
                print(f"  Collection efficiency: {trash_collected/(frame_count/30):.2f} items/second")
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...")
        
        if self.camera is not None:
            self.camera.release()
            print("  Camera released")
        
        if self.arduino is not None:
            self.arduino.close()
            print("  Arduino connection closed")
        
        self.running = False
        print("  Cleanup complete")
    
    def start(self):
        """Start the Trash Collector Robot"""
        print("Trash Collector Robot Starting...")
        print("=" * 50)
        print("  AI Engine: YOLOv8n")
        print("  Camera: Logitech C270 HD Webcam (with simulation fallback)")
        print("  Target Objects: Bottles, Cups, Cell Phones, Books, Laptops")
        print("=" * 50)
        
        # Load configuration
        if not self.load_configuration():
            print("  No config file found, using defaults")
        
        # Initialize components
        if not self.initialize_ai_framework():
            print("  AI framework not available, using basic detection")
        
        if not self.load_model():
            print("  Model not loaded, using simulation")
        
        if not self.initialize_camera():
            return False
        
        if not self.initialize_arduino():
            print("  Arduino not available, running in standalone mode")
        
        # Start AI loop
        self.running = True
        self.run_ai_loop()
        
        return True

def main():
    """Main function"""
    robot = TrashCollectorRobot()
    
    try:
        # Start the robot
        if robot.start():
            print("Trash Collector Robot completed successfully!")
        else:
            print("Trash Collector Robot failed to start!")
            
    except KeyboardInterrupt:
        print("\nTrash Collector Robot stopped by user")
    except Exception as e:
        print(f"Trash Collector Robot error: {e}")
    finally:
        robot.cleanup()

if __name__ == "__main__":
    main()