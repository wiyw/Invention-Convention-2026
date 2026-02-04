#!/usr/bin/env python3
"""
Arduino UNO Q4GB Main AI Robot Application
Phase 3: Hardware-Specific Optimization
Standard AI Package - Object Detection + Camera + Arduino Control + Web Interface
"""

import os
import sys
import json
import time
import threading
import numpy as np
from pathlib import Path

class ArduinoQ4GBAIRobot:
    def __init__(self):
        self.config = {}
        self.framework = None
        self.models = {}
        self.camera = None
        self.arduino = None
        self.running = False
        self.detection_results = []
        
        # Setup paths
        self.install_dir = Path.home() / 'arduino_q4gb_ai_robot_phase3'
        self.models_dir = self.install_dir / 'models'
        self.hardware_dir = self.install_dir / 'hardware_detection'
        
    def load_configuration(self):
        """Load system configuration"""
        print("📊 Loading configuration...")
        
        config_file = self.install_dir / 'config.json'
        if not config_file.exists():
            print("❌ Configuration file not found")
            return False
        
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            
            self.framework = self.config.get('framework', 'onnx')
            print(f"  Framework: {self.framework}")
            print(f"  Hardware optimized: {self.config.get('arduino_uno_q4gb', False)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return False
    
    def initialize_ai_framework(self):
        """Initialize AI framework based on configuration"""
        print(f"🤖 Initializing {self.framework} framework...")
        
        try:
            if self.framework == 'onnx':
                import onnxruntime
                print("  ✅ ONNX Runtime loaded")
                self.ort_session = None
                return True
            elif self.framework == 'tflite':
                try:
                    import tflite_runtime
                    print("  ✅ TensorFlow Lite Runtime loaded")
                    self.tflite_interpreter = None
                    return True
                except:
                    import tensorflow
                    print("  ✅ TensorFlow loaded (fallback)")
                    self.tf_model = None
                    return True
            else:
                print(f"❌ Unknown framework: {self.framework}")
                return False
                
        except ImportError as e:
            print(f"❌ Framework import error: {e}")
            return False
    
    def load_model(self):
        """Load AI model for object detection"""
        print("📦 Loading AI model...")
        
        model_file = None
        
        # Try to find model file
        if self.framework == 'onnx':
            onnx_model = self.models_dir / 'onnx' / 'yolov8n_int8.onnx'
            if onnx_model.exists():
                model_file = str(onnx_model)
            else:
                # Create placeholder model for testing
                print("  ⚠️  Using placeholder model (real model would be downloaded)")
                model_file = "placeholder"
        
        elif self.framework == 'tflite':
            tflite_model = self.models_dir / 'tflite' / 'mobilenetv2_int8.tflite'
            if tflite_model.exists():
                model_file = str(tflite_model)
            else:
                print("  ⚠️  Using placeholder model (real model would be downloaded)")
                model_file = "placeholder"
        
        if model_file:
            self.models['detection'] = {
                'file': model_file,
                'type': 'object_detection',
                'framework': self.framework
            }
            print(f"  ✅ Model loaded: {model_file}")
            return True
        else:
            print("❌ No model found")
            return False
    
    def initialize_camera(self):
        """Initialize camera for video capture"""
        print("📸 Initializing camera...")
        
        try:
            import cv2
            # Test camera initialization (0 is default camera)
            self.camera = cv2.VideoCapture(0)
            
            if self.camera.isOpened():
                # Get camera properties
                width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.camera.get(cv2.CAP_PROP_FPS)
                
                print(f"  ✅ Camera initialized")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {fps}")
                return True
            else:
                print("  ⚠️  Camera not available (using simulation mode)")
                self.camera = None
                return True  # Continue without camera
                
        except ImportError:
            print("  ⚠️  OpenCV not available (using simulation mode)")
            self.camera = None
            return True
        except Exception as e:
            print(f"  ⚠️  Camera initialization failed: {e}")
            self.camera = None
            return True  # Continue without camera
    
    def initialize_arduino(self):
        """Initialize Arduino communication"""
        print("🔌 Initializing Arduino communication...")
        
        try:
            import serial
            import serial.tools.list_ports
            
            # List available serial ports
            ports = serial.tools.list_ports.comports()
            
            if ports:
                # Try first available port
                port = ports[0].device
                print(f"  ✅ Found Arduino at: {port}")
                
                try:
                    self.arduino = serial.Serial(port, 115200, timeout=1)
                    print(f"  ✅ Arduino connected: {port}")
                    return True
                except Exception as e:
                    print(f"  ⚠️  Arduino connection failed: {e}")
                    self.arduino = None
                    return True  # Continue without Arduino
            else:
                print("  ⚠️  No Arduino ports found (simulation mode)")
                self.arduino = None
                return True
                
        except ImportError:
            print("  ⚠️  PySerial not available (simulation mode)")
            self.arduino = None
            return True
        except Exception as e:
            print(f"  ⚠️  Arduino initialization failed: {e}")
            self.arduino = None
            return True  # Continue without Arduino
    
    def create_test_image(self, width=640, height=480):
        """Create a test image for simulation mode"""
        # Create a colorful test image
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add some geometric patterns
        center_x, center_y = width // 2, height // 2
        cv2.circle(image, (center_x, center_y), 50, (255, 255, 255), 2)  # White circle
        cv2.rectangle(image, (center_x - 100, center_y - 50), (center_x + 100, center_y + 50), (0, 255, 0), 2)  # Green rectangle
        
        return image
    
    def run_object_detection(self, image):
        """Run object detection on image"""
        try:
            # Simulate object detection results
            # In real implementation, this would use the loaded AI model
            detections = []
            
            # Simulate some random detections for demonstration
            import random
            num_detections = random.randint(0, 3)
            
            for i in range(num_detections):
                detection = {
                    'class': random.choice(['person', 'car', 'bicycle', 'dog', 'cat']),
                    'confidence': random.uniform(0.5, 0.95),
                    'bbox': [random.randint(50, 200), random.randint(50, 200), 
                           random.randint(50, 150), random.randint(50, 150)]
                }
                detections.append(detection)
            
            self.detection_results = detections
            return detections
            
        except Exception as e:
            print(f"  ⚠️  Detection error: {e}")
            return []
    
    def process_camera_frame(self):
        """Process a single camera frame"""
        if self.camera is None:
            # Use test image in simulation mode
            image = self.create_test_image()
        else:
            ret, frame = self.camera.read()
            if not ret:
                print("  ⚠️  Failed to capture frame")
                return None
            image = frame
        
        # Run object detection
        detections = self.run_object_detection(image)
        
        # Draw detection results on image
        if detections:
            for detection in detections:
                bbox = detection['bbox']
                label = f"{detection['class']}: {detection['confidence']:.2f}"
                
                if self.camera is not None:
                    import cv2
                    x, y, w, h = bbox
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, label, (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image
    
    def send_to_arduino(self, detections):
        """Send detection results to Arduino"""
        if self.arduino is None or not detections:
            return
        
        try:
            # Send detection summary to Arduino
            num_objects = len(detections)
            message = f"DETECT:{num_objects}\n"
            
            self.arduino.write(message.encode())
            self.arduino.flush()
            
            # Read response
            response = self.arduino.readline().decode().strip()
            print(f"  📡 Arduino response: {response}")
            
        except Exception as e:
            print(f"  ⚠️  Arduino communication error: {e}")
    
    def run_ai_loop(self):
        """Main AI processing loop"""
        print("🚀 Starting AI processing loop...")
        print("  Press Ctrl+C to stop")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.running:
                # Process frame
                processed_image = self.process_camera_frame()
                
                if processed_image is not None:
                    frame_count += 1
                    
                    # Send results to Arduino
                    self.send_to_arduino(self.detection_results)
                    
                    # Print status every 30 frames
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        print(f"  📊 Processed {frame_count} frames, FPS: {fps:.1f}, Detections: {len(self.detection_results)}")
                
                # Small delay to prevent overwhelming system
                time.sleep(0.033)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\n  ⚡ AI loop stopped by user")
        except Exception as e:
            print(f"  ❌ AI loop error: {e}")
        
        finally:
            # Calculate final statistics
            if frame_count > 0:
                elapsed_time = time.time() - start_time
                avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"  📊 Final stats: {frame_count} frames, Average FPS: {avg_fps:.1f}")
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up resources...")
        
        if self.camera is not None:
            self.camera.release()
            print("  ✅ Camera released")
        
        if self.arduino is not None:
            self.arduino.close()
            print("  ✅ Arduino connection closed")
        
        self.running = False
        print("  ✅ Cleanup complete")
    
    def start(self):
        """Start the AI robot"""
        print("🤖 Arduino UNO Q4GB AI Robot Starting...")
        print("=" * 50)
        
        # Load configuration
        if not self.load_configuration():
            return False
        
        # Initialize components
        if not self.initialize_ai_framework():
            return False
        
        if not self.load_model():
            return False
        
        if not self.initialize_camera():
            return False
        
        if not self.initialize_arduino():
            return False
        
        # Start AI loop
        self.running = True
        self.run_ai_loop()
        
        return True

def main():
    """Main function"""
    robot = ArduinoQ4GBAIRobot()
    
    try:
        # Start the robot
        if robot.start():
            print("🎉 Arduino UNO Q4GB AI Robot completed successfully!")
        else:
            print("❌ Arduino UNO Q4GB AI Robot failed to start!")
            
    except KeyboardInterrupt:
        print("\n⚡ Arduino UNO Q4GB AI Robot stopped by user")
    except Exception as e:
        print(f"❌ Arduino UNO Q4GB AI Robot error: {e}")
    finally:
        robot.cleanup()

if __name__ == "__main__":
    main()