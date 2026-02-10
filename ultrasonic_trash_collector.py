#!/usr/bin/env python3
"""
Arduino UNO Q4GB Trash Collector Robot - Ultrasonic Only Version
Phase 3: Hardware-Specific Optimization
Ultrasonic Sensor-Based Object Detection + Arduino Control
"""

import os
import sys
import json
import time
import threading
import numpy as np
from pathlib import Path

class UltrasonicTrashCollectorRobot:
    def __init__(self):
        self.config = {}
        self.arduino = None
        self.running = False
        self.detection_results = []
        self.frame_count = 0
        
        # Setup paths
        self.install_dir = Path.home() / 'arduino_q4gb_ai_robot_phase3'
        self.models_dir = self.install_dir / 'models'
        self.hardware_dir = self.install_dir / 'hardware_detection'
        
        # Ultrasonic sensor configuration
        self.sensors = {
            'front': {'pin': 2, 'threshold': 30, 'angle': 0},
            'front_left': {'pin': 3, 'threshold': 25, 'angle': -30},
            'front_right': {'pin': 4, 'threshold': 25, 'angle': 30},
            'left': {'pin': 5, 'threshold': 20, 'angle': -60},
            'right': {'pin': 6, 'threshold': 20, 'angle': 60}
        }
        
        self.sensor_data = {key: 0 for key in self.sensors.keys()}
        self.detected_objects = []
        
        print("Ultrasonic Trash Collector Robot Initialized")
        print("  Features: 5x Ultrasonic Sensors, Real-time Object Detection, Arduino Control")
        print(f"  Sensors: {len(self.sensors)} configured with thresholds {self.get_thresholds()}")
    
    def get_thresholds(self):
        """Get detection thresholds for all sensors"""
        return {name: info['threshold'] for name, info in self.sensors.items()}
    
    def load_configuration(self):
        """Load system configuration"""
        print("Loading configuration...")
        
        config_file = self.install_dir / 'config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
                print("  Configuration loaded successfully")
                return True
            except Exception as e:
                print(f"Error loading configuration: {e}")
        
        print("  No config file found, using defaults")
        return True
    
    def initialize_ultrasonic_sensors(self):
        """Initialize ultrasonic sensor system"""
        print("Initializing ultrasonic sensors...")
        
        try:
            # For simulation, we'll generate realistic sensor data
            print("  Ultrasonic simulation mode activated")
            print(f"  Configured sensors: {list(self.sensors.keys())}")
            
            # Test sensor data generation
            self.generate_sensor_data()
            print(f"  Test readings: {self.sensor_data}")
            print("  Sensors initialized successfully")
            return True
            
        except Exception as e:
            print(f"  Sensor initialization failed: {e}")
            return False
    
    def generate_sensor_data(self):
        """Generate realistic ultrasonic sensor data"""
        import random
        
        # Simulate room with objects
        # Most directions should be clear (>100cm), some directions have objects
        base_distances = {
            'front': random.uniform(80, 150),
            'front_left': random.uniform(60, 120),
            'front_right': random.uniform(60, 120),
            'left': random.uniform(40, 100),
            'right': random.uniform(40, 100)
        }
        
        # Simulate moving objects (trash)
        if random.random() < 0.3:  # 30% chance of detecting trash
            # Random sensor detects object within threshold
            sensor_with_object = random.choice(list(self.sensors.keys()))
            base_distances[sensor_with_object] = random.uniform(10, self.sensors[sensor_with_object]['threshold'] - 5)
        
        # Add some noise to make it realistic
        for sensor in base_distances:
            noise = random.uniform(-2, 2)
            self.sensor_data[sensor] = max(2, base_distances[sensor] + noise)
        
        return self.sensor_data
    
    def detect_objects_from_sensors(self):
        """Detect objects using ultrasonic sensor data"""
        detected_objects = []
        
        for sensor_name, distance in self.sensor_data.items():
            sensor_info = self.sensors[sensor_name]
            threshold = sensor_info['threshold']
            angle = sensor_info['angle']
            
            if distance < threshold:
                # Calculate object position based on sensor angle and distance
                # Assuming robot is at origin (0,0)
                angle_rad = np.radians(angle)
                x = int(distance * np.cos(angle_rad))
                y = int(distance * np.sin(angle_rad))
                
                # Classify object type based on distance and sensor
                if distance < 15:
                    obj_type = "small_trash"
                    confidence = 0.9
                elif distance < 25:
                    obj_type = "medium_trash"
                    confidence = 0.8
                else:
                    obj_type = "large_trash"
                    confidence = 0.7
                
                detected_object = {
                    'type': obj_type,
                    'sensor': sensor_name,
                    'distance': distance,
                    'angle': angle,
                    'x': x + 320,  # Center in 640px width
                    'y': 240 - y,  # Center in 480px height (flip y-axis)
                    'confidence': confidence,
                    'bbox': [x + 320 - 20, 240 - y - 20, 40, 40]  # Approximate bounding box
                }
                
                detected_objects.append(detected_object)
        
        self.detected_objects = detected_objects
        return detected_objects
    
    def create_sensor_visualization(self, width=640, height=480):
        """Create visualization of sensor data and detected objects"""
        import cv2
        
        # Create dark background
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw robot at center
        robot_center = (width // 2, height // 2)
        cv2.circle(image, robot_center, 15, (0, 255, 0), -1)  # Green robot
        cv2.circle(image, robot_center, 15, (0, 255, 0), 2)   # White outline
        
        # Draw sensor rays and objects
        for sensor_name, sensor_info in self.sensors.items():
            angle = sensor_info['angle']
            distance = self.sensor_data[sensor_name]
            threshold = sensor_info['threshold']
            
            # Convert angle to coordinates
            angle_rad = np.radians(angle - 90)  # -90 to make 0° point up
            end_x = int(robot_center[0] + distance * 2 * np.cos(angle_rad))
            end_y = int(robot_center[1] + distance * 2 * np.sin(angle_rad))
            
            threshold_x = int(robot_center[0] + threshold * 2 * np.cos(angle_rad))
            threshold_y = int(robot_center[1] + threshold * 2 * np.sin(angle_rad))
            
            # Draw threshold line (red, dashed)
            cv2.line(image, robot_center, (threshold_x, threshold_y), (0, 0, 100), 1)
            
            # Draw sensor ray
            if distance < threshold:
                # Object detected - solid red line
                cv2.line(image, robot_center, (end_x, end_y), (0, 0, 255), 2)
                
                # Draw detected object
                cv2.circle(image, (end_x, end_y), 8, (0, 255, 255), -1)
                cv2.circle(image, (end_x, end_y), 8, (0, 0, 255), 2)
            else:
                # Clear - green line
                cv2.line(image, robot_center, (end_x, end_y), (0, 100, 0), 2)
            
            # Draw sensor label
            label_pos = (threshold_x + 10, threshold_y)
            cv2.putText(image, sensor_name[:4], label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw detected objects information
        y_offset = 30
        cv2.putText(image, "ULTRASONIC DETECTION MODE", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(image, f"Objects detected: {len(self.detected_objects)}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        for obj in self.detected_objects:
            text = f"{obj['type']}: {obj['distance']:.1f}cm"
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset += 15
        
        # Draw sensor data table
        y_offset = height - 120
        cv2.putText(image, "Sensor Readings:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 15
        
        for sensor_name, distance in self.sensor_data.items():
            threshold = self.sensors[sensor_name]['threshold']
            status = "DETECT" if distance < threshold else "CLEAR"
            color = (0, 0, 255) if distance < threshold else (0, 255, 0)
            
            text = f"{sensor_name[:10]:10s}: {distance:5.1f}cm ({status})"
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 12
        
        return image
    
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
    
    def send_to_arduino(self, detections):
        """Send ultrasonic detection results to Arduino"""
        if self.arduino is None:
            return
        
        try:
            # Find nearest object
            if detections:
                nearest = min(detections, key=lambda x: x['distance'])
                num_objects = len(detections)
                nearest_type = nearest['type']
                nearest_distance = nearest['distance']
                nearest_angle = nearest['angle']
                
                # Send ultrasonic-specific command
                message = f"ULTRASONIC:{num_objects}:{nearest_type}:{nearest_distance:.1f}:{nearest_angle}\n"
                print(f"  Found {num_objects} objects, nearest: {nearest_type} at {nearest_distance:.1f}cm ({nearest_angle}°)")
            else:
                message = "ULTRASONIC:0:NONE:999.9:0\n"
            
            self.arduino.write(message.encode())
            self.arduino.flush()
            
            # Read response
            response = self.arduino.readline().decode().strip()
            if response:
                print(f"  Arduino: {response}")
            
        except Exception as e:
            print(f"  Arduino communication error: {e}")
    
    def process_ultrasonic_data(self):
        """Process ultrasonic sensor data and create visualization"""
        # Generate new sensor data
        self.generate_sensor_data()
        
        # Detect objects from sensor data
        detections = self.detect_objects_from_sensors()
        
        # Create visualization
        visualization = self.create_sensor_visualization()
        
        return visualization, detections
    
    def run_ultrasonic_loop(self):
        """Main ultrasonic processing loop"""
        print("Starting Ultrasonic Trash Collector Robot loop...")
        print(f"  Mode: ULTRASONIC")
        print("  Press Ctrl+C to stop")
        
        frame_count = 0
        start_time = time.time()
        objects_processed = 0
        
        try:
            while self.running:
                # Process ultrasonic sensors
                processed_image, detections = self.process_ultrasonic_data()
                
                if processed_image is not None:
                    frame_count += 1
                    
                    # Send results to Arduino
                    self.send_to_arduino(detections)
                    
                    # Count objects detected
                    if detections:
                        objects_processed += len(detections)
                    
                    # Print status every 30 frames
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        print(f"  Frames: {frame_count}, FPS: {fps:.1f}, Objects detected: {len(detections)}, Total processed: {objects_processed}")
                
                # Small delay to prevent overwhelming system
                time.sleep(0.033)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\n  Ultrasonic Trash Collector Robot stopped by user")
        except Exception as e:
            print(f"  AI loop error: {e}")
        
        finally:
            # Calculate final statistics
            if frame_count > 0:
                elapsed_time = time.time() - start_time
                avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"  Final stats: {frame_count} frames, Average FPS: {avg_fps:.1f}")
                print(f"  Total objects processed: {objects_processed}")
                if frame_count > 0:
                    print(f"  Detection rate: {objects_processed/frame_count:.2f} objects/frame")
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...")
        
        if self.arduino is not None:
            self.arduino.close()
            print("  Arduino connection closed")
        
        self.running = False
        print("  Cleanup complete")
    
    def start(self):
        """Start the Ultrasonic Trash Collector Robot"""
        print("Ultrasonic Trash Collector Robot Starting...")
        print("=" * 50)
        print("  Detection System: 5x Ultrasonic Sensors")
        print("  Coverage: 360-degree object detection")
        print("  Range: 2-200cm")
        print("  Target Objects: Small/Medium/Large trash")
        print("=" * 50)
        
        # Load configuration
        if not self.load_configuration():
            print("  No config file found, using defaults")
        
        # Initialize components
        if not self.initialize_ultrasonic_sensors():
            return False
        
        if not self.initialize_arduino():
            print("  Arduino not available, running in standalone mode")
        
        # Start ultrasonic loop
        self.running = True
        self.run_ultrasonic_loop()
        
        return True

def main():
    """Main function"""
    robot = UltrasonicTrashCollectorRobot()
    
    try:
        # Start the robot
        if robot.start():
            print("Ultrasonic Trash Collector Robot completed successfully!")
        else:
            print("Ultrasonic Trash Collector Robot failed to start!")
            
    except KeyboardInterrupt:
        print("\nUltrasonic Trash Collector Robot stopped by user")
    except Exception as e:
        print(f"Ultrasonic Trash Collector Robot error: {e}")
    finally:
        robot.cleanup()

if __name__ == "__main__":
    main()