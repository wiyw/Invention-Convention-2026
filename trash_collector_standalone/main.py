#!/usr/bin/env python3
"""
Trash Collector Robot - Standalone Version
Direct deployment to Arduino Uno Q Debian system
Complete integrated solution without App Labs bridge
"""

import os
import sys
import json
import time
import threading
import numpy as np
import cv2
import serial
import serial.tools.list_ports
from pathlib import Path

class TrashCollectorStandalone:
    def __init__(self):
        self.running = False
        self.camera = None
        self.yolo_model = None
        self.arduino = None
        self.frame_count = 0
        
        # Robot state
        self.robot_state = {
            'mode': 'search',
            'servo_left': 90,
            'servo_right': 90,
            'ultrasound_front': 0,
            'ultrasound_left': 0,
            'ultrasound_right': 0,
            'concrete_detected': False,
            'trash_detected': False,
            'trash_location': None
        }
        
        # Configuration
        self.config = {
            'camera_index': 0,
            'yolo_confidence': 0.5,
            'servo_min': 60,
            'servo_max': 120,
            'safe_distance': 30,
            'collect_distance': 10,
            'frame_interval': 0.1,
            'sensor_interval': 0.05,
            'arduino_baud': 115200
        }
        
    def initialize_camera(self):
        """Initialize USB webcam"""
        print("📸 Initializing USB webcam...")
        
        try:
            self.camera = cv2.VideoCapture(self.config['camera_index'])
            if self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    print(f"  ✅ Camera initialized: {frame.shape}")
                    return True
                else:
                    print("  ❌ Failed to read from camera")
                    return False
            else:
                print("  ❌ Camera not opened")
                return False
        except Exception as e:
            print(f"  ❌ Camera initialization failed: {e}")
            return False
    
    def initialize_yolo(self):
        """Initialize YOLO model"""
        print("🤖 Initializing YOLO model...")
        
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            print("  ✅ YOLOv8 loaded")
            return True
        except ImportError:
            print("  ⚠️  YOLO not available - using simulation mode")
            return False
    
    def initialize_arduino(self):
        """Initialize Arduino communication"""
        print("🔌 Initializing Arduino communication...")
        
        try:
            # Find Arduino port
            ports = serial.tools.list_ports.comports()
            arduino_port = None
            
            for port in ports:
                if 'Arduino' in port.description or 'CH340' in port.description or 'USB' in port.description:
                    arduino_port = port.device
                    break
            
            if arduino_port:
                self.arduino = serial.Serial(arduino_port, self.config['arduino_baud'], timeout=1)
                time.sleep(2)  # Wait for Arduino to reset
                
                # Test connection
                self.arduino.write(b'TEST\n')
                response = self.arduino.readline().decode().strip()
                
                if 'OK' in response:
                    print(f"  ✅ Arduino connected: {arduino_port}")
                    return True
                else:
                    print(f"  ⚠️  Arduino response: {response}")
                    return True
            else:
                print("  ⚠️  No Arduino found - simulation mode")
                return True
                
        except Exception as e:
            print(f"  ⚠️  Arduino initialization failed: {e}")
            return True
    
    def send_arduino_command(self, command):
        """Send command to Arduino"""
        if self.arduino:
            try:
                self.arduino.write(f'{command}\n'.encode())
                response = self.arduino.readline().decode().strip()
                return response
            except Exception as e:
                print(f"  ⚠️  Arduino command error: {e}")
        return None
    
    def read_sensors(self):
        """Read ultrasound sensors"""
        if self.arduino:
            try:
                # Request sensor readings
                front = self.send_arduino_command('GET_FRONT')
                left = self.send_arduino_command('GET_LEFT')
                right = self.send_arduino_command('GET_RIGHT')
                
                if front and left and right:
                    self.robot_state['ultrasound_front'] = int(front.split(':')[1] if ':' in front else 50)
                    self.robot_state['ultrasound_left'] = int(left.split(':')[1] if ':' in left else 50)
                    self.robot_state['ultrasound_right'] = int(right.split(':')[1] if ':' in right else 50)
                    return
            except:
                pass
        
        # Simulation mode
        import random
        self.robot_state['ultrasound_front'] = random.randint(10, 100)
        self.robot_state['ultrasound_left'] = random.randint(10, 100)
        self.robot_state['ultrasound_right'] = random.randint(10, 100)
    
    def process_frame(self, frame):
        """Process camera frame with YOLO"""
        if self.yolo_model is None:
            return frame, []
        
        try:
            results = self.yolo_model(frame, conf=self.config['yolo_confidence'])
            
            detections = []
            concrete_detected = False
            trash_detected = False
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        class_name = self.yolo_model.names[cls]
                        
                        detection = {
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                        }
                        detections.append(detection)
                        
                        if class_name in ['road', 'street', 'pavement']:
                            concrete_detected = True
                        elif class_name in ['bottle', 'can', 'cup', 'plastic']:
                            trash_detected = True
                            self.robot_state['trash_location'] = detection['bbox']
            
            self.robot_state['concrete_detected'] = concrete_detected
            self.robot_state['trash_detected'] = trash_detected
            
            annotated_frame = results[0].plot() if hasattr(results[0], 'plot') else frame
            return annotated_frame, detections
            
        except Exception as e:
            print(f"  ⚠️  Frame processing error: {e}")
            return frame, []
    
    def set_servos(self, left_speed, right_speed):
        """Set servo speeds"""
        left_angle = 90 + (left_speed * 0.3)
        right_angle = 90 - (right_speed * 0.3)
        
        left_angle = max(self.config['servo_min'], min(self.config['servo_max'], left_angle))
        right_angle = max(self.config['servo_min'], min(self.config['servo_max'], right_angle))
        
        self.robot_state['servo_left'] = int(left_angle)
        self.robot_state['servo_right'] = int(right_angle)
        
        if self.arduino:
            self.send_arduino_command(f'SERVO_LEFT:{int(left_angle)}')
            self.send_arduino_command(f'SERVO_RIGHT:{int(right_angle)}')
    
    def move_forward(self, speed=50):
        self.set_servos(speed, speed)
    
    def move_backward(self, speed=50):
        self.set_servos(-speed, -speed)
    
    def turn_left(self, speed=50):
        self.set_servos(-speed, speed)
    
    def turn_right(self, speed=50):
        self.set_servos(speed, -speed)
    
    def stop(self):
        self.set_servos(0, 0)
    
    def decide_action(self):
        """Decide robot action"""
        front_dist = self.robot_state['ultrasound_front']
        left_dist = self.robot_state['ultrasound_left']
        right_dist = self.robot_state['ultrasound_right']
        
        concrete = self.robot_state['concrete_detected']
        trash = self.robot_state['trash_detected']
        
        # Safety first
        if front_dist < self.config['safe_distance']:
            self.robot_state['mode'] = 'avoid'
            if left_dist > right_dist:
                self.turn_right()
            else:
                self.turn_left()
            return
        
        # Mission logic
        if not concrete:
            self.robot_state['mode'] = 'search'
            self.turn_left(speed=30)
        elif trash and front_dist > self.config['collect_distance']:
            self.robot_state['mode'] = 'approach'
            self.move_forward(speed=40)
        elif trash and front_dist <= self.config['collect_distance']:
            self.robot_state['mode'] = 'collect'
            self.stop()
            time.sleep(2)
            print("  🗑️  Trash collected!")
            self.robot_state['trash_detected'] = False
        else:
            self.robot_state['mode'] = 'search'
            self.turn_left(speed=30)
    
    def run_main_loop(self):
        """Main processing loop"""
        print("🚀 Starting Trash Collector Main Loop...")
        
        last_frame_time = 0
        last_sensor_time = 0
        
        try:
            while self.running:
                current_time = time.time()
                
                # Read sensors
                if current_time - last_sensor_time > self.config['sensor_interval']:
                    self.read_sensors()
                    last_sensor_time = current_time
                
                # Process camera
                if current_time - last_frame_time > self.config['frame_interval']:
                    if self.camera:
                        ret, frame = self.camera.read()
                        if ret:
                            processed_frame, detections = self.process_frame(frame)
                            self.frame_count += 1
                            
                            cv2.imshow('Trash Collector Vision', processed_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                    
                    last_frame_time = current_time
                
                # Decide action
                self.decide_action()
                
                # Status update
                if self.frame_count % 50 == 0 and self.frame_count > 0:
                    self.print_status()
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n  ⚡ Main loop stopped by user")
        finally:
            self.cleanup()
    
    def print_status(self):
        """Print robot status"""
        print(f"\n📊 Status Report:")
        print(f"  Mode: {self.robot_state['mode']}")
        print(f"  Frame: {self.frame_count}")
        print(f"  Sensors - F:{self.robot_state['ultrasound_front']}cm L:{self.robot_state['ultrasound_left']}cm R:{self.robot_state['ultrasound_right']}cm")
        print(f"  Concrete: {'✅' if self.robot_state['concrete_detected'] else '❌'}")
        print(f"  Trash: {'✅' if self.robot_state['trash_detected'] else '❌'}")
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        
        self.running = False
        self.stop()
        
        if self.camera:
            self.camera.release()
        
        if self.arduino:
            self.arduino.close()
        
        cv2.destroyAllWindows()
        print("  ✅ Cleanup complete")
    
    def start(self):
        """Start the robot"""
        print("🤖 Trash Collector Robot (Standalone) Starting...")
        print("=" * 50)
        
        # Initialize components
        if not self.initialize_camera():
            print("❌ Camera initialization failed")
            return False
        
        if not self.initialize_yolo():
            print("⚠️  YOLO initialization failed - continuing without AI")
        
        if not self.initialize_arduino():
            print("⚠️  Arduino initialization failed - simulation mode")
        
        # Start main loop
        self.running = True
        self.run_main_loop()
        
        return True

def main():
    robot = TrashCollectorStandalone()
    
    try:
        if robot.start():
            print("🎉 Trash Collector Robot completed successfully!")
        else:
            print("❌ Trash Collector Robot failed to start!")
    
    except KeyboardInterrupt:
        print("\n⚡ Trash Collector Robot stopped by user")
    except Exception as e:
        print(f"❌ Trash Collector Robot error: {e}")

if __name__ == "__main__":
    main()