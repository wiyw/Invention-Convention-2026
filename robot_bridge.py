#!/usr/bin/env python3
"""
Robot Bridge - Communication between Arduino and AI Models
Handles serial communication, camera capture, YOLO detection, and Qwen decision making
"""

import serial
import json
import time
import threading
import subprocess
import os
import sys
from datetime import datetime

class RobotBridge:
    def __init__(self, arduino_port='COM3', baud_rate=115200):
        self.arduino = None
        self.arduino_port = arduino_port
        self.baud_rate = baud_rate
        self.running = False
        self.latest_sensors = {}
        self.last_command_time = 0
        
    def connect_arduino(self):
        """Connect to Arduino via serial"""
        try:
            self.arduino = serial.Serial(self.arduino_port, self.baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            print(f"Connected to Arduino on {self.arduino_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            return False
    
    def read_arduino_data(self):
        """Read and parse data from Arduino"""
        while self.running and self.arduino and self.arduino.in_waiting:
            try:
                line = self.arduino.readline().decode('utf-8').strip()
                if line.startswith('SENSORS'):
                    # Parse sensor data: SENSORS {"left45":12.3,"right45":15.7,"center":25.1,"timestamp":123456}
                    json_data = line.replace('SENSORS ', '')
                    self.latest_sensors = json.loads(json_data)
                elif line.startswith('MOTORS'):
                    # Command execution confirmation
                    print(f"Arduino: {line}")
                else:
                    # General Arduino messages
                    print(f"Arduino: {line}")
            except Exception as e:
                print(f"Error reading Arduino data: {e}")
    
    def send_command_to_arduino(self, command):
        """Send motor command to Arduino"""
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write((command + '\n').encode('utf-8'))
                self.arduino.flush()
                self.last_command_time = time.time()
                return True
            except Exception as e:
                print(f"Error sending command to Arduino: {e}")
                return False
        return False
    
    def capture_and_detect(self, camera_index=0, output_dir='yolo26n'):
        """Capture image and run YOLO26n detection"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = os.path.join(output_dir, f'capture_{timestamp}.jpg')
        
        # Capture image using OpenCV
        import cv2
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Failed to open camera")
            return None
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to capture image")
            return None
        
        cv2.imwrite(image_path, frame)
        print(f"Captured image: {image_path}")
        
        # Run YOLO detection
        try:
            result = subprocess.run([
                sys.executable, 'test.py'
            ], capture_output=True, text=True, cwd=output_dir)
            
            if result.returncode == 0:
                # Find the most recent JSON result file
                import glob
                json_files = glob.glob(os.path.join(output_dir, '*_result.json'))
                if json_files:
                    latest_json = max(json_files, key=os.path.getctime)
                    with open(latest_json, 'r') as f:
                        detection_data = json.load(f)
                    return detection_data
            else:
                print(f"YOLO detection failed: {result.stderr}")
        except Exception as e:
            print(f"Error running YOLO detection: {e}")
        
        return None
    
    def make_decision(self, detection_data):
        """Use decide.py to make robot decision"""
        try:
            # Save detection data temporarily
            temp_result_path = os.path.join('yolo26n', 'temp_detection.json')
            with open(temp_result_path, 'w') as f:
                json.dump(detection_data, f)
            
            # Run decision making
            cmd = [
                sys.executable, 'decide.py',
                '--result', temp_result_path,
                '--ultrasonic', json.dumps(self.latest_sensors)
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd='yolo26n'
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    command = lines[0]
                    summary = json.loads(lines[1])
                    return command, summary
            else:
                print(f"Decision making failed: {result.stderr}")
                
        except Exception as e:
            print(f"Error making decision: {e}")
        
        return None, None
    
    def autonomous_loop(self):
        """Main autonomous control loop"""
        cycle_count = 0
        
        while self.running:
            try:
                print(f"\n--- Cycle {cycle_count + 1} ---")
                
                # Capture and detect objects
                detection_data = self.capture_and_detect()
                if detection_data:
                    print(f"Detections: {len(detection_data.get('detections', []))}")
                    
                    # Make decision
                    command, summary = self.make_decision(detection_data)
                    if command:
                        print(f"Decision: {summary}")
                        print(f"Command: {command}")
                        
                        # Send command to Arduino
                        if self.send_command_to_arduino(command):
                            print("Command sent to Arduino")
                        else:
                            print("Failed to send command")
                    else:
                        print("No decision made")
                else:
                    print("No detection data available")
                
                cycle_count += 1
                
                # Wait between cycles
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\nStopping autonomous mode...")
                break
            except Exception as e:
                print(f"Error in autonomous loop: {e}")
                time.sleep(1)
    
    def start_autonomous(self):
        """Start autonomous control mode"""
        if not self.connect_arduino():
            return False
        
        self.running = True
        
        # Start Arduino reading thread
        read_thread = threading.Thread(target=self._arduino_read_loop)
        read_thread.daemon = True
        read_thread.start()
        
        # Start autonomous control
        self.autonomous_loop()
        
        return True
    
    def _arduino_read_loop(self):
        """Background thread for reading Arduino data"""
        while self.running:
            self.read_arduino_data()
            time.sleep(0.01)
    
    def stop(self):
        """Stop the robot bridge"""
        self.running = False
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Arduino connection closed")

def main():
    bridge = RobotBridge()
    
    try:
        print("Starting Robot Bridge...")
        bridge.start_autonomous()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        bridge.stop()

if __name__ == "__main__":
    main()