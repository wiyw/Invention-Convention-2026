#!/usr/bin/env python3
"""
Enhanced Robot Bridge with Qwen2.5-0.5B-Instruct Integration
Combines YOLO26n vision, Qwen reasoning, and ultrasonic sensors for intelligent navigation
"""

import serial
import json
import time
import threading
import subprocess
import os
import sys
from datetime import datetime
from qwen_integration import QwenDecisionEngine

class EnhancedRobotBridge:
    def __init__(self, arduino_port='COM3', baud_rate=115200, use_qwen=True):
        self.arduino = None
        self.arduino_port = arduino_port
        self.baud_rate = baud_rate
        self.running = False
        self.latest_sensors = {}
        self.last_command_time = 0
        self.use_qwen = use_qwen
        self.qwen_engine = QwenDecisionEngine() if use_qwen else None
        
        # Sensor fusion state
        self.sensor_history = []
        self.detection_history = []
        self.max_history = 10
        
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
                    self._update_sensor_history()
                elif line.startswith('MOTORS'):
                    # Command execution confirmation
                    print(f"Arduino: {line}")
                else:
                    # General Arduino messages
                    print(f"Arduino: {line}")
            except Exception as e:
                print(f"Error reading Arduino data: {e}")
    
    def _update_sensor_history(self):
        """Update sensor reading history for fusion"""
        self.sensor_history.append({
            'data': self.latest_sensors.copy(),
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.sensor_history) > self.max_history:
            self.sensor_history.pop(0)
    
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
            # Modify test.py to accept image path as argument
            result = subprocess.run([
                sys.executable, 'test.py', image_path
            ], capture_output=True, text=True, cwd=output_dir)
            
            if result.returncode == 0:
                # Find the most recent JSON result file
                import glob
                json_files = glob.glob(os.path.join(output_dir, '*_result.json'))
                if json_files:
                    latest_json = max(json_files, key=os.path.getctime)
                    with open(latest_json, 'r') as f:
                        detection_data = json.load(f)
                    
                    # Update detection history
                    self._update_detection_history(detection_data)
                    return detection_data
            else:
                print(f"YOLO detection failed: {result.stderr}")
        except Exception as e:
            print(f"Error running YOLO detection: {e}")
        
        return None
    
    def _update_detection_history(self, detection_data):
        """Update detection history for fusion"""
        self.detection_history.append({
            'data': detection_data,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
    
    def fuse_sensor_data(self):
        """Perform sensor fusion combining vision and ultrasonic data"""
        if not self.detection_history:
            return None
        
        # Get latest detection
        latest_detection = self.detection_history[-1]['data']
        
        # Add sensor fusion context
        fused_data = {
            'detections': latest_detection.get('detections', []),
            'width': latest_detection.get('width'),
            'height': latest_detection.get('height'),
            'ultrasonic_history': self.sensor_history[-3:] if self.sensor_history else [],  # Last 3 readings
            'detection_history': self.detection_history[-2:] if self.detection_history else [],  # Last 2 detections
            'fusion_timestamp': time.time()
        }
        
        # Enhanced object tracking with velocity estimation
        if len(self.detection_history) >= 2:
            prev_detection = self.detection_history[-2]['data']
            fused_data['object_velocity'] = self._estimate_object_velocity(
                prev_detection, latest_detection
            )
        
        # Safety assessment
        fused_data['safety_assessment'] = self._assess_safety()
        
        return fused_data
    
    def _estimate_object_velocity(self, prev_data, current_data):
        """Estimate velocity of detected objects"""
        prev_detections = {d['id']: d for d in prev_data.get('detections', [])}
        current_detections = {d['id']: d for d in current_data.get('detections', [])}
        
        velocities = []
        time_delta = current_data.get('timestamp', 0) - prev_data.get('timestamp', 0)
        
        if time_delta <= 0:
            return velocities
        
        for obj_id in current_detections:
            if obj_id in prev_detections:
                prev_bbox = prev_detections[obj_id]['bbox_norm']
                curr_bbox = current_detections[obj_id]['bbox_norm']
                
                # Calculate center movement
                dx = curr_bbox['cx'] - prev_bbox['cx']
                dy = curr_bbox['cy'] - prev_bbox['cy']
                
                # Calculate size change (approaching/receding)
                size_change = (curr_bbox['w'] * curr_bbox['h']) - (prev_bbox['w'] * prev_bbox['h'])
                
                velocities.append({
                    'object_id': obj_id,
                    'label': current_detections[obj_id]['label'],
                    'velocity_x': dx / time_delta,
                    'velocity_y': dy / time_delta,
                    'size_change_rate': size_change / time_delta,
                    'confidence': current_detections[obj_id]['confidence']
                })
        
        return velocities
    
    def _assess_safety(self):
        """Assess overall safety based on sensor data"""
        if not self.latest_sensors:
            return {'status': 'unknown', 'risk_level': 0}
        
        safety = {
            'status': 'safe',
            'risk_level': 0,
            'warnings': [],
            'critical_obstacles': []
        }
        
        # Check ultrasonic sensors
        for sensor_name, distance in self.latest_sensors.items():
            if sensor_name != 'timestamp' and distance > 0:
                if distance < 15:  # Critical threshold
                    safety['status'] = 'critical'
                    safety['risk_level'] = 10
                    safety['critical_obstacles'].append(f"{sensor_name}: {distance:.1f}cm")
                elif distance < 30:  # Warning threshold
                    safety['warnings'].append(f"{sensor_name}: {distance:.1f}cm")
                    safety['risk_level'] = max(safety['risk_level'], 5)
        
        # Consider object velocities
        if len(self.detection_history) >= 2:
            latest_fused = self.fuse_sensor_data()
            if latest_fused and latest_fused.get('object_velocity'):
                for velocity in latest_fused['object_velocity']:
                    # Objects approaching quickly
                    if velocity['size_change_rate'] > 0.1:  # Growing rapidly
                        safety['warnings'].append(f"Approaching {velocity['label']}")
                        safety['risk_level'] = max(safety['risk_level'], 7)
        
        if safety['risk_level'] > 7:
            safety['status'] = 'danger'
        elif safety['risk_level'] > 3:
            safety['status'] = 'caution'
        
        return safety
    
    def make_decision(self, fused_data):
        """Make decision using Qwen or fallback to rule-based"""
        if not fused_data:
            return None, None
        
        if self.use_qwen and self.qwen_engine:
            try:
                # Use Qwen for enhanced decision making
                return self.qwen_engine.analyze_with_qwen(
                    fused_data, 
                    fused_data.get('ultrasonic_history', [{}])[-1].get('data', {})
                )
            except Exception as e:
                print(f"Qwen decision failed: {e}")
                print("Falling back to rule-based decision...")
        
        # Fallback to original decide.py
        try:
            # Extract latest sensor data
            latest_ultrasonic = self.latest_sensors
            
            cmd = [
                sys.executable, 'yolo26n/decide.py',
                '--result', json.dumps({
                    'detections': fused_data.get('detections', []),
                    'width': fused_data.get('width'),
                    'height': fused_data.get('height')
                }),
                '--ultrasonic', json.dumps(latest_ultrasonic)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    command = lines[0]
                    summary = json.loads(lines[1])
                    summary["decision_engine"] = "rule_based"
                    return command, summary
        except Exception as e:
            print(f"Rule-based decision failed: {e}")
        
        # Emergency fallback
        command = "CMD L0 R0 T200"
        summary = {
            "action": "stop", 
            "left": 0, 
            "right": 0, 
            "duration_ms": 200, 
            "reason": "emergency_fallback",
            "decision_engine": "emergency"
        }
        return command, summary
    
    def autonomous_loop(self):
        """Main autonomous control loop with enhanced sensor fusion"""
        cycle_count = 0
        
        while self.running:
            try:
                print(f"\n--- Cycle {cycle_count + 1} ---")
                
                # Capture and detect objects
                detection_data = self.capture_and_detect()
                if detection_data:
                    # Perform sensor fusion
                    fused_data = self.fuse_sensor_data()
                    
                    if fused_data:
                        print(f"Detections: {len(fused_data.get('detections', []))}")
                        print(f"Safety Status: {fused_data.get('safety_assessment', {}).get('status', 'unknown')}")
                        
                        # Make decision
                        command, summary = self.make_decision(fused_data)
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
                        print("Sensor fusion failed")
                else:
                    print("No detection data available")
                
                cycle_count += 1
                
                # Adaptive wait time based on safety
                safety_level = 0
                if self.fuse_sensor_data() and self.fuse_sensor_data().get('safety_assessment'):
                    safety_level = self.fuse_sensor_data()['safety_assessment'].get('risk_level', 0)
                
                wait_time = 0.2 if safety_level > 5 else 0.5  # Faster cycles when in danger
                time.sleep(wait_time)
                
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Robot Bridge with Qwen Integration')
    parser.add_argument('--port', default='COM3', help='Arduino port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--no-qwen', action='store_true', help='Disable Qwen, use rule-based only')
    parser.add_argument('--test-fusion', action='store_true', help='Test sensor fusion only')
    
    args = parser.parse_args()
    
    if args.test_fusion:
        # Test sensor fusion without Arduino
        bridge = EnhancedRobotBridge(use_qwen=True)
        print("Sensor fusion test mode")
        return
    
    bridge = EnhancedRobotBridge(
        arduino_port=args.port,
        baud_rate=args.baud,
        use_qwen=not args.no_qwen
    )
    
    try:
        print(f"Starting Enhanced Robot Bridge (Qwen: {not args.no_qwen})...")
        bridge.start_autonomous()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        bridge.stop()

if __name__ == "__main__":
    main()