#!/usr/bin/env python3
"""
Arduino Q4GB Configuration and Monitoring Tool
Provides simple interface for configuring and monitoring the on-device AI robot
"""

import serial
import json
import time
import argparse
import sys
from datetime import datetime

class ArduinoQ4GBInterface:
    def __init__(self, port='COM3', baud_rate=115200):
        self.arduino = None
        self.port = port
        self.baud_rate = baud_rate
        self.connected = False
        self.monitoring = False
        
        # Statistics
        self.stats = {
            'total_cycles': 0,
            'detections': 0,
            'actions': {'forward': 0, 'turn_left': 0, 'turn_right': 0, 'stop': 0},
            'last_update': None
        }
    
    def connect(self):
        """Connect to Arduino UNO Q4GB"""
        try:
            self.arduino = serial.Serial(self.port, self.baud_rate, timeout=2)
            time.sleep(3)  # Wait for Arduino to boot
            print(f"Connected to Arduino UNO Q4GB on {self.port}")
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            self.connected = False
            print("Disconnected from Arduino")
    
    def send_command(self, command):
        """Send command to Arduino"""
        if not self.connected:
            print("Not connected to Arduino")
            return False
        
        try:
            self.arduino.write((command + '\n').encode('utf-8'))
            self.arduino.flush()
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def read_data(self):
        """Read data from Arduino"""
        if not self.connected or not self.arduino.in_waiting:
            return None
        
        try:
            line = self.arduino.readline().decode('utf-8').strip()
            return line
        except Exception as e:
            print(f"Error reading from Arduino: {e}")
            return None
    
    def parse_status_line(self, line):
        """Parse status output from Arduino"""
        if "Cycle" in line and "Action=" in line:
            parts = line.split(':')
            cycle_part = parts[0]
            action_part = parts[1] if len(parts) > 1 else ""
            
            # Extract cycle number
            try:
                cycle_num = int(cycle_part.split()[1])
                self.stats['total_cycles'] = cycle_num
            except:
                pass
            
            # Extract action
            action = action_part.split('=')[1].split(',')[0] if '=' in action_part else "unknown"
            if action in self.stats['actions']:
                self.stats['actions'][action] += 1
            
            # Extract detection count
            if "Detections=" in action_part:
                try:
                    detections = int(action_part.split('Detections=')[1].split(',')[0])
                    self.stats['detections'] = detections
                except:
                    pass
            
            # Extract center distance
            if "CenterDist=" in action_part:
                try:
                    center_dist = float(action_part.split('CenterDist=')[1].replace('cm', ''))
                    self.stats['center_distance'] = center_dist
                except:
                    pass
            
            self.stats['last_update'] = datetime.now()
            return True
        
        return False
    
    def monitor_mode(self, duration=None):
        """Monitor Arduino status in real-time"""
        print(f"Monitoring Arduino UNO Q4GB... (Ctrl+C to stop)")
        if duration:
            print(f"Will stop after {duration} seconds")
        
        start_time = time.time()
        self.monitoring = True
        
        try:
            while self.monitoring:
                line = self.read_data()
                if line:
                    print(f"Arduino: {line}")
                    self.parse_status_line(line)
                
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    break
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nStopping monitor...")
        
        self.monitoring = False
        self.display_stats()
    
    def display_stats(self):
        """Display collected statistics"""
        print("\n=== Arduino UNO Q4GB Statistics ===")
        print(f"Total Cycles: {self.stats['total_cycles']}")
        print(f"Current Detections: {self.stats['detections']}")
        print(f"Center Distance: {self.stats.get('center_distance', 'N/A')}cm")
        print("\nAction Distribution:")
        total_actions = sum(self.stats['actions'].values())
        for action, count in self.stats['actions'].items():
            percentage = (count / total_actions * 100) if total_actions > 0 else 0
            print(f"  {action:12s}: {count:4d} ({percentage:5.1f}%)")
        
        if self.stats['last_update']:
            print(f"\nLast Update: {self.stats['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    def configure_ai_parameters(self, params):
        """Configure AI parameters on Arduino"""
        print(f"Configuring AI parameters: {params}")
        
        for param, value in params.items():
            command = f"SET {param}={value}"
            if self.send_command(command):
                print(f"  Set {param} = {value}")
            else:
                print(f"  Failed to set {param}")
        
        # Request configuration confirmation
        self.send_command("GET_CONFIG")
        time.sleep(1)
    
    def run_calibration(self):
        """Run sensor calibration routine"""
        print("Starting sensor calibration...")
        print("Please ensure robot is in an open area with no obstacles.")
        
        input("Press Enter to begin calibration...")
        
        if self.send_command("CALIBRATE"):
            print("Calibration started. Please wait...")
            
            # Monitor calibration progress
            calibrating = True
            while calibrating:
                line = self.read_data()
                if line:
                    print(f"Arduino: {line}")
                    if "CALIBRATION" in line and "COMPLETE" in line:
                        calibrating = False
                        print("Calibration completed successfully!")
                    elif "ERROR" in line:
                        calibrating = False
                        print("Calibration failed!")
                
                time.sleep(0.1)
        
        return True
    
    def test_mode(self):
        """Run AI test mode with various scenarios"""
        print("Starting AI Test Mode...")
        print("This will test various scenarios to verify AI performance.")
        
        test_scenarios = [
            "TEST CLEAR_PATH",
            "TEST OBSTACLE_FRONT", 
            "TEST OBJECT_LEFT",
            "TEST OBJECT_RIGHT",
            "TEST MULTIPLE_OBJECTS",
            "TEST CRITICAL_OBSTACLE"
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n--- Test Scenario {i}: {scenario} ---")
            input("Press Enter to run this test...")
            
            if self.send_command(scenario):
                print(f"Executing {scenario}...")
                time.sleep(5)  # Wait for test completion
            else:
                print(f"Failed to execute {scenario}")
        
        print("\nAll test scenarios completed!")
        self.display_stats()
    
    def firmware_update(self, hex_file):
        """Update Arduino firmware (requires avrdude)"""
        print(f"Updating firmware with {hex_file}")
        
        import subprocess
        
        try:
            cmd = [
                'avrdude',
                '-p', 'atmega328p',  # Adjust for your board
                '-c', 'arduino',
                '-P', self.port,
                '-U', f'flash:w:{hex_file}:i'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Firmware update successful!")
                return True
            else:
                print(f"Firmware update failed: {result.stderr}")
                return False
        
        except FileNotFoundError:
            print("avrdude not found. Please install Arduino IDE or avrdude.")
            return False
    
    def get_system_info(self):
        """Get system information from Arduino"""
        print("Getting system information...")
        
        if self.send_command("SYS_INFO"):
            # Read response
            info_lines = []
            start_time = time.time()
            
            while time.time() - start_time < 3:  # 3 second timeout
                line = self.read_data()
                if line:
                    info_lines.append(line)
                    if "END_INFO" in line:
                        break
                time.sleep(0.1)
            
            if info_lines:
                print("\n=== System Information ===")
                for line in info_lines:
                    if not line.startswith("END_INFO"):
                        print(line)
            else:
                print("No system information received")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Arduino UNO Q4GB Interface Tool')
    parser.add_argument('--port', default='COM3', help='Arduino serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--monitor', action='store_true', help='Monitor mode')
    parser.add_argument('--duration', type=int, help='Monitor duration in seconds')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration')
    parser.add_argument('--test', action='store_true', help='Run AI test mode')
    parser.add_argument('--info', action='store_true', help='Get system information')
    parser.add_argument('--configure', nargs='+', help='Configure parameters (param=value)')
    parser.add_argument('--update', help='Update firmware with .hex file')
    
    args = parser.parse_args()
    
    interface = ArduinoQ4GBInterface(args.port, args.baud)
    
    # Connect to Arduino
    if not interface.connect():
        print("Failed to connect. Please check port and connections.")
        sys.exit(1)
    
    try:
        if args.update:
            interface.firmware_update(args.update)
        elif args.calibrate:
            interface.run_calibration()
        elif args.test:
            interface.test_mode()
        elif args.info:
            interface.get_system_info()
        elif args.configure:
            params = {}
            for param in args.configure:
                if '=' in param:
                    key, value = param.split('=', 1)
                    params[key] = value
            interface.configure_ai_parameters(params)
        elif args.monitor:
            interface.monitor_mode(args.duration)
        else:
            # Default: interactive mode
            print("Arduino UNO Q4GB Interactive Interface")
            print("Available commands:")
            print("  monitor     - Start monitoring mode")
            print("  calibrate   - Run sensor calibration")
            print("  test        - Run AI test scenarios")
            print("  info        - Get system information")
            print("  stats       - Display statistics")
            print("  quit        - Exit")
            
            while True:
                try:
                    cmd = input("\n> ").strip().lower()
                    
                    if cmd == 'quit':
                        break
                    elif cmd == 'monitor':
                        interface.monitor_mode()
                    elif cmd == 'calibrate':
                        interface.run_calibration()
                    elif cmd == 'test':
                        interface.test_mode()
                    elif cmd == 'info':
                        interface.get_system_info()
                    elif cmd == 'stats':
                        interface.display_stats()
                    else:
                        # Send custom command to Arduino
                        if interface.send_command(cmd):
                            print(f"Sent: {cmd}")
                            time.sleep(1)
                            # Read any response
                            for _ in range(5):
                                line = interface.read_data()
                                if line:
                                    print(f"Arduino: {line}")
                                time.sleep(0.1)
                        else:
                            print(f"Failed to send: {cmd}")
                
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\nUse 'quit' to exit")
    
    finally:
        interface.disconnect()

if __name__ == "__main__":
    main()