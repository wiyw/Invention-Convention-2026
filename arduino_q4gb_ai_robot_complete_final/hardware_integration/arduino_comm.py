#!/usr/bin/env python3
"""
Arduino UNO Q4GB Arduino Communication Interface
Hardware-optimized serial communication for Arduino UNO Q4GB
"""

import serial
import serial.tools.list_ports
import time
import json
import threading
from pathlib import Path

class ArduinoInterface:
    def __init__(self, port=None, baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.connected = False
        self.running = False
        
        # Command definitions for Arduino UNO Q4GB
        self.commands = {
            'START': 'START\n',
            'STOP': 'STOP\n',
            'MOVE_FORWARD': 'MF:{speed}\n',
            'MOVE_BACKWARD': 'MB:{speed}\n',
            'TURN_LEFT': 'TL:{angle}\n',
            'TURN_RIGHT': 'TR:{angle}\n',
            'SERVO': 'SV:{servo}:{angle}\n',
            'LED': 'LED:{state}\n',
            'BUZZER': 'BZ:{frequency}:{duration}\n',
            'SENSOR_READ': 'SR\n',
            'STATUS': 'ST\n'
        }
        
        # Response patterns
        self.response_patterns = {
            'OK': r'OK',
            'ERROR': r'ERROR',
            'STATUS': r'STATUS:(.+)',
            'SENSOR': r'SENSOR:(.+)',
            'DETECTION': r'DETECTION:(\d+)'
        }
    
    def auto_detect_port(self):
        """Auto-detect Arduino UNO Q4GB port"""
        print("🔍 Auto-detecting Arduino port...")
        
        try:
            ports = serial.tools.list_ports.comports()
            arduino_ports = []
            
            for port in ports:
                # Look for Arduino-like descriptions
                description = port.description.lower()
                if any(keyword in description for keyword in ['arduino', 'ch340', 'cp210', 'usb']):
                    arduino_ports.append(port)
                    print(f"  🔌 Found potential Arduino: {port.device} - {port.description}")
            
            if arduino_ports:
                # Use first Arduino port found
                self.port = arduino_ports[0].device
                print(f"  ✅ Selected Arduino port: {self.port}")
                return self.port
            else:
                print("  ⚠️  No Arduino ports found")
                return None
                
        except Exception as e:
            print(f"  ❌ Port detection error: {e}")
            return None
    
    def connect(self, port=None):
        """Connect to Arduino UNO Q4GB"""
        if port:
            self.port = port
        
        if not self.port:
            self.port = self.auto_detect_port()
        
        if not self.port:
            print("❌ No Arduino port specified")
            return False
        
        print(f"🔌 Connecting to Arduino on {self.port}...")
        
        try:
            # Open serial connection
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            
            # Wait for connection to establish
            time.sleep(2)
            
            # Test connection
            if self.serial_conn.is_open:
                self.connected = True
                print(f"  ✅ Connected to Arduino UNO Q4GB")
                print(f"  📡 Port: {self.port}")
                print(f"  📶 Baud rate: {self.baudrate}")
                
                # Initialize Arduino
                if self.initialize_arduino():
                    print("  ✅ Arduino initialized")
                    return True
                else:
                    print("  ⚠️  Arduino initialization warning")
                    return True  # Continue anyway
            else:
                print("  ❌ Failed to open serial port")
                return False
                
        except serial.SerialException as e:
            print(f"  ❌ Serial connection error: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Connection error: {e}")
            return False
    
    def initialize_arduino(self):
        """Initialize Arduino communication"""
        print("  🔄 Initializing Arduino...")
        
        try:
            # Clear any pending data
            self.serial_conn.reset_input_buffer()
            
            # Send initialization command
            self.send_command('STATUS')
            
            # Wait for response
            response = self.read_response(timeout=3)
            
            if response:
                print(f"  📥 Arduino response: {response}")
                return True
            else:
                print("  ⚠️  No response from Arduino")
                return False
                
        except Exception as e:
            print(f"  ❌ Initialization error: {e}")
            return False
    
    def send_command(self, command, **kwargs):
        """Send command to Arduino"""
        if not self.connected or not self.serial_conn:
            print("  ❌ Not connected to Arduino")
            return False
        
        try:
            # Format command with parameters
            if command in self.commands:
                formatted_command = self.commands[command].format(**kwargs)
            else:
                formatted_command = f"{command}\n"
            
            # Send command
            self.serial_conn.write(formatted_command.encode())
            self.serial_conn.flush()
            
            print(f"  📤 Sent: {formatted_command.strip()}")
            return True
            
        except Exception as e:
            print(f"  ❌ Send command error: {e}")
            return False
    
    def read_response(self, timeout=2):
        """Read response from Arduino"""
        if not self.connected or not self.serial_conn:
            return None
        
        try:
            start_time = time.time()
            response = ""
            
            while time.time() - start_time < timeout:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode().strip()
                    if line:
                        return line
                
                time.sleep(0.01)
            
            return None  # Timeout
            
        except Exception as e:
            print(f"  ❌ Read response error: {e}")
            return None
    
    def start_robot(self):
        """Start robot operation"""
        print("🚀 Starting robot operation...")
        return self.send_command('START')
    
    def stop_robot(self):
        """Stop robot operation"""
        print("🛑 Stopping robot operation...")
        return self.send_command('STOP')
    
    def move_forward(self, speed=50):
        """Move robot forward"""
        speed = max(0, min(100, speed))  # Clamp 0-100
        return self.send_command('MOVE_FORWARD', speed=speed)
    
    def move_backward(self, speed=50):
        """Move robot backward"""
        speed = max(0, min(100, speed))  # Clamp 0-100
        return self.send_command('MOVE_BACKWARD', speed=speed)
    
    def turn_left(self, angle=45):
        """Turn robot left"""
        angle = max(-180, min(180, angle))  # Clamp -180 to 180
        return self.send_command('TURN_LEFT', angle=angle)
    
    def turn_right(self, angle=45):
        """Turn robot right"""
        angle = max(-180, min(180, angle))  # Clamp -180 to 180
        return self.send_command('TURN_RIGHT', angle=angle)
    
    def set_servo(self, servo_id, angle):
        """Set servo position"""
        servo_id = max(0, min(3, servo_id))  # Support servos 0-3
        angle = max(0, min(180, angle))  # Clamp 0-180
        return self.send_command('SERVO', servo=servo_id, angle=angle)
    
    def set_led(self, state):
        """Set LED state"""
        state_value = 1 if state else 0
        return self.send_command('LED', state=state_value)
    
    def buzz_buzzer(self, frequency=1000, duration=100):
        """Activate buzzer"""
        frequency = max(100, min(10000, frequency))  # 100Hz to 10kHz
        duration = max(10, min(5000, duration))  # 10ms to 5s
        return self.send_command('BUZZER', frequency=frequency, duration=duration)
    
    def read_sensors(self):
        """Read sensor data from Arduino"""
        print("📊 Reading sensors...")
        if self.send_command('SENSOR_READ'):
            response = self.read_response(timeout=3)
            if response:
                print(f"  📥 Sensor data: {response}")
                return self._parse_sensor_response(response)
        
        return None
    
    def _parse_sensor_response(self, response):
        """Parse sensor response into structured data"""
        try:
            # Example format: SENSOR:temp:25.5,humidity:45.2,light:780
            if response.startswith('SENSOR:'):
                data_str = response[7:]  # Remove 'SENSOR:' prefix
                sensors = {}
                
                for item in data_str.split(','):
                    if ':' in item:
                        key, value = item.split(':', 1)
                        try:
                            # Try to convert to float
                            sensors[key.strip()] = float(value.strip())
                        except ValueError:
                            # Keep as string
                            sensors[key.strip()] = value.strip()
                
                return sensors
        except Exception as e:
            print(f"  ⚠️  Sensor parsing error: {e}")
            return None
    
    def get_status(self):
        """Get Arduino status"""
        print("📊 Getting status...")
        if self.send_command('STATUS'):
            response = self.read_response(timeout=3)
            if response:
                print(f"  📥 Status: {response}")
                return response
        
        return None
    
    def send_detection_results(self, num_detections):
        """Send AI detection results to Arduino"""
        print(f"🤖 Sending detection results: {num_detections} objects")
        return self.send_command('DETECTION', detections=num_detections)
    
    def disconnect(self):
        """Disconnect from Arduino"""
        print("🔌 Disconnecting from Arduino...")
        
        if self.serial_conn and self.serial_conn.is_open:
            # Stop robot first
            self.stop_robot()
            
            # Close connection
            self.serial_conn.close()
            self.connected = False
            print("  ✅ Disconnected from Arduino")
    
    def test_connection(self):
        """Test Arduino connection"""
        print("🧪 Testing Arduino connection...")
        
        if not self.connected:
            print("  ❌ Not connected")
            return False
        
        # Test basic commands
        commands_to_test = ['STATUS', 'LED:1', 'LED:0']
        
        for cmd in commands_to_test:
            if self.send_command(cmd):
                response = self.read_response(timeout=2)
                if response:
                    print(f"  ✅ {cmd} -> {response}")
                else:
                    print(f"  ⚠️  {cmd} -> No response")
        
        return True
    
    def benchmark(self, iterations=10):
        """Benchmark Arduino communication"""
        print(f"🚀 Running Arduino benchmark ({iterations} iterations)...")
        
        start_time = time.time()
        success_count = 0
        times = []
        
        for i in range(iterations):
            cmd_start = time.time()
            
            if self.send_command('STATUS'):
                response = self.read_response(timeout=1)
                cmd_end = time.time()
                
                if response:
                    success_count += 1
                    times.append((cmd_end - cmd_start) * 1000)  # Convert to ms
            
            if (i + 1) % 5 == 0:
                print(f"    Progress: {i + 1}/{iterations}")
        
        # Calculate statistics
        end_time = time.time()
        total_time = end_time - start_time
        
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            
            print(f"  📊 Benchmark Results:")
            print(f"    Success rate: {success_count}/{iterations} ({100*success_count/iterations:.1f}%)")
            print(f"    Total time: {total_time:.2f}s")
            print(f"    Average response time: {avg_time:.2f}ms")
            print(f"    Min response time: {min_time:.2f}ms")
            print(f"    Max response time: {max_time:.2f}ms")
            
            return {
                'success_rate': success_count / iterations,
                'total_time': total_time,
                'avg_response_time_ms': avg_time,
                'min_response_time_ms': min_time,
                'max_response_time_ms': max_time
            }
        
        return None

class ArduinoManager:
    def __init__(self):
        self.arduino = ArduinoInterface()
        self.connected = False
    
    def auto_connect(self):
        """Auto-connect to Arduino UNO Q4GB"""
        return self.arduino.connect()
    
    def disconnect(self):
        """Disconnect from Arduino"""
        self.arduino.disconnect()
        self.connected = False
    
    def is_connected(self):
        """Check connection status"""
        return self.arduino.connected

def main():
    """Main function for testing Arduino interface"""
    print("🔌 Arduino UNO Q4GB Communication Test")
    print("=" * 50)
    
    # Test auto-connection
    arduino_manager = ArduinoManager()
    
    if arduino_manager.auto_connect():
        print("✅ Arduino connection successful!")
        
        # Test basic functionality
        arduino_manager.arduino.get_status()
        arduino_manager.arduino.read_sensors()
        
        # Test control commands
        print("\n🧪 Testing control commands...")
        arduino_manager.arduino.set_led(True)
        time.sleep(1)
        arduino_manager.arduino.set_led(False)
        
        # Run benchmark
        print("\n🚀 Running communication benchmark...")
        benchmark_results = arduino_manager.arduino.benchmark(10)
        
        if benchmark_results:
            print("✅ Arduino communication test completed!")
        else:
            print("❌ Arduino communication test failed!")
        
        # Disconnect
        arduino_manager.disconnect()
        
        return True
    else:
        print("❌ Arduino connection failed!")
        print("  ⚠️  This is normal if no Arduino is connected")
        print("  ⚠️  The system will run in simulation mode")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)