#!/usr/bin/env python3
"""
Arduino UNO Q4GB AI Robot - Motor Controller Interface
Handles communication with Arduino STM32 coprocessor for motor control
"""

import serial
import time
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

class ArduinoMotorController:
    """Arduino motor controller for Arduino UNO Q4GB AI Robot"""
    
    # Motor command constants
    COMMANDS = {
        'FORWARD': 'FORWARD:{left_speed}:{right_speed}',
        'BACKWARD': 'BACKWARD:{left_speed}:{right_speed}',
        'LEFT': 'LEFT:{left_speed}:{right_speed}',
        'RIGHT': 'RIGHT:{left_speed}:{right_speed}',
        'STOP': 'STOP',
        'EMERGENCY_STOP': 'ESTOP',
        'SET_SPEED': 'SPEED:{left_speed}:{right_speed}',
        'GET_STATUS': 'STATUS',
        'CALIBRATE': 'CALIBRATE'
    }
    
    # Response codes
    RESPONSES = {
        'OK': 'OK',
        'ERROR': 'ERROR',
        'BUSY': 'BUSY',
        'READY': 'READY',
        'MOVING': 'MOVING',
        'STOPPED': 'STOPPED'
    }
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, timeout: float = 1.0):
        self.serial_port: Optional[serial.Serial] = None
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_connected = False
        self.is_emergency_stopped = False
        self.current_speed = (0, 0)
        self.current_direction = 'STOP'
        
        # Thread-safe communication
        self.lock = threading.Lock()
        
        # Configuration
        self.max_speed = 255
        self.speed_limits = {
            'min_forward': 50,
            'max_forward': 200,
            'min_turn': 30,
            'max_turn': 150
        }
        
    def auto_detect_port(self) -> Optional[str]:
        """Auto-detect Arduino port"""
        import serial.tools.list_ports
        
        ports = serial.tools.list_ports.comports()
        arduino_ports = []
        
        for port in ports:
            description = port.description.lower()
            if any(keyword in description for keyword in ['arduino', 'ch340', 'cp210', 'ftdi', 'usb-serial']):
                arduino_ports.append(port)
        
        if arduino_ports:
            # Return the first Arduino port found
            selected_port = arduino_ports[0]
            print(f"Auto-detected Arduino at {selected_port.device}: {selected_port.description}")
            return selected_port.device
        
        return None
    
    def connect(self, port: Optional[str] = None) -> bool:
        """Connect to Arduino"""
        with self.lock:
            if self.is_connected:
                return True
            
            try:
                # Use provided port or auto-detect
                target_port = port or self.port or self.auto_detect_port()
                
                if not target_port:
                    print("No Arduino port specified and auto-detection failed")
                    return False
                
                print(f"Connecting to Arduino at {target_port}...")
                
                self.serial_port = serial.Serial(
                    port=target_port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    write_timeout=self.timeout
                )
                
                # Wait for Arduino to be ready
                time.sleep(2)
                
                # Test connection
                if self._send_command('GET_STATUS'):
                    response = self._read_response()
                    if response and any(status in response for status in ['READY', 'OK']):
                        self.is_connected = True
                        print(f"Successfully connected to Arduino at {target_port}")
                        return True
                    else:
                        print(f"Arduino responded with: {response}")
                else:
                    print("Failed to send test command to Arduino")
                
                # If connection failed, cleanup
                self.serial_port.close()
                self.serial_port = None
                return False
                
            except serial.SerialException as e:
                print(f"Serial connection error: {e}")
                return False
            except Exception as e:
                print(f"Unexpected connection error: {e}")
                return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        with self.lock:
            if self.serial_port and self.serial_port.is_open:
                # Send emergency stop before disconnecting
                self.emergency_stop()
                self.serial_port.close()
                self.serial_port = None
                self.is_connected = False
                print("Disconnected from Arduino")
    
    def _send_command(self, command: str) -> bool:
        """Send command to Arduino"""
        if not self.is_connected or not self.serial_port:
            return False
        
        try:
            full_command = command + '\n'
            self.serial_port.write(full_command.encode())
            self.serial_port.flush()
            return True
        except serial.SerialException:
            print("Serial communication error during send")
            return False
        except Exception as e:
            print(f"Unexpected error during send: {e}")
            return False
    
    def _read_response(self, timeout: Optional[float] = None) -> Optional[str]:
        """Read response from Arduino"""
        if not self.is_connected or not self.serial_port:
            return None
        
        try:
            old_timeout = self.serial_port.timeout
            if timeout is not None:
                self.serial_port.timeout = timeout
            
            response = self.serial_port.readline().decode().strip()
            
            self.serial_port.timeout = old_timeout
            return response if response else None
            
        except serial.SerialException:
            print("Serial communication error during read")
            return None
        except Exception as e:
            print(f"Unexpected error during read: {e}")
            return None
    
    def _execute_command(self, command_template: str, *args) -> bool:
        """Execute command with arguments"""
        if self.is_emergency_stopped and not command_template.startswith('ESTOP'):
            print("Emergency stop active - command ignored")
            return False
        
        if not self.is_connected:
            print("Not connected to Arduino")
            return False
        
        try:
            command = command_template.format(*args)
            if self._send_command(command):
                response = self._read_response(timeout=2.0)
                return response and 'OK' in response
        except Exception as e:
            print(f"Command execution error: {e}")
        
        return False
    
    def emergency_stop(self) -> bool:
        """Emergency stop"""
        print("EMERGENCY STOP ACTIVATED!")
        self.is_emergency_stopped = True
        self.current_direction = 'STOP'
        self.current_speed = (0, 0)
        return self._execute_command(self.COMMANDS['EMERGENCY_STOP'])
    
    def clear_emergency_stop(self) -> bool:
        """Clear emergency stop"""
        self.is_emergency_stopped = False
        print("Emergency stop cleared")
        return self.stop()
    
    def stop(self) -> bool:
        """Stop motors"""
        self.current_direction = 'STOP'
        self.current_speed = (0, 0)
        return self._execute_command(self.COMMANDS['STOP'])
    
    def forward(self, speed: Optional[int] = None) -> bool:
        """Move forward"""
        if speed is None:
            speed = self.speed_limits['max_forward']
        
        speed = max(self.speed_limits['min_forward'], min(speed, self.speed_limits['max_forward']))
        
        self.current_direction = 'FORWARD'
        self.current_speed = (speed, speed)
        return self._execute_command(self.COMMANDS['FORWARD'], speed, speed)
    
    def backward(self, speed: Optional[int] = None) -> bool:
        """Move backward"""
        if speed is None:
            speed = self.speed_limits['max_forward']
        
        speed = max(self.speed_limits['min_forward'], min(speed, self.speed_limits['max_forward']))
        
        self.current_direction = 'BACKWARD'
        self.current_speed = (speed, speed)
        return self._execute_command(self.COMMANDS['BACKWARD'], speed, speed)
    
    def left(self, forward_speed: Optional[int] = None, turn_speed: Optional[int] = None) -> bool:
        """Turn left"""
        if forward_speed is None:
            forward_speed = self.speed_limits['min_turn']
        if turn_speed is None:
            turn_speed = self.speed_limits['max_turn']
        
        forward_speed = max(self.speed_limits['min_turn'], min(forward_speed, self.speed_limits['max_turn']))
        turn_speed = max(self.speed_limits['min_turn'], min(turn_speed, self.speed_limits['max_turn']))
        
        self.current_direction = 'LEFT'
        self.current_speed = (forward_speed, turn_speed)
        return self._execute_command(self.COMMANDS['LEFT'], forward_speed, turn_speed)
    
    def right(self, forward_speed: Optional[int] = None, turn_speed: Optional[int] = None) -> bool:
        """Turn right"""
        if forward_speed is None:
            forward_speed = self.speed_limits['min_turn']
        if turn_speed is None:
            turn_speed = self.speed_limits['max_turn']
        
        forward_speed = max(self.speed_limits['min_turn'], min(forward_speed, self.speed_limits['max_turn']))
        turn_speed = max(self.speed_limits['min_turn'], min(turn_speed, self.speed_limits['max_turn']))
        
        self.current_direction = 'RIGHT'
        self.current_speed = (turn_speed, forward_speed)
        return self._execute_command(self.COMMANDS['RIGHT'], turn_speed, forward_speed)
    
    def set_speed(self, left_speed: int, right_speed: int) -> bool:
        """Set individual motor speeds"""
        left_speed = max(0, min(left_speed, self.max_speed))
        right_speed = max(0, min(right_speed, self.max_speed))
        
        self.current_speed = (left_speed, right_speed)
        return self._execute_command(self.COMMANDS['SET_SPEED'], left_speed, right_speed)
    
    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get Arduino status"""
        if not self._execute_command(self.COMMANDS['GET_STATUS']):
            return None
        
        response = self._read_response(timeout=1.0)
        if not response:
            return None
        
        try:
            # Try to parse JSON response
            if response.startswith('{'):
                return json.loads(response)
            else:
                # Parse simple text response
                parts = response.split(',')
                status = {'raw_response': response}
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        status[key.strip()] = value.strip()
                return status
        except Exception as e:
            print(f"Status parsing error: {e}")
            return {'raw_response': response, 'parse_error': str(e)}
    
    def calibrate(self) -> bool:
        """Calibrate motors"""
        print("Starting motor calibration...")
        return self._execute_command(self.COMMANDS['CALIBRATE'])
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status"""
        return {
            'is_connected': self.is_connected,
            'port': self.port,
            'baudrate': self.baudrate,
            'current_direction': self.current_direction,
            'current_speed': self.current_speed,
            'emergency_stopped': self.is_emergency_stopped
        }
    
    def respond_to_detection(self, detection: Dict[str, Any]) -> bool:
        """Respond to AI object detection"""
        if not detection:
            return self.stop()
        
        class_name = detection.get('class', '').lower()
        confidence = detection.get('confidence', 0)
        
        # Ignore low confidence detections
        if confidence < 0.5:
            return self.stop()
        
        print(f"Responding to detection: {class_name} (confidence: {confidence:.2f})")
        
        # Define responses based on detected objects
        if class_name == 'person':
            # Move toward person slowly
            return self.forward(speed=80)
        elif class_name in ['car', 'truck', 'bus']:
            # Emergency stop for vehicles
            return self.emergency_stop()
        elif class_name in ['cup', 'bottle']:
            # Turn left for small objects
            return self.left(forward_speed=60, turn_speed=100)
        elif class_name in ['chair', 'table', 'couch']:
            # Turn right for furniture
            return self.right(forward_speed=60, turn_speed=100)
        elif class_name in ['dog', 'cat']:
            # Slow forward for pets
            return self.forward(speed=50)
        else:
            # Default slow forward
            return self.forward(speed=60)
    
    def test_motors(self, test_duration: float = 2.0) -> bool:
        """Test all motor functions"""
        if not self.is_connected:
            print("Not connected to Arduino - cannot test motors")
            return False
        
        print("Running motor test sequence...")
        
        test_sequence = [
            ("Forward", lambda: self.forward(100)),
            ("Stop", lambda: self.stop()),
            ("Backward", lambda: self.backward(100)),
            ("Stop", lambda: self.stop()),
            ("Left turn", lambda: self.left(80, 120)),
            ("Stop", lambda: self.stop()),
            ("Right turn", lambda: self.right(120, 80)),
            ("Stop", lambda: self.stop()),
        ]
        
        try:
            for test_name, test_func in test_sequence:
                print(f"  {test_name}...")
                if not test_func():
                    print(f"    Failed: {test_name}")
                    return False
                time.sleep(test_duration)
            
            print("Motor test completed successfully")
            return True
            
        except Exception as e:
            print(f"Motor test error: {e}")
            self.emergency_stop()
            return False

def main():
    """Test the motor controller"""
    controller = ArduinoMotorController()
    
    try:
        # Auto-connect
        if controller.connect():
            print("Connected to Arduino successfully!")
            
            # Test motors
            controller.test_motors(test_duration=1.0)
            
            # Get status
            status = controller.get_status()
            if status:
                print(f"Arduino status: {status}")
            
            # Test detection responses
            test_detections = [
                {'class': 'person', 'confidence': 0.9},
                {'class': 'car', 'confidence': 0.8},
                {'class': 'cup', 'confidence': 0.7}
            ]
            
            for detection in test_detections:
                print(f"\nTesting response to: {detection}")
                controller.respond_to_detection(detection)
                time.sleep(1.5)
            
            # Final stop
            controller.stop()
            
        else:
            print("Failed to connect to Arduino")
            return 1
            
    except KeyboardInterrupt:
        print("\nMotor test interrupted")
        controller.emergency_stop()
    finally:
        controller.disconnect()
    
    return 0

if __name__ == "__main__":
    exit(main())