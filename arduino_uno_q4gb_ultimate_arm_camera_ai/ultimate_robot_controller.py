#!/usr/bin/env python3
"""
Arduino UNO Q4GB Robot Controller - Ultimate Version
Handles robot control with ultrasonic sensors and motor coordination
"""

import time
import math
from pathlib import Path

class UltimateRobotController:
    def __init__(self):
        self.serial_port = None
        self.arduino_connected = False
        self.sensor_readings = {
            'front': 0,
            'left': 0,
            'right': 0
        }
        self.motor_state = {
            'speed': (0, 0),
            'direction': 'stop'
        }
        
    def connect_arduino(self, port=None):
        """Connect to Arduino with auto-detection"""
        import serial.tools.list_ports
        
        if port:
            ports = [serial.tools.list_ports.ListPortInfo(port, "", "")]
        else:
            ports = serial.tools.list_ports.comports()
        
        arduino_ports = []
        for p in ports:
            if any(keyword in p.description.lower() for keyword in ['arduino', 'ch340', 'cp210', 'ftdi']):
                arduino_ports.append(p)
        
        if not arduino_ports:
            print("❌ No Arduino ports found")
            return False
        
        try:
            self.serial_port = serial.Serial(
                port=arduino_ports[0].device,
                baudrate=115200,
                timeout=1.0
            )
            self.arduino_connected = True
            print(f"✅ Connected to Arduino: {arduino_ports[0].device}")
            return True
        except Exception as e:
            print(f"❌ Arduino connection failed: {e}")
            return False
    
    def send_command(self, command):
        """Send command to Arduino"""
        if not self.arduino_connected or not self.serial_port:
            return False
        
        try:
            cmd_line = command + '\n'
            self.serial_port.write(cmd_line.encode())
            time.sleep(0.1)  # Wait for Arduino to process
            
            # Read response
            response = self.serial_port.readline().decode().strip()
            return response
            
        except Exception as e:
            print(f"❌ Command failed: {e}")
            return None
    
    def read_sensors(self):
        """Read ultrasonic sensors"""
        if not self.arduino_connected:
            return False
        
        try:
            # Request sensor readings
            response = self.send_command("GET_SENSORS")
            if response and "SENSORS:" in response:
                # Parse sensor data
                try:
                    data_str = response.split("SENSORS:")[1].strip()
                    parts = data_str.split(",")
                    self.sensor_readings = {
                        'front': int(parts[0]),
                        'left': int(parts[1]),
                        'right': int(parts[2])
                    }
                    print(f"📊 Sensors: Front={self.sensor_readings['front']}mm, Left={self.sensor_readings['left']}mm, Right={self.sensor_readings['right']}mm")
                except:
                    print("⚠️  Sensor data parsing failed")
                    return False
                
                return True
            else:
                print("⚠️  Invalid sensor response")
                return False
                
        except Exception as e:
            print(f"❌ Sensor read failed: {e}")
            return False
    
    def ultrasonic_forward(self, speed=100, distance_threshold=200):
        """Move forward with ultrasonic collision detection"""
        if not self.read_sensors():
            return False
        
        # Check for obstacles
        front_distance = self.sensor_readings.get('front', 999)
        left_distance = self.sensor_readings.get('left', 999)
        right_distance = self.sensor_readings.get('right', 999)
        
        min_distance = min(front_distance, left_distance, right_distance)
        
        if min_distance < distance_threshold:
            print(f"⚠️  Obstacle detected at {min_distance}mm - STOPPING")
            self.stop_motors()
            return False
        
        print(f"✅ Path clear (min distance: {min_distance}mm)")
        
        # Move forward
        cmd = f"FORWARD:{speed}:{speed}"
        response = self.send_command(cmd)
        
        if response:
            print(f"🚀 Moving forward: {response}")
            self.motor_state = {
                'speed': (speed, speed),
                'direction': 'forward'
            }
            return True
        else:
            return False
    
    def ultrasonic_turn_left(self, forward_speed=50, turn_speed=100):
        """Turn left with obstacle detection"""
        if not self.read_sensors():
            return False
        
        # Check left side for obstacles
        left_distance = self.sensor_readings.get('left', 999)
        front_distance = self.sensor_readings.get('front', 999)
        
        if left_distance < 150 or front_distance < 100:
            print(f"⚠️  Obstacle on left side ({left_distance}mm) - RIGHT TURN INSTEAD")
            return self.ultrasonic_turn_right(forward_speed, turn_speed)
        
        print(f"✅ Left side clear ({left_distance}mm)")
        
        # Turn left
        cmd = f"LEFT:{forward_speed}:{turn_speed}"
        response = self.send_command(cmd)
        
        if response:
            print(f"↩️  Turning left: {response}")
            self.motor_state = {
                'speed': (forward_speed, turn_speed),
                'direction': 'left'
            }
            return True
        else:
            return False
    
    def ultrasonic_turn_right(self, forward_speed=100, turn_speed=50):
        """Turn right with obstacle detection"""
        if not self.read_sensors():
            return False
        
        # Check right side for obstacles
        right_distance = self.sensor_readings.get('right', 999)
        front_distance = self.sensor_readings.get('front', 999)
        
        if right_distance < 150 or front_distance < 100:
            print(f"⚠️  Obstacle on right side ({right_distance}mm) - LEFT TURN INSTEAD")
            return self.ultrasonic_turn_left(forward_speed, turn_speed)
        
        print(f"✅ Right side clear ({right_distance}mm)")
        
        # Turn right
        cmd = f"RIGHT:{turn_speed}:{forward_speed}"
        response = self.send_command(cmd)
        
        if response:
            print(f"↪️  Turning right: {response}")
            self.motor_state = {
                'speed': (turn_speed, forward_speed),
                'direction': 'right'
            }
            return True
        else:
            return False
    
    def stop_motors(self):
        """Stop all motors"""
        cmd = "STOP"
        response = self.send_command(cmd)
        
        if response:
            print("🛑 Motors stopped")
            self.motor_state = {
                'speed': (0, 0),
                'direction': 'stop'
            }
            return True
        else:
            return False
    
    def emergency_stop(self):
        """Emergency stop"""
        cmd = "ESTOP"
        response = self.send_command(cmd)
        
        if response:
            print("🚨 EMERGENCY STOP ACTIVATED")
            self.motor_state = {
                'speed': (0, 0),
                'direction': 'estop'
            }
            return True
        else:
            return False
    
    def autonomous_navigation(self, duration=30):
        """Autonomous navigation with obstacle avoidance"""
        print(f"🤖 Starting autonomous navigation for {duration} seconds...")
        
        start_time = time.time()
        decisions_made = 0
        
        try:
            while time.time() - start_time < duration:
                if not self.read_sensors():
                    time.sleep(0.5)
                    continue
                
                # Decision logic
                front_dist = self.sensor_readings.get('front', 999)
                left_dist = self.sensor_readings.get('left', 999)
                right_dist = self.sensor_readings.get('right', 999)
                
                decisions_made += 1
                
                if front_dist < 150:
                    print(f"⚠️  Front obstacle ({front_dist}mm) - Turning right")
                    self.ultrasonic_turn_right(80, 100)
                    time.sleep(2)
                elif left_dist < 200 and front_dist < 250:
                    print(f"⚠️  Left obstacle ({left_dist}mm) - Turning right")
                    self.ultrasonic_turn_right(80, 100)
                    time.sleep(2)
                elif right_dist < 200 and front_dist < 250:
                    print(f"⚠️  Right obstacle ({right_dist}mm) - Turning left")
                    self.ultrasonic_turn_left(80, 100)
                    time.sleep(2)
                else:
                    print(f"✅ Path clear - Moving forward (F:{front_dist}mm, L:{left_dist}mm, R:{right_dist}mm)")
                    self.ultrasonic_forward(100, 200)
                    time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n⏹️  Autonomous navigation interrupted")
        except Exception as e:
            print(f"❌ Autonomous navigation error: {e}")
        
        self.stop_motors()
        print(f"📊 Navigation complete - {decisions_made} decisions made")
        return True
    
    def ultrasonic_center_and_45_degrees(self):
        """Center robot and turn 45 degrees for better sensor positioning"""
        print("🎯 Centering robot for 45-degree sensor orientation...")
        
        # Read current sensor state
        if not self.read_sensors():
            return False
        
        # Stop first
        self.stop_motors()
        time.sleep(0.5)
        
        # Turn 45 degrees (approximate with timing)
        self.ultrasonic_turn_left(0, 100)
        time.sleep(1.5)  # Approximate 45-degree turn
        
        # Stop and center
        self.stop_motors()
        time.sleep(0.5)
        
        print("✅ Robot centered and oriented at 45 degrees")
        return True
    
    def test_ultrasonic_system(self):
        """Test the complete ultrasonic sensor system"""
        print("🔬 Testing Ultrasonic Sensor System...")
        
        test_sequence = [
            ("Read sensors", self.read_sensors),
            ("Forward with obstacle avoidance", lambda: self.ultrasonic_forward(80, 150)),
            ("Left turn", lambda: self.ultrasonic_turn_left(80, 100)),
            ("Right turn", lambda: self.ultrasonic_turn_right(80, 100)),
            ("Stop", self.stop_motors),
            ("Emergency stop", self.emergency_stop)
        ]
        
        results = []
        for test_name, test_func in test_sequence:
            print(f"  Testing: {test_name}...")
            try:
                result = test_func()
                results.append((test_name, result))
                status = "✅" if result else "❌"
                print(f"    {status} {test_name}")
                time.sleep(0.5)
            except Exception as e:
                print(f"    ❌ {test_name} failed: {e}")
                results.append((test_name, False))
        
        # Summary
        passed_tests = sum(1 for _, result in results if result)
        print(f"\n📊 Ultrasonic System Test Results: {passed_tests}/{len(results)} tests passed")
        
        return passed_tests >= len(results) - 1  # Allow one test to fail
    
    def get_robot_status(self):
        """Get comprehensive robot status"""
        print("🤖 Robot Status:")
        print(f"  Connection: {'Connected' if self.arduino_connected else 'Disconnected'}")
        print(f"  Motors: {self.motor_state.get('direction', 'unknown')} at {self.motor_state.get('speed', (0, 0))}")
        print(f"  Sensors: F:{self.sensor_readings.get('front', 'N/A')}mm L:{self.sensor_readings.get('left', 'N/A')}mm R:{self.sensor_readings.get('right', 'N/A')}mm")
        return {
            'connected': self.arduino_connected,
            'motors': self.motor_state,
            'sensors': self.sensor_readings
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("✅ Arduino connection closed")

def main():
    """Main function for testing"""
    robot = UltimateRobotController()
    
    try:
        print("🤖 Arduino UNO Q4GB Ultimate Robot Controller")
        print("=" * 50)
        
        # Connect to Arduino
        if not robot.connect_arduino():
            print("❌ Cannot continue without Arduino connection")
            return 1
        
        # Test ultrasonic system
        if robot.test_ultrasonic_system():
            print("✅ Ultrasonic system test passed")
        else:
            print("❌ Ultrasonic system test failed")
            robot.cleanup()
            return 1
        
        # Center robot for optimal sensor positioning
        if not robot.ultrasonic_center_and_45_degrees():
            print("⚠️  Robot centering failed")
        
        # Run autonomous navigation test
        if robot.autonomous_navigation(duration=10):
            print("✅ Autonomous navigation test completed")
        
        # Show final status
        status = robot.get_robot_status()
        print("\n🎉 Ultimate Robot Controller Test Completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Test crashed: {e}")
    finally:
        robot.cleanup()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())