#!/usr/bin/env python3
"""
Arduino UNO Q4GB CLI Interface
Command-line interface for robot control and monitoring
"""

import cmd
import json
import time
import sys
from pathlib import Path

# Try to import the main robot components
try:
    from main_ai_robot import ArduinoQ4GBAIRobot
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False

try:
    from hardware_integration.arduino_comm import ArduinoManager
    ARDUINO_AVAILABLE = True
except ImportError:
    ARDUINO_AVAILABLE = False

class ArduinoCLI(cmd.Cmd):
    """Arduino UNO Q4GB Command Line Interface"""
    
    intro = "🤖 Arduino UNO Q4GB AI Robot CLI"
    prompt = "arduino> "
    
    def __init__(self):
        super().__init__()
        self.robot = None
        self.arduino_manager = None
        self.running = False
        self.detection_history = []
        
        # Initialize components
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize robot and Arduino components"""
        print("🔧 Initializing components...")
        
        # Initialize main robot
        if ROBOT_AVAILABLE:
            try:
                self.robot = ArduinoQ4GBAIRobot()
                print("  ✅ AI Robot component loaded")
            except Exception as e:
                print(f"  ⚠️  AI Robot component error: {e}")
                self.robot = None
        
        # Initialize Arduino communication
        if ARDUINO_AVAILABLE:
            try:
                self.arduino_manager = ArduinoManager()
                print("  ✅ Arduino communication loaded")
            except Exception as e:
                print(f"  ⚠️  Arduino communication error: {e}")
                self.arduino_manager = None
        
        if not self.robot and not self.arduino_manager:
            print("  ⚠️  Running in simulation mode")
    
    def do_status(self, arg):
        """Show system status"""
        print("📊 System Status")
        print("-" * 30)
        
        if self.arduino_manager and self.arduino_manager.is_connected():
            print("✅ Arduino: Connected")
        elif self.arduino_manager:
            print("⚠️  Arduino: Disconnected")
        else:
            print("❌ Arduino: Not available")
        
        if self.robot:
            print("✅ AI Robot: Available")
        else:
            print("❌ AI Robot: Not available")
        
        print(f"📊 Detection history: {len(self.detection_history)} entries")
        print(f"⏱️  Uptime: {self._get_uptime()}")
    
    def do_connect(self, arg):
        """Connect to Arduino"""
        print("🔌 Connecting to Arduino...")
        
        if not self.arduino_manager:
            print("❌ Arduino communication not available")
            return
        
        if self.arduino_manager.auto_connect():
            print("✅ Connected to Arduino!")
            self._log_event("Arduino connected")
        else:
            print("❌ Failed to connect to Arduino")
            print("  💡 Make sure Arduino is connected and powered on")
    
    def do_disconnect(self, arg):
        """Disconnect from Arduino"""
        print("🔌 Disconnecting from Arduino...")
        
        if self.arduino_manager:
            self.arduino_manager.disconnect()
            print("✅ Disconnected from Arduino!")
            self._log_event("Arduino disconnected")
        else:
            print("❌ Arduino communication not available")
    
    def do_start(self, arg):
        """Start AI robot"""
        print("🚀 Starting AI robot...")
        
        if self.robot:
            if self.robot.start():
                self.running = True
                print("✅ AI robot started!")
                self._log_event("AI robot started")
            else:
                print("❌ Failed to start AI robot")
        else:
            print("❌ AI robot not available")
    
    def do_stop(self, arg):
        """Stop AI robot"""
        print("🛑 Stopping AI robot...")
        
        if self.robot:
            self.robot.running = False
            self.robot.cleanup()
            self.running = False
            print("✅ AI robot stopped!")
            self._log_event("AI robot stopped")
        else:
            print("❌ AI robot not available")
        
        if self.arduino_manager:
            self.arduino_manager.stop_robot()
    
    def do_forward(self, arg):
        """Move robot forward"""
        speed = int(arg) if arg.isdigit() else 50
        speed = max(0, min(100, speed))
        
        print(f"➡️  Moving forward at speed {speed}%")
        
        if self.arduino_manager and self.arduino_manager.is_connected():
            self.arduino_manager.arduino.move_forward(speed)
            self._log_event(f"Moved forward at {speed}% speed")
        else:
            print("  ⚠️  Arduino not connected")
    
    def do_backward(self, arg):
        """Move robot backward"""
        speed = int(arg) if arg.isdigit() else 50
        speed = max(0, min(100, speed))
        
        print(f"⬅️  Moving backward at speed {speed}%")
        
        if self.arduino_manager and self.arduino_manager.is_connected():
            self.arduino_manager.arduino.move_backward(speed)
            self._log_event(f"Moved backward at {speed}% speed")
        else:
            print("  ⚠️  Arduino not connected")
    
    def do_left(self, arg):
        """Turn robot left"""
        angle = int(arg) if arg.isdigit() else 45
        angle = max(-180, min(180, angle))
        
        print(f"↩️  Turning left {angle}°")
        
        if self.arduino_manager and self.arduino_manager.is_connected():
            self.arduino_manager.arduino.turn_left(angle)
            self._log_event(f"Turned left {angle}°")
        else:
            print("  ⚠️  Arduino not connected")
    
    def do_right(self, arg):
        """Turn robot right"""
        angle = int(arg) if arg.isdigit() else 45
        angle = max(-180, min(180, angle))
        
        print(f"↪️  Turning right {angle}°")
        
        if self.arduino_manager and self.arduino_manager.is_connected():
            self.arduino_manager.arduino.turn_right(angle)
            self._log_event(f"Turned right {angle}°")
        else:
            print("  ⚠️  Arduino not connected")
    
    def do_led(self, arg):
        """Control LED"""
        if arg.lower() in ['on', '1', 'true']:
            state = True
            state_str = "ON"
        elif arg.lower() in ['off', '0', 'false']:
            state = False
            state_str = "OFF"
        else:
            print("❌ Usage: led <on|off>")
            return
        
        print(f"💡 LED {state_str}")
        
        if self.arduino_manager and self.arduino_manager.is_connected():
            self.arduino_manager.arduino.set_led(state)
            self._log_event(f"LED turned {state_str}")
        else:
            print("  ⚠️  Arduino not connected")
    
    def do_sensors(self, arg):
        """Read sensors"""
        print("📊 Reading sensors...")
        
        if self.arduino_manager and self.arduino_manager.is_connected():
            sensor_data = self.arduino_manager.arduino.read_sensors()
            if sensor_data:
                print("  📡 Sensor Data:")
                for key, value in sensor_data.items():
                    print(f"    {key}: {value}")
                self._log_event(f"Sensors read: {sensor_data}")
            else:
                print("  ⚠️  No sensor data received")
        else:
            print("  ⚠️  Arduino not connected")
    
    def do_detections(self, arg):
        """Show detection history"""
        print("🤖 Detection History")
        print("-" * 30)
        
        if not self.detection_history:
            print("  No recent detections")
            return
        
        for i, detection in enumerate(self.detection_history[-10:], 1):  # Last 10 detections
            timestamp = detection.get('timestamp', 'Unknown')
            objects = detection.get('objects', [])
            
            print(f"{i}. {timestamp}")
            for obj in objects:
                class_name = obj.get('class', 'Unknown')
                confidence = obj.get('confidence', 0)
                print(f"   - {class_name}: {confidence:.2f}")
    
    def do_benchmark(self, arg):
        """Run performance benchmark"""
        print("🚀 Running performance benchmark...")
        
        iterations = int(arg) if arg.isdigit() else 20
        
        if self.arduino_manager and self.arduino_manager.is_connected():
            results = self.arduino_manager.arduino.benchmark(iterations)
            if results:
                print("✅ Arduino benchmark completed!")
                print(f"  Success rate: {results.get('success_rate', 0):.2f}")
                print(f"  Avg response time: {results.get('avg_response_time_ms', 0):.2f}ms")
            else:
                print("❌ Arduino benchmark failed!")
        else:
            print("  ⚠️  Arduino not connected")
    
    def do_clear(self, arg):
        """Clear detection history"""
        self.detection_history.clear()
        print("🗑️  Detection history cleared")
    
    def do_help(self, arg):
        """Show help"""
        print("🤖 Arduino UNO Q4GB AI Robot - CLI Help")
        print("=" * 50)
        print("Basic Commands:")
        print("  status        - Show system status")
        print("  connect       - Connect to Arduino")
        print("  disconnect    - Disconnect from Arduino")
        print("  start         - Start AI robot")
        print("  stop          - Stop AI robot")
        print()
        print("Movement Commands:")
        print("  forward [speed]  - Move forward (0-100%)")
        print("  backward [speed] - Move backward (0-100%)")
        print("  left [angle]     - Turn left (-180 to 180°)")
        print("  right [angle]    - Turn right (-180 to 180°)")
        print("  led <on|off>    - Control LED")
        print()
        print("Information Commands:")
        print("  sensors       - Read sensor data")
        print("  detections    - Show detection history")
        print("  benchmark [n]  - Run Arduino benchmark (n iterations)")
        print("  clear         - Clear detection history")
        print("  help          - Show this help")
        print("  quit/exit     - Exit CLI")
        print()
        print("Examples:")
        print("  forward 80    - Move forward at 80% speed")
        print("  left 90       - Turn left 90 degrees")
        print("  led on         - Turn LED on")
        print("  benchmark 50   - Run 50 Arduino tests")
    
    def do_quit(self, arg):
        """Exit CLI"""
        print("👋 Goodbye!")
        return True
    
    def do_exit(self, arg):
        """Exit CLI"""
        return self.do_quit(arg)
    
    def _log_event(self, event):
        """Log an event"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'event': event
        }
        
        self.detection_history.append(log_entry)
        
        # Keep only last 100 entries
        if len(self.detection_history) > 100:
            self.detection_history = self.detection_history[-100:]
    
    def _get_uptime(self):
        """Get system uptime"""
        if hasattr(self, '_start_time'):
            uptime = int(time.time() - self._start_time)
            return f"{uptime}s"
        else:
            self._start_time = time.time()
            return "0s"
    
    def default(self, line):
        """Handle unknown commands"""
        if line.strip():
            print(f"❌ Unknown command: {line}")
            print("  Type 'help' for available commands")
    
    def postcmd(self, stop, line):
        """Called after each command"""
        if stop:
            print()  # Add newline after command

def main():
    """Main function"""
    print("🤖 Arduino UNO Q4GB AI Robot CLI Interface")
    print("=" * 50)
    print("Type 'help' for available commands")
    print("Type 'quit' or 'exit' to leave")
    print()
    
    try:
        cli = ArduinoCLI()
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\n⚡ CLI interrupted by user")
    except Exception as e:
        print(f"❌ CLI error: {e}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)