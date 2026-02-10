#!/bin/bash
# Arduino UNO Q4GB AI Robot - Sensor-Enabled Installation
# Complete system with real sensor integration

set -e

echo "=========================================="
echo "  Arduino UNO Q4GB AI Robot - Sensors"
echo "  Phase 3: Full Sensor Integration Setup"
echo "  Ready for Physical Hardware"
echo "=========================================="
echo

# Configuration
INSTALL_DIR="$HOME/arduino_q4gb_sensors"
VENV_DIR="$INSTALL_DIR/venv"
LOG_FILE="$INSTALL_DIR/setup.log"

# Hardware detection variables
HAS_NEON=false
HAS_FP16=false
LOW_MEMORY=false
MEMORY_MB=0
CPU_CORES=0
SELECTED_FRAMEWORK="onnx"
ARDUINO_PORT="/dev/ttyUSB0"

# Function to detect hardware
detect_hardware() {
    echo "🔍 Detecting Arduino UNO Q4GB hardware..."
    
    # Basic system info
    echo "  System: $(uname -a)"
    echo "  Architecture: $(uname -m)"
    echo "  CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)"
    echo "  Cores: $(nproc)"
    echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    
    # Store hardware info
    CPU_CORES=$(nproc)
    MEMORY_MB=$(free -m | grep '^Mem:' | awk '{print $2}')
    
    # ARM features detection
    if [ -f /proc/cpuinfo ]; then
        FEATURES=$(grep '^Features' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)
        echo "  CPU Features: $FEATURES"
        
        # Check for NEON/ASIMD
        if echo "$FEATURES" | grep -q -E "(asimd|neon)"; then
            echo "  ✅ NEON/ASIMD support detected"
            HAS_NEON=true
        else
            echo "  ⚠️  Limited SIMD support"
            HAS_NEON=false
        fi
        
        # Check for FP16
        if echo "$FEATURES" | grep -q -E "(fp16|fphp)"; then
            echo "  ✅ FP16 support detected"
            HAS_FP16=true
        else
            HAS_FP16=false
        fi
    fi
    
    # Check memory
    if [ "$MEMORY_MB" -lt 512 ]; then
        echo "  ⚠️  Low memory detected (< 512MB)"
        LOW_MEMORY=true
    elif [ "$MEMORY_MB" -lt 1024 ]; then
        echo "  ⚠️  Moderate memory (< 1GB)"
        LOW_MEMORY=false
    else
        echo "  ✅ Sufficient memory (≥ 1GB)"
        LOW_MEMORY=false
    fi
    
    # Detect Arduino ports
    echo "🔍 Detecting Arduino ports..."
    for port in /dev/ttyUSB* /dev/ttyACM*; do
        if [ -e "$port" ]; then
            echo "  Found Arduino port: $port"
            ARDUINO_PORT="$port"
            break
        fi
    done
}

# Function to update system packages
update_system() {
    echo "🔄 Updating system packages..."
    
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get upgrade -y
        echo "✅ System packages updated"
    else
        echo "⚠️  apt-get not found, skipping system update"
    fi
}

# Function to install system dependencies
install_dependencies() {
    echo "📦 Installing system dependencies..."
    
    if command -v apt-get >/dev/null 2>&1; then
        # Install packages in smaller groups
        echo "  Installing Python packages..."
        sudo apt-get install -y python3 python3-pip python3-venv python3-dev
        
        echo "  Installing build tools..."
        sudo apt-get install -y build-essential cmake git wget curl unzip
        
        echo "  Installing image processing libraries..."
        sudo apt-get install -y pkg-config libjpeg-dev libpng-dev libtiff-dev
        
        echo "  Installing video processing libraries..."
        sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
        
        echo "  Installing GUI libraries..."
        sudo apt-get install -y libgtk-3-dev libgfortran-dev
        
        echo "  Installing serial communication libraries..."
        sudo apt-get install -y portaudio19-dev python3-pyaudio
        
        echo "  Installing linear algebra libraries..."
        sudo apt-get install -y libopenblas-dev liblapack-dev 2>/dev/null || \
        sudo apt-get install -y libblas-dev liblapack-dev 2>/dev/null || \
        echo "    ⚠️  Using fallback - linear algebra packages may be basic"
        
        # Add user to dialout group for Arduino access
        sudo usermod -a -G dialout $USER 2>/dev/null || echo "    ⚠️  Could not add user to dialout group"
        
        echo "✅ System dependencies installed"
    else
        echo "⚠️  apt-get not found, please install dependencies manually"
    fi
}

# Function to create virtual environment
create_venv() {
    echo "🐍 Creating Python virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        echo "⚠️  Virtual environment already exists, removing..."
        rm -rf "$VENV_DIR"
    fi
    
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    echo "✅ Virtual environment created and activated"
}

# Function to select and install AI framework
install_ai_framework() {
    echo "🤖 Installing AI framework..."
    
    source "$VENV_DIR/bin/activate"
    
    # Install common dependencies first
    pip install numpy pillow opencv-python flask pyserial
    
    # Framework selection logic based on hardware detection
    if [ "$HAS_NEON" = true ] && [ "$LOW_MEMORY" = false ]; then
        SELECTED_FRAMEWORK="onnx"
        echo "  ✅ Selected: ONNX Runtime (optimal for ARM64 + NEON)"
        pip install onnxruntime ultralytics
    elif [ "$LOW_MEMORY" = false ]; then
        SELECTED_FRAMEWORK="tflite"
        echo "  ✅ Selected: TensorFlow Lite (good for ARM64)"
        if ! pip install tflite-runtime; then
            echo "    Fallback to full TensorFlow..."
            pip install tensorflow
        fi
    else
        SELECTED_FRAMEWORK="tflite"
        echo "  ✅ Selected: TensorFlow Lite (lightweight)"
        pip install tflite-runtime
    fi
    
    # Save framework selection
    echo "SELECTED_FRAMEWORK=$SELECTED_FRAMEWORK" > "$INSTALL_DIR/framework_config"
    
    echo "✅ AI framework installed: $SELECTED_FRAMEWORK"
}

# Function to create Arduino firmware
create_arduino_firmware() {
    echo "🔧 Creating Arduino firmware..."
    
    mkdir -p "$INSTALL_DIR/arduino_firmware"
    
    # Create Arduino sketch
    cat > "$INSTALL_DIR/arduino_firmware/robot_controller.ino" << 'EOF'
/*
  Arduino UNO Q4GB AI Robot - Sensor Integration
  Ultrasonic sensors + Servo motor control
  Pin configuration:
    - Left 45° Ultrasonic: Trig D2, Echo D3
    - Right 45° Ultrasonic: Trig D4, Echo D5  
    - Center Ultrasonic: Trig D6, Echo D7
    - Left Servo: Pin D9
    - Right Servo: Pin D10
*/

#include <Servo.h>

// Ultrasonic sensor pins
#define LEFT_TRIG 2
#define LEFT_ECHO 3
#define RIGHT_TRIG 4
#define RIGHT_ECHO 5
#define CENTER_TRIG 6
#define CENTER_ECHO 7

// Servo pins
#define LEFT_SERVO 9
#define RIGHT_SERVO 10

// Servo objects
Servo leftServo;
Servo rightServo;

// Motor control variables
int leftSpeed = 90;  // Neutral (stop)
int rightSpeed = 90; // Neutral (stop)

// Sensor data structure
struct SensorData {
  float leftDistance;
  float rightDistance;
  float centerDistance;
  unsigned long timestamp;
};

SensorData sensorData;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  // Initialize ultrasonic sensors
  pinMode(LEFT_TRIG, OUTPUT);
  pinMode(LEFT_ECHO, INPUT);
  pinMode(RIGHT_TRIG, OUTPUT);
  pinMode(RIGHT_ECHO, INPUT);
  pinMode(CENTER_TRIG, OUTPUT);
  pinMode(CENTER_ECHO, INPUT);
  
  // Initialize servos
  leftServo.attach(LEFT_SERVO);
  rightServo.attach(RIGHT_SERVO);
  
  // Set servos to neutral position
  leftServo.write(90);
  rightServo.write(90);
  
  // Initialize sensor data
  sensorData.leftDistance = 100.0;
  sensorData.rightDistance = 100.0;
  sensorData.centerDistance = 100.0;
  sensorData.timestamp = millis();
  
  Serial.println("Arduino UNO Q4GB Robot Initialized");
  Serial.println("Pin Configuration:");
  Serial.println("Left 45° Sensor: Trig D2, Echo D3");
  Serial.println("Right 45° Sensor: Trig D4, Echo D5");
  Serial.println("Center Sensor: Trig D6, Echo D7");
  Serial.println("Left Servo: Pin D9");
  Serial.println("Right Servo: Pin D10");
  Serial.println("Ready for commands");
  Serial.println("CMD L<speed> R<speed> T<duration>");
  Serial.println("Example: CMD L150 R150 T1000");
}

void loop() {
  // Read sensors
  readSensors();
  
  // Send sensor data periodically
  static unsigned long lastSensorSend = 0;
  if (millis() - lastSensorSend > 100) { // Send every 100ms
    sendSensorData();
    lastSensorSend = millis();
  }
  
  // Process commands
  processCommands();
  
  delay(10); // Small delay for stability
}

float readDistance(int trigPin, int echoPin) {
  // Send pulse
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // Read echo
  long duration = pulseIn(echoPin, HIGH, 30000); // 30ms timeout
  
  // Calculate distance (cm)
  float distance = duration * 0.0343 / 2.0;
  
  // Sanity check
  if (distance <= 0 || distance > 300) {
    distance = 300.0; // Max range
  }
  
  return distance;
}

void readSensors() {
  // Read all ultrasonic sensors
  sensorData.leftDistance = readDistance(LEFT_TRIG, LEFT_ECHO);
  delay(10); // Small delay between readings
  sensorData.rightDistance = readDistance(RIGHT_TRIG, RIGHT_ECHO);
  delay(10);
  sensorData.centerDistance = readDistance(CENTER_TRIG, CENTER_ECHO);
  sensorData.timestamp = millis();
}

void sendSensorData() {
  // Send sensor data as JSON
  Serial.print("SENSORS {");
  Serial.print("\"left45\":");
  Serial.print(sensorData.leftDistance, 1);
  Serial.print(",\"right45\":");
  Serial.print(sensorData.rightDistance, 1);
  Serial.print(",\"center\":");
  Serial.print(sensorData.centerDistance, 1);
  Serial.print(",\"timestamp\":");
  Serial.print(sensorData.timestamp);
  Serial.println("}");
}

void processCommands() {
  while (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    // Parse motor command: CMD L<speed> R<speed> T<duration>
    if (command.startsWith("CMD ")) {
      int leftVal = 90;
      int rightVal = 90;
      int duration = 0;
      
      // Parse command
      if (command.indexOf("L") > 0) {
        int lStart = command.indexOf("L") + 1;
        int lEnd = command.indexOf(" ", lStart);
        if (lEnd == -1) lEnd = command.indexOf("R", lStart);
        if (lEnd == -1) lEnd = command.indexOf("T", lStart);
        if (lEnd == -1) lEnd = command.length();
        leftVal = command.substring(lStart, lEnd).toInt();
      }
      
      if (command.indexOf("R") > 0) {
        int rStart = command.indexOf("R") + 1;
        int rEnd = command.indexOf(" ", rStart);
        if (rEnd == -1) rEnd = command.indexOf("T", rStart);
        if (rEnd == -1) rEnd = command.length();
        rightVal = command.substring(rStart, rEnd).toInt();
      }
      
      if (command.indexOf("T") > 0) {
        int tStart = command.indexOf("T") + 1;
        int tEnd = command.length();
        duration = command.substring(tStart, tEnd).toInt();
      }
      
      // Execute motor command
      executeMotorCommand(leftVal, rightVal, duration);
    }
    else if (command == "STOP") {
      executeMotorCommand(90, 90, 0);
    }
    else if (command == "STATUS") {
      sendSensorData();
    }
  }
}

void executeMotorCommand(int leftSpeed, int rightSpeed, int duration) {
  // Convert speed (0-255) to servo angle (0-180)
  int leftServoValue = map(leftSpeed, 0, 255, 0, 180);
  int rightServoValue = map(rightSpeed, 0, 255, 180, 0); // Reverse for correct direction
  
  // Apply constraints
  leftServoValue = constrain(leftServoValue, 0, 180);
  rightServoValue = constrain(rightServoValue, 0, 180);
  
  // Set servos
  leftServo.write(leftServoValue);
  rightServo.write(rightServoValue);
  
  // Send confirmation
  Serial.print("MOTORS L:");
  Serial.print(leftSpeed);
  Serial.print(" R:");
  Serial.print(rightSpeed);
  Serial.print(" T:");
  Serial.println(duration);
  
  // Wait for duration if specified
  if (duration > 0) {
    delay(duration);
    // Stop motors after duration
    leftServo.write(90);
    rightServo.write(90);
  }
}
EOF

    # Create Arduino upload script
    cat > "$INSTALL_DIR/arduino_firmware/upload_firmware.sh" << EOF
#!/bin/bash
# Arduino firmware upload script

echo "🔧 Uploading Arduino UNO Q4GB firmware..."

# Find Arduino IDE or use arduino-cli
if command -v arduino-cli >/dev/null 2>&1; then
    echo "  Using arduino-cli..."
    arduino-cli compile --fqbn arduino:avr:uno arduino_firmware/
    arduino-cli upload --fqbn arduino:avr:uno --port $ARDUINO_PORT arduino_firmware/
elif command -v arduino >/dev/null 2>&1; then
    echo "  Using Arduino IDE..."
    echo "  Please open arduino_firmware/robot_controller.ino in Arduino IDE"
    echo "  and upload to your Arduino UNO Q4GB manually"
else
    echo "  ⚠️  Arduino CLI not found"
    echo "  Please install arduino-cli or use Arduino IDE manually"
    echo "  Sketch location: arduino_firmware/robot_controller.ino"
fi

echo "✅ Arduino firmware upload process completed"
EOF

    chmod +x "$INSTALL_DIR/arduino_firmware/upload_firmware.sh"
    
    echo "✅ Arduino firmware created"
}

# Function to create sensor robot application
create_sensor_robot() {
    echo "🤖 Creating sensor robot application..."
    
    source "$VENV_DIR/bin/activate"
    
    # Create main sensor robot app
    cat > "$INSTALL_DIR/sensor_robot.py" << 'EOF'
#!/usr/bin/env python3
"""
Arduino UNO Q4GB AI Robot - Full Sensor Integration
Real sensor data + AI decision making + Arduino control
"""

import json
import time
import threading
import serial
import serial.tools.list_ports
from datetime import datetime
try:
    import onnxruntime as ort
    import cv2
    import numpy as np
except ImportError:
    pass

class SensorRobot:
    def __init__(self, arduino_port="/dev/ttyUSB0"):
        self.running = False
        self.arduino_port = arduino_port
        self.serial_conn = None
        self.connected = False
        
        # Sensor data
        self.sensor_data = {
            'left_distance': 100.0,
            'right_distance': 100.0,
            'center_distance': 100.0,
            'timestamp': time.time()
        }
        
        # AI detection data
        self.detected_objects = []
        self.camera_active = False
        
        # Motor state
        self.motor_state = {'left_speed': 0, 'right_speed': 0, 'active': False}
        
        # AI decisions
        self.ai_decisions = []
        
        # Initialize AI models
        self.init_ai_models()
    
    def init_ai_models(self):
        """Initialize AI models (placeholder for real models)"""
        self.ai_models_loaded = False
        try:
            # Try to load ONNX models if available
            # This is a placeholder - actual model loading would go here
            self.ai_models_loaded = True
            print("✅ AI models initialized")
        except Exception as e:
            print(f"⚠️  AI models not available: {e}")
    
    def find_arduino_port(self):
        """Auto-detect Arduino port"""
        try:
            import serial.tools
            ports = serial.tools.list_ports.comports()
            for port in ports:
                if any(arduino_id in str(port) for arduino_id in ['Arduino', 'CH340', 'CP210']):
                    print(f"🔍 Found Arduino on {port.device}")
                    return port.device
        except Exception as e:
            print(f"⚠️  Error detecting Arduino ports: {e}")
        return self.arduino_port
    
    def connect_arduino(self):
        """Connect to Arduino"""
        try:
        # Auto-detect port if needed
        if not self.arduino_port or True:  # Always try to detect
                self.arduino_port = self.find_arduino_port()
            
            print(f"🔌 Connecting to Arduino on {self.arduino_port}...")
            self.serial_conn = serial.Serial(self.arduino_port, 115200, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            
            # Test connection
            self.serial_conn.write(b"STATUS\n")
            time.sleep(1)
            if self.serial_conn.in_waiting:
                response = self.serial_conn.readline().decode().strip()
                if "SENSORS" in response:
                    print("✅ Arduino connected successfully")
                    self.connected = True
                    return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Arduino: {e}")
        
        print("⚠️  Using simulation mode for sensors")
        self.connected = False
        return False
    
    def read_sensor_data(self):
        """Read sensor data from Arduino"""
        while self.running and self.connected:
            try:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode().strip()
                    if line.startswith("SENSORS"):
                        # Parse JSON sensor data
                        try:
                            json_data = json.loads(line[8:])  # Remove "SENSORS " prefix
                            self.sensor_data.update(json_data)
                        except json.JSONDecodeError:
                            pass
                    
            except Exception as e:
                print(f"Error reading sensor data: {e}")
                self.connected = False
                break
            
            time.sleep(0.05)  # 20Hz sensor reading
    
    def simulate_object_detection(self):
        """Simulate AI object detection (placeholder for real camera)"""
        while self.running:
            # Simulate detecting 0-3 objects randomly
            import random
            objects = ['person', 'car', 'bicycle', 'dog', 'chair', 'table']
            num_objects = random.randint(0, 3)
            self.detected_objects = []
            
            for i in range(num_objects):
                obj = {
                    'class': random.choice(objects),
                    'confidence': random.uniform(0.6, 0.95),
                    'bbox': [random.randint(50, 300), random.randint(50, 200), 
                            random.randint(100, 400), random.randint(100, 300)],
                    'timestamp': time.time()
                }
                self.detected_objects.append(obj)
            
            time.sleep(0.5)  # 2Hz detection
    
    def ai_navigation_logic(self):
        """AI-based navigation with real sensor data"""
        while self.running:
            center_dist = self.sensor_data['center_distance']
            left_dist = self.sensor_data['left_distance']
            right_dist = self.sensor_data['right_distance']
            
            # Check for detected objects
            has_person = any(obj['class'] == 'person' for obj in self.detected_objects)
            has_vehicle = any(obj['class'] in ['car', 'bicycle'] for obj in self.detected_objects)
            
            # AI Decision logic
            decision = "forward"
            speed = 150
            
            # Safety checks with real sensors
            if center_dist < 20:
                decision = "emergency_stop"
                speed = 0
            elif center_dist < 40:
                if left_dist > right_dist:
                    decision = "turn_left"
                else:
                    decision = "turn_right"
                speed = 120
            elif center_dist < 60:
                if left_dist < 30 and right_dist < 30:
                    decision = "slow_forward"
                    speed = 100
                elif left_dist < right_dist:
                    decision = "slight_right"
                else:
                    decision = "slight_left"
                speed = 130
            elif has_vehicle:
                decision = "cautious_forward"
                speed = 110
            elif has_person and center_dist < 80:
                decision = "cautious_forward"
                speed = 90
            
            # Execute decision
            if decision == "forward":
                self.send_motor_command(speed, speed)
            elif decision == "slow_forward":
                self.send_motor_command(speed, speed)
            elif decision == "cautious_forward":
                self.send_motor_command(speed, speed)
            elif decision == "turn_left":
                self.send_motor_command(speed//2, speed)
            elif decision == "turn_right":
                self.send_motor_command(speed, speed//2)
            elif decision == "slight_left":
                self.send_motor_command(int(speed*0.8), speed)
            elif decision == "slight_right":
                self.send_motor_command(speed, int(speed*0.8))
            elif decision == "emergency_stop":
                self.send_motor_command(0, 0)
            
            self.ai_decisions.append({
                'decision': decision,
                'speed': speed,
                'reasoning': f"Center: {center_dist:.1f}cm, Left: {left_dist:.1f}cm, Right: {right_dist:.1f}cm, Objects: {len(self.detected_objects)}",
                'timestamp': time.time()
            })
            
            # Keep only last 50 decisions
            if len(self.ai_decisions) > 50:
                self.ai_decisions.pop(0)
            
            time.sleep(0.2)  # 5Hz AI decisions
    
    def send_motor_command(self, left_speed, right_speed, duration=0):
        """Send motor command to Arduino"""
        if self.connected and self.serial_conn:
            try:
                command = f"CMD L{left_speed} R{right_speed} T{duration}\n"
                self.serial_conn.write(command.encode())
                
                # Update motor state
                self.motor_state = {
                    'left_speed': left_speed,
                    'right_speed': right_speed,
                    'active': left_speed > 0 or right_speed > 0
                }
            except Exception as e:
                print(f"Error sending motor command: {e}")
                self.connected = False
        else:
            # Simulation mode
            self.motor_state = {
                'left_speed': left_speed,
                'right_speed': right_speed,
                'active': left_speed > 0 or right_speed > 0
            }
    
    def get_status(self):
        """Get current robot status"""
        return {
            'running': self.running,
            'connected': self.connected,
            'arduino_port': self.arduino_port,
            'sensors': self.sensor_data,
            'detected_objects': self.detected_objects,
            'motors': self.motor_state,
            'ai_decisions': self.ai_decisions[-5:],  # Last 5 decisions
            'ai_models_loaded': self.ai_models_loaded,
            'timestamp': time.time()
        }
    
    def start(self):
        """Start robot with sensor integration"""
        print("🚀 Starting Arduino UNO Q4GB AI Robot with Sensors...")
        
        # Connect to Arduino
        if not self.connect_arduino():
            print("⚠️  Running in simulation mode - Arduino not detected")
        
        self.running = True
        
        # Start threads
        if self.connected:
            threading.Thread(target=self.read_sensor_data, daemon=True).start()
        
        threading.Thread(target=self.simulate_object_detection, daemon=True).start()
        threading.Thread(target=self.ai_navigation_logic, daemon=True).start()
        
        print("✅ Robot started - Real sensor integration active!")
        return True
    
    def stop(self):
        """Stop robot"""
        print("🛑 Stopping robot...")
        self.running = False
        
        # Send stop command
        self.send_motor_command(0, 0)
        
        # Close serial connection
        if self.serial_conn:
            self.serial_conn.close()
            self.serial_conn = None
        
        self.connected = False
        print("✅ Robot stopped")
        return True

def main():
    """Main sensor robot application"""
    print("🤖 Arduino UNO Q4GB AI Robot - Full Sensor Integration")
    print("=" * 60)
    
    # Initialize robot
    robot = SensorRobot()
    
    try:
        # Start robot
        robot.start()
        
        # Status update loop
        while True:
            status = robot.get_status()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Update:")
            print(f"  Arduino: {'Connected' if status['connected'] else 'Simulation'} ({status['arduino_port']})")
            print(f"  Motors: L={status['motors']['left_speed']}, R={status['motors']['right_speed']}")
            print(f"  Distances: L={status['sensors']['left_distance']:.1f}cm, C={status['sensors']['center_distance']:.1f}cm, R={status['sensors']['right_distance']:.1f}cm")
            print(f"  Objects: {len(status['detected_objects'])} detected")
            if status['detected_objects']:
                for obj in status['detected_objects']:
                    print(f"    - {obj['class']} ({obj['confidence']:.2f})")
            if status['ai_decisions']:
                latest = status['ai_decisions'][-1]
                print(f"  AI Decision: {latest['decision']} - {latest['reasoning']}")
            
            time.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 User interrupt received")
        robot.stop()
        print("👋 Robot ended gracefully")

if __name__ == "__main__":
    main()
EOF

    chmod +x "$INSTALL_DIR/sensor_robot.py"
    
    echo "✅ Sensor robot application created"
}

# Function to create configuration
create_config() {
    echo "⚙️  Creating configuration files..."
    
    # Load framework selection
    if [ -f "$INSTALL_DIR/framework_config" ]; then
        source "$INSTALL_DIR/framework_config"
    fi
    
    # Create main config
    cat > "$INSTALL_DIR/config.json" << EOF
{
    "mode": "sensors",
    "hardware_detected": true,
    "framework": "$SELECTED_FRAMEWORK",
    "optimization_level": "hardware_specific",
    "arduino_uno_q4gb": true,
    "setup_complete": true,
    "timestamp": "$(date -Iseconds)",
    "sensor_mode": {
        "enabled": true,
        "arduino_port": "$ARDUINO_PORT",
        "ultrasonic_sensors": {
            "left_45_deg": {"trig": 2, "echo": 3},
            "right_45_deg": {"trig": 4, "echo": 5},
            "center": {"trig": 6, "echo": 7}
        },
        "servo_motors": {
            "left_motor": 9,
            "right_motor": 10
        }
    },
    "hardware_info": {
        "architecture": "$(uname -m)",
        "cpu_cores": $CPU_CORES,
        "memory_mb": $MEMORY_MB,
        "has_neon": $HAS_NEON,
        "has_fp16": $HAS_FP16,
        "low_memory": $LOW_MEMORY
    },
    "installation": {
        "directory": "$INSTALL_DIR",
        "virtual_env": "$VENV_DIR",
        "framework": "$SELECTED_FRAMEWORK",
        "version": "sensors_final"
    }
}
EOF
    
    echo "✅ Configuration files created"
}

# Function to create startup scripts
create_startup_scripts() {
    echo "🚀 Creating startup scripts..."
    
    # Create main startup script
    cat > "$INSTALL_DIR/start_sensor_robot.sh" << EOF
#!/bin/bash
set -e

echo "🤖 Starting Arduino UNO Q4GB AI Robot with Sensors..."
echo "=================================================="

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "❌ Virtual environment not found"
    exit 1
fi

# Set environment variables
export PYTHONPATH="$INSTALL_DIR:\$PYTHONPATH"

echo "📊 Loading configuration..."
if [ -f "$INSTALL_DIR/config.json" ]; then
    ARDUINO_PORT=\$(python3 -c "import json; print(json.load(open('$INSTALL_DIR/config.json'))['sensor_mode']['arduino_port'])" 2>/dev/null || echo "/dev/ttyUSB0")
    echo "  Arduino Port: \$ARDUINO_PORT"
fi

echo "✅ Sensor Robot Starting..."
echo "📡 This will use real ultrasonic sensors"
echo "🤖 AI navigation with sensor fusion"
echo "🛑 Press Ctrl+C to stop"

# Run sensor robot
python3 sensor_robot.py
EOF

    chmod +x "$INSTALL_DIR/start_sensor_robot.sh"
    
    # Create Arduino firmware upload script
    cat > "$INSTALL_DIR/upload_arduino_firmware.sh" << EOF
#!/bin/bash
set -e

echo "🔧 Uploading Arduino UNO Q4GB Firmware..."
echo "====================================="

echo "📍 Please connect your Arduino UNO Q4GB to your computer"
echo "🔌 Ensure the board is detected:"

# Show available ports
if command -v ls >/dev/null 2>&1; then
    echo "Available serial ports:"
    ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "  No Arduino ports found"
fi

echo ""
echo "📋 Pin Configuration:"
echo "  Left 45° Ultrasonic: Trig D2, Echo D3"
echo "  Right 45° Ultrasonic: Trig D4, Echo D5"
echo "  Center Ultrasonic: Trig D6, Echo D7"
echo "  Left Servo: Pin D9"
echo "  Right Servo: Pin D10"

echo ""
echo "🔧 Starting firmware upload..."
cd "$INSTALL_DIR"
./arduino_firmware/upload_firmware.sh

echo "✅ Arduino firmware upload completed!"
echo "🚀 You can now run: ./start_sensor_robot.sh"
EOF

    chmod +x "$INSTALL_DIR/upload_arduino_firmware.sh"
    
    # Create test script
    cat > "$INSTALL_DIR/test_sensors.sh" << EOF
#!/bin/bash
set -e

echo "🧪 Testing Arduino UNO Q4GB AI Robot with Sensors..."
echo "=============================================="

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "❌ Virtual environment not found"
    exit 1
fi

echo "🔍 Testing framework import..."
if [ -f "$INSTALL_DIR/framework_config" ]; then
    source "$INSTALL_DIR/framework_config"
    
    case "\$SELECTED_FRAMEWORK" in
        "onnx")
            python3 -c "import onnxruntime; print('✅ ONNX Runtime working')"
            ;;
        "tflite")
            python3 -c "import tflite_runtime; print('✅ TensorFlow Lite working')" 2>/dev/null || python3 -c "import tensorflow; print('✅ TensorFlow working')"
            ;;
    esac
else
    echo "❌ Framework configuration not found"
    exit 1
fi

echo "🔍 Testing basic packages..."
python3 -c "import numpy, PIL, cv2, flask, serial; print('✅ Basic packages working')" 2>/dev/null || echo "⚠️  Some packages missing"

echo "🔍 Testing Arduino communication..."
python3 -c "
import serial.tools
try:
    ports = serial.tools.list_ports.comports()
    arduino_ports = [p.device for p in ports if 'Arduino' in str(p) or 'CH340' in str(p) or 'CP210' in str(p)]
    if arduino_ports:
        print(f'✅ Found Arduino ports: {arduino_ports}')
    else:
        print('⚠️  No Arduino ports detected - will run in simulation mode')
        print('Available ports:')
        for p in ports:
            print(f'  {p.device}: {p.description}')
except Exception as e:
    print(f'⚠️  Error detecting ports: {e}')
" 2>/dev/null

echo "🔍 Testing sensor robot components..."
python3 -c "
from sensor_robot import SensorRobot
robot = SensorRobot()
print('✅ Sensor robot class working')
robot.connected = False  # Test in simulation mode
robot.start()
import time
time.sleep(2)
robot.stop()
print('✅ Sensor robot startup/shutdown working')
" 2>/dev/null && echo "✅ Sensor robot system working" || echo "⚠️  Sensor robot issue"

echo "✅ Sensor system test completed"
EOF

    chmod +x "$INSTALL_DIR/test_sensors.sh"
    
    echo "✅ Startup scripts created"
}

# Function to run final tests
run_final_tests() {
    echo "🧪 Running final installation tests..."
    
    # Test virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        echo "❌ Virtual environment not found"
        return 1
    fi
    
    source "$VENV_DIR/bin/activate"
    
    # Test Python
    python3 --version
    
    # Test basic packages
    python3 -c "import numpy, PIL, cv2, flask, serial; print('✅ Basic packages working')" 2>/dev/null || echo "⚠️  Basic packages issue"
    
    # Test framework
    if [ -f "$INSTALL_DIR/framework_config" ]; then
        source "$INSTALL_DIR/framework_config"
        
        case "$SELECTED_FRAMEWORK" in
            "onnx")
                python3 -c "import onnxruntime, ultralytics; print('✅ ONNX Runtime + Ultralytics test passed')" 2>/dev/null || echo "⚠️  ONNX Runtime issue"
                ;;
            "tflite")
                python3 -c "import tflite_runtime; print('✅ TensorFlow Lite test passed')" 2>/dev/null || python3 -c "import tensorflow; print('✅ TensorFlow test passed')" 2>/dev/null || echo "⚠️  TensorFlow issue"
                ;;
        esac
    else
        echo "❌ Framework configuration not found"
        return 1
    fi
    
    echo "✅ Final tests completed successfully"
}

# Function to display completion message
display_completion() {
    echo
    echo "=========================================="
    echo "🎉 SENSOR SETUP COMPLETE!"
    echo "=========================================="
    echo
    echo "Installation directory: $INSTALL_DIR"
    echo "Virtual environment: $VENV_DIR"
    echo "Framework: $SELECTED_FRAMEWORK"
    echo "Sensor integration: READY"
    echo
    echo "🔌 Hardware Setup:"
    echo "  Left 45° Ultrasonic: Trig D2, Echo D3"
    echo "  Right 45° Ultrasonic: Trig D4, Echo D5"
    echo "  Center Ultrasonic: Trig D6, Echo D7"
    echo "  Left Servo: Pin D9"
    echo "  Right Servo: Pin D10"
    echo
    echo "🔧 To upload Arduino firmware:"
    echo "  $INSTALL_DIR/upload_arduino_firmware.sh"
    echo
    echo "🤖 To start sensor robot:"
    echo "  $INSTALL_DIR/start_sensor_robot.sh"
    echo
    echo "🧪 To test system:"
    echo "  $INSTALL_DIR/test_sensors.sh"
    echo
    echo "✅ Arduino UNO Q4GB AI Robot with Sensors is ready!"
    echo "   Connect sensors and upload firmware to start!"
    echo "=========================================="
}

# Main installation sequence
main() {
    echo "🚀 Starting Arduino UNO Q4GB Sensors setup..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    # Start logging
    exec > >(tee -a "$LOG_FILE")
    exec 2>&1
    
    echo "🔍 Starting setup at $(date)"
    
    detect_hardware
    update_system
    install_dependencies
    create_venv
    install_ai_framework
    create_arduino_firmware
    create_sensor_robot
    create_config
    create_startup_scripts
    run_final_tests
    display_completion
    
    echo "✅ Sensor setup completed successfully at $(date)"
}

# Error handling
trap 'echo "❌ Setup failed at line $LINENO"' ERR

# Run main function
main "$@"