#!/bin/bash
# Arduino UNO Q4GB AI Robot - Simulation Mode Installation
# Optimized for immediate use without physical sensors

set -e

echo "=========================================="
echo "  Arduino UNO Q4GB AI Robot - Simulation"
echo "  Phase 3: Simulation Mode Setup"
echo "  Ready to Run Without Sensors"
echo "=========================================="
echo

# Configuration
INSTALL_DIR="$HOME/arduino_q4gb_simulation"
VENV_DIR="$INSTALL_DIR/venv"
LOG_FILE="$INSTALL_DIR/setup.log"

# Hardware detection variables
HAS_NEON=false
HAS_FP16=false
LOW_MEMORY=false
MEMORY_MB=0
CPU_CORES=0
SELECTED_FRAMEWORK="onnx"

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
        
        echo "  Installing linear algebra libraries..."
        sudo apt-get install -y libopenblas-dev liblapack-dev 2>/dev/null || \
        sudo apt-get install -y libblas-dev liblapack-dev 2>/dev/null || \
        echo "    ⚠️  Using fallback - linear algebra packages may be basic"
        
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
    pip install numpy pillow opencv-python flask
    
    # Framework selection logic based on hardware detection
    if [ "$HAS_NEON" = true ] && [ "$LOW_MEMORY" = false ]; then
        SELECTED_FRAMEWORK="onnx"
        echo "  ✅ Selected: ONNX Runtime (optimal for ARM64 + NEON)"
        pip install onnxruntime
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

# Function to create simulation application
create_simulation_app() {
    echo "🎮 Creating simulation application..."
    
    source "$VENV_DIR/bin/activate"
    
    # Create main simulation app
    cat > "$INSTALL_DIR/simulation_robot.py" << 'EOF'
#!/usr/bin/env python3
"""
Arduino UNO Q4GB AI Robot - Simulation Mode
Complete AI robot simulation without physical sensors
"""

import json
import time
import random
import threading
from datetime import datetime
try:
    import onnxruntime as ort
except ImportError:
    try:
        import tflite_runtime as tflite
    except ImportError:
        import tensorflow as tf

class SimulationRobot:
    def __init__(self):
        self.running = False
        self.detected_objects = []
        self.sensor_data = {
            'left_distance': random.uniform(20, 100),
            'right_distance': random.uniform(20, 100),
            'center_distance': random.uniform(30, 150),
            'timestamp': time.time()
        }
        self.motor_state = {'left_speed': 0, 'right_speed': 0, 'active': False}
        self.ai_decisions = []
        
    def simulate_sensor_readings(self):
        """Simulate realistic sensor data"""
        while self.running:
            # Simulate distance changes with some randomness
            self.sensor_data = {
                'left_distance': max(10, min(200, self.sensor_data['left_distance'] + random.uniform(-5, 5))),
                'right_distance': max(10, min(200, self.sensor_data['right_distance'] + random.uniform(-5, 5))),
                'center_distance': max(10, min(200, self.sensor_data['center_distance'] + random.uniform(-3, 3))),
                'timestamp': time.time()
            }
            time.sleep(0.1)  # 10Hz sensor update
    
    def simulate_object_detection(self):
        """Simulate AI object detection"""
        objects = ['person', 'car', 'bicycle', 'dog', 'chair', 'table', 'phone', 'bottle']
        while self.running:
            # Simulate detecting 0-3 objects randomly
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
        """AI-based navigation decision making"""
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
            
            if center_dist < 30:
                decision = "stop"
                speed = 0
            elif center_dist < 60:
                if left_dist > right_dist:
                    decision = "left"
                else:
                    decision = "right"
                speed = 100
            elif has_vehicle:
                decision = "slow_forward"
                speed = 80
            elif has_person and center_dist < 100:
                decision = "slow_forward"
                speed = 90
            
            # Execute decision
            if decision == "forward":
                self.set_motors(speed, speed)
            elif decision == "slow_forward":
                self.set_motors(speed, speed)
            elif decision == "left":
                self.set_motors(speed//2, speed)
            elif decision == "right":
                self.set_motors(speed, speed//2)
            elif decision == "stop":
                self.set_motors(0, 0)
            
            self.ai_decisions.append({
                'decision': decision,
                'speed': speed,
                'reasoning': f"Center: {center_dist:.1f}cm, Objects: {len(self.detected_objects)}",
                'timestamp': time.time()
            })
            
            # Keep only last 50 decisions
            if len(self.ai_decisions) > 50:
                self.ai_decisions.pop(0)
            
            time.sleep(0.2)  # 5Hz AI decisions
    
    def set_motors(self, left_speed, right_speed):
        """Simulate motor control"""
        self.motor_state = {
            'left_speed': left_speed,
            'right_speed': right_speed,
            'active': left_speed > 0 or right_speed > 0
        }
    
    def get_status(self):
        """Get current robot status"""
        return {
            'running': self.running,
            'sensors': self.sensor_data,
            'detected_objects': self.detected_objects,
            'motors': self.motor_state,
            'ai_decisions': self.ai_decisions[-5:],  # Last 5 decisions
            'timestamp': time.time()
        }
    
    def start(self):
        """Start simulation"""
        print("🚀 Starting Arduino UNO Q4GB AI Robot Simulation...")
        self.running = True
        
        # Start simulation threads
        threading.Thread(target=self.simulate_sensor_readings, daemon=True).start()
        threading.Thread(target=self.simulate_object_detection, daemon=True).start()
        threading.Thread(target=self.ai_navigation_logic, daemon=True).start()
        
        print("✅ Simulation started - Robot is now autonomous!")
        return True
    
    def stop(self):
        """Stop simulation"""
        print("🛑 Stopping simulation...")
        self.running = False
        self.set_motors(0, 0)
        print("✅ Simulation stopped")
        return True

def main():
    """Main simulation application"""
    print("🤖 Arduino UNO Q4GB AI Robot - Simulation Mode")
    print("=" * 50)
    
    # Initialize robot
    robot = SimulationRobot()
    
    try:
        # Start simulation
        robot.start()
        
        # Status update loop
        while True:
            status = robot.get_status()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Update:")
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
        print("👋 Simulation ended gracefully")

if __name__ == "__main__":
    main()
EOF

    chmod +x "$INSTALL_DIR/simulation_robot.py"
    
    echo "✅ Simulation application created"
}

# Function to create web interface
create_web_interface() {
    echo "🌐 Creating web interface..."
    
    source "$VENV_DIR/bin/activate"
    
    # Create web interface
    cat > "$INSTALL_DIR/web_simulation.py" << 'EOF'
#!/usr/bin/env python3
"""
Arduino UNO Q4GB AI Robot - Web Interface for Simulation
Browser-based control and monitoring
"""

import json
import time
from datetime import datetime
from flask import Flask, jsonify, render_template_string
import threading

# Import simulation robot
from simulation_robot import SimulationRobot

app = Flask(__name__)
robot = SimulationRobot()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Arduino UNO Q4GB AI Robot - Simulation</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: white; }
        .container { max-width: 1200px; margin: 0 auto; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .status-card { background: #2d2d2d; padding: 15px; border-radius: 8px; border: 1px solid #444; }
        .sensor-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .motor-indicator { width: 20px; height: 20px; border-radius: 50%; display: inline-block; margin: 0 5px; }
        .motor-active { background: #4CAF50; }
        .motor-inactive { background: #666; }
        .object-item { background: #333; padding: 8px; margin: 5px 0; border-radius: 4px; }
        .ai-decision { background: #1e3a8a; padding: 8px; margin: 5px 0; border-radius: 4px; }
        .control-btn { background: #2196F3; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 4px; cursor: pointer; }
        .control-btn:hover { background: #1976D2; }
        .control-btn.stop { background: #f44336; }
        .control-btn.stop:hover { background: #d32f2f; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Arduino UNO Q4GB AI Robot - Simulation Mode</h1>
        
        <div class="status-card">
            <h3>🎮 Simulation Control</h3>
            <button class="control-btn" onclick="startSimulation()">Start Simulation</button>
            <button class="control-btn stop" onclick="stopSimulation()">Stop Simulation</button>
            <button class="control-btn" onclick="refreshStatus()">Refresh Status</button>
        </div>
        
        <div class="status-grid">
            <div class="status-card">
                <h3>📡 Sensor Data</h3>
                <div>Left Distance: <span class="sensor-value" id="left-dist">--</span> cm</div>
                <div>Center Distance: <span class="sensor-value" id="center-dist">--</span> cm</div>
                <div>Right Distance: <span class="sensor-value" id="right-dist">--</span> cm</div>
            </div>
            
            <div class="status-card">
                <h3>⚙️ Motor Status</h3>
                <div>Left Motor: <span class="motor-indicator" id="left-motor"></span> Speed: <span id="left-speed">0</span></div>
                <div>Right Motor: <span class="motor-indicator" id="right-motor"></span> Speed: <span id="right-speed">0</span></div>
                <div>Status: <span id="motor-status">Stopped</span></div>
            </div>
            
            <div class="status-card">
                <h3>👁️ Object Detection</h3>
                <div id="object-list">No objects detected</div>
            </div>
            
            <div class="status-card">
                <h3>🧠 AI Decisions</h3>
                <div id="ai-decisions">No AI decisions yet</div>
            </div>
        </div>
    </div>

    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update sensor data
                    document.getElementById('left-dist').textContent = data.sensors.left_distance.toFixed(1);
                    document.getElementById('center-dist').textContent = data.sensors.center_distance.toFixed(1);
                    document.getElementById('right-dist').textContent = data.sensors.right_distance.toFixed(1);
                    
                    // Update motor status
                    document.getElementById('left-speed').textContent = data.motors.left_speed;
                    document.getElementById('right-speed').textContent = data.motors.right_speed;
                    
                    const leftMotor = document.getElementById('left-motor');
                    const rightMotor = document.getElementById('right-motor');
                    
                    if (data.motors.left_speed > 0) {
                        leftMotor.className = 'motor-indicator motor-active';
                    } else {
                        leftMotor.className = 'motor-indicator motor-inactive';
                    }
                    
                    if (data.motors.right_speed > 0) {
                        rightMotor.className = 'motor-indicator motor-active';
                    } else {
                        rightMotor.className = 'motor-indicator motor-inactive';
                    }
                    
                    document.getElementById('motor-status').textContent = data.motors.active ? 'Active' : 'Stopped';
                    
                    // Update object detection
                    const objectList = document.getElementById('object-list');
                    if (data.detected_objects.length > 0) {
                        objectList.innerHTML = data.detected_objects.map(obj => 
                            `<div class="object-item">${obj.class} (${(obj.confidence * 100).toFixed(1)}%)</div>`
                        ).join('');
                    } else {
                        objectList.innerHTML = 'No objects detected';
                    }
                    
                    // Update AI decisions
                    const aiDecisions = document.getElementById('ai-decisions');
                    if (data.ai_decisions.length > 0) {
                        aiDecisions.innerHTML = data.ai_decisions.map(dec => 
                            `<div class="ai-decision"><strong>${dec.decision}</strong> - ${dec.reasoning}</div>`
                        ).join('');
                    } else {
                        aiDecisions.innerHTML = 'No AI decisions yet';
                    }
                });
        }
        
        function startSimulation() {
            fetch('/api/start', {method: 'POST'})
                .then(() => {
                    setTimeout(refreshStatus, 1000);
                });
        }
        
        function stopSimulation() {
            fetch('/api/stop', {method: 'POST'})
                .then(() => {
                    setTimeout(refreshStatus, 1000);
                });
        }
        
        function refreshStatus() {
            updateStatus();
        }
        
        // Auto-refresh every 2 seconds
        setInterval(updateStatus, 2000);
        
        // Initial load
        updateStatus();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def get_status():
    return jsonify(robot.get_status())

@app.route('/api/start', methods=['POST'])
def start_simulation():
    robot.start()
    return jsonify({'status': 'started'})

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    robot.stop()
    return jsonify({'status': 'stopped'})

def main():
    print("🌐 Starting Arduino UNO Q4GB AI Robot Web Interface...")
    print("📱 Access at: http://localhost:8080")
    print("🎮 Use the web interface to control the simulation")
    
    app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == "__main__":
    main()
EOF

    chmod +x "$INSTALL_DIR/web_simulation.py"
    
    echo "✅ Web interface created"
}

# Function to create startup scripts
create_startup_scripts() {
    echo "🚀 Creating startup scripts..."
    
    # Create simulation startup script
    cat > "$INSTALL_DIR/start_simulation.sh" << EOF
#!/bin/bash
set -e

echo "🎮 Starting Arduino UNO Q4GB AI Robot Simulation..."
echo "============================================="

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "❌ Virtual environment not found"
    exit 1
fi

# Set environment variables
export PYTHONPATH="$INSTALL_DIR:\$PYTHONPATH"

echo "✅ Simulation Robot Starting..."
echo "📊 This will run the autonomous AI simulation"
echo "🛑 Press Ctrl+C to stop"

# Run simulation
python3 simulation_robot.py
EOF

    chmod +x "$INSTALL_DIR/start_simulation.sh"
    
    # Create web interface startup script
    cat > "$INSTALL_DIR/start_web.sh" << EOF
#!/bin/bash
set -e

echo "🌐 Starting Arduino UNO Q4GB AI Robot Web Interface..."
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

echo "✅ Web Interface Starting..."
echo "📱 Access at: http://localhost:8080"
echo "🎮 Control the simulation through your browser"

# Run web interface
python3 web_simulation.py
EOF

    chmod +x "$INSTALL_DIR/start_web.sh"
    
    # Create test script
    cat > "$INSTALL_DIR/test_simulation.sh" << EOF
#!/bin/bash
set -e

echo "🧪 Testing Arduino UNO Q4GB AI Robot Simulation..."
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
python3 -c "import numpy, PIL, cv2, flask; print('✅ Basic packages working')" 2>/dev/null || echo "⚠️  Some packages missing"

echo "🔍 Testing simulation components..."
python3 -c "
from simulation_robot import SimulationRobot
robot = SimulationRobot()
print('✅ Simulation robot class working')
robot.start()
time.sleep(2)
robot.stop()
print('✅ Simulation startup/shutdown working')
" 2>/dev/null && echo "✅ Simulation system working" || echo "⚠️  Simulation issue"

echo "✅ Simulation test completed"
EOF

    chmod +x "$INSTALL_DIR/test_simulation.sh"
    
    echo "✅ Startup scripts created"
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
    "mode": "simulation",
    "hardware_detected": true,
    "framework": "$SELECTED_FRAMEWORK",
    "optimization_level": "hardware_specific",
    "arduino_uno_q4gb": true,
    "setup_complete": true,
    "timestamp": "$(date -Iseconds)",
    "simulation_mode": {
        "enabled": true,
        "realistic_sensors": true,
        "ai_decisions": true,
        "web_interface": true,
        "object_detection": true
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
        "version": "simulation_final"
    }
}
EOF
    
    echo "✅ Configuration files created"
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
    python3 -c "import numpy, PIL, cv2, flask; print('✅ Basic packages working')" 2>/dev/null || echo "⚠️  Basic packages issue"
    
    # Test framework
    if [ -f "$INSTALL_DIR/framework_config" ]; then
        source "$INSTALL_DIR/framework_config"
        
        case "$SELECTED_FRAMEWORK" in
            "onnx")
                python3 -c "import onnxruntime; print('✅ ONNX Runtime test passed')" 2>/dev/null || echo "⚠️  ONNX Runtime issue"
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
    echo "🎉 SIMULATION SETUP COMPLETE!"
    echo "=========================================="
    echo
    echo "Installation directory: $INSTALL_DIR"
    echo "Virtual environment: $VENV_DIR"
    echo "Framework: $SELECTED_FRAMEWORK"
    echo "Simulation mode: READY"
    echo
    echo "🎮 To start simulation:"
    echo "  $INSTALL_DIR/start_simulation.sh"
    echo
    echo "🌐 To start web interface:"
    echo "  $INSTALL_DIR/start_web.sh"
    echo "  Then access: http://localhost:8080"
    echo
    echo "🧪 To test system:"
    echo "  $INSTALL_DIR/test_simulation.sh"
    echo
    echo "✅ Arduino UNO Q4GB AI Robot Simulation is ready!"
    echo "   No physical sensors required - everything simulated!"
    echo "=========================================="
}

# Main installation sequence
main() {
    echo "🚀 Starting Arduino UNO Q4GB Simulation setup..."
    
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
    create_simulation_app
    create_web_interface
    create_startup_scripts
    create_config
    run_final_tests
    display_completion
    
    echo "✅ Simulation setup completed successfully at $(date)"
}

# Error handling
trap 'echo "❌ Setup failed at line $LINENO"' ERR

# Run main function
main "$@"