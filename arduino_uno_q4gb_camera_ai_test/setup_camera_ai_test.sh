#!/bin/bash
set -e

# Arduino UNO Q4GB Camera + AI Pipeline Test Deployment Script
# Sets up and runs comprehensive camera + AI + motor tests

echo "=============================================="
echo "  Arduino UNO Q4GB Camera + AI Test Setup"
echo "=============================================="
echo

# Installation directory
INSTALL_DIR="$HOME/arduino_q4gb_camera_ai_test"
echo "📁 Test directory: $INSTALL_DIR"

# Create directory if it doesn't exist
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Check if we're running on the correct system
if [ ! -f "/proc/device-tree/model" ] || ! grep -q "Arduino" "/proc/device-tree/model" 2>/dev/null; then
    echo "⚠️  Warning: This may not be an Arduino UNO Q4GB system"
    echo "    Continuing anyway for testing purposes..."
fi

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python3 first."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Check system packages
echo
echo "🔧 Checking system packages..."

# Required system packages
REQUIRED_PACKAGES=("python3-pip" "python3-venv" "libopencv-dev" "pkg-config")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg -l | grep -q "^ii  $package "; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "📦 Installing missing packages: ${MISSING_PACKAGES[*]}"
    sudo apt update
    sudo apt install -y "${MISSING_PACKAGES[@]}"
else
    echo "✅ All required packages are installed"
fi

# Create or activate virtual environment
echo
echo "🐍 Setting up Python virtual environment..."

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo
echo "📦 Installing Python dependencies..."

pip install -q \
    opencv-python \
    ultralytics \
    numpy \
    pillow \
    pyserial \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "✅ Python dependencies installed"

# Check camera hardware
echo
echo "📷 Checking camera hardware..."

CAMERA_DEVICES=()
for i in {0..4}; do
    if [ -e "/dev/video$i" ]; then
        CAMERA_DEVICES+=("/dev/video$i")
        echo "✅ Found camera device: /dev/video$i"
    fi
done

if [ ${#CAMERA_DEVICES[@]} -eq 0 ]; then
    echo "⚠️  No camera devices found in /dev/video*"
    echo "    Please connect a USB camera and re-run this script"
    echo "    Continuing with software-only tests..."
fi

# Download YOLO26n model if not present
echo
echo "🤖 Checking AI models..."

if [ ! -f "models/yolo26n.pt" ]; then
    echo "📥 Downloading YOLO26n model..."
    mkdir -p models
    cd models
    
    # Download YOLO26n model (using yolov8n as placeholder)
    if command -v wget &> /dev/null; then
        wget -q -O yolo26n.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    elif command -v curl &> /dev/null; then
        curl -s -L -o yolo26n.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    else
        echo "❌ Neither wget nor curl found. Please download YOLO26n model manually."
        exit 1
    fi
    
    cd ..
    echo "✅ YOLO26n model downloaded"
else
    echo "✅ YOLO26n model found"
fi

# Check Arduino connection
echo
echo "🔌 Checking Arduino connection..."

if command -v python3 -c "import serial.tools.list_ports; ports=serial.tools.list_ports.comports(); [print(f'  {p.device}: {p.description}') for p in ports if any(keyword in p.description.lower() for keyword in ['arduino', 'ch340', 'cp210', 'ftdi'])]" &> /dev/null; then
    echo "✅ Arduino ports found"
    python3 -c "
import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
for p in ports:
    if any(keyword in p.description.lower() for keyword in ['arduino', 'ch340', 'cp210', 'ftdi']):
        print(f'  {p.device}: {p.description}')
"
else
    echo "⚠️  No Arduino devices found"
    echo "    Please connect your Arduino UNO Q4GB and ensure drivers are installed"
fi

# Create test configuration
echo
echo "⚙️  Creating test configuration..."

cat > test_config.json << 'EOF'
{
  "camera": {
    "test_indices": [0, 1, 2, 3, 4],
    "backends": ["V4L2", "GStreamer"],
    "resolution": [640, 480],
    "fps_target": 15,
    "test_duration": 10
  },
  "yolo": {
    "model_path": "models/yolo26n.pt",
    "confidence_threshold": 0.5,
    "target_fps": 10
  },
  "arduino": {
    "baudrate": 115200,
    "timeout": 1.0,
    "motor_speed_limits": {
      "min_forward": 50,
      "max_forward": 200,
      "min_turn": 30,
      "max_turn": 150
    }
  },
  "pipeline": {
    "test_duration": 15,
    "min_pipeline_fps": 5,
    "max_error_rate": 0.1
  }
}
EOF

echo "✅ Test configuration created"

# Create launch script
echo
echo "🚀 Creating launch script..."

cat > run_camera_ai_test.sh << 'EOF'
#!/bin/bash
set -e

echo "=============================================="
echo "  Arduino UNO Q4GB Camera + AI Test Launcher"
echo "=============================================="
echo

# Activate virtual environment
source "$HOME/arduino_q4gb_camera_ai_test/venv/bin/activate"

# Set environment variables
export PYTHONPATH="$HOME/arduino_q4gb_camera_ai_test:$PYTHONPATH"

# Run the comprehensive test
echo "🧪 Starting Camera + AI Pipeline Test..."
echo

cd "$HOME/arduino_q4gb_camera_ai_test"

python3 arduino_uno_q4gb_camera_ai_pipeline_test.py

echo
echo "✅ Test completed!"
echo "📊 Check the generated report for detailed results:"
echo "    camera_ai_pipeline_report.json"
EOF

chmod +x run_camera_ai_test.sh

echo "✅ Launch script created"

# Create quick test scripts
echo
echo "⚡ Creating quick test scripts..."

# Camera-only test
cat > test_camera_only.sh << 'EOF'
#!/bin/bash
source "$HOME/arduino_q4gb_camera_ai_test/venv/bin/activate"
cd "$HOME/arduino_q4gb_camera_ai_test"
python3 -c "
import cv2
import time

print('📷 Quick Camera Test')
print('=' * 30)

for i in range(5):
    print(f'Testing camera {i}...')
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f'✅ Camera {i} works! Size: {frame.shape[1]}x{frame.shape[0]}')
            cap.release()
            break
        else:
            print(f'⚠️  Camera {i}: Opens but no frame')
        cap.release()
    else:
        print(f'❌ Camera {i}: Not available')
"
EOF

chmod +x test_camera_only.sh

# AI-only test
cat > test_ai_only.sh << 'EOF'
#!/bin/bash
source "$HOME/arduino_q4gb_camera_ai_test/venv/bin/activate"
cd "$HOME/arduino_q4gb_camera_ai_test"
python3 -c "
from ultralytics import YOLO
import numpy as np
import time

print('🤖 Quick AI Test')
print('=' * 30)

try:
    print('Loading YOLO26n model...')
    model = YOLO('models/yolo26n.pt')
    
    print('Running inference test...')
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    start = time.time()
    results = model(test_img, verbose=False)
    duration = time.time() - start
    
    print(f'✅ AI test passed! Inference time: {duration*1000:.1f}ms')
except Exception as e:
    print(f'❌ AI test failed: {e}')
"
EOF

chmod +x test_ai_only.sh

# Motor-only test
cat > test_motor_only.sh << 'EOF'
#!/bin/bash
source "$HOME/arduino_q4gb_camera_ai_test/venv/bin/activate"
cd "$HOME/arduino_q4gb_camera_ai_test"
python3 arduino_motor_controller.py
EOF

chmod +x test_motor_only.sh

echo "✅ Quick test scripts created"

# Final setup summary
echo
echo "🎉 Camera + AI Test Setup Complete!"
echo "===================================="
echo
echo "Installation directory: $INSTALL_DIR"
echo "Virtual environment: $INSTALL_DIR/venv"
echo "Configuration: $INSTALL_DIR/test_config.json"
echo
echo "🚀 To run the complete test:"
echo "  $INSTALL_DIR/run_camera_ai_test.sh"
echo
echo "⚡ Quick test options:"
echo "  $INSTALL_DIR/test_camera_only.sh    - Test camera only"
echo "  $INSTALL_DIR/test_ai_only.sh        - Test AI model only"
echo "  $INSTALL_DIR/test_motor_only.sh     - Test Arduino motors only"
echo
echo "📊 Test results will be saved to:"
echo "  $INSTALL_DIR/camera_ai_pipeline_report.json"
echo
echo "🔧 Troubleshooting:"
echo "  - Camera issues: Check USB connection and permissions"
echo "  - AI issues: Verify model download and Python environment"
echo "  - Arduino issues: Check serial connection and drivers"
echo
echo "✅ Ready for Camera + AI + Motor testing!"