#!/bin/bash
set -e

# Arduino UNO Q4GB ARM-Fixed Camera + AI Test Deployment Script
# Comprehensive fix for illegal instruction errors on ARM systems

echo "=============================================="
echo "  Arduino UNO Q4GB ARM-Fixed Deployment"
echo "=============================================="
echo

# Installation directory
INSTALL_DIR="$HOME/arduino_uno_q4gb_arm_fixed"
echo "📁 Deployment directory: $INSTALL_DIR"

# Create directory if it doesn't exist
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Check ARM architecture
ARCH=$(uname -m)
echo "🔍 Detected architecture: $ARCH"

if [[ ! "$ARCH" =~ (arm|aarch64) ]]; then
    echo "⚠️  Warning: This is designed for ARM systems"
    echo "    Current: $ARCH - continuing for testing"
fi

# Check system capabilities
echo
echo "🔧 Checking system capabilities..."

# CPU info
if [ -f "/proc/cpuinfo" ]; then
    CORES=$(grep -c "^processor" /proc/cpuinfo)
    echo "✅ CPU cores: $CORES"
    
    if grep -q "neon" /proc/cpuinfo; then
        echo "✅ NEON SIMD: Supported"
    else
        echo "⚠️  NEON SIMD: Not detected"
    fi
    
    if grep -q "vfp" /proc/cpuinfo; then
        echo "✅ VFP floating point: Supported"
    else
        echo "⚠️  VFP floating point: Not detected"
    fi
else
    echo "⚠️  Cannot read CPU info"
fi

# Python availability
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

echo "✅ Python3: $(python3 --version)"

# Check if we can install packages
echo
echo "📦 Setting up ARM-optimized Python environment..."

# Create ARM-optimized virtual environment
if [ ! -d "venv_arm" ]; then
    echo "Creating ARM-optimized virtual environment..."
    
    # Set ARM optimization before creating venv
    export OPENBLAS_CORETYPE=ARMV8
    export OMP_NUM_THREADS=1
    
    python3 -m venv venv_arm
fi

echo "Activating ARM virtual environment..."
source venv_arm/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install ARM-optimized dependencies
echo
echo "🔧 Installing ARM-optimized dependencies..."

# System package installation (if possible)
if command -v apt-get &> /dev/null; then
    echo "Installing system packages for ARM..."
    sudo apt-get update
    sudo apt-get install -y \
        python3-dev \
        build-essential \
        cmake \
        pkg-config \
        libjpeg-dev \
        libpng-dev \
        libatlas-base-dev \
        libopenblas-dev \
        liblapack-dev
fi

# Install Python packages with ARM compatibility
echo "Installing NumPy (ARM compatible)..."
pip uninstall -y numpy || true
pip install numpy==1.24.3

echo "Installing OpenCV (ARM optimized)..."
pip uninstall -y opencv-python opencv-contrib-python || true
pip install opencv-python-headless==4.7.1.72

echo "Installing PyTorch (ARM compatible)..."
pip uninstall -y torch torchvision torchaudio || true

# Try multiple PyTorch installation methods
PYTORCH_INSTALLED=false

# Method 1: Latest CPU build
echo "Trying PyTorch CPU build..."
if pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
    PYTORCH_INSTALLED=true
    echo "✅ PyTorch CPU build installed"
else
    # Method 2: Specific ARM-compatible version
    echo "Trying fallback PyTorch version..."
    if pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cpu; then
        PYTORCH_INSTALLED=true
        echo "✅ Fallback PyTorch installed"
    else
        echo "⚠️  PyTorch installation failed - will use fallbacks"
    fi
fi

# Install other libraries
echo "Installing additional libraries..."
pip install Pillow
pip install pyserial

# Install Ultralytics if PyTorch worked
if [ "$PYTORCH_INSTALLED" = true ]; then
    echo "Installing Ultralytics..."
    if pip install ultralytics; then
        echo "✅ Ultralytics installed"
    else
        echo "⚠️  Ultralytics installation failed"
    fi
else
    echo "⚠️  Skipping Ultralytics (PyTorch not available)"
fi

# Install fallback AI libraries
echo "Installing fallback AI libraries..."
pip install onnxruntime
pip install tflite-runtime

# Test installations
echo
echo "🧪 Testing ARM-optimized installations..."

test_result() {
    local lib_name="$1"
    local test_code="$2"
    
    echo "Testing $lib_name..."
    if python3 -c "$test_code" 2>/dev/null; then
        echo "✅ $lib_name: Working"
        return 0
    else
        echo "❌ $lib_name: Failed"
        return 1
    fi
}

# Test each library
NUMPY_OK=false
OPENCV_OK=false
PYTORCH_OK=false
ULTRALYTICS_OK=false

if test_result "NumPy" "import numpy as np; print('NumPy OK')"; then
    NUMPY_OK=true
fi

if test_result "OpenCV" "import cv2; import numpy as np; img = np.zeros((10, 10, 3), dtype=np.uint8); cv2.resize(img, (5, 5)); print('OpenCV OK')"; then
    OPENCV_OK=true
fi

if test_result "PyTorch" "import torch; x = torch.rand(5, 5); print('PyTorch OK')"; then
    PYTORCH_OK=true
fi

if test_result "Ultralytics" "from ultralytics import YOLO; print('Ultralytics OK')" 2>/dev/null; then
    ULTRALYTICS_OK=true
fi

# Download YOLO26n model if needed
echo
echo "🤖 Setting up AI models..."
mkdir -p models

if [ ! -f "models/yolo26n.pt" ]; then
    echo "Downloading YOLO26n model..."
    if command -v wget &> /dev/null; then
        wget -q -O models/yolo26n.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        echo "✅ Model downloaded"
    elif command -v curl &> /dev/null; then
        curl -s -L -o models/yolo26n.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        echo "✅ Model downloaded"
    else
        echo "❌ Cannot download model (no wget/curl)"
    fi
else
    echo "✅ YOLO26n model already exists"
fi

# Create configuration file
echo
echo "⚙️  Creating ARM-optimized configuration..."

cat > arm_config.json << 'EOF'
{
  "arm_optimized": true,
  "camera": {
    "resolution": [320, 240],
    "fps_target": 8,
    "backend": "V4L2",
    "buffer_size": 1
  },
  "yolo": {
    "confidence_threshold": 0.6,
    "max_detections": 3,
    "imgsz": [320, 240],
    "device": "cpu"
  },
  "performance": {
    "max_threads": 2,
    "omp_num_threads": 1,
    "openblas_coretype": "ARMV8"
  }
}
EOF

# Create test launchers
echo
echo "🚀 Creating test launchers..."

# Diagnostic launcher
cat > run_arm_diagnostic.sh << 'EOF'
#!/bin/bash
set -e
echo "🔍 Running ARM Compatibility Diagnostic..."
cd "$(dirname "$0")"
source venv_arm/bin/activate
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=1
python3 arm_diagnostic.py
EOF

chmod +x run_arm_diagnostic.sh

# Fallback test launcher
cat > run_fallback_test.sh << 'EOF'
#!/bin/bash
set -e
echo "🔄 Running ARM-Fallback Camera + AI Test..."
cd "$(dirname "$0")"
source venv_arm/bin/activate
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=1
python3 arm_fallback_ai_pipeline.py
EOF

chmod +x run_fallback_test.sh

# Optimized test launcher
cat > run_optimized_test.sh << 'EOF'
#!/bin/bash
set -e
echo "⚡ Running ARM-Optimized Camera + AI Test..."
cd "$(dirname "$0")"
source venv_arm/bin/activate
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=1
python3 arm_optimized_camera_ai_test.py
EOF

chmod +x run_optimized_test.sh

# Main launcher (tries best option first)
cat > run_arm_camera_ai_test.sh << 'EOF'
#!/bin/bash
set -e
echo "🚀 Arduino UNO Q4GB ARM-Fixed Camera + AI Test"
echo "============================================="
echo

cd "$(dirname "$0")"

# Activate ARM-optimized environment
source venv_arm/bin/activate

# Set ARM optimization variables
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "🔧 ARM optimizations enabled"
echo "📊 Test options:"
echo "  1. ARM Compatibility Diagnostic"
echo "  2. ARM-Fallback Camera + AI Test"
echo "  3. ARM-Optimized Camera + AI Test"
echo
echo "Choose test (1-3) or press Enter for recommended test:"

read -r choice
echo

case $choice in
    1)
        echo "🔍 Running ARM Compatibility Diagnostic..."
        python3 arm_diagnostic.py
        ;;
    2)
        echo "🔄 Running ARM-Fallback Test..."
        python3 arm_fallback_ai_pipeline.py
        ;;
    3)
        echo "⚡ Running ARM-Optimized Test..."
        python3 arm_optimized_camera_ai_test.py
        ;;
    *)
        echo "🎯 Running recommended ARM-Optimized Test..."
        python3 arm_optimized_camera_ai_test.py
        ;;
esac
EOF

chmod +x run_arm_camera_ai_test.sh

# Final summary
echo
echo "🎉 ARM-Fixed Deployment Complete!"
echo "================================="
echo
echo "📁 Installation directory: $INSTALL_DIR"
echo "🐍 ARM virtual environment: venv_arm"
echo "⚙️  Configuration: arm_config.json"
echo
echo "🚀 Test Options:"
echo "  ./run_arm_diagnostic.sh     - ARM compatibility diagnostic"
echo "  ./run_fallback_test.sh      - Camera + AI with fallbacks"
echo "  ./run_optimized_test.sh     - ARM-optimized full test"
echo "  ./run_arm_camera_ai_test.sh  - Interactive menu (recommended)"
echo
echo "📊 Installation Status:"
echo "  NumPy: $([ "$NUMPY_OK" = true ] && echo '✅ Working' || echo '❌ Failed')"
echo "  OpenCV: $([ "$OPENCV_OK" = true ] && echo '✅ Working' || echo '❌ Failed')"
echo "  PyTorch: $([ "$PYTORCH_OK" = true ] && echo '✅ Working' || echo '❌ Failed')"
echo "  Ultralytics: $([ "$ULTRALYTICS_OK" = true ] && echo '✅ Working' || echo '❌ Failed')"
echo
if [ "$PYTORCH_OK" = true ] && [ "$OPENCV_OK" = true ] && [ "$NUMPY_OK" = true ]; then
    echo "✅ Core libraries working - try optimized test first"
elif [ "$OPENCV_OK" = true ] && [ "$NUMPY_OK" = true ]; then
    echo "⚠️  Some libraries failed - try fallback test"
else
    echo "❌ Critical libraries failed - run diagnostic first"
fi
echo
echo "🔧 If you still get illegal instruction errors:"
echo "   1. Run: ./run_arm_diagnostic.sh"
echo "   2. Check CPU: cat /proc/cpuinfo"
echo "   3. Report errors with CPU model and library versions"
echo
echo "✅ Ready for ARM-optimized Camera + AI testing!"