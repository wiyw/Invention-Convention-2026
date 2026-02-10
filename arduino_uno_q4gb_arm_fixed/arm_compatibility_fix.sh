#!/bin/bash
set -e

# Arduino UNO Q4GB ARM Compatibility Fix Script
# Specifically addresses illegal instruction errors on ARM processors

echo "=============================================="
echo "  Arduino UNO Q4GB ARM Compatibility Fix"
echo "=============================================="
echo

# Check ARM architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

if [[ ! "$ARCH" =~ (arm|aarch64) ]]; then
    echo "⚠️  This script is designed for ARM architectures"
    echo "    Current architecture: $ARCH"
    echo "    Proceeding anyway for testing..."
fi

# Set environment variables for ARM optimization
echo "🔧 Setting ARM optimization environment..."
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Create or activate virtual environment
echo
echo "🐍 Setting up Python virtual environment..."

if [ ! -d "venv_arm" ]; then
    echo "Creating ARM-optimized virtual environment..."
    python3 -m venv venv_arm
fi

echo "Activating virtual environment..."
source venv_arm/bin/activate

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install ARM-compatible system packages
echo
echo "📦 Installing system dependencies for ARM..."

if command -v apt-get &> /dev/null; then
    echo "Installing ARM-optimized packages..."
    sudo apt-get update
    sudo apt-get install -y \
        python3-dev \
        build-essential \
        cmake \
        pkg-config \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libgtk-3-dev \
        libatlas-base-dev \
        gfortran \
        libopenblas-dev \
        liblapack-dev
fi

# Remove incompatible packages first
echo
echo "🗑️  Removing potentially incompatible packages..."
pip uninstall -y torch torchvision torchaudio opencv-python opencv-contrib-python || true

# Install ARM-compatible NumPy and SciPy first
echo
echo "🔢 Installing ARM-compatible NumPy and SciPy..."
pip install numpy==1.24.3
pip install scipy

# Install ARM-compatible PyTorch
echo
echo "🔥 Installing ARM-compatible PyTorch..."

# Try different PyTorch installation methods
PYTORCH_INSTALLED=false

# Method 1: CPU-only build (most compatible)
echo "Trying CPU-only PyTorch build..."
if pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
    echo "✅ CPU-only PyTorch installed successfully"
    PYTORCH_INSTALLED=true
else
    echo "❌ CPU-only PyTorch installation failed"
fi

# Method 2: ARM-specific build if CPU build fails
if [ "$PYTORCH_INSTALLED" = false ]; then
    echo "Trying ARM-specific PyTorch build..."
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu || \
    pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu \
        -f https://download.pytorch.org/whl/torch_stable.html || \
    {
        echo "⚠️  All PyTorch installation methods failed"
        echo "    Installing minimal CPU-optimized alternative..."
        pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1+cpu \
            -f https://download.pytorch.org/whl/torch_stable.html || \
        {
            echo "❌ PyTorch installation completely failed"
            echo "    Consider using ONNX Runtime as alternative"
        }
    }
fi

# Install ARM-compatible OpenCV
echo
echo "📷 Installing ARM-compatible OpenCV..."

# Remove any existing OpenCV installations
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true

# Install headless OpenCV (more compatible)
echo "Installing OpenCV headless version..."
pip install opencv-python-headless==4.7.1.72 || \
pip install opencv-python-headless || \
pip install opencv-python==4.7.1.72 || \
{
    echo "⚠️  OpenCV installation failed, trying alternative..."
    pip install opencv-contrib-python-headless || \
    pip install opencv-contrib-python
}

# Install other ARM-compatible libraries
echo
echo "📚 Installing other ARM-compatible libraries..."

# PIL/Pillow
pip install Pillow==9.5.0

# PySerial
pip install pyserial

# Try to install Ultralytics (requires working PyTorch)
echo
echo "🤖 Attempting to install Ultralytics..."
if python3 -c "import torch" 2>/dev/null; then
    pip install ultralytics
    echo "✅ Ultralytics installed successfully"
else
    echo "⚠️  PyTorch not working, skipping Ultralytics installation"
    echo "    Install Ultralytics after fixing PyTorch"
fi

# Alternative AI libraries as fallback
echo
echo "🔄 Installing fallback AI libraries..."
pip install onnxruntime  # ONNX Runtime for ARM
pip install tflite-runtime  # TensorFlow Lite for ARM

# Test installations
echo
echo "🧪 Testing ARM-compatible installations..."
echo

# Test Python
echo "Testing Python..."
python3 --version

# Test NumPy
echo "Testing NumPy..."
python3 -c "
import numpy as np
print(f'NumPy {np.__version__}')
try:
    x = np.array([1, 2, 3])
    y = np.sum(x)
    print('NumPy test: ✅ PASS')
except Exception as e:
    print(f'NumPy test: ❌ FAIL - {e}')
"

# Test PyTorch
echo "Testing PyTorch..."
python3 -c "
try:
    import torch
    print(f'PyTorch {torch.__version__}')
    x = torch.rand(10, 10)
    y = torch.sum(x)
    print('PyTorch test: ✅ PASS')
except Exception as e:
    print(f'PyTorch test: ❌ FAIL - {e}')
"

# Test OpenCV
echo "Testing OpenCV..."
python3 -c "
try:
    import cv2
    print(f'OpenCV {cv2.__version__}')
    import numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    resized = cv2.resize(img, (50, 50))
    print('OpenCV test: ✅ PASS')
except Exception as e:
    print(f'OpenCV test: ❌ FAIL - {e}')
"

# Test Ultralytics
echo "Testing Ultralytics..."
python3 -c "
try:
    from ultralytics import YOLO
    print(f'Ultralytics {YOLO.__version__}')
    print('Ultralytics test: ✅ PASS')
except Exception as e:
    print(f'Ultralytics test: ❌ FAIL - {e}')
"

# Create ARM-optimized startup script
echo
echo "📝 Creating ARM-optimized startup script..."

cat > start_arm_optimized.sh << 'EOF'
#!/bin/bash
set -e

# Arduino UNO Q4GB ARM-Optimized Startup Script
echo "🚀 Starting ARM-optimized Camera + AI Pipeline..."

# Set ARM optimization variables
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Activate virtual environment
source "$(dirname "$0")/venv_arm/bin/activate"

# Set Python path
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

# Run the main application
cd "$(dirname "$0")"

echo "✅ ARM optimization enabled"
echo "🧪 Starting Camera + AI test..."

# Run the main test
python3 arduino_uno_q4gb_camera_ai_pipeline_test.py
EOF

chmod +x start_arm_optimized.sh

# Summary
echo
echo "🎉 ARM Compatibility Fix Complete!"
echo "================================="
echo
echo "✅ ARM-optimized virtual environment: venv_arm"
echo "✅ ARM-compatible libraries installed"
echo "✅ Startup script created: start_arm_optimized.sh"
echo
echo "🚀 To run the Camera + AI test:"
echo "   ./start_arm_optimized.sh"
echo
echo "🔧 To manually activate environment:"
echo "   source venv_arm/bin/activate"
echo
echo "⚠️  If you still get illegal instruction errors:"
echo "   1. Check CPU: cat /proc/cpuinfo"
echo "   2. Test individual: python3 -c 'import torch; print(torch.rand(5,5))'"
echo "   3. Consider alternative AI frameworks (ONNX Runtime, TensorFlow Lite)"
echo
echo "📊 For troubleshooting:"
echo "   - Run diagnostic: python3 arm_diagnostic.py"
echo "   - Check logs for specific error locations"
echo "   - Report CPU model and error messages"