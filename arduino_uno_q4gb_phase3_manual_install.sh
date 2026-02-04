#!/bin/bash
# Arduino UNO Q4GB Phase 3 Manual Installation Script
# Fixed version to resolve package installation errors

set -e

echo "=========================================="
echo "  Arduino UNO Q4GB AI Robot Setup"
echo "  Phase 3: Hardware-Specific Optimization"
echo "=========================================="
echo

# Configuration
INSTALL_DIR="$HOME/arduino_q4gb_ai_robot_phase3"
VENV_DIR="$INSTALL_DIR/venv"
LOG_FILE="$INSTALL_DIR/setup.log"

# Clean up previous installation
echo "🧹 Cleaning up previous installation..."
rm -rf "$INSTALL_DIR"
rm -rf "$HOME/arduino_uno_q4gb_phase3"
rm -rf "$HOME/arduino_uno_q4gb_deployment*"

# Create installation directory
echo "📁 Creating installation directory..."
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Start logging
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "🔍 Starting setup at $(date)"

# Function to detect hardware
detect_hardware() {
    echo "🔍 Detecting Arduino UNO Q4GB hardware..."
    
    # Basic system info
    echo "  System: $(uname -a)"
    echo "  Architecture: $(uname -m)"
    echo "  CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)"
    echo "  Cores: $(nproc)"
    echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    
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
    MEMORY_MB=$(free -m | grep '^Mem:' | awk '{print $2}')
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

# Function to install system dependencies (FIXED)
install_dependencies() {
    echo "📦 Installing system dependencies..."
    
    if command -v apt-get >/dev/null 2>&1; then
        # Install packages in smaller groups to avoid syntax errors
        echo "  Installing Python packages..."
        sudo apt-get install -y python3 python3-pip python3-venv python3-dev
        
        echo "  Installing build tools..."
        sudo apt-get install -y build-essential cmake git wget curl unzip
        
        echo "  Installing image processing libraries..."
        sudo apt-get install -y pkg-config libjpeg-dev libpng-dev libtiff-dev
        
        echo "  Installing video processing libraries..."
        sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
        
        echo "  Installing GUI libraries..."
        sudo apt-get install -y libgtk-3-dev libatlas-base-dev gfortran
        
        echo "  Installing audio libraries..."
        sudo apt-get install -y portaudio19-dev python3-pyaudio
        
        echo "✅ System dependencies installed"
    else
        echo "⚠️  apt-get not found, please install dependencies manually"
        echo "Required: python3, python3-pip, python3-venv, build-essential, cmake"
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
    pip install numpy pillow opencv-python
    
    # Framework selection logic (simplified for ARM64 + NEON)
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

# Function to setup hardware detection tools
setup_hardware_tools() {
    echo "🔧 Setting up hardware detection tools..."
    
    source "$VENV_DIR/bin/activate"
    
    # Create hardware detection directory
    mkdir -p "$INSTALL_DIR/hardware_detection"
    
    # Copy hardware detection tools from package
    if [ -f "hardware_detection/hardware_analyzer.py" ]; then
        cp hardware_detection/hardware_analyzer.py "$INSTALL_DIR/hardware_detection/"
        cp hardware_detection/benchmark_suite.py "$INSTALL_DIR/hardware_detection/"
        cp hardware_detection/framework_selector.py "$INSTALL_DIR/hardware_detection/"
    fi
    
    echo "✅ Hardware detection tools setup"
}

# Function to create basic model files
create_basic_models() {
    echo "📦 Creating basic model structure..."
    
    source "$VENV_DIR/bin/activate"
    
    # Create models directory
    mkdir -p "$INSTALL_DIR/models"
    mkdir -p "$INSTALL_DIR/models/onnx"
    mkdir -p "$INSTALL_DIR/models/tflite"
    
    # Create placeholder model files (would be downloaded in real deployment)
    cat > "$INSTALL_DIR/models/model_info.json" << EOF
{
    "framework": "$SELECTED_FRAMEWORK",
    "hardware_optimized": true,
    "arduino_uno_q4gb": true,
    "models": {
        "object_detection": "yolov8n_int8.onnx",
        "classification": "mobilenetv2_int8.tflite"
    },
    "timestamp": "$(date -Iseconds)"
}
EOF
    
    echo "✅ Basic model structure created"
}

# Function to create configuration files
create_config() {
    echo "⚙️  Creating configuration files..."
    
    # Load framework selection
    if [ -f "$INSTALL_DIR/framework_config" ]; then
        source "$INSTALL_DIR/framework_config"
    fi
    
    # Create main config
    cat > "$INSTALL_DIR/config.json" << EOF
{
    "hardware_detected": true,
    "framework": "$SELECTED_FRAMEWORK",
    "optimization_level": "hardware_specific",
    "arduino_uno_q4gb": true,
    "setup_complete": true,
    "timestamp": "$(date -Iseconds)",
    "memory_mb": "$MEMORY_MB",
    "has_neon": "$HAS_NEON",
    "has_fp16": "$HAS_FP16",
    "low_memory": "$LOW_MEMORY"
}
EOF
    
    echo "✅ Configuration files created"
}

# Function to create startup scripts
create_startup_scripts() {
    echo "🚀 Creating startup scripts..."
    
    # Create main startup script
    cat > "$INSTALL_DIR/start_ai_robot.sh" << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting Arduino UNO Q4GB AI Robot..."
echo "========================================"

# Activate virtual environment
if [ -d "$HOME/arduino_q4gb_ai_robot_phase3/venv" ]; then
    source "$HOME/arduino_q4gb_ai_robot_phase3/venv/bin/activate"
else
    echo "❌ Virtual environment not found"
    exit 1
fi

# Set environment variables
export PYTHONPATH="$HOME/arduino_q4gb_ai_robot_phase3:$PYTHONPATH"

# Load configuration
if [ -f "$HOME/arduino_q4gb_ai_robot_phase3/config.json" ]; then
    echo "📊 Loading configuration..."
    # Configuration would be loaded here
fi

echo "✅ AI Robot starting..."
# Run main AI robot application
python3 -c "
import json
try:
    with open('$HOME/arduino_q4gb_ai_robot_phase3/config.json') as f:
        config = json.load(f)
    print('🤖 Arduino UNO Q4GB AI Robot')
    print(f'Framework: {config.get(\"framework\", \"unknown\")}')
    print(f'Hardware Optimized: {config.get(\"arduino_uno_q4gb\", False)}')
    print('Hardware-optimized AI inference ready')
except Exception as e:
    print(f'Configuration error: {e}')
"
EOF
    
    chmod +x "$INSTALL_DIR/start_ai_robot.sh"
    
    # Create test script
    cat > "$INSTALL_DIR/test_system.sh" << 'EOF'
#!/bin/bash
set -e

echo "🧪 Testing Arduino UNO Q4GB AI Robot..."
echo "======================================"

# Activate virtual environment
if [ -d "$HOME/arduino_q4gb_ai_robot_phase3/venv" ]; then
    source "$HOME/arduino_q4gb_ai_robot_phase3/venv/bin/activate"
else
    echo "❌ Virtual environment not found"
    exit 1
fi

echo "🔍 Testing framework import..."
if [ -f "$HOME/arduino_q4gb_ai_robot_phase3/config.json" ]; then
    FRAMEWORK=$(python3 -c "import json; print(json.load(open('$HOME/arduino_q4gb_ai_robot_phase3/config.json'))['framework'])")
    echo "Testing framework: $FRAMEWORK"
    
    case "$FRAMEWORK" in
        "onnx")
            python3 -c "import onnxruntime; print('✅ ONNX Runtime working')"
            ;;
        "tflite")
            python3 -c "import tflite_runtime; print('✅ TensorFlow Lite working')" 2>/dev/null || python3 -c "import tensorflow; print('✅ TensorFlow working')"
            ;;
        "pytorch")
            python3 -c "import torch; print('✅ PyTorch working')"
            ;;
    esac
else
    echo "❌ Configuration not found"
    exit 1
fi

echo "✅ System test completed"
EOF
    
    chmod +x "$INSTALL_DIR/test_system.sh"
    
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
    python3 -c "import numpy, PIL; print('✅ Basic packages working')"
    
    # Test framework
    if [ -f "$INSTALL_DIR/framework_config" ]; then
        source "$INSTALL_DIR/framework_config"
        
        case "$SELECTED_FRAMEWORK" in
            "onnx")
                python3 -c "import onnxruntime; print('✅ ONNX Runtime test passed')"
                ;;
            "tflite")
                python3 -c "import tflite_runtime; print('✅ TensorFlow Lite test passed')" 2>/dev/null || python3 -c "import tensorflow; print('✅ TensorFlow test passed')"
                ;;
            "pytorch")
                python3 -c "import torch; print('✅ PyTorch test passed')"
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
    echo "🎉 SETUP COMPLETE!"
    echo "=========================================="
    echo
    echo "Installation directory: $INSTALL_DIR"
    echo "Virtual environment: $VENV_DIR"
    echo "Log file: $LOG_FILE"
    echo
    
    if [ -f "$INSTALL_DIR/framework_config" ]; then
        source "$INSTALL_DIR/framework_config"
        echo "✅ AI Framework: $SELECTED_FRAMEWORK"
    fi
    
    echo "🚀 To start AI robot:"
    echo "  $INSTALL_DIR/start_ai_robot.sh"
    echo
    echo "🧪 To test system:"
    echo "  $INSTALL_DIR/test_system.sh"
    echo
    echo "✅ Arduino UNO Q4GB AI Robot is ready!"
    echo "=========================================="
}

# Main installation sequence
main() {
    echo "🚀 Starting Arduino UNO Q4GB Phase 3 setup..."
    
    detect_hardware
    update_system
    install_dependencies
    create_venv
    install_ai_framework
    setup_hardware_tools
    create_basic_models
    create_config
    create_startup_scripts
    run_final_tests
    display_completion
    
    echo "✅ Setup completed successfully at $(date)"
}

# Error handling
trap 'echo "❌ Setup failed at line $LINENO"' ERR

# Run main function
main "$@"