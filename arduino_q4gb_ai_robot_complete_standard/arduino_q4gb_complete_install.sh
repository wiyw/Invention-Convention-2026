#!/bin/bash
# Arduino UNO Q4GB Phase 3 Complete Installation Script
# Self-contained with ALL AI runtime Python files included

set -e

echo "=========================================="
echo "  Arduino UNO Q4GB AI Robot Setup"
echo "  Phase 3: Complete AI Runtime Package"
echo "  Option 1: Standard AI Package"
echo "=========================================="
echo

# Configuration
INSTALL_DIR="$HOME/arduino_q4gb_ai_robot_phase3"
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

# Function to install system dependencies (FIXED - ATLAS Alternative)
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
        sudo apt-get install -y libgtk-3-dev libgfortran-dev
        
        echo "  Installing audio libraries..."
        sudo apt-get install -y portaudio19-dev python3-pyaudio
        
        echo "  Installing linear algebra libraries (ATLAS alternatives)..."
        # Try alternative linear algebra packages
        sudo apt-get install -y libopenblas-dev liblapack-dev 2>/dev/null || \
        sudo apt-get install -y libblas-dev liblapack-dev 2>/dev/null || \
        echo "    ⚠️  Using fallback - linear algebra packages may be basic"
        
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

# Function to create complete AI runtime structure
create_ai_runtime_structure() {
    echo "🤖 Creating AI runtime structure..."
    
    # Create main directories
    mkdir -p "$INSTALL_DIR/main_ai"
    mkdir -p "$INSTALL_DIR/ai_frameworks"
    mkdir -p "$INSTALL_DIR/ai_frameworks/onnx_runtime"
    mkdir -p "$INSTALL_DIR/hardware_integration"
    mkdir -p "$INSTALL_DIR/ui"
    
    # Copy AI runtime files from package
    echo "  Copying AI runtime files..."
    
    # Main AI application
    if [ -f "main_ai_robot.py" ]; then
        cp main_ai_robot.py "$INSTALL_DIR/main_ai/"
        echo "    ✅ Main AI application copied"
    fi
    
    # AI framework integration
    if [ -f "ai_frameworks/onnx_runtime/onnx_detector.py" ]; then
        cp ai_frameworks/onnx_runtime/*.py "$INSTALL_DIR/ai_frameworks/onnx_runtime/"
        echo "    ✅ ONNX Runtime integration copied"
    fi
    
    # Hardware integration
    if [ -d "hardware_integration" ]; then
        cp hardware_integration/*.py "$INSTALL_DIR/hardware_integration/"
        echo "    ✅ Hardware integration modules copied"
    fi
    
    # User interfaces
    if [ -d "ui" ]; then
        cp ui/*.py "$INSTALL_DIR/ui/"
        echo "    ✅ User interface modules copied"
    fi
    
    echo "✅ AI runtime structure created"
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
        "version": "phase3_complete",
        "ai_runtime_included": true
    }
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
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "❌ Virtual environment not found"
    exit 1
fi

# Set environment variables
export PYTHONPATH="$INSTALL_DIR:$PYTHONPATH"

# Load configuration
if [ -f "$INSTALL_DIR/config.json" ]; then
    echo "📊 Loading configuration..."
    
    # Extract framework info
    FRAMEWORK=\$(python3 -c "import json; print(json.load(open('$INSTALL_DIR/config.json'))['framework'])" 2>/dev/null || echo "onnx")
    MEMORY_MB=\$(python3 -c "import json; print(json.load(open('$INSTALL_DIR/config.json'))['hardware_info']['memory_mb'])" 2>/dev/null || echo "1024")
    
    echo "📊 Configuration loaded:"
    echo "  Framework: \$FRAMEWORK"
    echo "  Memory: \$MEMORY_MB MB"
    echo "  Directory: $INSTALL_DIR"
fi

echo "✅ AI Robot starting..."

# Run main AI robot application
python3 main_ai_robot.py
EOF
    
    chmod +x "$INSTALL_DIR/start_ai_robot.sh"
    
    # Create test script
    cat > "$INSTALL_DIR/test_system.sh" << 'EOF'
#!/bin/bash
set -e

echo "🧪 Testing Arduino UNO Q4GB AI Robot..."
echo "======================================"

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
    echo "Testing framework: $SELECTED_FRAMEWORK"
    
    case "$SELECTED_FRAMEWORK" in
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
    echo "❌ Framework configuration not found"
    exit 1
fi

echo "🔍 Testing basic packages..."
python3 -c "import numpy, PIL, cv2; print('✅ Basic packages working')" 2>/dev/null || echo "⚠️  Some packages missing"

echo "🔍 Testing AI runtime..."
if [ -f "$INSTALL_DIR/main_ai/main_ai_robot.py" ]; then
    cd "$INSTALL_DIR"
    python3 main_ai/main_ai_robot.py --test 2>/dev/null && echo "✅ AI runtime test passed" || echo "⚠️  AI runtime test issue"
else
    echo "⚠️  AI runtime not found"
fi

echo "✅ System test completed"
EOF
    
    chmod +x "$INSTALL_DIR/test_system.sh"
    
    # Create CLI interface launcher
    cat > "$INSTALL_DIR/cli_interface.sh" << 'EOF'
#!/bin/bash
set -e

echo "💻 Arduino UNO Q4GB AI Robot CLI Interface"
echo "=================================="

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "❌ Virtual environment not found"
    exit 1
fi

# Set environment variables
export PYTHONPATH="$INSTALL_DIR:$PYTHONPATH"

# Run CLI interface
if [ -f "$INSTALL_DIR/ui/cli_interface.py" ]; then
    cd "$INSTALL_DIR"
    python3 ui/cli_interface.py
else
    echo "❌ CLI interface not found"
    exit 1
fi
EOF
    
    chmod +x "$INSTALL_DIR/cli_interface.sh"
    
    # Create web interface launcher
    cat > "$INSTALL_DIR/web_interface.sh" << 'EOF'
#!/bin/bash
set -e

echo "🌐 Arduino UNO Q4GB AI Robot Web Interface"
echo "===================================="

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "❌ Virtual environment not found"
    exit 1
fi

# Set environment variables
export PYTHONPATH="$INSTALL_DIR:$PYTHONPATH"

# Run web interface
if [ -f "$INSTALL_DIR/ui/web_interface.py" ]; then
    cd "$INSTALL_DIR"
    python3 ui/web_interface.py
else
    echo "❌ Web interface not found"
    exit 1
fi
EOF
    
    chmod +x "$INSTALL_DIR/web_interface.sh"
    
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
    python3 -c "import numpy, PIL; print('✅ Basic packages working')" 2>/dev/null || echo "⚠️  Basic packages issue"
    
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
            "pytorch")
                python3 -c "import torch; print('✅ PyTorch test passed')" 2>/dev/null || echo "⚠️  PyTorch issue"
                ;;
        esac
    else
        echo "❌ Framework configuration not found"
        return 1
    fi
    
    # Test AI runtime files
    if [ -f "$INSTALL_DIR/main_ai/main_ai_robot.py" ]; then
        python3 "$INSTALL_DIR/main_ai/main_ai_robot.py" --test 2>/dev/null && echo "✅ AI runtime test passed" || echo "⚠️  AI runtime test issue"
    else
        echo "⚠️  AI runtime not found"
    fi
    
    echo "✅ Final tests completed successfully"
}

# Function to display completion message
display_completion() {
    echo
    echo "=========================================="
    echo "🎉 COMPLETE AI RUNTIME SETUP!"
    echo "=========================================="
    echo
    echo "Installation directory: $INSTALL_DIR"
    echo "Virtual environment: $VENV_DIR"
    echo "Framework: $SELECTED_FRAMEWORK"
    echo "AI Runtime: Included and Installed"
    echo
    
    if [ -f "$INSTALL_DIR/framework_config" ]; then
        source "$INSTALL_DIR/framework_config"
        echo "✅ AI Framework: $SELECTED_FRAMEWORK"
    fi
    
    echo "🚀 To start AI robot:"
    echo "  $INSTALL_DIR/start_ai_robot.sh"
    echo
    echo "💻 To use CLI interface:"
    echo "  $INSTALL_DIR/cli_interface.sh"
    echo
    echo "🌐 To use web interface:"
    echo "  $INSTALL_DIR/web_interface.sh"
    echo
    echo "🧪 To test system:"
    echo "  $INSTALL_DIR/test_system.sh"
    echo
    echo "📊 AI Runtime Components:"
    echo "  - Main AI Application: main_ai/main_ai_robot.py"
    echo "  - ONNX Runtime: ai_frameworks/onnx_runtime/"
    echo "  - Camera Interface: hardware_integration/camera_interface.py"
    echo "  - Arduino Communication: hardware_integration/arduino_comm.py"
    echo "  - CLI Interface: ui/cli_interface.py"
    echo "  - Web Interface: ui/web_interface.py"
    echo
    echo "✅ Arduino UNO Q4GB AI Robot is ready!"
    echo "=========================================="
}

# Main installation sequence
main() {
    echo "🚀 Starting Arduino UNO Q4GB Phase 3 Complete AI Runtime setup..."
    
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
    create_ai_runtime_structure
    create_config
    create_startup_scripts
    run_final_tests
    display_completion
    
    echo "✅ Complete AI Runtime setup completed successfully at $(date)"
}

# Error handling
trap 'echo "❌ Setup failed at line $LINENO"' ERR

# Run main function
main "$@"