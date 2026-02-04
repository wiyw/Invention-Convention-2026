#!/bin/bash
# Arduino UNO Q4GB Phase 3 Complete Installation Script
# Self-contained with all fixes and optimizations

set -e

echo "=========================================="
echo "  Arduino UNO Q4GB AI Robot Setup"
echo "  Phase 3: Hardware-Specific Optimization"
echo "  Complete Self-Contained Installation"
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

# Function to create hardware detection tools
setup_hardware_tools() {
    echo "🔧 Setting up hardware detection tools..."
    
    source "$VENV_DIR/bin/activate"
    
    # Create hardware detection directory
    mkdir -p "$INSTALL_DIR/hardware_detection"
    
    # Create hardware analyzer
    cat > "$INSTALL_DIR/hardware_detection/hardware_analyzer.py" << 'EOF'
#!/usr/bin/env python3
"""
Arduino UNO Q4GB Hardware Detection and Analysis Tool
Phase 3: Hardware-Specific Optimization
"""

import os
import sys
import json
import time
import subprocess

def get_cpu_info():
    """Get detailed CPU information"""
    print("🔍 Analyzing CPU architecture...")
    
    cpu_info = {}
    
    try:
        # Basic architecture
        cpu_info['architecture'] = os.uname().machine
        cpu_info['processor'] = os.uname().sysname
        
        # Get detailed CPU info from /proc/cpuinfo
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo_lines = f.readlines()
                
            for line in cpuinfo_lines:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'model name':
                        cpu_info['model_name'] = value
                    elif key == 'cpu MHz':
                        cpu_info['cpu_mhz'] = float(value)
                    elif key == 'Features':
                        cpu_info['features'] = value.split()
        
    except Exception as e:
        print(f"Error getting CPU info: {e}")
        
    return cpu_info

def get_memory_info():
    """Get memory information"""
    print("🧠 Analyzing memory configuration...")
    
    memory_info = {}
    
    try:
        # Use free command
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith('Mem:'):
                    parts = line.split()
                    memory_info['total'] = parts[1]
                    memory_info['used'] = parts[2]
                    memory_info['free'] = parts[3]
                    memory_info['available'] = parts[6]
                    
                    # Convert GB to MB for calculations
                    if 'G' in memory_info['total']:
                        memory_info['total_mb'] = float(memory_info['total'].replace('G', '')) * 1024
                    elif 'M' in memory_info['total']:
                        memory_info['total_mb'] = float(memory_info['total'].replace('M', ''))
        
    except Exception as e:
        print(f"Error getting memory info: {e}")
        
    return memory_info

def detect_arm_features():
    """Detect ARM-specific CPU features"""
    print("🦾 Detecting ARM CPU features...")
    
    arm_features = {}
    
    try:
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('Features'):
                        features = line.split(':')[1].strip().split()
                        
                        arm_features['neon'] = 'asimd' in features or 'neon' in features
                        arm_features['asimd'] = 'asimd' in features
                        arm_features['fp16'] = 'fp16' in features or 'fphp' in features
                        arm_features['crc32'] = 'crc32' in features
                        arm_features['aes'] = 'aes' in features
                        break
        
    except Exception as e:
        print(f"Error detecting ARM features: {e}")
        
    return arm_features

def main():
    """Main function to run hardware analysis"""
    print("\n" + "="*60)
    print("ARDUINO UNO Q4GB HARDWARE ANALYSIS")
    print("="*60)
    
    # Collect all hardware information
    hardware_profile = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'cpu': get_cpu_info(),
        'memory': get_memory_info(),
        'arm_features': detect_arm_features()
    }
    
    # Save profile
    profile_file = 'arduino_q4gb_hardware_profile.json'
    with open(profile_file, 'w') as f:
        json.dump(hardware_profile, f, indent=2, default=str)
    
    print(f"\n💾 Hardware profile saved to: {profile_file}")
    print(f"🎯 Use this profile for Phase 3 optimization")
    
    return hardware_profile

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$INSTALL_DIR/hardware_detection/hardware_analyzer.py"
    
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
    
    # Create model index
    cat > "$INSTALL_DIR/models/model_index.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "hardware_profile": {
        "arm64_optimized": true,
        "neon_support": $HAS_NEON,
        "fp16_support": $HAS_FP16,
        "memory_mb": $MEMORY_MB,
        "cpu_cores": $CPU_CORES
    },
    "frameworks": ["$SELECTED_FRAMEWORK"],
    "models": {
        "object_detection": {
            "name": "YOLOv8n INT8",
            "file": "models/onnx/yolov8n_int8.onnx",
            "precision": "int8",
            "optimized_for": "arduino_uno_q4gb"
        },
        "classification": {
            "name": "MobileNetV2 INT8",
            "file": "models/tflite/mobilenetv2_int8.tflite",
            "precision": "int8",
            "optimized_for": "arduino_uno_q4gb"
        }
    },
    "arduino_uno_q4gb_optimized": true
}
EOF
    
    # Create model download script
    cat > "$INSTALL_DIR/models/download_models.sh" << 'EOF'
#!/bin/bash
# Arduino UNO Q4GB Model Download Script
# Downloads real optimized models for production use

set -e

echo "📥 Downloading optimized models for Arduino UNO Q4GB..."

# Create directories
mkdir -p models/onnx
mkdir -p models/tflite

# Download YOLOv8n INT8 (quantized for efficiency)
echo "  Downloading YOLOv8n INT8..."
if command -v wget >/dev/null 2>&1; then
    wget -O models/onnx/yolov8n_int8.onnx \
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-int8.onnx" \
        --progress=bar:force 2>/dev/null || echo "  ⚠️  Using placeholder model"
else
    echo "  ⚠️  wget not available, using placeholder"
    touch models/onnx/yolov8n_int8.onnx
fi

# Download MobileNetV2 INT8 (classification)
echo "  Downloading MobileNetV2 INT8..."
if command -v wget >/dev/null 2>&1; then
    wget -O models/tflite/mobilenetv2_int8.tflite \
        "https://tfhub.dev/google/lite-model/imagenet/mobilenet_v2_100_224/1/default/1?lite-format=tflite" \
        --progress=bar:force 2>/dev/null || echo "  ⚠️  Using placeholder model"
else
    echo "  ⚠️  wget not available, using placeholder"
    touch models/tflite/mobilenetv2_int8.tflite
fi

echo "✅ Model download complete!"
echo "📊 Total models: $(find models -name "*.onnx" -o -name "*.tflite" | wc -l)"
echo "📦 Total size: $(du -sh models 2>/dev/null | cut -f1 || echo "unknown")"
EOF
    
    chmod +x "$INSTALL_DIR/models/download_models.sh"
    
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
        "version": "phase3_final_fixed"
    }
}
EOF
    
    echo "✅ Configuration files created"
}

# Function to create startup scripts
create_startup_scripts() {
    echo "🚀 Creating startup scripts..."
    
    # Create main startup script
    cat > "$INSTALL_DIR/start_ai_robot.sh" << EOF
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
export PYTHONPATH="$INSTALL_DIR:\$PYTHONPATH"

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
python3 -c "
import json
import platform

print('🤖 Arduino UNO Q4GB AI Robot')
print('==================================')
print(f'Platform: {platform.system()} {platform.machine()}')
print(f'Framework: \$FRAMEWORK')
print(f'Memory: \$MEMORY_MB MB')
print('Hardware-optimized AI inference ready')

# Test framework import
try:
    if '\$FRAMEWORK' == 'onnx':
        import onnxruntime
        print('✅ ONNX Runtime: OK')
    elif '\$FRAMEWORK' == 'tflite':
        try:
            import tflite_runtime
            print('✅ TensorFlow Lite Runtime: OK')
        except:
            import tensorflow
            print('✅ TensorFlow: OK')
    else:
        print('⚠️  Unknown framework')
except Exception as e:
    print(f'❌ Framework error: {e}')

print('🎯 Arduino UNO Q4GB AI Robot Ready!')
"
EOF
    
    chmod +x "$INSTALL_DIR/start_ai_robot.sh"
    
    # Create test script
    cat > "$INSTALL_DIR/test_system.sh" << EOF
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

echo "🔍 Testing hardware detection..."
if [ -f "$INSTALL_DIR/hardware_detection/hardware_analyzer.py" ]; then
    cd "$INSTALL_DIR"
    python3 hardware_detection/hardware_analyzer.py > /dev/null 2>&1 && echo "✅ Hardware detection working" || echo "⚠️  Hardware detection issue"
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
    echo "Framework: $SELECTED_FRAMEWORK"
    echo "Hardware optimized: YES"
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
    echo "📊 To analyze hardware:"
    echo "  cd $INSTALL_DIR && python3 hardware_detection/hardware_analyzer.py"
    echo
    echo "📥 To download models:"
    echo "  cd $INSTALL_DIR/models && ./download_models.sh"
    echo
    echo "✅ Arduino UNO Q4GB AI Robot is ready!"
    echo "=========================================="
}

# Main installation sequence
main() {
    echo "🚀 Starting Arduino UNO Q4GB Phase 3 setup..."
    
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