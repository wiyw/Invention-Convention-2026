# Arduino UNO Q4GB AI Robot - Windows Setup Guide

## 🪟 Windows Dependencies Installation

### Automated Installation (Recommended)
1. **Run the automated installer:**
   ```cmd
   cd windows_setup
   install_dependencies.bat
   ```

2. **Verify installation:**
   ```cmd
   test_setup.bat
   ```

### Manual Installation

#### 1. Python Environment
```cmd
# Download Python 3.10
# Visit: https://www.python.org/downloads/release/python-31011/
# Download: python-3.10.11-amd64.exe

# During installation:
# ☑️ Add Python to PATH
# ☑️ Install for all users
```

#### 2. Required Python Packages
```cmd
py -m pip install opencv-python ultralytics pyserial numpy matplotlib
py -m pip install torch torchvision tensorflow
py -m pip install pillow scipy scikit-learn tqdm
py -m pip install pyyaml requests psutil
```

#### 3. Arduino IDE
```cmd
# Download Arduino IDE 2.0+
# Visit: https://www.arduino.cc/en/software
# Download: arduino-ide_2.0.4_Windows_64bit.exe

# Install with default settings
```

#### 4. Hardware Drivers
```cmd
# Arduino UNO Q4GB drivers (automatically installed with IDE)
# USB to UART drivers:
# - CP210x: https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers
# - CH340: http://www.wch.cn/downloads/CH341SER_ZIP.html
```

## 🧪 Testing Setup

### Quick Test (No Hardware Required)
```cmd
# Test simulation mode
cd python_tools\testing
python ai_test_suite_windows.py --simulate --quick

# Check dependencies
python ai_test_suite_windows.py --check-deps
```

### Full Test Suite
```cmd
# Complete simulation testing
python ai_test_suite_windows.py --simulate

# This tests:
# - Detection accuracy (simulated)
# - Response time performance
# - Memory usage simulation
# - Safety system reliability
# - Decision consistency
```

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10 (64-bit) or higher
- **RAM**: 4GB (8GB+ recommended for simulation)
- **Storage**: 2GB free space
- **USB**: Available USB 2.0+ port

### Recommended Specifications
- **OS**: Windows 11 (64-bit)
- **RAM**: 16GB
- **CPU**: Intel i5+ or AMD Ryzen 5+
- **GPU**: NVIDIA GPU with CUDA support (optional, faster ML)
- **Storage**: 5GB free space

## 🔧 Configuration

### Arduino IDE Setup
1. **Open Arduino IDE 2.0+**
2. **Install Board Package:**
   ```
   Tools → Board → Boards Manager
   Search: "Arduino UNO Q4GB"
   Install: Latest version
   ```
3. **Select Board:**
   ```
   Tools → Board → Arduino UNO Q4GB
   ```
4. **Configure Port:**
   ```
   Tools → Port → COMx (Arduino UNO Q4GB)
   ```

### Python Virtual Environment
```cmd
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate

# Install requirements
py -m pip install -r requirements.txt
```

## 🎯 Getting Started

### 1. Simulation Testing (No Hardware)
```cmd
# Navigate to project directory
cd arduino_uno_q4gb_ai_robot

# Run simulation tests
python python_tools\testing\ai_test_suite_windows.py --simulate

# This simulates complete AI robot operation:
# - Camera capture simulation
# - TinyYOLO object detection
# - TinyQwen decision making
# - Sensor fusion
# - Safety system testing
```

### 2. Model Conversion
```cmd
# Convert full YOLO26n to TinyML format
python python_tools\model_conversion\tinyml_converter.py

# Generate Arduino-compatible weights
python python_tools\model_conversion\generate_weights.py
```

### 3. Arduino Firmware Testing
```cmd
# Open Arduino IDE
# Load: arduino_firmware\core\ai_robot_controller.ino

# Test compilation (no upload needed)
# Click: Verify button

# This tests:
# - Syntax correctness
# - Memory usage estimation
# - Library compatibility
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Python Installation
```
Error: 'python' is not recognized
Solution: 
1. Reinstall Python with "Add to PATH" checked
2. Or add to PATH manually: C:\Python310\
```

#### 2. Package Installation
```
Error: "Microsoft Visual C++ 14.0 required"
Solution: Install Microsoft C++ Build Tools
Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### 3. Arduino IDE Issues
```
Error: "Board not found"
Solution:
1. Install board package via Boards Manager
2. Check USB cable connection
3. Install driver: windows_setup\drivers\arduino.inf
```

#### 4. Serial Port Issues
```
Error: "Access denied to COM port"
Solution:
1. Close Arduino IDE Serial Monitor
2. Run as Administrator
3. Check device manager for port conflicts
```

#### 5. OpenCV Installation
```
Error: "numpy version mismatch"
Solution:
py -m pip uninstall opencv-python numpy
py -m pip install numpy==1.24.0
py -m pip install opencv-python
```

### Performance Issues

#### Slow Response Times
```
Symptoms: >200ms response time
Causes:
- Insufficient RAM (close other programs)
- Missing CUDA acceleration (install NVIDIA drivers)
- Large model sizes (use quantized models)

Solutions:
- Increase virtual memory
- Install GPU drivers
- Use --optimize flag
```

#### Memory Issues
```
Symptoms: "Out of memory" errors
Causes:
- Insufficient system RAM
- Memory leaks in simulation
- Too large model files

Solutions:
- Increase page file size
- Restart simulation between tests
- Use reduced model size
```

### Hardware Issues

#### Arduino Not Detected
```
Symptoms: "No Arduino UNO Q4GB found"
Solutions:
1. Check USB cable (try different cable)
2. Try different USB port
3. Install/reinstall drivers
4. Check Device Manager
```

#### Sensor Issues
```
Symptoms: Incorrect sensor readings
Solutions:
1. Check wiring connections
2. Verify power supply (5V)
3. Calibrate sensors
4. Check for interference
```

## 📞 Support Resources

### Documentation
- **Main README**: `../README.md`
- **Hardware Setup**: `../docs/hardware/wiring_guide.md`
- **API Reference**: `../docs/api/arduino_api.md`
- **Troubleshooting**: `../docs/troubleshooting/common_issues.md`

### Online Resources
- **Arduino Documentation**: https://docs.arduino.cc/
- **TinyML Resources**: https://www.tensorflow.org/lite/microcontrollers
- **OpenCV Installation**: https://opencv.org/releases/

### Community Support
- **Arduino Forum**: https://forum.arduino.cc/
- **Stack Overflow**: https://stackoverflow.com/questions/tagged/arduino
- **GitHub Issues**: Project repository issues

## 🚀 Next Steps

### After Setup Completion:
1. **Run Simulation Tests:**
   ```cmd
   python python_tools\testing\ai_test_suite_windows.py --simulate
   ```

2. **Model Conversion:**
   ```cmd
   python python_tools\model_conversion\tinyml_converter.py
   ```

3. **Hardware Testing:**
   ```cmd
   # Connect Arduino UNO Q4GB
   # Load firmware in Arduino IDE
   # Upload to board
   ```

4. **Performance Validation:**
   ```cmd
   python python_tools\interfaces\arduino_monitor.py --port COM3
   ```

This setup provides complete Windows testing capability without requiring actual Arduino hardware, while maintaining full compatibility for real hardware implementation.