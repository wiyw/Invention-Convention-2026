# Arduino UNO Q4GB AI Robot Project

A comprehensive on-device AI implementation running YOLO26n + Qwen2.5-0.5B-Instruct entirely on Arduino UNO Q4GB with 4GB RAM.

## 📁 Project Structure

```
arduino_uno_q4gb_ai_robot/
├── 📋 README.md                    # This file
├── 📄 LICENSE                      # MIT License
├── 📄 .gitignore                   # Git ignore file
│
├── 🤖 arduino_firmware/            # Arduino sketches and libraries
│   ├── 📁 core/                   # Main Arduino firmware
│   ├── 📁 libraries/               # Custom Arduino libraries
│   ├── 📁 models/                  # AI model files
│   ├── 📁 headers/                 # Header files
│   └── 📁 utils/                  # Utility functions
│
├── 🐍 python_tools/                # Python development tools
│   ├── 📁 model_conversion/        # TinyML model conversion
│   ├── 📁 testing/                # Test suites and validation
│   ├── 📁 simulation/             # Hardware simulation
│   ├── 📁 interfaces/             # Communication interfaces
│   └── 📁 utils/                 # Python utilities
│
├── 🪟 windows_setup/              # Windows dependencies and setup
│   ├── 📁 drivers/                # Hardware drivers
│   ├── 📁 software/               # Required software installers
│   └── 📁 installation_scripts/   # Automated setup scripts
│
├── 📚 docs/                      # Comprehensive documentation
│   ├── 📁 api/                    # API documentation
│   ├── 📁 hardware/               # Hardware guides
│   ├── 📁 tutorials/              # Step-by-step tutorials
│   └── 📁 troubleshooting/        # Common issues and solutions
│
├── 🧪 tests/                     # Automated testing
│   ├── 📁 unit_tests/             # Unit test suites
│   ├── 📁 integration_tests/      # Integration tests
│   └── 📁 performance_tests/       # Performance benchmarks
│
├── 🎯 examples/                  # Example projects and code
│   ├── 📁 basic_navigation/        # Simple navigation example
│   ├── 📁 object_tracking/        # Object tracking demo
│   └── 📁 safety_demo/           # Safety system demonstration
│
└── 📦 requirements/              # Dependency management
    ├── 📄 requirements.txt       # Python packages
    ├── 📄 requirements-dev.txt   # Development dependencies
    └── 📁 platformio/            # PlatformIO configuration
```

## 🚀 Quick Start

### For Windows Testing (No Hardware Required)
1. Navigate to `windows_setup/`
2. Run `install_dependencies.bat`
3. Use `python_tools/testing/` for simulation testing

### For Arduino Implementation
1. Install Arduino IDE from `windows_setup/software/`
2. Convert AI models using `python_tools/model_conversion/`
3. Upload firmware from `arduino_firmware/core/`

## 📋 Requirements

### Windows System Requirements
- **OS**: Windows 10/11 (64-bit)
- **RAM**: 8GB+ recommended for simulation
- **Storage**: 2GB free space
- **Python**: 3.9+ (auto-installed)

### Hardware Requirements (for actual implementation)
- Arduino UNO Q4GB
- USB camera module
- 3x HC-SR04 ultrasonic sensors
- 2x servo motors
- Motor driver and power supply

## 🛠️ Development Workflow

### 1. Setup Environment
```bash
# Install all dependencies
cd windows_setup/
install_dependencies.bat

# Or manual installation
pip install -r requirements.txt
```

### 2. Convert AI Models
```bash
cd python_tools/model_conversion/
python convert_tinyml.py
python generate_weights.py
```

### 3. Test in Simulation
```bash
cd python_tools/testing/
python test_suite.py --simulate
```

### 4. Upload to Arduino
```bash
# Open Arduino IDE
# Load arduino_firmware/core/ai_robot_controller.ino
# Upload to Arduino UNO Q4GB
```

### 5. Validate Performance
```bash
cd python_tools/interfaces/
python arduino_monitor.py --port COM3
```

## 🔧 Key Features

- **On-Device AI**: Complete AI inference on Arduino UNO Q4GB
- **TinyML Models**: Optimized INT8 quantized models
- **Real-Time Control**: 10Hz decision cycle with <100ms latency
- **Safety First**: Multi-layer protection and obstacle avoidance
- **Memory Optimized**: 512KB total AI memory usage
- **Windows Compatible**: Full testing and development on Windows

## 📖 Documentation

- **Getting Started**: `docs/tutorials/getting_started.md`
- **Hardware Setup**: `docs/hardware/wiring_guide.md`
- **API Reference**: `docs/api/arduino_api.md`
- **Troubleshooting**: `docs/troubleshooting/common_issues.md`

## 🧪 Testing

Run comprehensive test suite:
```bash
cd tests/
python run_all_tests.py
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

MIT License - see `LICENSE` file for details.

## 📞 Support

For questions and issues:
- Check `docs/troubleshooting/` first
- Open issue on GitHub repository
- Join Discord community (link in docs)