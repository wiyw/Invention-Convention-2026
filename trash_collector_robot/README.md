# Trash Collector Robot

Autonomous trash collector robot for Arduino Uno Q with YOLO computer vision, servo wheel control, and ultrasound sensors.

## Features

- **Computer Vision**: YOLO object detection for identifying concrete paths and trash items
- **Servo Control**: Two-wheel drive with precise servo control for navigation
- **Ultrasound Sensors**: Three sensors for obstacle detection and distance measurement
- **Autonomous Navigation**: AI-driven decision making for path following and trash collection
- **Real-time Processing**: Live camera feed with detection overlay

## Hardware Requirements

### Arduino Uno Q Board
- Qualcomm QRB2210 microcomputer (Debian Linux)
- STM32U585 microcontroller

### Components
- 2x Servo motors (for wheel control)
- 3x Ultrasound sensors (HC-SR04)
- 1x USB webcam
- Servo wheels and chassis
- Power supply

### Wiring
```
Servo Left    -> Pin 9
Servo Right   -> Pin 10
Ultrasound Front Trig -> Pin 2
Ultrasound Front Echo -> Pin 3
Ultrasound Left Trig  -> Pin 4
Ultrasound Left Echo  -> Pin 5
Ultrasound Right Trig -> Pin 6
Ultrasound Right Echo -> Pin 7
```

## Software Architecture

### Dual Processor Design
- **Python Side** (Microcomputer): Computer vision, AI processing, decision making
- **C++ Side** (Microcontroller): Real-time servo control, sensor reading

### Bridge Communication
The Arduino Router Bridge enables communication between Python and C++ sides:
- Python calls C++ functions to read sensors and control servos
- C++ provides real-time hardware control

## Installation

### Option 1: Arduino App Labs (Recommended)

1. Install Arduino App Lab from Arduino's website
2. Connect Arduino Uno Q via USB-C or network
3. Copy the `trash_collector_robot` folder to `/home/arduino/arduino_apps/`
4. Open App Lab and select the "trash-collector-robot" app
5. Click "Run" to deploy and start

### Option 2: Standalone Deployment

1. Copy `trash_collector_standalone` folder to Arduino Uno Q
2. Install dependencies: `pip install -r requirements.txt`
3. Upload `arduino_sketch.ino` using Arduino IDE
4. Run: `python main.py`

## Dependencies

### Python Packages
```bash
pip install opencv-python numpy ultralytics pyserial pillow
```

### Arduino Libraries
- Servo (built-in)
- Arduino_RouterBridge (included with App Lab)

## Usage

### Starting the Robot
```bash
# App Labs - Click Run in App Lab interface
# Standalone - Run from command line
python main.py
```

### Controls
- **Automatic Mode**: Robot operates autonomously
- **Manual Override**: Press 'q' in vision window to stop
- **Emergency Stop**: Ctrl+C in terminal

### Robot Modes
1. **Search**: Spiral search pattern for concrete and trash
2. **Approach**: Navigate toward detected trash
3. **Collect**: Stop and collect trash (simulated)
4. **Avoid**: Obstacle avoidance using ultrasound sensors

## AI Model Configuration

### YOLO Detection Classes
- **Concrete**: road, street, pavement
- **Trash**: bottle, can, cup, plastic

### Model Options
- **Default**: YOLOv8n (nano model for speed)
- **Fallback**: OpenCV DNN (if ultralytics unavailable)
- **Simulation**: Random detections for testing

## Configuration

### Python Settings
```python
config = {
    'camera_index': 0,           # USB webcam device
    'yolo_confidence': 0.5,      # Detection threshold
    'servo_min': 60,            # Minimum servo angle
    'servo_max': 120,           # Maximum servo angle
    'safe_distance': 30,        # Obstacle avoidance distance (cm)
    'collect_distance': 10,     # Trash collection distance (cm)
    'frame_interval': 0.1,      # Vision processing rate (seconds)
    'sensor_interval': 0.05      # Sensor reading rate (seconds)
}
```

### Arduino Settings
- Baud rate: 115200
- Servo range: 60-120 degrees
- Ultrasound range: 2-400 cm

## Troubleshooting

### Common Issues

**Camera not detected**
- Check USB webcam connection
- Verify camera index in config
- Try different USB port

**YOLO model fails to load**
- Install ultralytics: `pip install ultralytics`
- Check internet connection for model download
- Use fallback OpenCV DNN mode

**Arduino communication error**
- Verify USB connection to Arduino
- Check correct serial port
- Ensure Arduino sketch is uploaded

**Servo not responding**
- Check servo wiring to pins 9 and 10
- Verify power supply (servos need external power)
- Check servo calibration

### Debug Mode
Enable verbose output by setting debug flag in main.py:
```python
DEBUG = True
```

## Development

### Modifying Behavior
- **AI Logic**: Edit `decide_action()` in main.py
- **Sensor Processing**: Modify `read_sensors()` and `process_frame()`
- **Servo Control**: Update servo functions in sketch.ino
- **Detection Classes**: Modify YOLO class mapping

### Adding Features
- New sensors: Add to Arduino sketch and bridge functions
- Additional AI models: Integrate in `process_frame()`
- Custom behaviors: Implement in `decide_action()`

## Performance

### Specifications
- **Vision Processing**: ~10 FPS (YOLOv8n)
- **Sensor Reading**: 20 Hz
- **Servo Response**: Real-time
- **Battery Life**: ~2 hours (depends on hardware)

### Optimization Tips
- Use YOLOv8n for faster processing
- Reduce frame resolution if needed
- Adjust detection confidence threshold
- Optimize servo update frequency

## Safety

### Precautions
- Ensure clear operating area
- Monitor for overheating
- Check battery levels
- Emergency stop accessible

### Limitations
- Not for outdoor use (weather protection needed)
- Limited to flat surfaces
- Requires adequate lighting for vision
- Maximum speed: 1 m/s

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Check troubleshooting section
- Review Arduino App Labs documentation
- Test with simulation mode first

---

**Invention Convention 2026**  
*Autonomous Robotics & Computer Vision*