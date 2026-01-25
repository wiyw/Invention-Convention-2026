# YOLO26n + Qwen2.5-0.5B-Instruct Robot Implementation

## Overview
This implementation combines YOLO26n object detection, Qwen2.5-0.5B-Instruct reasoning, and ultrasonic sensors for intelligent robot navigation using Arduino Uno Q4GB.

## Hardware Setup

### Arduino Uno Q4GB Connections

#### Ultrasonic Sensors
- **Left 45° Sensor**: Trig Pin D2, Echo Pin D3
- **Right 45° Sensor**: Trig Pin D4, Echo Pin D5  
- **Center Sensor**: Trig Pin D6, Echo Pin D7

#### Servo Motors (Wheels)
- **Left Servo**: Pin D9
- **Right Servo**: Pin D10

#### Camera
- USB camera connected to host computer (not Arduino)

### Sensor Mounting
1. **Camera**: Front-facing, centered
2. **Center Ultrasonic**: Directly under camera
3. **Left Ultrasonic**: 45° angle to left
4. **Right Ultrasonic**: 45° angle to right

## Software Installation

### 1. Python Dependencies
```bash
pip install opencv-python ultralytics pyserial numpy transformers torch
```

### 2. Arduino Setup
1. Open `arduino_controller/robot_controller.ino` in Arduino IDE
2. Upload to Arduino Uno Q4GB
3. Verify serial connection (default COM3, 115200 baud)

### 3. Model Setup
Ensure you have:
- `yolo26n.pt` in the project root
- `yolo26n/decide.py` and `yolo26n/test.py` present

## Running the Robot

### Option 1: Basic Bridge (Rule-based)
```bash
python robot_bridge.py --port COM3
```

### Option 2: Enhanced Bridge (Qwen + Sensor Fusion)
```bash
python enhanced_robot_bridge.py --port COM3
```

### Option 3: Disable Qwen (Rule-based only)
```bash
python enhanced_robot_bridge.py --port COM3 --no-qwen
```

## File Structure
```
InventionConvention2026/
├── arduino_controller/
│   └── robot_controller.ino          # Arduino firmware
├── yolo26n/
│   ├── decide.py                     # Rule-based decision logic
│   ├── test.py                       # YOLO detection
│   ├── yolo26n.pt                    # YOLO model
│   └── prompt.txt                     # Qwen prompt template
├── robot_bridge.py                    # Basic bridge
├── enhanced_robot_bridge.py          # Enhanced bridge with Qwen
├── qwen_integration.py               # Qwen decision engine
└── yolo26n.pt                        # YOLO model weights
```

## Communication Protocol

### Arduino to Host
- **Sensor Data**: `SENSORS {"left45":12.3,"right45":15.7,"center":25.1,"timestamp":123456}`
- **Motor Confirmation**: `MOTORS L:150 R:150 T:500`

### Host to Arduino
- **Motor Command**: `CMD L<speed> R<speed> T<duration_ms>`
  - Speed: 0-255 (0=stop, 255=max)
  - Duration: milliseconds

## Sensor Fusion Features

### Enhanced Robot Bridge Includes:
1. **Object Velocity Estimation**: Tracks movement of detected objects
2. **Safety Assessment**: Multi-level risk evaluation (safe/caution/danger/critical)
3. **Historical Context**: Uses recent sensor readings for better decisions
4. **Adaptive Timing**: Faster reaction times when danger detected

### Safety Levels
- **Safe**: No obstacles detected
- **Caution**: Obstacles 20-30cm away
- **Danger**: Approaching objects, warnings present
- **Critical**: Obstacles <15cm, immediate stop required

## Qwen Integration

### Features:
- Natural language reasoning for complex scenarios
- Context-aware decision making
- Fallback to rule-based system if unavailable

### Model Requirements:
- `Qwen/Qwen2.5-0.5B-Instruct` (automatically downloaded)
- ~2GB disk space
- PyTorch with CUDA support recommended

## Troubleshooting

### Common Issues:
1. **Arduino Connection**: Check COM port and baud rate
2. **Camera Access**: Verify camera index (usually 0)
3. **YOLO Model**: Ensure `yolo26n.pt` is in correct location
4. **Memory**: Qwen model may require additional RAM

### Debug Mode:
```bash
python enhanced_robot_bridge.py --test-fusion
```

## Performance Notes

- **Cycle Time**: ~500ms with Qwen, ~200ms rule-based
- **Memory Usage**: ~3GB with Qwen, ~1GB rule-based
- **CPU**: Intel i5+ recommended for smooth operation

## Safety Features

1. **Ultrasonic Override**: Always stops if center obstacle <20cm
2. **Emergency Stop**: Falls back to safe behavior on errors
3. **Confidence Scaling**: Adjusts speed based on detection confidence
4. **Timeout Protection**: Commands expire after specified duration

## Customization

### Modify Speed Limits:
Edit `arduino_controller/robot_controller.ino` servo mapping:
```cpp
int leftServoValue = map(leftSpeed, 0, 255, 0, 180);
```

### Adjust Safety Thresholds:
Modify `enhanced_robot_bridge.py` safety assessment:
```python
if distance < 15:  # Critical threshold
if distance < 30:  # Warning threshold
```

### Change Detection Priority:
Edit `yolo26n/decide.py` priority list:
```python
priority = ['person', 'bicycle', 'car', 'truck']
```