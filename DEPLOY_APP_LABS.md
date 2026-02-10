# Ultrasonic Trash Collector - Arduino App Labs

## Quick Deployment Guide

### 1. Transfer to Arduino UNO Q
```bash
# Method A: Direct copy to App Labs directory
scp -r app_labs/* arduino@<arduino-ip>:/home/arduino/arduino_apps/ultrasonic-trash-collector/

# Method B: Upload via Arduino App Labs web interface
# Upload app_labs/ directory contents
```

### 2. Setup on Arduino UNO Q
```bash
# SSH into Arduino UNO Q
ssh arduino@<arduino-ip>

# Navigate to app directory
cd /home/arduino/arduino_apps/ultrasonic-trash-collector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r Python/requirements.txt
```

### 3. Run in Arduino App Labs
1. Open Arduino App Labs
2. Select `ultrasonic-trash-collector` app
3. Click "Run"
4. Monitor console output

### 4. Test Commands
```bash
# In App Labs console:
TEST      # Run self-test
STATUS     # Show current status
HELP       # Show commands
```

## Expected Output
```
=== App Labs Ultrasonic Trash Collector ===
5-Sensor Array | Real-time Detection | Arduino Control
Sensors: 5 | Arduino: Auto-detect
Starting App Labs Ultrasonic Trash Collector...
=======================================================
  Platform: Arduino App Labs
  Detection: 5x Ultrasonic Sensors  
  Coverage: 360-degree
  Integration: Arduino UNO R4
=======================================================
Starting App Labs Ultrasonic Trash Collector...
Mode: APP_LABS_ULTRASONIC | Press Ctrl+C to stop
  Frame: 30 | FPS: 29.8 | Objects: 2 | Total: 15
  Objects: 2, Nearest: medium_trash at 18.3cm (-30°)
  Arduino: App Labs Navigation: medium_trash @ 18.3cm, -30°
```

## Files Ready for Deployment

- ✅ `app_labs/app.yaml` - App configuration
- ✅ `app_labs/Python/main.py` - Main application
- ✅ `app_labs/Python/requirements.txt` - Dependencies
- ✅ `app_labs/Sketch/ultrasonic_controller.ino` - Arduino code
- ✅ `app_labs/README.md` - Complete documentation

Deploy `app_labs/` directory to Arduino App Labs and start your ultrasonic trash collector!