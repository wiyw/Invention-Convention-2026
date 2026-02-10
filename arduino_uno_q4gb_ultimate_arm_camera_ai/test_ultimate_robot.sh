#!/bin/bash
set -e

# Arduino UNO Q4GB Ultimate Robot Test
# Tests motor control, ultrasonic sensors, and robot navigation

echo "🤖 Arduino UNO Q4GB Ultimate Robot Test"
echo "====================================="
echo

# Activate environment if available
if [ -f "venv_ultimate/bin/activate" ]; then
    echo "🐍 Activating ultimate virtual environment..."
    source venv_ultimate/bin/activate
elif [ -f "venv_arm/bin/activate" ]; then
    echo "🐍 Activating ARM virtual environment..."
    source venv_arm/bin/activate
else
    echo "⚠️  No virtual environment found - using system Python"
fi

# Test robot controller
echo "🤖 Testing Ultimate Robot Controller..."
echo

python3 -c "
import sys
import time
from pathlib import Path

# Import the ultimate robot controller
sys.path.append('.')
try:
    from ultimate_robot_controller import UltimateRobotController
    print('✅ Ultimate robot controller imported')
except ImportError as e:
    print(f'❌ Robot controller import failed: {e}')
    sys.exit(1)

# Create robot instance
robot = UltimateRobotController()

# Run comprehensive test
try:
    # Test connection
    if robot.connect_arduino():
        print('✅ Arduino connection test passed')
        
        # Test sensor readings
        if robot.read_sensors():
            print('✅ Sensor reading test passed')
            
            # Test motor control
            if robot.ultrasonic_forward(speed=50):
                time.sleep(2)
                if robot.stop_motors():
                    print('✅ Motor control test passed')
                else:
                    print('❌ Motor control test failed')
            else:
                print('❌ Forward movement test failed')
            
            # Test turning
            if robot.ultrasonic_turn_left(forward_speed=60, turn_speed=80):
                time.sleep(2)
                if robot.stop_motors():
                    print('✅ Left turn test passed')
                else:
                    print('❌ Left turn test failed')
            
            # Test emergency stop
            if robot.emergency_stop():
                print('✅ Emergency stop test passed')
                time.sleep(1)
            else:
                print('❌ Emergency stop test failed')
            
            # Final status
            status = robot.get_robot_status()
            print(f'📊 Final Robot Status: Connected={status[\"connected\"]}')
            print(f'📊 Final Motor State: {status[\"motors\"][\"direction\"]}')
            print(f'📊 Final Sensor Readings: {status[\"sensors\"]}')
            
            print('🎉 Ultimate robot controller test completed successfully!')
        else:
            print('❌ Sensor reading test failed')
    
except KeyboardInterrupt:
    print('\n⏹️  Robot test interrupted by user')
except Exception as e:
    print(f'❌ Robot test crashed: {e}')
finally:
    robot.cleanup()

echo
echo "🎯 Robot Controller Test Complete!"
echo "====================================="
echo

# Test Arduino sketch compilation (if Arduino IDE available)
if command -v arduino-cli &> /dev/null; then
    echo "📦 Testing Arduino sketch compilation..."
    if [ -f "arduino_q4gb_motor_controller.ino" ]; then
        if arduino-cli compile --fqbn arduino:avr:uno arduino_q4gb_motor_controller.ino; then
            echo "✅ Arduino sketch compiles successfully"
        else
            echo "❌ Arduino sketch compilation failed"
    else
        echo "⚠️  Arduino sketch not found"
else
    echo "⚠️  arduino-cli not available - skipping Arduino sketch test"
fi

echo
echo "🎉 Ultimate Robot Controller Test Suite Complete!"
echo "💡 To run robot manually: python3 ultimate_robot_controller.py"
echo "🔧 To upload Arduino sketch: arduino-cli upload --port /dev/ttyACM0 --fqbn arduino:avr:uno arduino_q4gb_motor_controller.ino"
echo