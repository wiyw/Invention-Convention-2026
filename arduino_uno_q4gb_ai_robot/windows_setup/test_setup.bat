@echo off
echo ========================================
echo Arduino UNO Q4GB AI Robot - Quick Test
echo ========================================
echo.

echo Testing Python installation...
C:\Python310\python.exe --version
if errorlevel 1 (
    echo Python not found! Please run install_dependencies.bat first
    pause
    exit /b 1
)

echo Testing required packages...
C:\Python310\python.exe -c "import cv2; print('OpenCV:', cv2.__version__)"
C:\Python310\python.exe -c "import torch; print('PyTorch:', torch.__version__)"
C:\Python310\python.exe -c "import serial; print('PySerial installed')"
C:\Python310\python.exe -c "import numpy; print('NumPy:', numpy.__version__)"

echo.
echo Testing AI model conversion...
cd python_tools\model_conversion
C:\Python310\python.exe tinyml_converter.py --test

echo.
echo Testing Arduino UNO Q4GB detection...
C:\Python310\python.exe -c "
import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
arduino_ports = [port for port in ports if 'Arduino' in port.description]
if arduino_ports:
    print('Arduino found on:', arduino_ports[0].device)
else:
    print('No Arduino UNO Q4GB detected. Check USB connection.')
"

echo.
echo Testing simulation mode...
cd ..\testing
C:\Python310\python.exe ai_test_suite.py --simulate --quick

echo.
echo ========================================
echo Quick Test Complete!
echo ========================================
echo.
echo If all tests passed, you're ready to:
echo 1. Open Arduino IDE
echo 2. Load arduino_firmware\core\ai_robot_controller.ino
echo 3. Upload to Arduino UNO Q4GB
echo.
echo If tests failed, see: docs\troubleshooting\windows_setup.md
echo.
pause