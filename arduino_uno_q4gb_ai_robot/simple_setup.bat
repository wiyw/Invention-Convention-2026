@echo off
echo ========================================
echo Arduino UNO Q4GB AI - SIMPLE SETUP
echo ========================================
echo.

echo Installing required packages...
py -m pip install --upgrade pip
py -m pip install opencv-python ultralytics numpy

echo.
echo Creating simple camera test...
echo.

REM Create simple working script
echo @echo off > simple_camera_test.bat
echo cd python_tools\testing >> simple_camera_test.bat
echo python simple_camera_test.py --test >> simple_camera_test.bat
echo pause >> simple_camera_test.bat

echo.
echo Running simple camera test...
echo.

cd python_tools\testing
python simple_camera_test.py --test

echo.
echo Setup complete!
echo.
echo ========================================
echo SIMPLE SETUP COMPLETE!
echo.
echo Quick Start:
echo 1. Double-click simple_camera_test.bat
echo.
echo Features:
echo - YOLO model auto-detection
echo - Simple camera interface
echo - Basic object detection
echo - Real-time display
echo.
echo If camera fails, the script will show available camera IDs
echo.
pause