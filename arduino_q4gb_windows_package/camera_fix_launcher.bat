@echo off
REM Arduino UNO Q4GB AI Robot - Camera Fix Launcher
REM Fixes Windows camera issues before running main app

echo 📷 Arduino UNO Q4GB - Camera Fix Tool
echo ====================================
echo This tool will test and fix camera issues.
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found - install from https://python.org
    pause
    exit /b 1
)

REM Check dependencies
python -c "import cv2" 2>nul || (
    echo ❌ OpenCV not found - installing...
    python -m pip install opencv-python
)

python -c "import tkinter" 2>nul || (
    echo ❌ Tkinter not available
    pause
    exit /b 1
)

echo ✅ Dependencies checked
echo.
echo 📷 Starting camera diagnostic tool...
echo.
echo 🔍 This will test all camera configurations
echo 📊 Find the best settings for your system
echo 🔧 Provide solutions for camera issues
echo.

python camera_fix.py

echo.
echo 📊 Camera test completed!
echo.
echo 🎯 If you found a working configuration:
echo    1. Note the Camera number and Backend that worked
echo    2. The main app will use that configuration
echo    3. If issues persist, run this tool again
echo.
echo 🚀 You can now try the main application:
echo    windows_webcam_robot.bat
echo.

pause