@echo off
REM Arduino UNO Q4GB AI Robot - Windows Quick Test
REM Quick dependency check and start

echo 🧪 Arduino UNO Q4GB AI Robot - Quick System Check
echo =================================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found - install from https://python.org
    pause
    exit /b 1
)
echo ✅ Python: 
python --version

REM Check core packages
echo 📦 Checking dependencies...

python -c "import cv2; print('✅ OpenCV:', cv2.__version__)" 2>nul || (
    echo ⚠️  OpenCV not found - will install
    python -m pip install opencv-python
)

python -c "import numpy; print('✅ NumPy:', numpy.__version__)" 2>nul || (
    echo ⚠️  NumPy not found - will install  
    python -m pip install numpy
)

python -c "import PIL; print('✅ PIL: OK')" 2>nul || (
    echo ⚠️  PIL not found - will install
    python -m pip install pillow
)

python -c "import tkinter; print('✅ Tkinter: OK')" 2>nul || (
    echo ⚠️  Tkinter not available - GUI may not work
)

echo.
echo 🚀 Starting AI Robot Test...
echo.

python windows_webcam_robot.py

pause