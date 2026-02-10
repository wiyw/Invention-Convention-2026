@echo off
REM Arduino UNO Q4GB AI Robot - Windows Launcher
REM Start the webcam AI robot on Windows

echo 🚀 Starting Arduino UNO Q4GB AI Robot - Windows Webcam Version
echo =============================================================

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo Please run install_windows.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if main script exists
if not exist "windows_webcam_robot.py" (
    echo ❌ windows_webcam_robot.py not found!
    echo Ensure all files are in the same directory
    pause
    exit /b 1
)

REM Start the AI robot
echo 🤖 Starting AI Robot with webcam support...
echo.
echo 📷 Make sure your webcam is connected and not in use by other applications
echo 🖥️  A GUI window will open with live camera feed and AI detection
echo 🛑 Close the window or click Stop button to exit
echo.

python windows_webcam_robot.py

REM Cleanup
echo.
echo 👋 AI Robot stopped
echo 🔄 Deactivating virtual environment...
call venv\Scripts\deactivate.bat

pause