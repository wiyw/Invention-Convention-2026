@echo off
REM Arduino UNO Q4GB AI Robot - Fixed Windows Installation
REM Enhanced camera compatibility

echo ==========================================
echo   Arduino UNO Q4GB AI Robot - Windows
echo   Enhanced Camera Compatibility Version
echo ==========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python detected
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 🐍 Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment activated

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ⚠️  Pip upgrade failed, continuing...
)

REM Install core dependencies
echo 📦 Installing core dependencies...
python -m pip install opencv-python numpy pillow
if errorlevel 1 (
    echo ❌ Failed to install core dependencies
    pause
    exit /b 1
)
echo ✅ Core dependencies installed

REM Install AI framework
echo 🤖 Installing AI framework...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ⚠️  PyTorch installation failed, using basic detection...
    echo ⚠️  AI detection will be simplified but still works
) else (
    echo ✅ PyTorch installed
)

echo.
echo ==========================================
echo 🎉 ENHANCED WINDOWS SETUP COMPLETE!
echo ==========================================
echo.
echo 🎮 Run options:
echo   windows_webcam_robot.bat     - Main application
echo   camera_diagnostic.bat        - Test camera compatibility
echo   quick_test_windows.bat      - Quick dependency test
echo.
echo 🔧 If camera doesn't work:
echo   1. Run camera_diagnostic.bat
echo   2. Close other camera apps
echo   3. Check Windows camera permissions
echo   4. Click "Switch Camera" button
echo.
echo ✅ Enhanced camera detection for all Windows systems!
echo ==========================================
pause