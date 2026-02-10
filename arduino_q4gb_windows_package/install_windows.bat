@echo off
REM Arduino UNO Q4GB AI Robot - Windows Installation Script
REM Automated setup for Windows systems with webcam support

echo ==========================================
echo   Arduino UNO Q4GB AI Robot - Windows
echo   Webcam AI Detection Version
echo   Ready to Test on Windows
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

REM Check if virtual environment exists
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

REM Install AI framework (PyTorch - works well on Windows)
echo 🤖 Installing AI framework...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ⚠️  PyTorch installation failed, using basic detection...
    echo ⚠️  AI detection will be simplified
) else (
    echo ✅ PyTorch installed
)

REM Install optional dependencies for enhanced features
echo 📦 Installing optional dependencies...
python -m pip install matplotlib seaborn
if errorlevel 1 (
    echo ⚠️  Optional dependencies failed, continuing...
)

echo.
echo ==========================================
echo 🎉 WINDOWS SETUP COMPLETE!
echo ==========================================
echo.
echo 🎮 To run the Windows AI Robot:
echo   windows_webcam_robot.bat
echo.
echo ✅ Your Windows system is ready!
echo   - Real webcam integration
echo   - AI object detection simulation
echo   - Interactive GUI interface
echo   - No Arduino hardware required
echo ==========================================
pause