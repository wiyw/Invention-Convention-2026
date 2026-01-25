@echo off
echo ========================================
echo Arduino UNO Q4GB AI Robot Setup
echo ========================================
echo.

echo [1/6] Installing Python 3.10...
if not exist "C:\Python310" (
    echo Downloading Python 3.10 installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe' -OutFile 'python_installer.exe'"
    echo Installing Python 3.10...
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    del python_installer.exe
) else (
    echo Python 3.10 already installed
)

echo.
echo [2/6] Installing required Python packages...
C:\Python310\python.exe -m pip install --upgrade pip
C:\Python310\python.exe -m pip install opencv-python ultralytics pyserial numpy matplotlib torch torchvision tensorflow
C:\Python310\python.exe -m pip install -r requirements.txt

echo.
echo [3/6] Installing Arduino IDE 2.0...
if not exist "C:\Program Files\Arduino IDE" (
    echo Downloading Arduino IDE 2.0...
    powershell -Command "Invoke-WebRequest -Uri 'https://downloads.arduino.cc/arduino-ide/arduino-ide_2.0.4_Windows_64bit.exe' -OutFile 'arduino_ide_installer.exe'"
    echo Installing Arduino IDE...
    start /wait arduino_ide_installer.exe /S
    del arduino_ide_installer.exe
) else (
    echo Arduino IDE already installed
)

echo.
echo [4/6] Installing USB Drivers...
echo Installing Arduino drivers...
pnputil /add "drivers\arduino.inf" /install

echo Installing CP210x USB to UART drivers...
pnputil /add "drivers\cp210x.inf" /install

echo Installing CH340/CH341 drivers...
pnputil /add "drivers\ch341.inf" /install

echo.
echo [5/6] Installing PlatformIO...
C:\Python310\python.exe -m pip install platformio

echo.
echo [6/6] Setting up development environment...
echo Creating desktop shortcuts...
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\Arduino AI Robot.lnk'); $Shortcut.TargetPath = 'C:\Program Files\Arduino IDE\arduino-ide.exe'; $Shortcut.Save()"

echo Creating virtual environment...
C:\Python310\python.exe -m venv venv
call venv\Scripts\activate
C:\Python310\python.exe -m pip install -r requirements.txt

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Quick Start:
echo 1. Open Arduino IDE from desktop shortcut
echo 2. Load: arduino_firmware\core\ai_robot_controller.ino
echo 3. Install Arduino UNO Q4GB board package
echo 4. Test simulation: python python_tools\ai_test_suite.py --simulate
echo.
echo For troubleshooting, see: docs\troubleshooting\windows_setup.md
echo.
pause