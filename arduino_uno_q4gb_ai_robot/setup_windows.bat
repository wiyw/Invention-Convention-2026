@echo off
echo ========================================
echo Arduino UNO Q4GB AI Robot - Setup
echo ========================================
echo.

echo Creating Desktop Shortcuts...
echo.

cd /d "%~dp0"

REM Check if in correct directory
if not exist "python_tools\testing" (
    echo Error: Not in correct project directory
    echo Please run this from the project root
    pause
    exit /b 1
)

REM Create shortcuts using PowerShell (no extra dependencies)
powershell -Command "& {$WshShell = New-Object -comObject WScript.Shell; $desktop = [Environment]::GetFolderPath('Desktop'); $projectRoot = (Get-Location).Path; $shortcuts = @{'Arduino AI Robot'=($projectRoot); 'Camera Testing'=($projectRoot + '\python_tools\testing'); 'Arduino Firmware'=($projectRoot + '\arduino_firmware'); 'Documentation'=($projectRoot + '\docs'); 'Windows Setup'=($projectRoot + '\windows_setup')}; foreach ($name in $shortcuts.Keys) { $path = $shortcuts[$name]; $shortcut = $WshShell.CreateShortcut($desktop + '\' + $name + '.lnk'); $shortcut.TargetPath = $path; $shortcut.WorkingDirectory = $path; $shortcut.IconLocation = 'shell32.dll,3'; $shortcut.Save(); Write-Host 'Created: ' + $name }}"

echo.
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Installing Python 3.10...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe' -OutFile 'python_installer.exe'"
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python_installer.exe
    echo Python 3.10 installed
) else (
    echo Python already installed
)

echo.
echo Installing required packages...
pip install --upgrade pip
pip install opencv-python ultralytics numpy matplotlib pyserial

echo.
echo Creating quick test script...
echo @echo off > quick_test.bat
echo echo Running quick camera test... >> quick_test.bat
echo cd python_tools\testing >> quick_test.bat
echo python camera_only_tester.py --test >> quick_test.bat
echo pause >> quick_test.bat

echo.
echo Creating benchmark script...
echo @echo off > benchmark.bat
echo echo Running 30-second benchmark... >> benchmark.bat
echo cd python_tools\testing >> benchmark.bat
echo python camera_only_tester.py --benchmark 30 >> benchmark.bat
echo pause >> benchmark.bat

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Quick Start Options:
echo 1. Double-click "quick_test.bat" for interactive camera test
echo 2. Double-click "benchmark.bat" for performance testing
echo 3. Use desktop shortcuts to navigate folders
echo.
echo Desktop Shortcuts Created:
echo - Arduino AI Robot (main project folder)
echo - Camera Testing (test scripts)
echo - Arduino Firmware (Arduino code)
echo - Documentation (user guides)
echo - Windows Setup (installers and tools)
echo.
echo For detailed testing:
echo python python_tools\testing\camera_only_tester.py --test
echo.
echo For performance benchmark:
echo python python_tools\testing\camera_only_tester.py --benchmark 30
echo.
pause