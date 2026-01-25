@echo off
echo ========================================
echo Arduino UNO Q4GB AI - Enhanced Setup
echo ========================================
echo.

echo Installing additional packages for high resolution testing...
pip install --upgrade pip
pip install opencv-python ultralytics numpy matplotlib pyserial pillow

echo.
echo Setting up enhanced camera testing...
echo.

REM Check if yolo26n.pt exists, if not create placeholder
if not exist "yolo26n.pt" (
    if exist "..\yolo26n.pt" (
        echo Copying yolo26n.pt from parent directory...
        copy "..\yolo26n.pt" "yolo26n.pt"
    ) else if exist "..\..\yolo26n.pt" (
        echo Copying yolo26n.pt from grandparent directory...
        copy "..\..\yolo26n.pt" "yolo26n.pt"
    ) else (
        echo YOLO model not found. Testing with placeholder detection.
        echo To fix: Copy yolo26n.pt to project directory.
    )
)

echo.
echo Creating enhanced test scripts...

REM Create enhanced quick test
echo @echo off > enhanced_quick_test.bat
echo echo Starting Enhanced Camera Test... >> enhanced_quick_test.bat
echo cd python_tools\testing >> enhanced_quick_test.bat
echo echo Using high resolution display (1280x720) >> enhanced_quick_test.bat
echo python camera_only_tester.py --test >> enhanced_quick_test.bat
echo pause >> enhanced_quick_test.bat

REM Create enhanced benchmark
echo @echo off > enhanced_benchmark.bat
echo echo Running Enhanced 30-second benchmark... >> enhanced_benchmark.bat
echo cd python_tools\testing >> enhanced_benchmark.bat
echo echo High resolution display + YOLO model search >> enhanced_benchmark.bat
echo python camera_only_tester.py --benchmark 30 >> enhanced_benchmark.bat
echo pause >> enhanced_benchmark.bat

echo.
echo Creating desktop shortcuts...
powershell -Command "& {$WshShell = New-Object -comObject WScript.Shell; $desktop = [Environment]::GetFolderPath('Desktop'); $projectRoot = (Get-Location).Path; $shortcuts = @{'Arduino AI Robot'=($projectRoot); 'Camera Testing'=($projectRoot + '\python_tools\testing'); 'Enhanced Tests'=($projectRoot + '\python_tools\testing')}; foreach ($name in $shortcuts.Keys) { $path = $shortcuts[$name]; $shortcut = $WshShell.CreateShortcut($desktop + '\' + $name + '.lnk'); $shortcut.TargetPath = $path; $shortcut.WorkingDirectory = $path; $shortcut.IconLocation = 'shell32.dll,3'; $shortcut.Save(); Write-Host 'Created: ' + $name }}"

echo.
echo Creating configuration file...
echo # Camera Testing Configuration > camera_config.txt
echo display_resolution=1280x720 >> camera_config.txt
echo input_resolution=160x120 >> camera_config.txt
echo fps_target=15 >> camera_config.txt
echo yolo_model_search=true >> camera_config.txt
echo mock_ultrasonic=true >> camera_config.txt

echo.
echo ========================================
echo Enhanced Setup Complete!
echo ========================================
echo.
echo NEW FEATURES:
echo - High Resolution Display (1280x720)
echo - Automatic YOLO Model Search
echo - Enhanced Error Messages
echo - Better Visual Feedback
echo.
echo Quick Start:
echo 1. Double-click "enhanced_quick_test.bat" for high resolution test
echo 2. Double-click "enhanced_benchmark.bat" for performance test
echo.
echo Desktop Shortcuts Created:
echo - Arduino AI Robot (main project folder)
echo - Camera Testing (test scripts)
echo - Enhanced Tests (enhanced scripts)
echo.
echo For troubleshooting, see: camera_config.txt
echo.
pause