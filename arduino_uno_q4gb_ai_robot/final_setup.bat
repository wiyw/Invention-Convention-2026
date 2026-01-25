@echo off
echo ========================================
echo Arduino UNO Q4GB AI - Final Setup
echo ========================================
echo.

echo Installing required packages...
pip install --upgrade pip
pip install opencv-python ultralytics numpy matplotlib pyserial pillow

echo.
echo Setting up reasonable resolution testing...
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
echo Creating final test scripts...

REM Create final quick test
echo @echo off > final_quick_test.bat
echo echo Starting Final Camera Test... >> final_quick_test.bat
echo cd python_tools\testing >> final_quick_test.bat
echo echo Using 800x600 display size (reasonable for testing) >> final_quick_test.bat
echo echo Auto-searching YOLO model... >> final_quick_test.bat
echo python camera_only_tester.py --test >> final_quick_test.bat
echo pause >> final_quick_test.bat

REM Create final benchmark
echo @echo off > final_benchmark.bat
echo echo Running Final 30-second benchmark... >> final_benchmark.bat
echo cd python_tools\testing >> final_benchmark.bat
echo echo Reasonable resolution + YOLO search >> final_benchmark.bat
echo python camera_only_tester.py --benchmark 30 >> final_benchmark.bat
echo pause >> final_benchmark.bat

echo.
echo Creating desktop shortcuts...
powershell -Command "& {$WshShell = New-Object -comObject WScript.Shell; $desktop = [Environment]::GetFolderPath('Desktop'); $projectRoot = (Get-Location).Path; $shortcuts = @{'Arduino AI Robot'=($projectRoot); 'Camera Testing'=($projectRoot + '\python_tools\testing'); 'Final Tests'=($projectRoot + '\python_tools\testing')}; foreach ($name in $shortcuts.Keys) { $path = $shortcuts[$name]; $shortcut = $WshShell.CreateShortcut($desktop + '\' + $name + '.lnk'); $shortcut.TargetPath = $path; $shortcut.WorkingDirectory = $path; $shortcut.IconLocation = 'shell32.dll,3'; $shortcut.Save(); Write-Host 'Created: ' + $name }}"

echo.
echo Creating configuration file...
echo # Final Camera Testing Configuration > camera_config.txt
echo display_resolution=800x600 >> camera_config.txt
echo input_resolution=160x120 >> camera_config.txt
echo fps_target=15 >> camera_config.txt
echo yolo_model_search=true >> camera_config.txt
echo mock_ultrasonic=true >> camera_config.txt
echo setup_version=final >> camera_config.txt

echo.
echo ========================================
echo Final Setup Complete!
echo ========================================
echo.
echo FINAL FEATURES:
echo - Reasonable Display Size (800x600) - Perfect for laptops
echo - Compact Text Display - Small, readable labels, no overlap
echo - Automatic YOLO Model Search - Finds model automatically
echo - Professional Interface - Clean, uncluttered layout
echo.
echo Quick Start:
echo 1. Double-click "final_quick_test.bat" for camera test
echo 2. Double-click "final_benchmark.bat" for performance test
echo.
echo Desktop Shortcuts Created:
echo - Arduino AI Robot (main project folder)
echo - Camera Testing (test scripts)
echo - Final Tests (final scripts)
echo.
echo Display Resolution: 800x600 (perfect for laptop screens)
echo AI Processing: 160x120 (optimized for speed)
echo YOLO Search: Automatic (finds model in multiple locations)
echo.
echo For configuration, see: camera_config.txt
echo.
pause