@echo off
echo ========================================
echo Arduino UNO Q4GB AI - Camera Test
echo ========================================
echo.

REM Check if in correct directory
if not exist "python_tools\testing\camera_only_tester.py" (
    echo Error: camera_only_tester.py not found
    echo Please run setup_windows.bat first
    pause
    exit /b 1
)

echo Starting camera test...
echo.
echo Instructions:
echo - Position objects in front of camera
echo - Press 's' to save screenshots
echo - Press 't' to run test sequence
echo - Press 'q' to quit
echo.

cd python_tools\testing
python camera_only_tester.py --test

echo.
echo Camera test completed
pause