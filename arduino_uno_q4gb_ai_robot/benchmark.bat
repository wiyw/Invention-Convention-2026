@echo off
echo ========================================
echo Arduino UNO Q4GB AI - Performance Benchmark
echo ========================================
echo.

REM Check if in correct directory
if not exist "python_tools\testing\camera_only_tester.py" (
    echo Error: camera_only_tester.py not found
    echo Please run setup_windows.bat first
    pause
    exit /b 1
)

set DURATION=30
if not "%1"=="" set DURATION=%1

echo Running %DURATION% second performance benchmark...
echo This will test:
echo - Camera capture speed
echo - Object detection performance  
echo - AI decision making speed
echo - Memory usage patterns
echo.

cd python_tools\testing
python camera_only_tester.py --benchmark %DURATION%

echo.
echo Benchmark completed
echo Results are shown above
pause