@echo off
REM Arduino UNO Q4GB AI Robot - Camera Diagnostic Tool
REM Helps troubleshoot webcam issues on Windows

echo 📷 Arduino UNO Q4GB - Camera Diagnostic Tool
echo ============================================
echo.

REM Check Python and dependencies
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found
    pause
    exit /b 1
)

echo 🔍 Checking dependencies...
python -c "import cv2; print('✅ OpenCV:', cv2.__version__)" 2>nul || (
    echo ❌ OpenCV not found
    echo Installing OpenCV...
    python -m pip install opencv-python
)

echo.
echo 📷 Scanning for available cameras...
echo.

python -c "
import cv2
print('Scanning cameras 0-9...')
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f'✅ Camera {i}: Working (Frame size: {frame.shape[1]}x{frame.shape[0]})')
        else:
            print(f'⚠️  Camera {i}: Open but no frame')
        cap.release()
    else:
        print(f'❌ Camera {i}: Not available')
"

echo.
echo 🔍 Testing different backends...
echo.

python -c "
import cv2
backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
backend_names = ['DShow', 'Media Foundation', 'V4L2']

for i, backend in enumerate(backends):
    print(f'Testing backend: {backend_names[i]} (Backend {backend})')
    try:
        cap = cv2.VideoCapture(0 + backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f'  ✅ Works with camera 0')
            else:
                print(f'  ⚠️  Opens but cannot read frame')
            cap.release()
        else:
            print(f'  ❌ Cannot open camera')
    except Exception as e:
        print(f'  ❌ Error: {e}')
"

echo.
echo 🔍 Checking Windows camera permissions...
echo.

python -c "
import cv2
import time

print('Testing camera access...')
try:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        print('Camera opened successfully')
        
        # Try reading for 3 seconds
        start_time = time.time()
        frames_read = 0
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if ret:
                frames_read += 1
                if frames_read == 1:
                    print(f'First frame read successfully (Size: {frame.shape[1]}x{frame.shape[0]})')
            else:
                print(f'Failed to read frame')
            time.sleep(0.1)
        
        cap.release()
        print(f'Read {frames_read} frames in 3 seconds')
        
        if frames_read > 10:
            print('✅ Camera working properly!')
        elif frames_read > 0:
            print('⚠️  Camera working but slow')
        else:
            print('❌ Camera not providing frames')
    else:
        print('❌ Cannot open camera with DSHOW backend')
        
except Exception as e:
    print(f'❌ Error testing camera: {e}')
"

echo.
echo 🔧 Troubleshooting Tips:
echo 1. Close other apps using camera (Zoom, Teams, Skype, etc.)
echo 2. Check Windows Settings ^> Privacy ^> Camera
echo 3. Ensure camera drivers are up to date
echo 4. Try different USB ports if using external camera
echo 5. Restart computer if camera was recently disconnected
echo.
echo 📊 If cameras were found above, the app should work.
echo    If no cameras were found, check hardware connections.
echo.

pause