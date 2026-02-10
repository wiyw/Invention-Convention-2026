@echo off
REM Arduino UNO Q4GB AI Robot - Minimal Windows Test
REM Simple camera test with minimal dependencies

echo 📷 Arduino UNO Q4GB - Minimal Camera Test
echo =========================================
echo.

python -c "
import cv2
import sys

print('Testing camera with minimal setup...')
print('Checking camera indices 0-4...')

for i in range(5):
    print(f'Testing camera {i}...')
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f'✅ Camera {i} works! Size: {frame.shape[1]}x{frame.shape[0]}')
        else:
            print(f'⚠️  Camera {i}: Opens but no frame')
        cap.release()
    else:
        print(f'❌ Camera {i}: Not available')

print()
print('Trying different backends for camera 0...')

backends = [
    (cv2.CAP_DSHOW, 'DShow'),
    (cv2.CAP_MSMF, 'Media Foundation'),
    (cv2.CAP_FFMPEG, 'FFMPEG')
]

for backend, name in backends:
    cap = cv2.VideoCapture(0 + backend)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f'✅ {name} works! Size: {frame.shape[1]}x{frame.shape[0]}')
        else:
            print(f'⚠️  {name}: Opens but no frame')
        cap.release()
    else:
        print(f'❌ {name}: Not available')
"

pause