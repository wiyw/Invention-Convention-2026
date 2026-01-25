@echo off 
echo Starting Final Camera Test... 
cd python_tools\testing 
echo Using 800x600 display size (reasonable for testing) 
echo Auto-searching YOLO model... 
python camera_only_tester.py --test 
pause 
