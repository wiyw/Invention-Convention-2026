@echo off 
echo Running Enhanced 30-second benchmark... 
cd python_tools\testing 
echo High resolution display + YOLO model search 
python camera_only_tester.py --benchmark 30 
pause 
