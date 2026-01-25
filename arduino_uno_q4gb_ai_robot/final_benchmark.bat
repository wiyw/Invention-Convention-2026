@echo off 
echo Running Final 30-second benchmark... 
cd python_tools\testing 
echo Reasonable resolution + YOLO search 
python camera_only_tester.py --benchmark 30 
pause 
