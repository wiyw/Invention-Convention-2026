#!/usr/bin/env python3
"""
Arduino UNO Q4GB AI Robot - ONE-CLICK VERSION
"""

import subprocess
import sys
import os

def main():
    print("Arduino UNO Q4GB AI Robot - One-Click Setup")
    print("="*60)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if simple_camera_test.py exists
    if not os.path.exists("python_tools/testing/simple_camera_test.py"):
        print("Error: simple_camera_test.py not found")
        return
    
    # Run simple setup
    print("Running simple camera setup...")
    try:
        result = subprocess.run(
            ["cmd", "/c", "simple_setup.bat"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print("✅ Setup completed successfully!")
            print("\nTo test your camera:")
            print("1. Double-click simple_camera_test.bat")
            print("2. Or run: python python_tools/testing/simple_camera_test.py --test")
            print("\nFeatures:")
            print("- Auto-camera detection (tries IDs 0, 1, 2)")
            print("- Simple YOLO model (no external model needed)")
            print("- Clean 800x600 display")
            print("- Real-time FPS monitoring")
            print("- Save screenshots (press 's')")
        else:
            print(f"❌ Setup failed with return code: {result.returncode}")
            print("Error output:")
            print(result.stderr)
    
    except Exception as e:
        print(f"❌ Error during setup: {e}")

if __name__ == "__main__":
    main()