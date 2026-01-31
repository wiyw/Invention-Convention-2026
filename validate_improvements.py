#!/usr/bin/env python3
"""
Simple test to validate camera improvements
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'arduino_uno_q4gb_ai_robot'))

def test_camera_backends():
    """Test camera backend fallback"""
    print("=== Testing Camera Backend Fallback ===")
    
    import cv2
    
    # Test backend selection
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    backend_names = ["DirectShow", "MSMF", "Auto"]
    
    for backend_idx, backend in enumerate(backends):
        print(f"Trying {backend_names[backend_idx]} backend...")
        
        for cam_id in range(3):
            try:
                camera = cv2.VideoCapture(cam_id, backend)
                
                if not camera.isOpened():
                    camera.release()
                    continue
                
                ret, frame = camera.read()
                if ret and frame is not None:
                    print(f"✓ Camera {cam_id} with {backend_names[backend_idx]} backend")
                    print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")
                    camera.release()
                    return True
                else:
                    camera.release()
                    
            except Exception as e:
                print(f"  ⚠ Camera {cam_id} failed: {e}")
                continue
    
    print("❌ No working camera found")
    return False

def test_model_loading():
    """Test model file locations"""
    print("\n=== Testing Model File Loading ===")
    
    # Check YOLO model
    yolo_paths = [
        "models/yolo26n.pt",
        "arduino_uno_q4gb_ai_robot/yolo26n.pt",
        "yolo26n.pt"
    ]
    
    yolo_found = False
    for path in yolo_paths:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✅ YOLO model found: {path} ({size//1024}KB)")
            yolo_found = True
            break
        else:
            print(f"❌ YOLO model not found: {path}")
    
    if not yolo_found:
        print("❌ No YOLO model found anywhere")
    
    # Check Qwen prompt
    prompt_paths = [
        "models/qwen_prompt.txt",
        "arduino_uno_q4gb_ai_robot/models/qwen_prompt.txt",
        "yolo26n/prompt.txt",
        "prompt.txt"
    ]
    
    prompt_found = False
    for path in prompt_paths:
        if os.path.exists(path):
            size = len(open(path).read())
            print(f"✅ Prompt found: {path} ({size} chars)")
            prompt_found = True
            break
        else:
            print(f"❌ Prompt not found: {path}")
    
    if not prompt_found:
        print("❌ No prompt file found anywhere")
    
    return yolo_found and prompt_found

def test_yolo_loading():
    """Test actual YOLO model loading"""
    print("\n=== Testing YOLO Model Loading ===")
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics library available")
    except ImportError as e:
        print(f"❌ Ultralytics not available: {e}")
        return False
    
    # Try to load YOLO model
    model_path = None
    for path in ["models/yolo26n.pt", "arduino_uno_q4gb_ai_robot/yolo26n.pt"]:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path:
        try:
            model = YOLO(model_path)
            print(f"✅ YOLO model loaded: {model_path}")
            print(f"  Model type: {type(model)}")
            return True
        except Exception as e:
            print(f"❌ YOLO loading failed: {e}")
            return False
    else:
        print("❌ No YOLO model available to load")
        return False

def main():
    """Run all tests"""
    print("Arduino UNO Q4GB AI Robot - Validation Test")
    print("="*60)
    
    results = {}
    
    # Test 1: Camera backends
    try:
        camera_ok = test_camera_backends()
        results['camera'] = "✅ Pass" if camera_ok else "❌ Fail"
    except Exception as e:
        results['camera'] = f"❌ Error: {e}"
    
    # Test 2: Model files
    try:
        files_ok = test_model_loading()
        results['files'] = "✅ Pass" if files_ok else "❌ Fail"
    except Exception as e:
        results['files'] = f"❌ Error: {e}"
    
    # Test 3: YOLO loading
    try:
        yolo_ok = test_yolo_loading()
        results['yolo'] = "✅ Pass" if yolo_ok else "❌ Fail"
    except Exception as e:
        results['yolo'] = f"❌ Error: {e}"
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    for test_name, result in results.items():
        print(f"{test_name.upper():15}: {result}")
    
    # Overall assessment
    all_good = all("✅" in result for result in results.values())
    if all_good:
        print(f"\n🎉 ALL SYSTEMS VALIDATED!")
        print(f"Your Arduino UNO Q4GB AI robot is ready:")
        print(f"✅ Robust camera initialization (DirectShow/MSMF fallback)")
        print(f"✅ Standardized model loading (models/ directory)")
        print(f"✅ YOLO model functional")
        print(f"✅ Ready for Arduino UNO Q4GB testing")
    else:
        print(f"\n⚠️  Some validation failed.")
        print(f"Check the details above for troubleshooting.")
    
    print(f"\nNext steps:")
    print(f"1. Run: python arduino_uno_q4gb_ai_robot/python_tools/testing/simple_camera_test.py")
    print(f"2. Test with trash objects to verify hybrid AI control")
    print(f"3. Connect Arduino UNO Q4GB for hardware testing")

if __name__ == "__main__":
    main()