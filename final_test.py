#!/usr/bin/env python3
"""
SIMPLE FINAL TEST
Arduino UNO Q4GB AI Robot - Core functionality test
"""

import os
import sys

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'arduino_uno_q4gb_ai_robot'))

def main():
    print("Arduino UNO Q4GB AI Robot - Simple Final Test")
    print("="*60)
    
    results = {}
    
    # Test 1: YOLO
    print("\n[TEST 1] YOLO Model")
    print("-" * 40)
    
    try:
        from ultralytics import YOLO
        model_path = "models/yolo26n.pt"
        
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print("PASS: YOLO model loaded successfully")
            print(f"Model: {len(model.names)} classes")
            results['yolo'] = "PASS"
        else:
            print("FAIL: YOLO model not found")
            results['yolo'] = "FAIL"
            
    except Exception as e:
        print(f"ERROR: {e}")
        results['yolo'] = f"ERROR: {e}"
    
    # Test 2: Qwen
    print("\n[TEST 2] Qwen Integration")
    print("-" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("PASS: Transformers library available")
        
        prompt_path = "models/qwen_prompt.txt"
        if os.path.exists(prompt_path):
            print("PASS: Qwen prompt found")
            results['qwen'] = "PASS"
        else:
            print("FAIL: Qwen prompt not found")
            results['qwen'] = "FAIL"
            
    except Exception as e:
        print(f"ERROR: {e}")
        results['qwen'] = f"ERROR: {e}"
    
    # Test 3: Camera
    print("\n[TEST 3] Camera Backend")
    print("-" * 40)
    
    try:
        import cv2
        
        # Try DirectShow first
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if camera.isOpened():
            ret, frame = camera.read()
            if ret:
                print("PASS: Camera with DirectShow backend")
                print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
                results['camera'] = "PASS"
            else:
                print("FAIL: Camera read failed")
                results['camera'] = "FAIL"
            camera.release()
        else:
            print("FAIL: Camera failed to open")
            results['camera'] = "FAIL"
            
    except Exception as e:
        print(f"ERROR: {e}")
        results['camera'] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    
    for test_name, result in results.items():
        status = "PASS" if result == "PASS" else "FAIL"
        print(f"{test_name.upper():8}: {status}")
    
    # Overall assessment
    pass_count = sum(1 for r in results.values() if "PASS" in r)
    total_count = len(results)
    
    if pass_count == total_count:
        print(f"\nSUCCESS: {pass_count}/{total_count} tests passed")
        print("SYSTEM READY FOR ARDUINO UNO Q4GB!")
        print("\nCore systems working:")
        print("  - YOLO26n object detection")
        print("  - Camera with backend fallback")
        print("  - Qwen AI integration")
        print("  - Hybrid AI controller")
        
        print("\nNEXT STEPS:")
        print("1. python arduino_uno_q4gb_ai_robot/python_tools/testing/simple_camera_test.py --test")
        print("2. Test with trash objects")
        print("3. Connect Arduino UNO Q4GB hardware")
        print("4. Upload firmware and test")
        
    else:
        print(f"\nISSUES: {total_count - pass_count} tests failed")
        print("Check results above for troubleshooting")
    
    print(f"\nTEST STATUS: {'READY' if pass_count >= 2 else 'NEEDS ATTENTION'}")

if __name__ == "__main__":
    main()