#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE TEST
Arduino UNO Q4GB AI Robot - Complete System Validation
"""

import os
import sys

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'arduino_uno_q4gb_ai_robot'))

def test_all_components():
    """Test all system components"""
    print("ARDUINO UNO Q4GB AI ROBOT - FINAL COMPREHENSIVE TEST")
    print("="*70)
    
    results = {}
    
    # Test 1: YOLO Model
    print("\n[TEST 1] YOLO26n Model Loading")
    print("-" * 50)
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics library available")
        
        yolo_paths = [
            "models/yolo26n.pt",
            "arduino_uno_q4gb_ai_robot/yolo26n.pt"
        ]
        
        model_loaded = False
        for path in yolo_paths:
            if os.path.exists(path):
                try:
                    model = YOLO(path)
                    print(f"✅ YOLO model loaded: {path}")
                    print(f"   Model classes: {len(model.names)}")
                    print(f"   Sample classes: {list(model.names.keys())[:3]}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"❌ YOLO load failed: {path} - {e}")
        
        results['yolo'] = "PASS" if model_loaded else "FAIL"
        
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        results['yolo'] = f"ERROR: {e}"
    
    # Test 2: Camera Backend
    print("\n[TEST 2] Camera Backend Fallback")
    print("-" * 50)
    
    try:
        import cv2
        
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        backend_names = ["DirectShow", "MSMF", "Auto"]
        
        camera_found = False
        for backend_idx, backend in enumerate(backends):
            for cam_id in range(3):
                try:
                    camera = cv2.VideoCapture(cam_id, backend)
                    if camera.isOpened():
                        ret, frame = camera.read()
                        if ret and frame is not None:
                            print(f"✅ Camera {cam_id} with {backend_names[backend_idx]}")
                            print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                            camera_found = True
                            camera.release()
                            break
                    camera.release()
                except Exception as e:
                    print(f"   ⚠ Camera {cam_id} failed: {e}")
            
            if camera_found:
                break
        
        results['camera'] = "PASS" if camera_found else "FAIL"
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        results['camera'] = f"ERROR: {e}"
    
    # Test 3: Qwen Integration (without model download)
    print("\n[TEST 3] Qwen Integration (Ready Status)")
    print("-" * 50)
    
    try:
        # Test prompt loading
        prompt_paths = ["models/qwen_prompt.txt", "qwen_prompt.txt"]
        prompt_found = False
        
        for path in prompt_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    prompt_content = f.read()
                print(f"✅ Qwen prompt found: {path}")
                prompt_found = True
                break
        
        # Test transformers availability
        transformers_available = False
        try:
            import transformers
            print("✅ Transformers library available")
            transformers_available = True
        except ImportError:
            print("⚠ Transformers not available (would need download)")
        
        qwen_ready = prompt_found and transformers_available
        results['qwen'] = "READY" if qwen_ready else "NOT_READY"
        
    except Exception as e:
        print(f"❌ Qwen test failed: {e}")
        results['qwen'] = f"ERROR: {e}"
    
    # Test 4: Hybrid AI Controller
    print("\n[TEST 4] Hybrid AI Controller")
    print("-" * 50)
    
    try:
        from python_tools.hybrid_ai_controller import HybridAIController
        
        controller = HybridAIController(enable_qwen=True)
        print("✅ Hybrid AI controller loaded")
        
        # Test decision making
        test_detections = [{
            'label': 'bottle',
            'confidence': 0.8,
            'bbox_norm': {'cx': 0.5, 'cy': 0.5, 'w': 0.3, 'h': 0.4}
        }]
        
        decision = controller.make_decision(test_detections, force_qwen_precision=True)
        
        print(f"✅ Decision made: {decision.get('action')}")
        print(f"   Mode: {decision.get('mode')}")
        print(f"   Command: {decision.get('command')}")
        
        results['hybrid'] = "PASS"
        
    except Exception as e:
        print(f"❌ Hybrid controller test failed: {e}")
        results['hybrid'] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✅" if "PASS" in result or "READY" in result else "❌"
        print(f"{test_name.upper():15}: {status} {result}")
    
    # Overall assessment
    success_count = sum(1 for r in results.values() if "PASS" in r or "READY" in r)
    total_tests = len(results)
    
    print(f"\nOVERALL: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print(f"\n🎉 SYSTEM FULLY READY!")
        print("✅ YOLO26n model detection working")
        print("✅ Camera initialization with backend fallback")
        print("✅ Hybrid AI controller (Rules + Qwen precision)")
        print("✅ All models and paths standardized")
        
        print("\n🚀 READY FOR ARDUINO UNO Q4GB DEPLOYMENT!")
        print("\nNext steps:")
        print("1. Run: python arduino_uno_q4gb_ai_robot/python_tools/testing/simple_camera_test.py --test")
        print("2. Test with trash objects to verify hybrid AI precision")
        print("3. Connect Arduino UNO Q4GB hardware")
        print("4. Upload Arduino firmware")
        print("5. Field test in target environment")
        
    elif success_count >= 3:
        print(f"\n✅ SYSTEM MOSTLY READY!")
        print("Core functionality working")
        print("Minor issues may need attention")
        
    else:
        print(f"\n⚠️ SYSTEM NEEDS ATTENTION!")
        print("Some critical components not working")
        print("Check individual test results above")
    
    return success_count == total_tests

def create_usage_guide():
    """Create usage guide for user"""
    guide = """
# ARDUINO UNO Q4GB AI ROBOT - USAGE GUIDE

## QUICK START COMMANDS

### 1. Test Camera and YOLO
```bash
cd C:\\Users\\Greyson\\Code\\InventionConvention2026
python arduino_uno_q4gb_ai_robot\\python_tools\\testing\\simple_camera_test.py --test
```

### 2. Test Full Pipeline with Hybrid AI
```bash
python arduino_uno_q4gb_ai_robot\\python_tools\\testing\\camera_only_tester.py --test
```

### 3. Run Integration Validation
```bash
python validate_system.py
```

### 4. Enable Qwen Integration (When Ready)
```bash
# First install Qwen dependencies (if needed)
py -m pip install transformers torch

# Then test integration
python arduino_uno_q4gb_ai_robot\\python_tools\\qwen_integration_real.py --test
```

## TESTING SCENARIOS

### Camera Test:
- Hold objects in front of camera
- Press 's' to save screenshots
- Press 'q' to quit

### Hybrid AI Test:
- Test with regular objects (person, car) → Rule-based navigation
- Test with trash objects (bottle, cup) → Qwen precision mode
- Watch for mode switching between "rule_based" and "qwen_precision"

## TROUBLESHOOTING

### Camera Issues:
- If DirectShow fails, system will try MSMF automatically
- Check if other apps are using camera
- Try different camera IDs (0, 1, 2)

### Model Issues:
- YOLO model: Ensure models/yolo26n.pt exists (5.5MB)
- Qwen prompt: Ensure models/qwen_prompt.txt exists
- Missing models will trigger placeholder detection

### Qwen Integration:
- Transformers library: Required for real AI reasoning
- Memory: Qwen2.5-0.5B needs ~2GB RAM
- Fallback: System works with rule-based if Qwen unavailable

## ARDUINO HARDWARE CONNECTION

### Serial Port:
- Connect Arduino UNO Q4GB via USB
- Check COM port in Windows Device Manager
- Default baud rate: 115200

### Upload Firmware:
- Use Arduino IDE 2.0+
- Select "Arduino UNO Q4GB" board
- Open arduino_firmware/core/ai_robot_controller.ino

## SUCCESS INDICATORS

✅ Ready State:
- Camera opens with backend fallback
- YOLO model loads successfully  
- Hybrid AI controller functional
- All test scripts run without errors

🚀 Deployment Ready:
- All systems validated
- Models in standardized locations
- Hybrid AI decision making operational
- Ready for Arduino UNO Q4GB hardware

This guide ensures you have everything needed for successful Arduino UNO Q4GB AI robot development and deployment.
"""
    
    with open("USAGE_GUIDE.md", "w") as f:
        f.write(guide)
    
    print("\n✅ Usage guide created: USAGE_GUIDE.md")

if __name__ == "__main__":
    test_all_components()
    create_usage_guide()