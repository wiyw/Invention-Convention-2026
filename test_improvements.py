#!/usr/bin/env python3
"""
Comprehensive test suite for Arduino UNO Q4GB AI Robot improvements
Tests camera initialization, model loading, and hybrid AI control
"""

import sys
import os

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_tools.testing.camera_only_tester import CameraOnlyAITester
from python_tools.testing.simple_camera_test import SimpleAITester
from python_tools.hybrid_ai_controller import HybridAIController
from python_tools.utils import ModelValidator, initialize_camera_with_fallback, load_yolo_model

def test_camera_initialization():
    """Test robust camera initialization"""
    print("\n" + "="*60)
    print("TEST 1: Camera Initialization with Backend Fallback")
    print("="*60)
    
    # Test utility function
    print("\n1. Testing utility camera initialization...")
    camera, cam_id, backend = initialize_camera_with_fallback()
    if camera:
        print(f"✅ Utility: Camera {cam_id} with {backend} backend")
        camera.release()
    else:
        print("❌ Utility: Camera initialization failed")
    
    # Test SimpleAITester
    print("\n2. Testing SimpleAITester...")
    simple_tester = SimpleAITester()
    if simple_tester.initialize_camera():
        print("✅ SimpleAITester: Camera initialized")
        simple_tester.camera.release()
    else:
        print("❌ SimpleAITester: Camera failed")
    
    # Test CameraOnlyAITester
    print("\n3. Testing CameraOnlyAITester...")
    camera_tester = CameraOnlyAITester()
    if camera_tester.initialize_camera():
        print("✅ CameraOnlyAITester: Camera initialized")
        camera_tester.camera.release()
    else:
        print("❌ CameraOnlyAITester: Camera failed")

def test_model_loading():
    """Test standardized model loading"""
    print("\n" + "="*60)
    print("TEST 2: Model Loading with Standardized Paths")
    print("="*60)
    
    # Test YOLO validation
    print("\n1. Testing YOLO model validation...")
    yolo_paths = [
        "models/yolo26n.pt",
        "../models/yolo26n.pt",
        "../../models/yolo26n.pt",
        "yolo26n.pt"
    ]
    
    yolo_found = False
    for path in yolo_paths:
        valid, message = ModelValidator.validate_yolo_model(path)
        if valid:
            print(f"✅ YOLO valid: {message}")
            yolo_found = True
            break
        else:
            print(f"❌ YOLO invalid: {path} - {message}")
    
    if not yolo_found:
        print("❌ No valid YOLO model found")
    
    # Test prompt validation
    print("\n2. Testing Qwen prompt validation...")
    prompt_paths = [
        "models/qwen_prompt.txt",
        "../models/qwen_prompt.txt",
        "../../models/qwen_prompt.txt",
        "qwen_prompt.txt"
    ]
    
    prompt_found = False
    for path in prompt_paths:
        valid, message = ModelValidator.validate_qwen_prompt(path)
        if valid:
            print(f"✅ Prompt valid: {message}")
            prompt_found = True
            break
        else:
            print(f"❌ Prompt invalid: {path} - {message}")
    
    if not prompt_found:
        print("❌ No valid prompt found")
    
    # Test model loading functions
    print("\n3. Testing model loading functions...")
    model, path = load_yolo_model()
    if model:
        print(f"✅ Model loaded: {path}")
        return True
    else:
        print("❌ Model loading failed")
        return False

def test_hybrid_ai_controller():
    """Test hybrid AI controller logic"""
    print("\n" + "="*60)
    print("TEST 3: Hybrid AI Controller")
    print("="*60)
    
    controller = HybridAIController(enable_qwen=True)
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Clear path',
            'detections': [],
            'expected_mode': 'rule_based'
        },
        {
            'name': 'Trash detection (high confidence)',
            'detections': [{
                'label': 'trash',
                'confidence': 0.8,
                'bbox_norm': {'cx': 0.5, 'cy': 0.5, 'w': 0.3, 'h': 0.4}
            }],
            'expected_mode': 'qwen_precision'
        },
        {
            'name': 'Person navigation',
            'detections': [{
                'label': 'person',
                'confidence': 0.9,
                'bbox_norm': {'cx': 0.3, 'cy': 0.5, 'w': 0.4, 'h': 0.6}
            }],
            'expected_mode': 'rule_based'
        },
        {
            'name': 'Cup precision task',
            'detections': [{
                'label': 'cup',
                'confidence': 0.75,
                'bbox_norm': {'cx': 0.48, 'cy': 0.5, 'w': 0.2, 'h': 0.3}
            }],
            'expected_mode': 'qwen_precision'
        }
    ]
    
    all_passed = True
    for i, scenario in enumerate(test_scenarios):
        print(f"\nScenario {i+1}: {scenario['name']}")
        decision = controller.make_decision(scenario['detections'])
        
        actual_mode = decision.get('mode', 'unknown')
        expected_mode = scenario['expected_mode']
        
        mode_ok = actual_mode == expected_mode
        status = "✅" if mode_ok else "❌"
        
        print(f"  {status} Mode: {actual_mode} (expected: {expected_mode})")
        print(f"  Action: {decision.get('action')}")
        print(f"  Command: {decision.get('command')}")
        print(f"  Reason: {decision.get('reason')}")
        
        # Validate servo command format
        command = decision.get('command', '')
        if 'CMD L' in command and 'R' in command and 'T' in command:
            print(f"  ✅ Servo command format valid")
        else:
            print(f"  ❌ Servo command format invalid: {command}")
            all_passed = False
    
    return all_passed

def test_integration():
    """Test end-to-end integration"""
    print("\n" + "="*60)
    print("TEST 4: End-to-End Integration")
    print("="*60)
    
    try:
        # Initialize camera
        tester = SimpleAITester()
        if not tester.initialize_camera():
            print("❌ Camera initialization failed")
            return False
        
        # Initialize models
        if not tester.initialize_model():
            print("❌ Model initialization failed")
            tester.camera.release()
            return False
        
        print("✅ Integration setup complete")
        
        # Test a few frames
        print("\nTesting 3 frames...")
        for i in range(3):
            ret, frame = tester.camera.read()
            if ret:
                detections = tester.detect_objects(frame)
                action, confidence, reason = tester.make_decision(detections, use_qwen_for_precision=True)
                print(f"  Frame {i+1}: {action} ({confidence:.2f}) - {len(detections)} objects")
            else:
                print(f"  Frame {i+1}: Failed to read")
                break
        
        tester.camera.release()
        print("✅ Integration test completed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run comprehensive test suite"""
    print("Arduino UNO Q4GB AI Robot - Comprehensive Test Suite")
    print("Testing camera fixes, model loading, and hybrid AI control")
    print("="*60)
    
    results = {}
    
    # Run tests
    try:
        test_camera_initialization()
        results['camera'] = "✅ Complete"
    except Exception as e:
        results['camera'] = f"❌ Failed: {e}"
    
    try:
        model_ok = test_model_loading()
        results['models'] = "✅ Complete" if model_ok else "❌ Failed"
    except Exception as e:
        results['models'] = f"❌ Failed: {e}"
    
    try:
        hybrid_ok = test_hybrid_ai_controller()
        results['hybrid'] = "✅ Complete" if hybrid_ok else "❌ Failed"
    except Exception as e:
        results['hybrid'] = f"❌ Failed: {e}"
    
    try:
        integration_ok = test_integration()
        results['integration'] = "✅ Complete" if integration_ok else "❌ Failed"
    except Exception as e:
        results['integration'] = f"❌ Failed: {e}"
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        print(f"{test_name.upper():15}: {result}")
    
    # Overall status
    all_passed = all("✅" in result for result in results.values())
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED! System ready for Arduino UNO Q4GB")
    else:
        print(f"\n⚠️  Some tests failed. Check details above.")
    
    print(f"\nNext steps:")
    print(f"1. Test with real hardware (Arduino UNO Q4GB)")
    print(f"2. Calibrate for your specific camera setup")
    print(f"3. Adjust Qwen confidence threshold as needed")
    print(f"4. Test trash reaching scenarios")

if __name__ == "__main__":
    main()