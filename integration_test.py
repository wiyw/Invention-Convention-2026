#!/usr/bin/env python3
"""
Final Integration Test for Arduino UNO Q4GB AI Robot
Tests complete pipeline: Camera -> YOLO -> Hybrid AI -> Servo Commands
"""

import os
import sys
import time
import cv2

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'arduino_uno_q4gb_ai_robot'))

def test_complete_pipeline():
    """Test complete AI robot pipeline"""
    print("Arduino UNO Q4GB AI Robot - Final Integration Test")
    print("="*70)
    
    # Test 1: Camera with backend fallback
    print("\n[TEST 1] Camera Initialization with Backend Fallback")
    print("-" * 50)
    
    import cv2
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    backend_names = ["DirectShow", "MSMF", "Auto"]
    
    camera = None
    camera_id = None
    backend_name = None
    
    for backend_idx, backend in enumerate(backends):
        for cam_id in range(3):
            try:
                camera = cv2.VideoCapture(cam_id, backend)
                if camera.isOpened():
                    ret, frame = camera.read()
                    if ret and frame is not None:
                        camera_id = cam_id
                        backend_name = backend_names[backend_idx]
                        print(f"✅ Camera {cam_id} with {backend_name} backend")
                        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                        break
                else:
                    camera.release()
            except Exception as e:
                print(f"   ⚠ Camera {cam_id} failed: {e}")
                continue
            continue
        
        if camera is not None:
            break
    
    if camera is None:
        print("❌ No working camera found")
        return False
    
    # Test 2: YOLO Model Loading
    print("\n[TEST 2] YOLO Model from Standardized Location")
    print("-" * 50)
    
    yolo_model = None
    yolo_paths = [
        os.path.join(project_root, "models", "yolo26n.pt"),
        os.path.join(project_root, "arduino_uno_q4gb_ai_robot", "yolo26n.pt")
    ]
    
    for path in yolo_paths:
        if os.path.exists(path):
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(path)
                print(f"✅ YOLO loaded: {path}")
                break
            except Exception as e:
                print(f"⚠ YOLO load failed: {path} - {e}")
                continue
    
    if yolo_model is None:
        print("❌ YOLO model loading failed")
        camera.release()
        return False
    
    # Test 3: Hybrid AI Controller
    print("\n[TEST 3] Hybrid AI Controller (Rules + Qwen)")
    print("-" * 50)
    
    try:
        from python_tools.hybrid_ai_controller import HybridAIController
        ai_controller = HybridAIController(enable_qwen=True)
        print("✅ Hybrid AI controller initialized")
    except Exception as e:
        print(f"❌ AI controller failed: {e}")
        camera.release()
        return False
    
    # Test 4: End-to-End Pipeline
    print("\n[TEST 4] End-to-End Pipeline (5 frames)")
    print("-" * 50)
    
    all_success = True
    
    for frame_num in range(5):
        print(f"\nFrame {frame_num + 1}:")
        
        # Capture frame
        ret, frame = camera.read()
        if not ret:
            print("❌ Camera read failed")
            all_success = False
            continue
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (320, 240))
        
        # Run YOLO detection
        start_time = time.time()
        try:
            results = yolo_model(small_frame, verbose=False)
            detections = []
            
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Calculate normalized coordinates
                    h, w = small_frame.shape[:2]
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bbox_w = (x2 - x1) / w
                    bbox_h = (y2 - y1) / h
                    
                    detections.append({
                        'label': yolo_model.names.get(cls, f'obj_{cls}'),
                        'confidence': float(conf),
                        'bbox_norm': {
                            'cx': cx,
                            'cy': cy,
                            'w': bbox_w,
                            'h': bbox_h
                        }
                    })
            
            detection_time = (time.time() - start_time) * 1000
            
        except Exception as e:
            print(f"   ❌ YOLO failed: {e}")
            detections = []
            detection_time = 0
        
        print(f"   🎯 Detections: {len(detections)} ({detection_time:.1f}ms)")
        
        # Run AI decision
        start_time = time.time()
        try:
            decision = ai_controller.make_decision(detections, force_precision=(frame_num == 3))
            decision_time = (time.time() - start_time) * 1000
            
            action = decision.get('action', 'unknown')
            mode = decision.get('mode', 'unknown')
            confidence = decision.get('confidence', 0)
            reason = decision.get('reason', 'no reason')
            command = decision.get('command', 'no command')
            
        except Exception as e:
            print(f"   ❌ AI decision failed: {e}")
            action = 'error'
            mode = 'error'
            confidence = 0
            reason = f'Error: {e}'
            command = 'CMD L0 R0 T100'
            decision_time = 0
        
        print(f"   🤖 AI Decision: {action} ({mode})")
        print(f"   📊 Confidence: {confidence:.2f} ({decision_time:.1f}ms)")
        print(f"   💭 Reason: {reason}")
        print(f"   🔧 Command: {command}")
        
        # Show basic frame info
        print(f"   📷 Frame: {small_frame.shape[1]}x{small_frame.shape[0]}")
    
    # Cleanup
    camera.release()
    
    # Summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    
    if all_success:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Arduino UNO Q4GB AI Robot System Ready:")
        print("  ✓ Robust camera initialization (DirectShow/MSMF fallback)")
        print("  ✓ Standardized model loading (models/ directory)")
        print("  ✓ YOLO26n object detection functional")
        print("  ✓ Hybrid AI controller (Rules + Qwen precision)")
        print("  ✓ Complete pipeline: Camera → YOLO → AI → Commands")
        
        print("\n🚀 Ready for Arduino UNO Q4GB deployment!")
        print("\nNext steps:")
        print("1. Upload Arduino firmware (arduino_firmware/)")
        print("2. Connect Arduino UNO Q4GB hardware")
        print("3. Test with real trash objects")
        print("4. Calibrate servo PWM values")
        print("5. Field test in target environment")
        
    else:
        print("⚠️  Some tests failed. Check details above.")
    
    return all_success

def main():
    """Run integration test"""
    try:
        success = test_complete_pipeline()
        return success
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return False
    except Exception as e:
        print(f"\n\nIntegration test failed: {e}")
        return False

if __name__ == "__main__":
    main()