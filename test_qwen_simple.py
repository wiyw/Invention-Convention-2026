#!/usr/bin/env python3
"""
Simple Qwen Test for Arduino UNO Q4GB AI Robot
Windows-compatible without unicode issues
"""

import os
import sys

def test_qwen_integration():
    """Test Qwen integration without unicode"""
    print("=== Qwen Integration Test ===")
    
    try:
        # Test transformers import
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("PASS: Transformers library available")
        
        # Test model loading (might be slow)
        print("Testing model loading...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct",
                torch_dtype="auto",  # Let it choose
                device_map="cpu"
            )
            print("PASS: Qwen model loads successfully")
            
            # Test basic inference
            test_input = tokenizer("Test", return_tensors="pt")
            with model.no_grad():
                output = model.generate(**test_input, max_new_tokens=5, do_sample=False)
            
            result = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"PASS: Qwen inference works - output: {result[:50]}...")
            
            return True
            
        except Exception as e:
            print(f"FAIL: Model loading failed - {e}")
            return False
            
    except ImportError as e:
        print(f"FAIL: Transformers not available - {e}")
        return False

def test_existing_system():
    """Test current system state"""
    print("\n=== Existing System Test ===")
    
    # Test YOLO
    try:
        from ultralytics import YOLO
        yolo_paths = [
            "models/yolo26n.pt",
            "arduino_uno_q4gb_ai_robot/yolo26n.pt",
            "yolo26n.pt"
        ]
        
        for path in yolo_paths:
            if os.path.exists(path):
                try:
                    model = YOLO(path)
                    print(f"PASS: YOLO loads from {path}")
                    return True
                except Exception as e:
                    print(f"FAIL: YOLO load failed from {path} - {e}")
                    continue
        
        print("FAIL: No YOLO model found")
        return False
        
    except ImportError as e:
        print(f"FAIL: Ultralytics not available - {e}")
        return False

def main():
    """Run tests"""
    print("Arduino UNO Q4GB AI Robot - Simple Test")
    print("="*60)
    
    results = {}
    
    # Test existing system
    try:
        yolo_ok = test_existing_system()
        results['yolo'] = "PASS" if yolo_ok else "FAIL"
    except Exception as e:
        results['yolo'] = f"ERROR: {e}"
    
    # Test Qwen integration
    try:
        qwen_ok = test_qwen_integration()
        results['qwen'] = "PASS" if qwen_ok else "FAIL"
    except Exception as e:
        results['qwen'] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    for test_name, result in results.items():
        print(f"{test_name.upper():8}: {result}")
    
    all_pass = all("PASS" in result for result in results.values())
    if all_pass:
        print(f"\nSUCCESS: All systems ready!")
        print("YOLO: Working")
        print("Qwen: Real AI integration available")
        print("Next: Run camera tests with hybrid AI")
    else:
        print(f"\nSOME ISSUES FOUND")
        print("Check results above")
    
    print(f"\nReady for Arduino UNO Q4GB: {all_pass}")

if __name__ == "__main__":
    main()