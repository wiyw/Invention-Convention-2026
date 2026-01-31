#!/usr/bin/env python3
"""
Qwen2.5-0.5B-Instruct Integration for Arduino UNO Q4GB AI Robot
Real AI reasoning for precision tasks (Windows CPU optimized)
"""

import json
import os
import sys
import time
import subprocess
from datetime import datetime

class QwenDecisionEngine:
    """Real Qwen model integration with transformers"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.prompt_template = self._load_prompt_template()
        self.is_available = False
        
        print("Initializing Qwen Decision Engine...")
        
        # Try to initialize transformers-based model
        try:
            self._initialize_transformers_model()
        except Exception as e:
            print(f"⚠ Transformers initialization failed: {e}")
            print("  Falling back to subprocess method...")
            try:
                self._initialize_subprocess_model()
            except Exception as e2:
                print(f"❌ Subprocess initialization also failed: {e2}")
                print("  Using rule-based reasoning only")
                self.is_available = False
    
    def _load_prompt_template(self):
        """Load Qwen prompt template"""
        prompt_paths = [
            "models/qwen_prompt.txt",
            "../models/qwen_prompt.txt",
            "qwen_prompt.txt"
        ]
        
        for path in prompt_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    print(f"⚠ Prompt read failed: {path} - {e}")
                    continue
        
        return self._get_default_prompt()
    
    def _get_default_prompt(self):
        """Default prompt if template not found"""
        return """You are an Arduino UNO Q4GB robot controller. Analyze detection data for navigation and trash grasping.

DETECTIONS_JSON:
<<DETECTIONS_JSON>>

Guidelines:
1. For trash objects (bottle, cup, can, plastic): use precision servo control
2. For navigation (person, car, etc): use fast rule-based decisions  
3. Object too close (>0.6 frame width): stop or retreat
4. Object centered and well-sized: slow approach for grasping

Respond with action in format: ACTION:<forward|turn_left|turn_right|stop|forward_slow>, CONFIDENCE:<0-1>, REASON:<brief reason>"""
    
    def _initialize_transformers_model(self):
        """Initialize Qwen model with transformers library"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise Exception("Transformers library not available")
        
        print(f"  Loading {self.model_name}...")
        
        # Load tokenizer and model with CPU optimization for Windows
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Half precision for memory efficiency
            device_map="cpu",        # Force CPU for Windows compatibility
            low_cpu_mem_usage=True  # Optimize for CPU
        )
        
        print("✅ Qwen model loaded with transformers")
        self.is_available = True
        
        # Warm up model (optional but helps)
        test_input = self.tokenizer("Test", return_tensors="pt")
        with torch.no_grad():
            _ = self.model.generate(**test_input, max_new_tokens=1, do_sample=False)
        print("  Qwen model warmed up")
    
    def _initialize_subprocess_model(self):
        """Initialize Qwen model via subprocess (fallback)"""
        print(f"  Attempting subprocess Qwen initialization...")
        
        # This would use a separate Python script to run Qwen
        # For now, mark as unavailable
        self.is_available = False
    
    def analyze_with_qwen(self, detections, ultrasonic_data=None, force_precision=False):
        """Use real Qwen model for decision making"""
        if not detections:
            return self._clear_path_response()
        
        # Check if we should use precision mode
        best_detection = max(detections, key=lambda d: d['confidence'])
        confidence = best_detection.get('confidence', 0.5)
        label = best_detection.get('label', 'object').lower()
        
        needs_precision = (
            force_precision or
            (label in ['trash', 'bottle', 'cup', 'can', 'plastic'] and confidence > 0.7) or
            best_detection.get('bbox_norm', {}).get('w', 0) > 0.2
        )
        
        if not self.is_available or not needs_precision:
            return self._rule_based_response(best_detection, detections, ultrasonic_data)
        
        # Use real Qwen for precision tasks
        return self._qwen_precision_decision(detections, ultrasonic_data)
    
    def _clear_path_response(self):
        """Response for clear path"""
        return {
            'action': 'forward',
            'confidence': 0.95,
            'reason': 'Clear path - rule based',
            'mode': 'rule_based',
            'left': 160,
            'right': 160,
            'duration_ms': 500,
            'command': 'CMD L160 R160 T500',
            'json': {"action":"forward","left":160,"right":160,"duration_ms":500,"reason":"clear path"}
        }
    
    def _rule_based_response(self, best_detection, detections, ultrasonic_data):
        """Fast rule-based response"""
        bbox_norm = best_detection.get('bbox_norm', {})
        confidence = best_detection.get('confidence', 0.5)
        label = best_detection.get('label', 'object')
        cx = bbox_norm.get('cx', 0.5)
        w = bbox_norm.get('w', 0.1)
        
        # Safety check with ultrasonic
        if ultrasonic_data:
            center_dist = ultrasonic_data.get('center', 100)
            if center_dist < 20:
                return {
                    'action': 'stop',
                    'confidence': 0.9,
                    'reason': f'Safety stop - obstacle {center_dist}cm',
                    'mode': 'rule_based_safety',
                    'left': 0,
                    'right': 0,
                    'duration_ms': 200,
                    'command': 'CMD L0 R0 T200',
                    'json': {"action":"stop","left":0,"right":0,"duration_ms":200,"reason":"ultrasonic safety"}
                }
        
        # Navigation decisions
        if w > 0.6:
            action = 'stop'
            reason = f'Object too close: {label}'
        elif 0.4 <= cx <= 0.6:
            action = 'forward'
            reason = f'Centered: {label}'
        elif cx < 0.4:
            action = 'turn_right'
            reason = f'Object left: {label}'
        else:
            action = 'turn_left'
            reason = f'Object right: {label}'
        
        # Calculate servo values
        base_speed = int(160 * confidence)
        steering_offset = int((0.5 - cx) * 2 * 80)
        left = max(0, min(255, base_speed - steering_offset))
        right = max(0, min(255, base_speed + steering_offset))
        duration = 500
        
        return {
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'mode': 'rule_based',
            'left': left,
            'right': right,
            'duration_ms': duration,
            'command': f'CMD L{left} R{right} T{duration}',
            'json': {"action":action,"left":left,"right":right,"duration_ms":duration,"reason":reason}
        }
    
    def _qwen_precision_decision(self, detections, ultrasonic_data):
        """Use real Qwen model for precision tasks"""
        if not self.is_available or not self.tokenizer or not self.model:
            return self._rule_based_response(detections[0], detections, ultrasonic_data)
        
        # Prepare data for Qwen
        detection_data = {
            'detections': detections,
            'width': 640,
            'height': 480
        }
        
        if ultrasonic_data:
            detection_data['ultrasonics'] = ultrasonic_data
        
        # Create prompt
        filled_prompt = self.prompt_template.replace('<<DETECTIONS_JSON>>', json.dumps(detection_data, indent=2))
        
        try:
            # Tokenize input
            inputs = self.tokenizer(filled_prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response with temperature for creativity
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,  # Some creativity for precision
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response to extract action
            return self._parse_qwen_response(response)
            
        except Exception as e:
            print(f"⚠ Qwen inference failed: {e}")
            # Fallback to rule-based
            best_detection = max(detections, key=lambda d: d['confidence'])
            return self._rule_based_response(best_detection, detections, ultrasonic_data)
    
    def _parse_qwen_response(self, response):
        """Parse Qwen model response"""
        try:
            # Look for ACTION, CONFIDENCE, REASON patterns
            action = 'forward'
            confidence = 0.5
            reason = 'Qwen decision'
            
            # Extract action
            if 'ACTION:' in response.upper():
                action_line = [line for line in response.split('\n') if 'ACTION:' in line.upper()]
                if action_line:
                    action_part = action_line[0].split('ACTION:')[1].strip()
                    if action_part:
                        action = action_part.split(',')[0].strip()
            
            # Extract confidence
            if 'CONFIDENCE:' in response.upper():
                conf_line = [line for line in response.split('\n') if 'CONFIDENCE:' in line.upper()]
                if conf_line:
                    conf_part = conf_line[0].split('CONFIDENCE:')[1].strip()
                    try:
                        confidence = float(conf_part.split(',')[0].strip())
                    except:
                        pass
            
            # Extract reason
            if 'REASON:' in response.upper():
                reason_line = [line for line in response.split('\n') if 'REASON:' in line.upper()]
                if reason_line:
                    reason = reason_line[0].split('REASON:')[1].strip()
            
            # Map to servo values (simplified)
            if 'forward' in action.lower():
                left = right = 140
            elif 'turn_left' in action.lower():
                left, right = 100, 180
            elif 'turn_right' in action.lower():
                left, right = 180, 100
            else:  # stop
                left = right = 0
            
            duration = 400 if 'turn' in action.lower() else 500
            
            return {
                'action': action,
                'confidence': confidence,
                'reason': reason,
                'mode': 'qwen_precision',
                'left': left,
                'right': right,
                'duration_ms': duration,
                'command': f'CMD L{left} R{right} T{duration}',
                'json': {"action":action,"left":left,"right":right,"duration_ms":duration,"reason":reason}
            }
            
        except Exception as e:
            print(f"⚠ Response parsing failed: {e}")
            # Default to safe forward
            return {
                'action': 'forward',
                'confidence': 0.7,
                'reason': 'Qwen response parse error',
                'mode': 'qwen_fallback',
                'left': 140,
                'right': 140,
                'duration_ms': 500,
                'command': 'CMD L140 R140 T500',
                'json': {"action":"forward","left":140,"right":140,"duration_ms":500,"reason":"parse error"}
            }
    
    def test_qwen_model(self):
        """Test Qwen model functionality"""
        print("=== Testing Qwen Model ===")
        
        if not self.is_available:
            print("❌ Qwen model not available")
            return False
        
        test_cases = [
            {
                'name': 'Trash precision',
                'detections': [{
                    'label': 'bottle',
                    'confidence': 0.8,
                    'bbox_norm': {'cx': 0.5, 'cy': 0.6, 'w': 0.25, 'h': 0.3}
                }]
            },
            {
                'name': 'Navigation',
                'detections': [{
                    'label': 'person',
                    'confidence': 0.9,
                    'bbox_norm': {'cx': 0.3, 'cy': 0.5, 'w': 0.4, 'h': 0.6}
                }]
            }
        ]
        
        for test_case in test_cases:
            print(f"\nTest: {test_case['name']}")
            start_time = time.time()
            
            result = self.analyze_with_qwen(
                test_case['detections'],
                force_precision=(test_case['name'] == 'Trash precision')
            )
            
            inference_time = (time.time() - start_time) * 1000
            print(f"  Mode: {result.get('mode')}")
            print(f"  Action: {result.get('action')}")
            print(f"  Confidence: {result.get('confidence'):.2f}")
            print(f"  Inference: {inference_time:.1f}ms")
            print(f"  Command: {result.get('command')}")
        
        print("\n✅ Qwen model test complete")
        return True

def test_qwen_integration():
    """Test Qwen integration"""
    print("=== Qwen Integration Test ===")
    
    try:
        engine = QwenDecisionEngine()
        
        # Test basic functionality
        if engine.is_available:
            print("✅ Qwen engine initialized successfully")
        else:
            print("⚠ Qwen engine using fallback mode")
        
        # Test decision making
        test_detections = [{
            'label': 'cup',
            'confidence': 0.75,
            'bbox_norm': {'cx': 0.48, 'cy': 0.5, 'w': 0.2, 'h': 0.25}
        }]
        
        result = engine.analyze_with_qwen(test_detections, force_precision=True)
        print(f"  Test result: {result.get('mode')}")
        print(f"  Action: {result.get('action')}")
        print(f"  Command: {result.get('command')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Qwen integration test failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Qwen Integration Test')
    parser.add_argument('--test', action='store_true', help='Test Qwen integration')
    parser.add_argument('--model-test', action='store_true', help='Test Qwen model functionality')
    
    args = parser.parse_args()
    
    print("Arduino UNO Q4GB - Qwen2.5-0.5B Integration")
    print("="*60)
    
    if args.model_test:
        engine = QwenDecisionEngine()
        engine.test_qwen_model()
    elif args.test:
        test_qwen_integration()
    else:
        # Initialize and show status
        engine = QwenDecisionEngine()
        status = "✅ Available" if engine.is_available else "⚠ Fallback"
        print(f"Qwen Engine Status: {status}")
        
        if engine.is_available:
            print("Model:", engine.model_name)
            print("Memory: CPU optimized")
            print("Precision: Real AI reasoning available")
        else:
            print("Mode: Rule-based with Qwen-style formatting")
        
        print("\nUsage examples:")
        print("  python qwen_integration_real.py --test")
        print("  python qwen_integration_real.py --model-test")