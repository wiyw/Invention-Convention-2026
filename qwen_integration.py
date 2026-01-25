#!/usr/bin/env python3
"""
Qwen2.5-0.5B-Instruct Integration for Robot Decision Making
Enhanced AI decision making with natural language reasoning
"""

import json
import os
import sys
import subprocess
import tempfile
from datetime import datetime

class QwenDecisionEngine:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.base_prompt = self._load_base_prompt()
    
    def _load_base_prompt(self):
        """Load the base prompt template"""
        try:
            with open('yolo26n/prompt.txt', 'r') as f:
                return f.read()
        except Exception:
            return self._get_default_prompt()
    
    def _get_default_prompt(self):
        """Default prompt if prompt.txt not available"""
        return """You are an autonomous robot controller. Analyze the detection data and ultrasonic sensor readings to make safe navigation decisions.

DETECTIONS_JSON:
<<DETECTIONS_JSON>>

Guidelines:
1. Priority: person > bicycle > car > truck > other objects
2. Safety: ultrasonic sensors override vision when obstacles are closer than 20cm
3. Steering: turn toward detected objects but maintain safe distance
4. Speed: adjust based on confidence and proximity

Respond with action in format: ACTION:<forward|turn_left|turn_right|stop>, SPEED:<0-255>, DURATION:<ms>"""
    
    def analyze_with_qwen(self, detection_data, ultrasonic_data):
        """Use Qwen model for enhanced decision making"""
        # Prepare the prompt with current data
        filled_prompt = self.base_prompt.replace('<<DETECTIONS_JSON>>', json.dumps(detection_data, indent=2))
        
        # Add ultrasonic sensor data
        if ultrasonic_data:
            filled_prompt += f"\n\nUltrasonic Readings:\nLeft45: {ultrasonic_data.get('left45', 'N/A')}cm\n"
            filled_prompt += f"Right45: {ultrasonic_data.get('right45', 'N/A')}cm\n"
            filled_prompt += f"Center: {ultrasonic_data.get('center', 'N/A')}cm\n"
        
        # Try to use local Qwen model first, fallback to rule-based
        try:
            return self._call_qwen_model(filled_prompt)
        except Exception as e:
            print(f"Qwen model unavailable: {e}")
            return self._fallback_decision(detection_data, ultrasonic_data)
    
    def _call_qwen_model(self, prompt):
        """Call Qwen model for decision making"""
        # This would typically use transformers or similar
        # For now, simulate the call and parse response
        try:
            # Attempt to use transformers if available
            cmd = [
                'python', '-c',
                f'''
import sys
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    model_name = "{self.model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    prompt = """{prompt}"""
    
    messages = [
        {{"role": "user", "content": prompt}}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    
except ImportError:
    print("TRANSFORMERS_UNAVAILABLE")
except Exception as e:
    print(f"ERROR: {{e}}")
                '''
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and "TRANSFORMERS_UNAVAILABLE" not in result.stdout:
                # Parse Qwen response to extract action
                return self._parse_qwen_response(result.stdout)
            else:
                raise Exception("Transformers not available or model loading failed")
                
        except subprocess.TimeoutExpired:
            raise Exception("Qwen model timeout")
        except Exception as e:
            raise e
    
    def _parse_qwen_response(self, response):
        """Parse Qwen model response to extract action command"""
        # Look for ACTION: format in response
        import re
        
        action_match = re.search(r'ACTION:(\w+)', response.upper())
        speed_match = re.search(r'SPEED:(\d+)', response)
        duration_match = re.search(r'DURATION:(\d+)', response)
        
        if action_match:
            action = action_match.group(1).lower()
            speed = int(speed_match.group(1)) if speed_match else 150
            duration = int(duration_match.group(1)) if duration_match else 500
            
            # Convert to motor command format
            left_speed, right_speed = self._action_to_motor_speeds(action, speed)
            command = f'CMD L{left_speed} R{right_speed} T{duration}'
            
            summary = {
                "action": action,
                "left": left_speed,
                "right": right_speed,
                "duration_ms": duration,
                "reason": "qwen_decision",
                "model_response": response[:200] + "..." if len(response) > 200 else response
            }
            
            return command, summary
        
        # Fallback if parsing fails
        return self._fallback_decision({}, {})
    
    def _action_to_motor_speeds(self, action, base_speed=150):
        """Convert action to left/right motor speeds"""
        if action == "forward":
            return base_speed, base_speed
        elif action == "turn_left":
            return base_speed - 40, base_speed + 40
        elif action == "turn_right":
            return base_speed + 40, base_speed - 40
        elif action == "stop":
            return 0, 0
        else:
            return base_speed, base_speed
    
    def _fallback_decision(self, detection_data, ultrasonic_data):
        """Fallback to rule-based decision making"""
        # Use the existing decide.py logic as fallback
        try:
            cmd = [
                sys.executable, 'yolo26n/decide.py',
                '--result', json.dumps(detection_data),
                '--ultrasonic', json.dumps(ultrasonic_data)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    command = lines[0]
                    summary = json.loads(lines[1])
                    summary["reason"] = "fallback_rule_based"
                    return command, summary
        except Exception as e:
            print(f"Fallback decision failed: {e}")
        
        # Ultimate fallback
        command = "CMD L0 R0 T200"
        summary = {"action": "stop", "left": 0, "right": 0, "duration_ms": 200, "reason": "emergency_stop"}
        return command, summary

def test_qwen_integration():
    """Test the Qwen integration with sample data"""
    engine = QwenDecisionEngine()
    
    # Sample detection data
    detection_data = {
        "detections": [
            {
                "id": 0,
                "label": "person",
                "confidence": 0.85,
                "bbox_norm": {"cx": 0.6, "cy": 0.5, "w": 0.2, "h": 0.4}
            }
        ],
        "width": 640,
        "height": 480
    }
    
    # Sample ultrasonic data
    ultrasonic_data = {
        "left45": 45.2,
        "right45": 38.7,
        "center": 52.1,
        "timestamp": int(datetime.now().timestamp() * 1000)
    }
    
    command, summary = engine.analyze_with_qwen(detection_data, ultrasonic_data)
    
    print(f"Command: {command}")
    print(f"Summary: {json.dumps(summary, indent=2)}")
    
    return command, summary

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_qwen_integration()
    else:
        print("Qwen Decision Engine ready. Use --test to run integration test.")