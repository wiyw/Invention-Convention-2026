#!/usr/bin/env python3
"""
Hybrid AI Controller for Arduino UNO Q4GB Robot
Combines fast rule-based navigation with Qwen precision for trash reaching
"""

import json
import time
import os
try:
    from ..utils import load_qwen_prompt, calculate_servo_commands, clamp
except ImportError:
    # Fallback if utils not available
    def load_qwen_prompt():
        paths = ["models/qwen_prompt.txt", "qwen_prompt.txt"]
        for path in paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return f.read()
        return None
    
    def calculate_servo_commands(action, confidence, bbox_norm):
        base_speed = max(50, min(220, int(160 * confidence)))
        if not bbox_norm:
            return base_speed, base_speed, 500
        
        cx = bbox_norm.get('cx', 0.5)
        steering_offset = int((0.5 - cx) * 2 * 80)
        left = max(0, min(255, base_speed - steering_offset))
        right = max(0, min(255, base_speed + steering_offset))
        duration = 500
        return left, right, duration
    
    def clamp(value, min_val, max_val):
        return max(min_val, min(max_val, value))

class HybridAIController:
    """Hybrid AI controller combining rules and Qwen reasoning"""
    
    def __init__(self, enable_qwen=True):
        self.enable_qwen = enable_qwen
        self.qwen_prompt = load_qwen_prompt() if enable_qwen else None
        self.rule_mode = True  # Default to fast rule-based
        self.qwen_confidence_threshold = 0.7
        self.precision_targets = ['trash', 'bottle', 'cup', 'can', 'plastic']
        
        print(f"Hybrid AI Controller initialized")
        print(f"  Rule-based: ✅ Always active")
        print(f"  Qwen precision: {'✅' if self.qwen_prompt else '❌'}")
    
    def make_decision(self, detections, ultrasonic_data=None, force_qwen_precision=False):
        """
        Make hybrid decision combining rules and Qwen
        
        Args:
            detections: List of detection dictionaries
            ultrasonic_data: Optional sensor readings
            force_precision: Force Qwen precision mode
            
        Returns:
            Dictionary with action, servo commands, and metadata
        """
        
        if not detections:
            return self._clear_path_response()
        
        # Find best detection
        best_detection = max(detections, key=lambda d: d['confidence'])
        bbox_norm = best_detection.get('bbox_norm', {})
        confidence = best_detection.get('confidence', 0.5)
        label = best_detection.get('label', 'object').lower()
        
        # Determine if we need precision mode
        needs_precision = (
            force_qwen_precision and  # User requested
            self.qwen_available and  # Qwen system available
            confidence > self.qwen_confidence_threshold and  # Good detection
            (label in self.precision_targets or w > 0.2)  # Trash-like or sizable object
        )
        
        if needs_precision and self.enable_qwen and self.qwen_prompt:
            return self._qwen_precision_decision(best_detection, detections, ultrasonic_data)
        else:
            return self._rule_based_decision(best_detection, detections, ultrasonic_data)
    
    def _clear_path_response(self):
        """Response for clear path ahead"""
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
    
    def _rule_based_decision(self, best_detection, all_detections, ultrasonic_data):
        """Fast rule-based decision for navigation"""
        bbox_norm = best_detection.get('bbox_norm', {})
        confidence = best_detection.get('confidence', 0.5)
        label = best_detection.get('label', 'object')
        
        cx = bbox_norm.get('cx', 0.5)
        w = bbox_norm.get('w', 0.1)
        
        # Safety check with ultrasonic data
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
            action, reason = 'stop', f'Object too close: {label}'
        elif 0.4 <= cx <= 0.6:
            action, reason = 'forward', f'Centered: {label}'
        elif cx < 0.4:
            action, reason = 'turn_right', f'Object left: {label}'
        else:
            action, reason = 'turn_left', f'Object right: {label}'
        
        # Calculate servo commands
        left, right, duration = calculate_servo_commands(action, confidence, bbox_norm)
        command = f'CMD L{left} R{right} T{duration}'
        
        return {
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'mode': 'rule_based',
            'left': left,
            'right': right,
            'duration_ms': duration,
            'command': command,
            'json': {"action":action,"left":left,"right":right,"duration_ms":duration,"reason":reason}
        }
    
    def _qwen_precision_decision(self, best_detection, all_detections, ultrasonic_data):
        """Use Qwen-style reasoning for precise trash reaching"""
        # This is a simplified version - real Qwen would call the model
        bbox_norm = best_detection.get('bbox_norm', {})
        confidence = best_detection.get('confidence', 0.5)
        label = best_detection.get('label', 'object')
        
        cx = bbox_norm.get('cx', 0.5)
        w = bbox_norm.get('w', 0.1)
        
        # Enhanced precision reasoning
        if w > 0.7:
            action = 'backward'
            reason = f'Qwen precision: {label} too close - retreat for better approach'
            duration = 300
        elif 0.3 <= cx <= 0.7 and w > 0.15:
            action = 'forward_slow'
            reason = f'Qwen precision: {label} positioned for grasp - slow approach'
            duration = 400
        elif 0.2 <= cx <= 0.8:
            if cx < 0.45:
                action = 'turn_left_slow'
                reason = f'Qwen precision: {label} slightly left - fine adjust'
                duration = 250
            elif cx > 0.55:
                action = 'turn_right_slow'
                reason = f'Qwen precision: {label} slightly right - fine adjust'
                duration = 250
            else:
                action = 'forward_slow'
                reason = f'Qwen precision: {label} nearly centered - micro adjust'
                duration = 350
        else:
            # Fall back to rule-based for badly positioned objects
            return self._rule_based_decision(best_detection, all_detections, ultrasonic_data)
        
        # Calculate precise servo values for precision mode
        base_speed = int(120 * confidence)  # Slower for precision
        steering_offset = int((0.5 - cx) * 1.5 * 60)  # Finer steering control
        
        left = clamp(base_speed - steering_offset, 40, 180)
        right = clamp(base_speed + steering_offset, 40, 180)
        
        command = f'CMD L{left} R{right} T{duration}'
        
        return {
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'mode': 'qwen_precision',
            'left': left,
            'right': right,
            'duration_ms': duration,
            'command': command,
            'json': {"action":action,"left":left,"right":right,"duration_ms":duration,"reason":reason}
        }
    
    def get_detection_summary(self, detections):
        """Get summary of current detections"""
        if not detections:
            return {"count": 0, "best": None, "targets": []}
        
        # Find best detection
        best = max(detections, key=lambda d: d['confidence'])
        
        # Find potential trash targets
        targets = [d for d in detections if d.get('label', '').lower() in self.precision_targets]
        
        return {
            "count": len(detections),
            "best": best,
            "targets": targets,
            "needs_precision": len(targets) > 0 and best.get('confidence', 0) > self.qwen_confidence_threshold
        }
    
    def should_use_qwen(self, detections, manual_override=False):
        """Determine if Qwen precision should be used"""
        if manual_override:
            return True
        
        if not self.enable_qwen or not self.qwen_prompt:
            return False
        
        summary = self.get_detection_summary(detections)
        return summary['needs_precision']

def test_hybrid_controller():
    """Test the hybrid AI controller"""
    print("=== Testing Hybrid AI Controller ===")
    
    controller = HybridAIController(enable_qwen=True)
    
    # Test scenarios
    test_cases = [
        {
            'name': 'Clear path',
            'detections': [],
            'expect_rule': True
        },
        {
            'name': 'Object centered (cup)',
            'detections': [{
                'label': 'cup',
                'confidence': 0.8,
                'bbox_norm': {'cx': 0.5, 'cy': 0.5, 'w': 0.25, 'h': 0.3}
            }],
            'expect_rule': False  # Should use Qwen for cup
        },
        {
            'name': 'Object left (person)',
            'detections': [{
                'label': 'person',
                'confidence': 0.9,
                'bbox_norm': {'cx': 0.2, 'cy': 0.5, 'w': 0.3, 'h': 0.6}
            }],
            'expect_rule': True  # Person navigation, use rules
        },
        {
            'name': 'Close object (trash)',
            'detections': [{
                'label': 'trash',
                'confidence': 0.7,
                'bbox_norm': {'cx': 0.5, 'cy': 0.5, 'w': 0.8, 'h': 0.6}
            }],
            'expect_rule': False  # Too close, use Qwen precision
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['name']}")
        decision = controller.make_decision(
            test_case['detections'],
            force_precision=False
        )
        
        expected_mode = 'qwen_precision' if not test_case['expect_rule'] else 'rule_based'
        actual_mode = decision.get('mode', 'unknown')
        
        status = "✅" if expected_mode in actual_mode else "❌"
        print(f"  {status} Mode: {actual_mode}")
        print(f"  Action: {decision.get('action')}")
        print(f"  Reason: {decision.get('reason')}")
        print(f"  Command: {decision.get('command')}")
    
    print("\n=== Hybrid Controller Test Complete ===")

if __name__ == "__main__":
    test_hybrid_controller()