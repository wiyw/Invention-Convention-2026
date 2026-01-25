#!/usr/bin/env python3
"""
Arduino UNO Q4GB AI Robot Testing Suite
Tests and validates on-device AI performance
"""

import time
import serial
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse
import sys

class AITestSuite:
    def __init__(self, port='COM3', baud_rate=115200):
        self.arduino = None
        self.port = port
        self.baud_rate = baud_rate
        self.connected = False
        self.test_results = {
            'detection_accuracy': [],
            'response_times': [],
            'memory_usage': [],
            'safety_reactions': [],
            'decision_consistency': []
        }
    
    def connect(self):
        """Connect to Arduino"""
        try:
            self.arduino = serial.Serial(self.port, self.baud_rate, timeout=3)
            time.sleep(2)
            self.connected = True
            print(f"Connected to Arduino UNO Q4GB on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            self.connected = False
    
    def send_command(self, command, wait_time=1.0):
        """Send command and wait for response"""
        if not self.connected:
            return None
        
        try:
            self.arduino.write((command + '\n').encode('utf-8'))
            self.arduino.flush()
            time.sleep(wait_time)
            
            # Read all available responses
            responses = []
            start_time = time.time()
            
            while time.time() - start_time < 2.0:  # 2 second timeout
                if self.arduino.in_waiting:
                    line = self.arduino.readline().decode('utf-8').strip()
                    if line:
                        responses.append(line)
                time.sleep(0.1)
            
            return responses
        
        except Exception as e:
            print(f"Error sending command: {e}")
            return None
    
    def test_detection_accuracy(self):
        """Test AI detection accuracy with simulated scenarios"""
        print("\n=== Testing Detection Accuracy ===")
        
        scenarios = [
            ("CLEAR_PATH", 0, "Clear path expected"),
            ("OBJECT_FRONT", 1, "Single object expected"),
            ("OBJECT_LEFT", 1, "Left object expected"),
            ("OBJECT_RIGHT", 1, "Right object expected"),
            ("MULTIPLE_OBJECTS", 2, "Multiple objects expected")
        ]
        
        accuracy_scores = []
        
        for scenario, expected_detections, description in scenarios:
            print(f"Testing {description}...")
            
            responses = self.send_command(f"TEST_{scenario}", wait_time=3)
            if responses:
                # Parse detection count from responses
                detected_count = 0
                for line in responses:
                    if "Detections=" in line:
                        try:
                            detected_count = int(line.split("Detections=")[1].split(",")[0])
                        except:
                            pass
                        break
                
                # Calculate accuracy
                if detected_count == expected_detections:
                    accuracy = 100.0
                elif abs(detected_count - expected_detections) <= 1:
                    accuracy = 75.0
                else:
                    accuracy = 0.0
                
                accuracy_scores.append(accuracy)
                print(f"  Expected: {expected_detections}, Detected: {detected_count}, Accuracy: {accuracy}%")
            else:
                print(f"  No response for {scenario}")
                accuracy_scores.append(0.0)
        
        avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0
        self.test_results['detection_accuracy'] = accuracy_scores
        print(f"Average Detection Accuracy: {avg_accuracy:.1f}%")
        
        return avg_accuracy
    
    def test_response_time(self):
        """Test AI response time"""
        print("\n=== Testing Response Time ===")
        
        response_times = []
        
        for i in range(10):
            print(f"Response test {i+1}/10...")
            
            start_time = time.time()
            responses = self.send_command("QUICK_TEST", wait_time=1)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            response_times.append(response_time)
            
            print(f"  Response time: {response_time:.1f}ms")
            time.sleep(0.5)  # Brief pause between tests
        
        avg_time = np.mean(response_times)
        max_time = np.max(response_times)
        
        self.test_results['response_times'] = response_times
        print(f"Average Response Time: {avg_time:.1f}ms")
        print(f"Maximum Response Time: {max_time:.1f}ms")
        
        return avg_time
    
    def test_memory_usage(self):
        """Test memory usage during AI operations"""
        print("\n=== Testing Memory Usage ===")
        
        memory_readings = []
        
        # Test memory usage under different loads
        test_commands = [
            "MEMORY_TEST_LIGHT",
            "MEMORY_TEST_MEDIUM", 
            "MEMORY_TEST_HEAVY"
        ]
        
        for cmd in test_commands:
            print(f"Testing {cmd}...")
            responses = self.send_command(cmd, wait_time=2)
            
            for line in responses:
                if "Memory:" in line:
                    try:
                        # Parse memory usage percentage
                        mem_str = line.split("(")[1].split(")")[0]
                        usage = float(mem_str.replace("%", ""))
                        memory_readings.append(usage)
                        print(f"  Memory usage: {usage}%")
                    except:
                        print(f"  Could not parse memory usage from: {line}")
            
            time.sleep(1)
        
        avg_memory = np.mean(memory_readings) if memory_readings else 0
        max_memory = np.max(memory_readings) if memory_readings else 0
        
        self.test_results['memory_usage'] = memory_readings
        print(f"Average Memory Usage: {avg_memory:.1f}%")
        print(f"Peak Memory Usage: {max_memory:.1f}%")
        
        return avg_memory
    
    def test_safety_reactions(self):
        """Test safety system reactions"""
        print("\n=== Testing Safety Reactions ===")
        
        safety_tests = [
            ("CRITICAL_OBSTACLE", "STOP", "Critical obstacle should trigger stop"),
            ("EMERGENCY_STOP", "STOP", "Emergency stop command"),
            ("SENSOR_FAILURE", "SAFE", "Sensor failure should trigger safe mode")
        ]
        
        safety_results = []
        
        for test, expected_action, description in safety_tests:
            print(f"Testing: {description}")
            
            responses = self.send_command(test, wait_time=2)
            safety_passed = False
            
            for line in responses:
                if "Action=" in line:
                    action = line.split("Action=")[1].split(",")[0]
                    if action == expected_action:
                        safety_passed = True
                        print(f"  ✓ Correct action: {action}")
                    else:
                        print(f"  ✗ Expected {expected_action}, got {action}")
                    break
            
            if not safety_passed and not any("Action=" in line for line in responses):
                print("  ✗ No action detected")
            
            safety_results.append(1 if safety_passed else 0)
        
        safety_score = np.mean(safety_results) * 100 if safety_results else 0
        self.test_results['safety_reactions'] = safety_results
        print(f"Safety System Score: {safety_score:.1f}%")
        
        return safety_score
    
    def test_decision_consistency(self):
        """Test decision making consistency"""
        print("\n=== Testing Decision Consistency ===")
        
        # Run the same test multiple times and check consistency
        test_command = "CONSISTENCY_TEST"
        decisions = []
        
        for i in range(5):
            print(f"Consistency test {i+1}/5...")
            responses = self.send_command(test_command, wait_time=2)
            
            for line in responses:
                if "Action=" in line:
                    action = line.split("Action=")[1].split(",")[0]
                    decisions.append(action)
                    print(f"  Decision: {action}")
                    break
        
        # Calculate consistency
        if decisions:
            most_common = max(set(decisions), key=decisions.count)
            consistency = decisions.count(most_common) / len(decisions) * 100
        else:
            consistency = 0
            most_common = "NONE"
        
        self.test_results['decision_consistency'] = decisions
        print(f"Most common decision: {most_common}")
        print(f"Decision consistency: {consistency:.1f}%")
        
        return consistency
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*50)
        print("COMPREHENSIVE AI TEST REPORT")
        print("="*50)
        
        # Calculate overall scores
        detection_score = np.mean(self.test_results['detection_accuracy']) if self.test_results['detection_accuracy'] else 0
        response_score = max(0, 100 - (np.mean(self.test_results['response_times']) if self.test_results['response_times'] else 0) / 10)  # 100ms = 0%
        memory_score = max(0, 100 - (np.mean(self.test_results['memory_usage']) if self.test_results['memory_usage'] else 0))
        safety_score = np.mean(self.test_results['safety_reactions']) * 100 if self.test_results['safety_reactions'] else 0
        consistency_score = self.test_results['decision_consistency'] if isinstance(self.test_results['decision_consistency'], (int, float)) else 0
        
        overall_score = (detection_score + response_score + memory_score + safety_score + consistency_score) / 5
        
        print(f"Detection Accuracy:     {detection_score:.1f}%")
        print(f"Response Time Score:     {response_score:.1f}%")
        print(f"Memory Efficiency:      {memory_score:.1f}%")
        print(f"Safety System:          {safety_score:.1f}%")
        print(f"Decision Consistency:    {consistency_score:.1f}%")
        print(f"\nOVERALL AI SCORE:       {overall_score:.1f}%")
        
        # Performance classification
        if overall_score >= 90:
            grade = "EXCELLENT"
        elif overall_score >= 80:
            grade = "GOOD"
        elif overall_score >= 70:
            grade = "ACCEPTABLE"
        elif overall_score >= 60:
            grade = "NEEDS IMPROVEMENT"
        else:
            grade = "POOR"
        
        print(f"Performance Grade:       {grade}")
        
        # Save report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'detection_score': detection_score,
            'response_score': response_score,
            'memory_score': memory_score,
            'safety_score': safety_score,
            'consistency_score': consistency_score,
            'overall_score': overall_score,
            'grade': grade,
            'detailed_results': self.test_results
        }
        
        with open(f"ai_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nDetailed report saved to: ai_test_report_*.json")
        
        return overall_score
    
    def run_all_tests(self):
        """Run complete test suite"""
        if not self.connect():
            print("Failed to connect to Arduino")
            return False
        
        try:
            print("Starting comprehensive AI test suite...")
            print("This will test all aspects of on-device AI performance.\n")
            
            # Run all tests
            self.test_detection_accuracy()
            self.test_response_time()
            self.test_memory_usage()
            self.test_safety_reactions()
            self.test_decision_consistency()
            
            # Generate final report
            overall_score = self.generate_report()
            
            return overall_score
        
        except KeyboardInterrupt:
            print("\nTest suite interrupted by user")
            return False
        except Exception as e:
            print(f"\nTest suite error: {e}")
            return False
        finally:
            self.disconnect()
    
    def plot_results(self):
        """Create visual plots of test results"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Arduino UNO Q4GB AI Performance Test Results', fontsize=16)
            
            # Detection accuracy
            if self.test_results['detection_accuracy']:
                axes[0, 0].bar(range(len(self.test_results['detection_accuracy'])), 
                             self.test_results['detection_accuracy'])
                axes[0, 0].set_title('Detection Accuracy by Scenario')
                axes[0, 0].set_ylabel('Accuracy (%)')
                axes[0, 0].set_ylim(0, 100)
            
            # Response times
            if self.test_results['response_times']:
                axes[0, 1].plot(self.test_results['response_times'], 'o-')
                axes[0, 1].set_title('Response Time Variability')
                axes[0, 1].set_ylabel('Time (ms)')
                axes[0, 1].set_xlabel('Test iteration')
            
            # Memory usage
            if self.test_results['memory_usage']:
                axes[0, 2].bar(range(len(self.test_results['memory_usage'])),
                             self.test_results['memory_usage'])
                axes[0, 2].set_title('Memory Usage Under Load')
                axes[0, 2].set_ylabel('Memory Usage (%)')
                axes[0, 2].set_ylim(0, 100)
            
            # Safety reactions
            if self.test_results['safety_reactions']:
                safety_labels = ['Critical', 'Emergency', 'Sensor Failure']
                axes[1, 0].bar(safety_labels, [x * 100 for x in self.test_results['safety_reactions']])
                axes[1, 0].set_title('Safety System Performance')
                axes[1, 0].set_ylabel('Success Rate (%)')
                axes[1, 0].set_ylim(0, 100)
            
            # Decision consistency
            if isinstance(self.test_results['decision_consistency'], list):
                unique_decisions = list(set(self.test_results['decision_consistency']))
                decision_counts = [self.test_results['decision_consistency'].count(d) for d in unique_decisions]
                axes[1, 1].pie(decision_counts, labels=unique_decisions, autopct='%1.1f%%')
                axes[1, 1].set_title('Decision Distribution')
            
            # Overall score gauge
            overall_score = 0
            if self.test_results['detection_accuracy']:
                overall_score += np.mean(self.test_results['detection_accuracy'])
            if self.test_results['response_times']:
                overall_score += max(0, 100 - np.mean(self.test_results['response_times']) / 10)
            if self.test_results['memory_usage']:
                overall_score += max(0, 100 - np.mean(self.test_results['memory_usage']))
            if self.test_results['safety_reactions']:
                overall_score += np.mean(self.test_results['safety_reactions']) * 100
            overall_score /= 5
            
            # Create gauge chart
            theta = np.linspace(0, np.pi, 100)
            r = 0.3
            axes[1, 2].fill_between(theta, 0, r, color='lightgray')
            score_theta = np.pi * (1 - overall_score / 100)
            score_r = np.linspace(0, r, 50)
            score_theta_fill = np.linspace(score_theta, np.pi, 50)
            axes[1, 2].fill_between(score_theta_fill, 0, r, color='green' if overall_score >= 80 else 'orange' if overall_score >= 60 else 'red')
            axes[1, 2].set_xlim(-0.5, 0.5)
            axes[1, 2].set_ylim(0, 0.4)
            axes[1, 2].set_aspect('equal')
            axes[1, 2].set_title(f'Overall Score: {overall_score:.1f}%')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'ai_test_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=150)
            print(f"Performance plots saved to: ai_test_performance_*.png")
            
        except Exception as e:
            print(f"Error creating plots: {e}")

def main():
    parser = argparse.ArgumentParser(description='Arduino UNO Q4GB AI Test Suite')
    parser.add_argument('--port', default='COM3', help='Arduino serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--plot', action='store_true', help='Generate performance plots')
    
    args = parser.parse_args()
    
    test_suite = AITestSuite(args.port, args.baud)
    
    print("Arduino UNO Q4GB AI Robot Test Suite")
    print("=====================================")
    print("This comprehensive test suite validates:")
    print("- Detection accuracy")
    print("- Response time performance")
    print("- Memory usage efficiency")
    print("- Safety system reliability")
    print("- Decision consistency")
    
    try:
        overall_score = test_suite.run_all_tests()
        
        if args.plot:
            test_suite.plot_results()
        
        if overall_score >= 80:
            print(f"\n🎉 EXCELLENT! Overall AI performance: {overall_score:.1f}%")
        elif overall_score >= 70:
            print(f"\n✅ GOOD! Overall AI performance: {overall_score:.1f}%")
        elif overall_score >= 60:
            print(f"\n⚠️  ACCEPTABLE. Overall AI performance: {overall_score:.1f}%")
        else:
            print(f"\n❌ NEEDS IMPROVEMENT. Overall AI performance: {overall_score:.1f}%")
    
    except KeyboardInterrupt:
        print("\nTest suite interrupted")
    except Exception as e:
        print(f"\nTest suite error: {e}")

if __name__ == "__main__":
    main()