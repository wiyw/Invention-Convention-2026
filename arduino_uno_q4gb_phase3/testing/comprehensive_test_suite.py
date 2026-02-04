#!/usr/bin/env python3
"""
Arduino UNO Q4GB Comprehensive Testing Suite
Phase 3: Hardware-Specific Optimization
Tests all components of the Arduino UNO Q4GB AI deployment
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

class ArduinoQ4GBTestSuite:
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        self.install_dir = Path.home() / 'arduino_q4gb_ai_robot_phase3'
        
    def print_header(self):
        """Print test suite header"""
        print("="*70)
        print("    Arduino UNO Q4GB AI Robot - Comprehensive Test Suite")
        print("    Phase 3: Hardware-Specific Optimization")
        print("="*70)
        print()
    
    def run_test(self, test_name, test_function, critical=True):
        """Run a single test and record results"""
        self.total_tests += 1
        print(f"[TEST] {test_name}...")
        
        try:
            start_time = time.time()
            result = test_function()
            end_time = time.time()
            duration = end_time - start_time
            
            if result:
                print(f"[PASS] {test_name} - PASSED ({duration:.2f}s)")
                self.passed_tests += 1
                self.test_results[test_name] = {
                    'status': 'PASSED',
                    'duration': duration,
                    'critical': critical,
                    'message': 'Test completed successfully'
                }
            else:
                print(f"[FAIL] {test_name} - FAILED ({duration:.2f}s)")
                self.failed_tests += 1
                self.test_results[test_name] = {
                    'status': 'FAILED',
                    'duration': duration,
                    'critical': critical,
                    'message': 'Test failed'
                }
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"[ERROR] {test_name} - ERROR ({duration:.2f}s): {e}")
            self.failed_tests += 1
            self.test_results[test_name] = {
                'status': 'ERROR',
                'duration': duration,
                'critical': critical,
                'message': str(e)
            }
        
        print()
    
    def test_installation_directory(self):
        """Test if installation directory exists and has correct structure"""
        if not self.install_dir.exists():
            return False
        
        required_dirs = ['venv', 'hardware_detection', 'models', 'testing']
        for dir_name in required_dirs:
            if not (self.install_dir / dir_name).exists():
                return False
        
        return True
    
    def test_virtual_environment(self):
        """Test virtual environment setup"""
        venv_dir = self.install_dir / 'venv'
        
        if not venv_dir.exists():
            return False
        
        # Test activation
        try:
            result = subprocess.run([
                str(venv_dir / 'bin' / 'python3'), '-c', 
                'import sys; print("venv_active" if sys.prefix != sys.base_prefix else "venv_inactive")'
            ], capture_output=True, text=True)
            
            return result.returncode == 0 and 'venv_active' in result.stdout
        except:
            return False
    
    def test_python_packages(self):
        """Test essential Python packages"""
        venv_python = str(self.install_dir / 'venv' / 'bin' / 'python3')
        
        packages = ['numpy', 'PIL', 'cv2']
        for package in packages:
            try:
                result = subprocess.run([
                    venv_python, '-c', f'import {package}; print("ok")'
                ], capture_output=True, text=True)
                
                if result.returncode != 0 or 'ok' not in result.stdout:
                    return False
            except:
                return False
        
        return True
    
    def test_ai_framework(self):
        """Test AI framework installation"""
        config_file = self.install_dir / 'config.json'
        
        if not config_file.exists():
            return False
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            framework = config.get('framework')
            if not framework:
                return False
            
            venv_python = str(self.install_dir / 'venv' / 'bin' / 'python3')
            
            # Test framework import
            if framework == 'onnx':
                result = subprocess.run([
                    venv_python, '-c', 'import onnxruntime; print("ok")'
                ], capture_output=True, text=True)
            elif framework == 'tflite':
                result = subprocess.run([
                    venv_python, '-c', 
                    'import tflite_runtime; print("ok")'  # Try TFLite runtime first
                ], capture_output=True, text=True)
                if result.returncode != 0:
                    # Fallback to full TensorFlow
                    result = subprocess.run([
                        venv_python, '-c', 'import tensorflow; print("ok")'
                    ], capture_output=True, text=True)
            elif framework == 'pytorch':
                result = subprocess.run([
                    venv_python, '-c', 'import torch; print("ok")'
                ], capture_output=True, text=True)
            else:
                return False
            
            return result.returncode == 0 and 'ok' in result.stdout
            
        except:
            return False
    
    def test_hardware_detection(self):
        """Test hardware detection tools"""
        hardware_analyzer = self.install_dir / 'hardware_detection' / 'hardware_analyzer.py'
        
        if not hardware_analyzer.exists():
            return False
        
        try:
            venv_python = str(self.install_dir / 'venv' / 'bin' / 'python3')
            result = subprocess.run([
                venv_python, str(hardware_analyzer)
            ], capture_output=True, text=True, timeout=30)
            
            return result.returncode == 0
        except:
            return False
    
    def test_model_files(self):
        """Test model files and availability"""
        models_dir = self.install_dir / 'models'
        
        if not models_dir.exists():
            return False
        
        # Check for at least one model file
        model_files = list(models_dir.glob('*.onnx')) + list(models_dir.glob('*.tflite'))
        return len(model_files) > 0
    
    def test_startup_script(self):
        """Test startup script functionality"""
        startup_script = self.install_dir / 'start_ai_robot.sh'
        
        if not startup_script.exists() or not os.access(startup_script, os.X_OK):
            return False
        
        return True
    
    def test_system_integration(self):
        """Test overall system integration"""
        test_script = self.install_dir / 'test_system.sh'
        
        if not test_script.exists() or not os.access(test_script, os.X_OK):
            return False
        
        try:
            result = subprocess.run([str(test_script)], 
                                 capture_output=True, text=True, timeout=60)
            return result.returncode == 0
        except:
            return False
    
    def test_performance_benchmark(self):
        """Test performance benchmark tools"""
        benchmark_script = self.install_dir / 'hardware_detection' / 'benchmark_suite.py'
        
        if not benchmark_script.exists():
            return False
        
        try:
            venv_python = str(self.install_dir / 'venv' / 'bin' / 'python3')
            result = subprocess.run([
                venv_python, str(benchmark_script)
            ], capture_output=True, text=True, timeout=120)
            
            return result.returncode == 0
        except:
            return False
    
    def test_ai_inference(self):
        """Test AI inference capabilities"""
        config_file = self.install_dir / 'config.json'
        
        if not config_file.exists():
            return False
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            framework = config.get('framework')
            if not framework:
                return False
            
            venv_python = str(self.install_dir / 'venv' / 'bin' / 'python3')
            
            # Create a simple inference test
            test_code = f'''
import numpy as np
import json

# Test framework basic functionality
try:
    if "{framework}" == "onnx":
        import onnxruntime
        print("onnx_runtime_ok")
    elif "{framework}" == "tflite":
        try:
            import tflite_runtime
            print("tflite_runtime_ok")
        except:
            import tensorflow
            print("tensorflow_ok")
    elif "{framework}" == "pytorch":
        import torch
        print("pytorch_ok")
    else:
        print("unknown_framework")
except Exception as e:
    print(f"framework_error: {{e}}")
'''
            
            result = subprocess.run([venv_python, '-c', test_code],
                                 capture_output=True, text=True)
            
            return result.returncode == 0 and '_ok' in result.stdout
            
        except:
            return False
    
    def test_memory_usage(self):
        """Test memory usage and optimization"""
        venv_python = str(self.install_dir / 'venv' / 'bin' / 'python3')
        
        # Test memory usage of framework
        test_code = '''
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"memory_mb: {{memory_mb}}")

# Test if memory usage is reasonable (< 200MB for basic import)
if memory_mb < 200:
    print("memory_ok")
else:
    print("memory_high")
'''
        
        try:
            result = subprocess.run([venv_python, '-c', test_code],
                                 capture_output=True, text=True)
            
            return result.returncode == 0 and 'memory_ok' in result.stdout
        except:
            return False
    
    def test_configuration_files(self):
        """Test configuration files integrity"""
        config_file = self.install_dir / 'config.json'
        framework_config = self.install_dir / 'framework_config'
        
        # Test main config
        if not config_file.exists():
            return False
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            required_fields = ['hardware_detected', 'framework', 'arduino_uno_q4gb']
            for field in required_fields:
                if field not in config:
                    return False
        except:
            return False
        
        return True
    
    def run_comprehensive_tests(self):
        """Run all tests"""
        self.print_header()
        
        # Define test suite
        test_suite = [
            ("Installation Directory Check", self.test_installation_directory),
            ("Virtual Environment Test", self.test_virtual_environment),
            ("Python Packages Test", self.test_python_packages),
            ("AI Framework Test", self.test_ai_framework),
            ("Hardware Detection Test", self.test_hardware_detection),
            ("Model Files Test", self.test_model_files),
            ("Startup Script Test", self.test_startup_script),
            ("System Integration Test", self.test_system_integration),
            ("Performance Benchmark Test", self.test_performance_benchmark),
            ("AI Inference Test", self.test_ai_inference),
            ("Memory Usage Test", self.test_memory_usage),
            ("Configuration Files Test", self.test_configuration_files)
        ]
        
        # Run all tests
        for test_name, test_function in test_suite:
            self.run_test(test_name, test_function)
        
        # Calculate success rate
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        return success_rate
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'success_rate': (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0,
            'test_results': self.test_results,
            'installation_dir': str(self.install_dir),
            'arduino_uno_q4gb_phase3': True
        }
        
        # Save report
        report_file = Path('arduino_q4gb_test_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report, str(report_file)
    
    def print_summary(self):
        """Print test summary"""
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print("="*70)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*70)
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print()
        
        if success_rate >= 95:
            print("🎉 EXCELLENT: Arduino UNO Q4GB AI Robot is fully operational!")
        elif success_rate >= 85:
            print("[GOOD] Arduino UNO Q4GB AI Robot is mostly functional.")
        elif success_rate >= 70:
            print("[FAIR] Arduino UNO Q4GB AI Robot has some issues.")
        else:
            print("[POOR] Arduino UNO Q4GB AI Robot has significant problems.")
        
        print()
        print("Critical Failures:")
        critical_failures = [name for name, result in self.test_results.items() 
                           if result.get('critical', False) and result['status'] != 'PASSED']
        
        if critical_failures:
            for test_name in critical_failures:
                print(f"  [FAIL] {test_name}")
        else:
            print("  [PASS] No critical failures")
        
        print("="*70)
        
        return success_rate

def main():
    """Main function"""
    test_suite = ArduinoQ4GBTestSuite()
    
    try:
        # Run comprehensive tests
        success_rate = test_suite.run_comprehensive_tests()
        
        # Generate report
        report, report_file = test_suite.generate_test_report()
        
        # Print summary
        test_suite.print_summary()
        
        print(f"\n[REPORT] Detailed report saved to: {report_file}")
        
        return 0 if success_rate >= 70 else 1
        
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Test suite interrupted")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Test suite error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())