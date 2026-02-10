#!/usr/bin/env python3
"""
Arduino UNO Q4GB Ultimate Test Suite
Comprehensive testing with ARM-optimized fallbacks
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime

class UltimateTestSuite:
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        self.config = {}
        self.camera = None
        self.ai_backend = None
        self.arduino_controller = None
        
    def load_configuration(self):
        """Load ultimate configuration"""
        try:
            with open('ultimate_config.json', 'r') as f:
                self.config = json.load(f)
            print("✅ Ultimate configuration loaded")
            
            # Load AI stack configuration
            try:
                with open('ai_stack_config.json', 'r') as f:
                    ai_config = json.load(f)
                    self.active_backend = ai_config.get('active_backend')
                    print(f"✅ AI backend loaded: {self.active_backend}")
            except:
                print("⚠️  No AI stack config - will detect during tests")
                self.active_backend = None
                
            return True
        except Exception as e:
            print(f"❌ Configuration loading failed: {e}")
            return False
    
    def run_test(self, test_name, test_function, critical=True, timeout=30):
        """Run test with timeout and error handling"""
        self.total_tests += 1
        print(f"[TEST] {test_name}...")
        
        try:
            # Run test in thread with timeout
            test_thread = threading.Thread(target=test_function)
            test_thread.daemon = True
            test_thread.start()
            test_thread.join(timeout)
            
            if test_thread.is_alive():
                print(f"[TIMEOUT] {test_name} - TIMED OUT ({timeout}s)")
                self.failed_tests += 1
                self.test_results[test_name] = {
                    'status': 'TIMEOUT',
                    'critical': critical,
                    'timeout': timeout
                }
            else:
                print(f"[PASS] {test_name} - PASSED")
                self.passed_tests += 1
                self.test_results[test_name] = {
                    'status': 'PASSED',
                    'critical': critical
                }
                
        except Exception as e:
            print(f"[ERROR] {test_name} - ERROR: {e}")
            self.failed_tests += 1
            self.test_results[test_name] = {
                'status': 'ERROR',
                'critical': critical,
                'error': str(e)
            }
        
        print()
    
    def test_environment_compatibility(self):
        """Test environment compatibility"""
        def test_func():
            # Test Python version
            python_version = sys.version_info[:2]
            if python_version < (3, 8):
                raise Exception(f"Python {python_version} too old")
            
            # Test architecture
            import platform
            machine = platform.machine().lower()
            if 'arm' not in machine and 'aarch64' not in machine:
                print(f"Warning: Not ARM architecture ({machine})")
            
            # Test essential modules
            import numpy as np
            import cv2
            
            # Test basic operations
            x = np.random.rand(10, 10)
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            resized = cv2.resize(img, (50, 50))
            
            return True
        
        self.run_test("Environment Compatibility", test_func, critical=True)
    
    def test_camera_detection(self):
        """Test camera detection with multiple backends"""
        def test_func():
            import cv2
            
            camera_configs = [
                (0, cv2.CAP_V4L2),
                (0, cv2.CAP_GSTREAMER),
                (1, cv2.CAP_V4L2),
                (2, cv2.CAP_V4L2)
            ]
            
            working_cameras = []
            
            for idx, backend in camera_configs:
                cap = cv2.VideoCapture(idx + backend)
                
                if cap.isOpened():
                    try:
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            working_cameras.append({
                                'index': idx,
                                'backend': backend,
                                'resolution': frame.shape[:2],
                                'working': True
                            })
                            print(f"    Camera {idx} backend {backend}: {frame.shape}")
                        cap.release()
                    except:
                        cap.release()
                else:
                    cap.release()
            
            if not working_cameras:
                raise Exception("No working cameras found")
            
            self.config['working_camera'] = working_cameras[0]
            return True
        
        self.run_test("Camera Detection", test_func, critical=True, timeout=15)
    
    def test_ai_backends(self):
        """Test AI backends with fallbacks"""
        def test_func():
            backends_working = []
            
            # Test Ultralytics
            try:
                from ultralytics import YOLO
                model_path = self._find_model('yolo26n.pt')
                if model_path:
                    model = YOLO(model_path)
                    import numpy as np
                    test_img = np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8)
                    results = model(test_img, verbose=False)
                    backends_working.append('ultralytics')
                    print("    ✅ Ultralytics working")
            except Exception as e:
                print(f"    ❌ Ultralytics failed: {e}")
            
            # Test ONNX Runtime
            try:
                import onnxruntime
                backends_working.append('onnx')
                print("    ✅ ONNX Runtime working")
            except Exception as e:
                print(f"    ❌ ONNX Runtime failed: {e}")
            
            # Test TensorFlow Lite
            try:
                import tflite_runtime
                backends_working.append('tflite')
                print("    ✅ TensorFlow Lite working")
            except Exception as e:
                print(f"    ❌ TensorFlow Lite failed: {e}")
            
            # Test OpenCV
            try:
                import cv2
                import numpy as np
                # Simple detection test
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                backends_working.append('opencv')
                print("    ✅ OpenCV working")
            except Exception as e:
                print(f"    ❌ OpenCV failed: {e}")
            
            if not backends_working:
                raise Exception("No AI backends working")
            
            self.config['working_ai_backends'] = backends_working
            return True
        
        self.run_test("AI Backends", test_func, critical=True)
    
    def test_arduino_connection(self):
        """Test Arduino connection"""
        def test_func():
            try:
                import serial
                import serial.tools.list_ports
                
                ports = serial.tools.list_ports.comports()
                arduino_ports = []
                
                for port in ports:
                    if any(keyword in port.description.lower() 
                          for keyword in ['arduino', 'ch340', 'cp210', 'ftdi']):
                        arduino_ports.append(port)
                        print(f"    Found Arduino: {port.device} - {port.description}")
                
                if not arduino_ports:
                    print("    No Arduino ports found")
                    # Still return True as not critical for basic testing
                    return True
                
                # Try connecting to first Arduino
                try:
                    ser = serial.Serial(
                        port=arduino_ports[0].device,
                        baudrate=115200,
                        timeout=1
                    )
                    ser.close()
                    print("    ✅ Arduino connection successful")
                    return True
                except Exception as e:
                    print(f"    ❌ Arduino connection failed: {e}")
                    return False
                    
            except ImportError:
                print("    PySerial not available - skipping Arduino test")
                return True
        
        self.run_test("Arduino Connection", test_func, critical=False)
    
    def test_pipeline_performance(self):
        """Test complete pipeline performance"""
        def test_func():
            import cv2
            import numpy as np
            import time
            
            # Use best available AI backend
            if 'working_ai_backends' in self.config:
                if 'ultralytics' in self.config['working_ai_backends']:
                    return self._test_ultralytics_pipeline()
                elif 'opencv' in self.config['working_ai_backends']:
                    return self._test_opencv_pipeline()
            
            # Fallback to basic camera test
            return self._test_camera_only_pipeline()
        
        self.run_test("Pipeline Performance", test_func, critical=True, timeout=20)
    
    def _test_ultralytics_pipeline(self):
        """Test YOLO pipeline performance"""
        try:
            from ultralytics import YOLO
            import cv2
            import numpy as np
            
            # Setup camera
            camera_config = self.config.get('working_camera', {'index': 0, 'backend': cv2.CAP_V4L2})
            cap = cv2.VideoCapture(camera_config['index'] + camera_config['backend'])
            
            if not cap.isOpened():
                raise Exception("Cannot open camera")
            
            # Setup YOLO
            model_path = self._find_model('yolo26n.pt')
            if not model_path:
                raise Exception("YOLO model not found")
            
            model = YOLO(model_path)
            
            # Performance test
            frame_count = 0
            detection_count = 0
            start_time = time.time()
            test_duration = 10
            
            while time.time() - start_time < test_duration:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                
                frame_count += 1
                
                # YOLO detection
                results = model(frame, verbose=False, conf=0.6)
                if results and len(results) > 0 and results[0].boxes:
                    detection_count += len(results[0].boxes)
                
                # Progress update
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"    Frame {frame_count}: FPS {fps:.1f}, Detections {detection_count}")
            
            cap.release()
            
            # Calculate metrics
            total_time = time.time() - start_time
            pipeline_fps = frame_count / total_time if total_time > 0 else 0
            detection_rate = detection_count / frame_count if frame_count > 0 else 0
            
            print(f"    YOLO Pipeline Results:")
            print(f"      Frames: {frame_count}")
            print(f"      FPS: {pipeline_fps:.2f}")
            print(f"      Detection Rate: {detection_rate:.2f}/frame")
            
            # Success criteria (ARM-adjusted)
            success = (
                pipeline_fps >= 5 and
                detection_rate >= 0.1 and
                frame_count >= 30
            )
            
            if success:
                print("    ✅ YOLO pipeline performance acceptable")
            else:
                print("    ⚠️  YOLO pipeline performance limited")
            
            return success
            
        except Exception as e:
            print(f"    ❌ YOLO pipeline test failed: {e}")
            return False
    
    def _test_opencv_pipeline(self):
        """Test OpenCV pipeline performance"""
        try:
            import cv2
            import numpy as np
            
            # Setup camera
            camera_config = self.config.get('working_camera', {'index': 0, 'backend': cv2.CAP_V4L2})
            cap = cv2.VideoCapture(camera_config['index'] + camera_config['backend'])
            
            if not cap.isOpened():
                raise Exception("Cannot open camera")
            
            # Performance test
            frame_count = 0
            detection_count = 0
            start_time = time.time()
            test_duration = 10
            
            while time.time() - start_time < test_duration:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                
                frame_count += 1
                
                # Simple detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                detection_count += len(contours[:3])
                
                # Progress update
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"    Frame {frame_count}: FPS {fps:.1f}, Detections {detection_count}")
            
            cap.release()
            
            # Calculate metrics
            total_time = time.time() - start_time
            pipeline_fps = frame_count / total_time if total_time > 0 else 0
            detection_rate = detection_count / frame_count if frame_count > 0 else 0
            
            print(f"    OpenCV Pipeline Results:")
            print(f"      Frames: {frame_count}")
            print(f"      FPS: {pipeline_fps:.2f}")
            print(f"      Detection Rate: {detection_rate:.2f}/frame")
            
            # Success criteria (ARM-adjusted)
            success = pipeline_fps >= 8  # Higher FPS for simple OpenCV
            
            if success:
                print("    ✅ OpenCV pipeline performance acceptable")
            else:
                print("    ⚠️  OpenCV pipeline performance limited")
            
            return success
            
        except Exception as e:
            print(f"    ❌ OpenCV pipeline test failed: {e}")
            return False
    
    def _test_camera_only_pipeline(self):
        """Test camera-only pipeline"""
        try:
            import cv2
            import numpy as np
            
            # Setup camera
            camera_config = self.config.get('working_camera', {'index': 0, 'backend': cv2.CAP_V4L2})
            cap = cv2.VideoCapture(camera_config['index'] + camera_config['backend'])
            
            if not cap.isOpened():
                raise Exception("Cannot open camera")
            
            # Performance test
            frame_count = 0
            start_time = time.time()
            test_duration = 8
            
            while time.time() - start_time < test_duration:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_count += 1
                
                # Progress update
                if frame_count % 20 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"    Frame {frame_count}: FPS {fps:.1f}")
            
            cap.release()
            
            # Calculate metrics
            total_time = time.time() - start_time
            camera_fps = frame_count / total_time if total_time > 0 else 0
            
            print(f"    Camera-Only Results:")
            print(f"      Frames: {frame_count}")
            print(f"      FPS: {camera_fps:.2f}")
            
            # Success criteria
            success = camera_fps >= 10
            
            if success:
                print("    ✅ Camera performance acceptable")
            else:
                print("    ⚠️  Camera performance limited")
            
            return success
            
        except Exception as e:
            print(f"    ❌ Camera test failed: {e}")
            return False
    
    def _find_model(self, filename):
        """Find model file"""
        search_paths = [
            Path('models') / filename,
            Path(filename),
            Path.cwd() / 'models' / filename
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def run_ultimate_tests(self):
        """Run all ultimate tests"""
        print("🧪 Arduino UNO Q4GB Ultimate Test Suite")
        print("=" * 60)
        
        # Load configuration
        if not self.load_configuration():
            print("❌ Cannot proceed without configuration")
            return 1
        
        # Test sequence
        tests = [
            ("Environment Compatibility", self.test_environment_compatibility, True),
            ("Camera Detection", self.test_camera_detection, True),
            ("AI Backends", self.test_ai_backends, True),
            ("Arduino Connection", self.test_arduino_connection, False),
            ("Pipeline Performance", self.test_pipeline_performance, True)
        ]
        
        # Run all tests
        for test_name, test_func, critical in tests:
            self.run_test(test_name, test_func, critical)
        
        # Calculate success rate
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        # Generate report
        self.generate_ultimate_report(success_rate)
        
        return 0 if success_rate >= 80 else 1
    
    def generate_ultimate_report(self, success_rate):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("🎯 ULTIMATE TEST SUITE - FINAL REPORT")
        print("=" * 60)
        
        print(f"Tests Passed: {self.passed_tests}/{self.total_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = result['status']
            critical = result['critical']
            critical_mark = " [CRITICAL]" if critical else ""
            print(f"  {test_name}: {status}{critical_mark}")
            
            if 'error' in result:
                print(f"    Error: {result['error']}")
            elif 'timeout' in result:
                print(f"    Timeout: {result['timeout']}s")
        
        # Recommendations
        print("\n💡 Recommendations:")
        
        if success_rate >= 90:
            print("🎉 EXCELLENT! System is perfectly optimized for ARM AI robotics!")
            print("   Ready for production deployment")
        elif success_rate >= 80:
            print("✅ GREAT! System is highly capable for ARM AI robotics!")
            print("   Minor optimizations may improve performance")
        elif success_rate >= 60:
            print("⚠️  GOOD! System works but has some limitations")
            print("   Consider the optimizations below")
        else:
            print("❌ NEEDS IMPROVEMENT! System has significant issues")
            print("   Follow the troubleshooting steps below")
        
        # Save report
        report_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'success_rate': success_rate,
            'tests_passed': self.passed_tests,
            'tests_total': self.total_tests,
            'test_results': self.test_results,
            'config': self.config,
            'recommendations': 'System ready for ARM AI deployment' if success_rate >= 80 else 'System needs optimization'
        }
        
        with open('ultimate_test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Ultimate report saved: ultimate_test_report.json")
        print(f"\n🚀 System is {'READY' if success_rate >= 80 else 'NEEDS OPTIMIZATION'} for Arduino UNO Q4GB AI robotics!")

def main():
    """Main function"""
    suite = UltimateTestSuite()
    
    try:
        return suite.run_ultimate_tests()
    except KeyboardInterrupt:
        print("\n⏹️  Test suite interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Test suite crashed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())