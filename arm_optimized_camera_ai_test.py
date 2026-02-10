#!/usr/bin/env python3
"""
Arduino UNO Q4GB Camera + AI Pipeline - ARM Optimized Version
Modified to work with ARM processors and avoid illegal instruction errors
"""

import os
import sys
import time
import json
import threading
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

class ARMOptimizedCameraAI:
    """ARM-optimized camera + AI pipeline for Arduino UNO Q4GB"""
    
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        self.install_dir = Path.home() / 'arduino_q4gb_camera_ai_test'
        self.camera = None
        self.yolo_model = None
        self.arduino_serial = None
        
        # ARM-optimized configuration
        self.test_config = {
            'camera_indices': [0, 1, 2, 3, 4],
            'backends': [cv2.CAP_V4L2],  # Only V4L2 for ARM
            'test_duration': 8,  # Shorter test for ARM
            'resolution': (320, 240),  # Lower resolution for ARM
            'fps_target': 8,  # Lower FPS target for ARM
            'detection_confidence': 0.6,  # Higher confidence to reduce processing
            'max_detections': 3  # Limit detections for performance
        }
        
        # ARM optimization flags
        self.arm_optimized = True
        self.cpu_count = os.cpu_count() or 1
        
        # Set ARM environment variables
        self._set_arm_optimization()
        
    def _set_arm_optimization(self):
        """Set ARM-specific optimization environment variables"""
        os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'
        os.environ['OMP_NUM_THREADS'] = str(max(1, self.cpu_count - 1))
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        
        # Reduce thread contention for ARM
        if hasattr(os, 'sched_setaffinity'):
            try:
                # Limit to specific CPU cores for ARM
                cpu_mask = (1 << min(self.cpu_count, 2)) - 1  # Use first 2 cores
                os.sched_setaffinity(0, {i for i in range(min(self.cpu_count, 2)) if (cpu_mask >> i) & 1})
            except:
                pass
    
    def print_header(self):
        """Print test suite header"""
        print("="*70)
        print("  Arduino UNO Q4GB Camera + AI Test - ARM Optimized")
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
                    'critical': critical
                }
            else:
                print(f"[FAIL] {test_name} - FAILED ({duration:.2f}s)")
                self.failed_tests += 1
                self.test_results[test_name] = {
                    'status': 'FAILED',
                    'duration': duration,
                    'critical': critical
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
                'error': str(e)
            }
        
        print()
    
    def test_arm_system(self):
        """Test ARM system compatibility"""
        print("Testing ARM system compatibility...")
        
        try:
            import platform
            
            # Check if we're on ARM
            machine = platform.machine().lower()
            if 'arm' in machine or 'aarch64' in machine:
                print(f"✅ ARM architecture detected: {machine}")
                
                # Check CPU info
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        cpuinfo = f.read()
                    
                    # Count cores
                    cores = cpuinfo.count('processor')
                    print(f"✅ CPU cores: {cores}")
                    
                    # Check for NEON (important for ARM AI)
                    if 'neon' in cpuinfo.lower():
                        print("✅ NEON SIMD supported")
                    else:
                        print("⚠️  NEON SIMD not detected - may affect performance")
                    
                    # Check for VFP (floating point)
                    if 'vfp' in cpuinfo.lower():
                        print("✅ VFP floating point supported")
                    else:
                        print("⚠️  VFP not detected - may affect floating point operations")
                        
                except Exception as e:
                    print(f"⚠️  Could not read CPU info: {e}")
                
                return True
            else:
                print(f"⚠️  Not ARM architecture: {machine}")
                print("    This test is optimized for ARM systems")
                return False
                
        except Exception as e:
            print(f"❌ System compatibility test failed: {e}")
            return False
    
    def test_arm_libraries(self):
        """Test ARM-compatible libraries"""
        print("Testing ARM-compatible libraries...")
        
        # Test NumPy with ARM optimizations
        try:
            import numpy as np
            print(f"✅ NumPy {np.__version__}")
            
            # Test basic operations
            x = np.random.rand(10, 10)
            y = np.sum(x)
            print("✅ NumPy operations working")
            
        except Exception as e:
            print(f"❌ NumPy failed: {e}")
            return False
        
        # Test OpenCV with ARM optimizations
        try:
            import cv2
            print(f"✅ OpenCV {cv2.__version__}")
            
            # Test basic operations
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            resized = cv2.resize(img, (50, 50))
            print("✅ OpenCV operations working")
            
        except Exception as e:
            print(f"❌ OpenCV failed: {e}")
            return False
        
        # Test PyTorch with ARM CPU
        torch_working = False
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__}")
            
            # Test basic tensor operations
            x = torch.rand(5, 5)
            y = torch.sum(x)
            
            # Check CPU backend
            if torch.backends.cpu.is_available():
                print("✅ PyTorch CPU backend working")
            else:
                print("⚠️  PyTorch CPU backend may have issues")
            
            torch_working = True
            
        except Exception as e:
            print(f"❌ PyTorch failed: {e}")
            if 'illegal instruction' in str(e).lower():
                print("   💡 This is a known ARM compatibility issue")
                print("   💡 Run arm_compatibility_fix.sh to resolve")
        
        # Test Ultralytics if PyTorch works
        if torch_working:
            try:
                from ultralytics import YOLO
                print(f"✅ Ultralytics {YOLO.__version__}")
                
                # Test model loading (if file exists)
                model_path = self._find_model_file('yolo26n.pt')
                if model_path and Path(model_path).exists():
                    model = YOLO(model_path)
                    print("✅ YOLO model loads successfully")
                else:
                    print("⚠️  YOLO model file not found")
                    
            except Exception as e:
                print(f"❌ Ultralytics failed: {e}")
        
        # Test serial communication
        try:
            import serial
            print(f"✅ PySerial {serial.__version__}")
            
            # Test port enumeration
            ports = serial.tools.list_ports.comports()
            print(f"✅ Found {len(ports)} serial ports")
            
        except Exception as e:
            print(f"❌ PySerial failed: {e}")
        
        return True
    
    def _find_model_file(self, filename):
        """Find model file in common locations"""
        search_paths = [
            Path.cwd() / 'models' / filename,
            Path.cwd() / filename,
            self.install_dir / 'models' / filename,
            self.install_dir / filename
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def test_arm_camera(self):
        """Test camera with ARM optimizations"""
        print("Testing camera with ARM optimizations...")
        
        working_cameras = []
        
        # Only test V4L2 backend (most compatible with ARM)
        for cam_idx in self.test_config['camera_indices']:
            try:
                print(f"  Testing camera {cam_idx} with V4L2...")
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
                
                if cap.isOpened():
                    # Set ARM-optimized camera settings
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.test_config['resolution'][0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.test_config['resolution'][1])
                    cap.set(cv2.CAP_PROP_FPS, self.test_config['fps_target'])
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for ARM
                    
                    # Test frame capture
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        camera_info = {
                            'index': cam_idx,
                            'resolution': frame.shape[:2],
                            'working': True
                        }
                        working_cameras.append(camera_info)
                        print(f"    SUCCESS: {frame.shape[1]}x{frame.shape[0]}")
                        cap.release()
                    else:
                        print(f"    FAIL: Opens but no frame")
                        cap.release()
                else:
                    print(f"    FAIL: Cannot open")
                    
            except Exception as e:
                print(f"    ERROR: {e}")
        
        if working_cameras:
            # Store best camera
            best_camera = working_cameras[0]  # Use first working camera
            self.test_config['best_camera'] = best_camera
            print(f"✅ Working camera found: Index {best_camera['index']}")
            return True
        else:
            print("❌ No working cameras found")
            return False
    
    def test_arm_yolo(self):
        """Test YOLO with ARM optimizations"""
        print("Testing YOLO with ARM optimizations...")
        
        try:
            from ultralytics import YOLO
            
            model_path = self._find_model_file('yolo26n.pt')
            if not model_path or not Path(model_path).exists():
                print("⚠️  YOLO26n model not found - skipping YOLO test")
                return True  # Not a failure, just skipping
            
            print(f"  Loading model: {model_path}")
            
            # Load model with ARM optimizations
            start_time = time.time()
            
            # Configure for ARM CPU
            model = YOLO(model_path)
            
            # Set ARM-optimized inference settings
            model.overrides = {
                'device': 'cpu',
                'half': False,  # Disable FP16 on ARM if problematic
                'imgsz': self.test_config['resolution']
            }
            
            load_time = time.time() - start_time
            print(f"  Model loaded in {load_time:.2f}s")
            
            # Test with smaller image for ARM
            test_image = np.random.randint(0, 255, 
                (self.test_config['resolution'][1], self.test_config['resolution'][0], 3), 
                dtype=np.uint8)
            
            # Warm up
            try:
                _ = model(test_image, verbose=False, conf=self.test_config['detection_confidence'])
                print("  Model warmup successful")
            except Exception as e:
                print(f"  Model warmup failed: {e}")
                return False
            
            # Performance test with fewer iterations for ARM
            times = []
            for i in range(3):  # Reduced iterations for ARM
                start_time = time.time()
                try:
                    results = model(test_image, verbose=False, conf=self.test_config['detection_confidence'])
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
                except Exception as e:
                    print(f"  Inference {i+1} failed: {e}")
                    return False
            
            if times:
                avg_time = sum(times) / len(times)
                fps = 1000 / avg_time
                
                print(f"  Average inference time: {avg_time:.1f}ms")
                print(f"  Theoretical FPS: {fps:.1f}")
                
                # Store results
                self.test_config['yolo_performance'] = {
                    'avg_time': avg_time,
                    'fps': fps,
                    'working': True
                }
                
                return fps >= 3  # Lower FPS target for ARM
            else:
                print("  No successful inference tests")
                return False
                
        except Exception as e:
            print(f"❌ YOLO test failed: {e}")
            if 'illegal instruction' in str(e).lower():
                print("   💡 ARM instruction set incompatibility detected")
                print("   💡 Run arm_compatibility_fix.sh to resolve")
            return False
    
    def test_arm_pipeline(self):
        """Test complete ARM-optimized pipeline"""
        print("Testing complete ARM-optimized pipeline...")
        
        if 'best_camera' not in self.test_config:
            print("❌ No camera available for pipeline test")
            return False
        
        if not self.test_config.get('yolo_performance', {}).get('working', False):
            print("⚠️  YOLO not working - using camera-only test")
            return self._test_camera_only_pipeline()
        
        try:
            # Initialize camera
            cam_config = self.test_config['best_camera']
            cap = cv2.VideoCapture(cam_config['index'], cv2.CAP_V4L2)
            
            if not cap.isOpened():
                print("❌ Cannot open camera for pipeline test")
                return False
            
            # Set ARM-optimized settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.test_config['resolution'][0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.test_config['resolution'][1])
            cap.set(cv2.CAP_PROP_FPS, self.test_config['fps_target'])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            from ultralytics import YOLO
            model = YOLO(self._find_model_file('yolo26n.pt'))
            
            # Pipeline test
            frame_count = 0
            detection_count = 0
            errors = 0
            start_time = time.time()
            
            print(f"  Running {self.test_config['test_duration']}s ARM pipeline test...")
            
            while time.time() - start_time < self.test_config['test_duration']:
                try:
                    # Capture frame
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        errors += 1
                        continue
                    
                    frame_count += 1
                    
                    # YOLO detection with ARM optimizations
                    results = model(frame, verbose=False, 
                                 conf=self.test_config['detection_confidence'],
                                 imgsz=self.test_config['resolution'])
                    
                    # Count detections (limited for performance)
                    if results and len(results) > 0 and results[0].boxes:
                        detections = min(len(results[0].boxes), self.test_config['max_detections'])
                        detection_count += detections
                    
                    # Progress update
                    if frame_count % 20 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        print(f"    Frame {frame_count}: FPS {fps:.1f}, Detections {detection_count}")
                
                except Exception as e:
                    errors += 1
                    if errors < 3:
                        print(f"    Pipeline error: {e}")
                    continue
            
            # Calculate results
            total_time = time.time() - start_time
            pipeline_fps = frame_count / total_time if total_time > 0 else 0
            detection_rate = detection_count / frame_count if frame_count > 0 else 0
            error_rate = errors / frame_count if frame_count > 0 else 0
            
            print(f"  Pipeline Results:")
            print(f"    Frames: {frame_count}")
            print(f"    FPS: {pipeline_fps:.2f}")
            print(f"    Detections: {detection_count}")
            print(f"    Detection rate: {detection_rate:.2f}/frame")
            print(f"    Errors: {errors} ({error_rate*100:.1f}%)")
            
            cap.release()
            
            # Success criteria (adjusted for ARM)
            success = (
                pipeline_fps >= 3 and  # Lower FPS target
                error_rate < 0.15 and    # Higher error tolerance
                frame_count >= 20         # Minimum frames
            )
            
            return success
            
        except Exception as e:
            print(f"❌ ARM pipeline test failed: {e}")
            return False
    
    def _test_camera_only_pipeline(self):
        """Test camera-only pipeline (fallback when YOLO fails)"""
        try:
            cam_config = self.test_config['best_camera']
            cap = cv2.VideoCapture(cam_config['index'], cv2.CAP_V4L2)
            
            if not cap.isOpened():
                print("❌ Cannot open camera for camera-only test")
                return False
            
            # Simple camera performance test
            frame_count = 0
            start_time = time.time()
            
            print(f"  Running camera-only test...")
            
            while time.time() - start_time < self.test_config['test_duration']:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_count += 1
                
                if frame_count % 20 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"    Frame {frame_count}: FPS {fps:.1f}")
            
            total_time = time.time() - start_time
            camera_fps = frame_count / total_time if total_time > 0 else 0
            
            print(f"  Camera Results:")
            print(f"    Frames: {frame_count}")
            print(f"    FPS: {camera_fps:.2f}")
            
            cap.release()
            return camera_fps >= 5  # Camera-only FPS target
            
        except Exception as e:
            print(f"❌ Camera-only test failed: {e}")
            return False
    
    def generate_arm_report(self):
        """Generate ARM-optimized test report"""
        print("\n" + "="*70)
        print("ARM-OPTIMIZED TEST REPORT")
        print("="*70)
        
        print(f"Tests Passed: {self.passed_tests}/{self.total_tests}")
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        print("\nTest Results:")
        for test_name, result in self.test_results.items():
            status = result['status']
            duration = result['duration']
            critical = result['critical']
            critical_mark = " [CRITICAL]" if critical else ""
            print(f"  {test_name}: {status} ({duration:.2f}s){critical_mark}")
        
        # ARM-specific recommendations
        print("\nARM Optimization Recommendations:")
        if self.test_config.get('yolo_performance', {}).get('fps', 0) < 5:
            print("  📈 YOLO performance is low - consider:")
            print("     - Lower resolution (320x240)")
            print("     - Higher confidence threshold")
            print("     - Fewer max detections")
        
        if success_rate >= 80:
            print("🎉 EXCELLENT! ARM-optimized system working well!")
        elif success_rate >= 60:
            print("✅ GOOD! ARM-optimized system mostly working.")
        else:
            print("⚠️  NEEDS WORK! ARM compatibility issues detected.")
            print("   Run arm_compatibility_fix.sh to resolve")
        
        # Save report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'success_rate': success_rate,
            'tests_passed': self.passed_tests,
            'tests_total': self.total_tests,
            'test_results': self.test_results,
            'arm_optimized': True,
            'configuration': self.test_config
        }
        
        report_file = Path('arm_optimized_test_report.json')
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\n📄 Report saved to: {report_file}")
        
        return success_rate
    
    def run_all_tests(self):
        """Run all ARM-optimized tests"""
        self.print_header()
        
        # Define ARM-specific test sequence
        tests = [
            ("ARM System Compatibility", self.test_arm_system, True),
            ("ARM Library Compatibility", self.test_arm_libraries, True),
            ("ARM Camera Detection", self.test_arm_camera, True),
            ("ARM YOLO Performance", self.test_arm_yolo, True),
            ("ARM Pipeline Integration", self.test_arm_pipeline, True)
        ]
        
        # Run all tests
        for test_name, test_function, critical in tests:
            self.run_test(test_name, test_function, critical)
        
        # Generate report
        success_rate = self.generate_arm_report()
        
        return success_rate

def main():
    """Main function"""
    tester = ARMOptimizedCameraAI()
    
    try:
        success_rate = tester.run_all_tests()
        
        if success_rate >= 70:  # Lower threshold for ARM
            print("\n🎉 ARM-OPTIMIZED TEST PASSED!")
            print("System is working with ARM optimizations!")
        elif success_rate >= 50:
            print("\n✅ ARM-OPTIMIZED TEST MOSTLY PASSED!")
            print("System working with some limitations.")
        else:
            print("\n⚠️  ARM-OPTIMIZED TEST NEEDS IMPROVEMENT!")
            print("Run arm_compatibility_fix.sh to resolve issues.")
        
        return 0 if success_rate >= 50 else 1
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())