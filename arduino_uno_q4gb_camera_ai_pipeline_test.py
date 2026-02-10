#!/usr/bin/env python3
"""
Arduino UNO Q4GB AI Robot - Complete Camera + AI Pipeline Test Suite
Tests USB camera, YOLO26n object detection, and Arduino motor responses
Optimized for Arduino UNO Q4GB Linux system
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

class CameraAIPipelineTester:
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        self.install_dir = Path.home() / 'arduino_q4gb_ai_robot_phase3'
        self.camera = None
        self.yolo_model = None
        self.arduino_serial = None
        self.test_config = {
            'camera_indices': [0, 1, 2, 3, 4],
            'backends': [cv2.CAP_V4L2, cv2.CAP_GSTREAMER],
            'test_duration': 10,  # seconds per test
            'resolution': (640, 480),
            'fps_target': 15,
            'detection_confidence': 0.5
        }
        
    def print_header(self):
        """Print test suite header"""
        print("="*80)
        print("    Arduino UNO Q4GB AI Robot - Complete Camera + AI Pipeline Test")
        print("    Phase 3: USB Camera + YOLO26n + Arduino Motor Control")
        print("="*80)
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
    
    def test_system_compatibility(self):
        """Test system compatibility for camera + AI pipeline"""
        print("Checking system compatibility...")
        
        # Check Python version
        if sys.version_info < (3, 7):
            print("Python 3.7+ required")
            return False
        
        # Check required packages
        required_packages = ['cv2', 'numpy', 'ultralytics', 'serial', 'PIL']
        for package in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                    print(f"OpenCV version: {cv2.__version__}")
                elif package == 'serial':
                    import serial
                    print(f"PySerial version: {serial.__version__}")
                else:
                    spec = importlib.util.find_spec(package)
                    if spec is None:
                        raise ImportError(f"{package} not found")
            except ImportError as e:
                print(f"Missing package: {package}")
                return False
        
        # Check camera devices
        camera_devices = []
        for i in range(5):
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                camera_devices.append(device_path)
        
        if not camera_devices:
            print("No camera devices found in /dev/video*")
            return False
        
        print(f"Found camera devices: {camera_devices}")
        return True
    
    def test_camera_detection(self):
        """Test USB camera detection and initialization"""
        print("Testing camera detection...")
        
        working_cameras = []
        
        # Test different camera indices and backends
        for cam_idx in self.test_config['camera_indices']:
            for backend in self.test_config['backends']:
                try:
                    print(f"  Testing camera {cam_idx} with backend {backend}...")
                    cap = cv2.VideoCapture(cam_idx + backend)
                    
                    if cap.isOpened():
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        # Try to capture a frame
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            camera_info = {
                                'index': cam_idx,
                                'backend': backend,
                                'resolution': (width, height),
                                'fps': fps,
                                'working': True
                            }
                            working_cameras.append(camera_info)
                            print(f"    SUCCESS: {width}x{height} @ {fps}fps")
                        else:
                            print(f"    FAIL: Opens but no frame")
                        
                        cap.release()
                    else:
                        print(f"    FAIL: Cannot open")
                        
                except Exception as e:
                    print(f"    ERROR: {e}")
        
        if working_cameras:
            # Store best camera (highest resolution)
            best_camera = max(working_cameras, key=lambda x: x['resolution'][0] * x['resolution'][1])
            self.test_config['best_camera'] = best_camera
            print(f"Best camera: Index {best_camera['index']}, Backend {best_camera['backend']}")
            return True
        else:
            print("No working cameras found")
            return False
    
    def test_camera_performance(self):
        """Test camera capture performance"""
        print("Testing camera performance...")
        
        if 'best_camera' not in self.test_config:
            print("No camera configuration available")
            return False
        
        cam_config = self.test_config['best_camera']
        backend_name = {cv2.CAP_V4L2: 'V4L2', cv2.CAP_GSTREAMER: 'GStreamer'}[cam_config['backend']]
        
        try:
            cap = cv2.VideoCapture(cam_config['index'] + cam_config['backend'])
            if not cap.isOpened():
                print("Cannot open camera for performance test")
                return False
            
            # Set camera parameters
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.test_config['resolution'][0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.test_config['resolution'][1])
            cap.set(cv2.CAP_PROP_FPS, self.test_config['fps_target'])
            
            # Performance test
            frame_count = 0
            start_time = time.time()
            test_duration = self.test_config['test_duration']
            
            print(f"  Running {test_duration}s performance test...")
            
            while time.time() - start_time < test_duration:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_count += 1
                else:
                    print("    Failed to capture frame")
                    break
            
            end_time = time.time()
            actual_duration = end_time - start_time
            actual_fps = frame_count / actual_duration if actual_duration > 0 else 0
            
            print(f"  Captured {frame_count} frames in {actual_duration:.2f}s")
            print(f"  Actual FPS: {actual_fps:.2f}")
            
            cap.release()
            
            # Store performance results
            self.test_config['camera_performance'] = {
                'fps': actual_fps,
                'frames_captured': frame_count,
                'duration': actual_duration
            }
            
            return actual_fps >= (self.test_config['fps_target'] * 0.8)  # 80% of target
            
        except Exception as e:
            print(f"Camera performance test error: {e}")
            return False
    
    def test_yolo_model_loading(self):
        """Test YOLO26n model loading"""
        print("Testing YOLO26n model loading...")
        
        try:
            from ultralytics import YOLO
            
            # Check model file
            model_path = self.install_dir / 'models' / 'yolo26n.pt'
            if not model_path.exists():
                # Try alternative paths
                alt_paths = [
                    'models/yolo26n.pt',
                    '/tmp/yolo26n.pt',
                    Path.cwd() / 'models' / 'yolo26n.pt'
                ]
                
                for alt_path in alt_paths:
                    if Path(alt_path).exists():
                        model_path = Path(alt_path)
                        break
                else:
                    print("YOLO26n model file not found")
                    return False
            
            print(f"  Loading model from: {model_path}")
            
            # Load model
            start_time = time.time()
            self.yolo_model = YOLO(str(model_path))
            load_time = time.time() - start_time
            
            print(f"  Model loaded in {load_time:.2f}s")
            print(f"  Model classes: {len(self.yolo_model.names)}")
            
            # Test model with dummy image
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            start_time = time.time()
            results = self.yolo_model(test_image, verbose=False)
            inference_time = time.time() - start_time
            
            print(f"  Test inference: {inference_time*1000:.1f}ms")
            
            self.test_config['yolo_loaded'] = True
            return True
            
        except Exception as e:
            print(f"YOLO model loading error: {e}")
            return False
    
    def test_yolo_detection_performance(self):
        """Test YOLO real-time detection performance"""
        print("Testing YOLO real-time detection performance...")
        
        if not self.test_config.get('yolo_loaded', False):
            print("YOLO model not loaded")
            return False
        
        if 'best_camera' not in self.test_config:
            print("No camera available")
            return False
        
        try:
            cam_config = self.test_config['best_camera']
            cap = cv2.VideoCapture(cam_config['index'] + cam_config['backend'])
            
            if not cap.isOpened():
                print("Cannot open camera for YOLO testing")
                return False
            
            # Detection performance test
            detection_count = 0
            frame_count = 0
            total_inference_time = 0
            start_time = time.time()
            test_duration = self.test_config['test_duration']
            
            print(f"  Running {test_duration}s detection test...")
            
            while time.time() - start_time < test_duration:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_count += 1
                    
                    # Run YOLO detection
                    inference_start = time.time()
                    results = self.yolo_model(frame, verbose=False, 
                                           conf=self.test_config['detection_confidence'])
                    inference_time = time.time() - inference_start
                    total_inference_time += inference_time
                    
                    # Count detections
                    if results and len(results) > 0:
                        detections = len(results[0].boxes) if results[0].boxes else 0
                        detection_count += detections
                else:
                    break
            
            end_time = time.time()
            actual_duration = end_time - start_time
            
            # Calculate metrics
            avg_inference_time = (total_inference_time / frame_count * 1000) if frame_count > 0 else 0
            detection_fps = frame_count / actual_duration if actual_duration > 0 else 0
            avg_detections_per_frame = detection_count / frame_count if frame_count > 0 else 0
            
            print(f"  Processed {frame_count} frames in {actual_duration:.2f}s")
            print(f"  Detection FPS: {detection_fps:.2f}")
            print(f"  Average inference time: {avg_inference_time:.1f}ms")
            print(f"  Average detections per frame: {avg_detections_per_frame:.1f}")
            
            cap.release()
            
            # Store results
            self.test_config['yolo_performance'] = {
                'fps': detection_fps,
                'avg_inference_time': avg_inference_time,
                'total_detections': detection_count,
                'frames_processed': frame_count
            }
            
            return detection_fps >= 10  # Minimum 10 FPS for real-time operation
            
        except Exception as e:
            print(f"YOLO detection test error: {e}")
            return False
    
    def test_arduino_connection(self):
        """Test Arduino serial connection"""
        print("Testing Arduino connection...")
        
        try:
            import serial.tools.list_ports
            
            # Scan for Arduino ports
            arduino_ports = []
            ports = serial.tools.list_ports.comports()
            
            for port in ports:
                if any(arduino_keyword in port.description.lower() 
                      for arduino_keyword in ['arduino', 'ch340', 'cp210', 'ftdi']):
                    arduino_ports.append(port)
                    print(f"  Found Arduino: {port.device} - {port.description}")
            
            if not arduino_ports:
                print("No Arduino ports found")
                return False
            
            # Try to connect to first Arduino
            arduino_port = arduino_ports[0]
            
            try:
                self.arduino_serial = serial.Serial(
                    port=arduino_port.device,
                    baudrate=115200,
                    timeout=1
                )
                
                print(f"  Connected to Arduino at {arduino_port.device}")
                
                # Test communication
                self.arduino_serial.write(b'TEST\n')
                time.sleep(0.1)
                response = self.arduino_serial.readline().decode().strip()
                
                if response:
                    print(f"  Arduino response: {response}")
                else:
                    print("  No response from Arduino (may be normal)")
                
                self.test_config['arduino_connected'] = True
                return True
                
            except serial.SerialException as e:
                print(f"  Serial connection failed: {e}")
                return False
                
        except Exception as e:
            print(f"Arduino connection test error: {e}")
            return False
    
    def test_motor_response(self):
        """Test Arduino motor response based on detections"""
        print("Testing Arduino motor response...")
        
        if not self.test_config.get('arduino_connected', False):
            print("Arduino not connected")
            return False
        
        if not self.test_config.get('yolo_loaded', False):
            print("YOLO model not loaded")
            return False
        
        try:
            # Test motor commands
            motor_commands = [
                'FORWARD:100:100\n',
                'STOP\n',
                'LEFT:100:50\n',
                'RIGHT:50:100\n',
                'STOP\n'
            ]
            
            print("  Testing motor commands...")
            
            for cmd in motor_commands:
                self.arduino_serial.write(cmd.encode())
                time.sleep(0.5)
                response = self.arduino_serial.readline().decode().strip()
                print(f"    Command: {cmd.strip()} -> Response: {response}")
            
            print("  Motor command test completed")
            return True
            
        except Exception as e:
            print(f"Motor response test error: {e}")
            return False
    
    def test_complete_pipeline(self):
        """Test complete camera + AI + Arduino pipeline"""
        print("Testing complete pipeline...")
        
        required_components = [
            self.test_config.get('best_camera'),
            self.test_config.get('yolo_loaded', False),
            self.test_config.get('arduino_connected', False)
        ]
        
        if not all(required_components):
            print("Missing required components for pipeline test")
            return False
        
        try:
            cam_config = self.test_config['best_camera']
            cap = cv2.VideoCapture(cam_config['index'] + cam_config['backend'])
            
            if not cap.isOpened():
                print("Cannot open camera for pipeline test")
                return False
            
            pipeline_stats = {
                'frames_processed': 0,
                'detections_made': 0,
                'motor_commands_sent': 0,
                'pipeline_fps': 0,
                'errors': 0
            }
            
            start_time = time.time()
            test_duration = self.test_config['test_duration']
            
            print(f"  Running {test_duration}s pipeline test...")
            
            while time.time() - start_time < test_duration:
                try:
                    # 1. Capture frame
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        pipeline_stats['errors'] += 1
                        continue
                    
                    pipeline_stats['frames_processed'] += 1
                    
                    # 2. YOLO detection
                    results = self.yolo_model(frame, verbose=False, 
                                           conf=self.test_config['detection_confidence'])
                    
                    # 3. Process detections
                    detected_objects = []
                    if results and len(results) > 0 and results[0].boxes:
                        for box in results[0].boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = self.yolo_model.names[class_id]
                            
                            detected_objects.append({
                                'class': class_name,
                                'confidence': confidence
                            })
                    
                    if detected_objects:
                        pipeline_stats['detections_made'] += len(detected_objects)
                        
                        # 4. Send motor commands based on detections
                        for obj in detected_objects[:3]:  # Limit to 3 objects
                            if obj['class'] == 'person':
                                cmd = 'FORWARD:80:80\n'
                            elif obj['class'] in ['car', 'truck', 'bus']:
                                cmd = 'STOP\n'
                            elif obj['class'] in ['cup', 'bottle']:
                                cmd = f'LEFT:60:40\n'
                            else:
                                cmd = 'FORWARD:50:50\n'
                            
                            self.arduino_serial.write(cmd.encode())
                            pipeline_stats['motor_commands_sent'] += 1
                            time.sleep(0.1)  # Small delay between commands
                            break  # Send one command per frame
                    
                except Exception as e:
                    pipeline_stats['errors'] += 1
                    print(f"    Pipeline error: {e}")
            
            end_time = time.time()
            actual_duration = end_time - start_time
            
            # Calculate final metrics
            pipeline_stats['pipeline_fps'] = pipeline_stats['frames_processed'] / actual_duration
            pipeline_stats['detection_rate'] = pipeline_stats['detections_made'] / pipeline_stats['frames_processed'] if pipeline_stats['frames_processed'] > 0 else 0
            
            print(f"  Pipeline Results:")
            print(f"    Frames processed: {pipeline_stats['frames_processed']}")
            print(f"    Detections made: {pipeline_stats['detections_made']}")
            print(f"    Motor commands sent: {pipeline_stats['motor_commands_sent']}")
            print(f"    Pipeline FPS: {pipeline_stats['pipeline_fps']:.2f}")
            print(f"    Detection rate: {pipeline_stats['detection_rate']:.2f} detections/frame")
            print(f"    Errors: {pipeline_stats['errors']}")
            
            cap.release()
            
            # Store pipeline results
            self.test_config['pipeline_results'] = pipeline_stats
            
            # Success criteria
            success = (
                pipeline_stats['pipeline_fps'] >= 5 and
                pipeline_stats['errors'] < pipeline_stats['frames_processed'] * 0.1
            )
            
            return success
            
        except Exception as e:
            print(f"Complete pipeline test error: {e}")
            return False
    
    def cleanup_resources(self):
        """Clean up all resources"""
        print("Cleaning up resources...")
        
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
        
        if self.arduino_serial:
            try:
                self.arduino_serial.close()
            except:
                pass
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        print(f"Tests Passed: {self.passed_tests}/{self.total_tests}")
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = result['status']
            duration = result['duration']
            critical = result['critical']
            critical_mark = " [CRITICAL]" if critical else ""
            print(f"  {test_name}: {status} ({duration:.2f}s){critical_mark}")
        
        # Performance summary
        if 'camera_performance' in self.test_config:
            cam_perf = self.test_config['camera_performance']
            print(f"\nCamera Performance:")
            print(f"  FPS: {cam_perf['fps']:.2f}")
            print(f"  Frames captured: {cam_perf['frames_captured']}")
        
        if 'yolo_performance' in self.test_config:
            yolo_perf = self.test_config['yolo_performance']
            print(f"\nYOLO Performance:")
            print(f"  Detection FPS: {yolo_perf['fps']:.2f}")
            print(f"  Avg inference time: {yolo_perf['avg_inference_time']:.1f}ms")
            print(f"  Total detections: {yolo_perf['total_detections']}")
        
        if 'pipeline_results' in self.test_config:
            pipeline = self.test_config['pipeline_results']
            print(f"\nComplete Pipeline Performance:")
            print(f"  Pipeline FPS: {pipeline['pipeline_fps']:.2f}")
            print(f"  Detection rate: {pipeline['detection_rate']:.2f} detections/frame")
            print(f"  Motor commands sent: {pipeline['motor_commands_sent']}")
            print(f"  Error rate: {pipeline['errors']}/{pipeline['frames_processed']} frames")
        
        # Save detailed report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'success_rate': success_rate,
            'tests_passed': self.passed_tests,
            'tests_total': self.total_tests,
            'test_results': self.test_results,
            'configuration': self.test_config
        }
        
        report_file = Path('camera_ai_pipeline_report.json')
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\nDetailed report saved to: {report_file}")
        except Exception as e:
            print(f"Failed to save report: {e}")
        
        return success_rate
    
    def run_all_tests(self):
        """Run all camera + AI pipeline tests"""
        self.print_header()
        
        # Define test sequence
        tests = [
            ("System Compatibility", self.test_system_compatibility, True),
            ("Camera Detection", self.test_camera_detection, True),
            ("Camera Performance", self.test_camera_performance, True),
            ("YOLO Model Loading", self.test_yolo_model_loading, True),
            ("YOLO Detection Performance", self.test_yolo_detection_performance, True),
            ("Arduino Connection", self.test_arduino_connection, True),
            ("Motor Response", self.test_motor_response, True),
            ("Complete Pipeline", self.test_complete_pipeline, True)
        ]
        
        # Run all tests
        for test_name, test_function, critical in tests:
            self.run_test(test_name, test_function, critical)
        
        # Cleanup and report
        self.cleanup_resources()
        success_rate = self.generate_report()
        
        return success_rate

def main():
    """Main function"""
    tester = CameraAIPipelineTester()
    
    try:
        success_rate = tester.run_all_tests()
        
        if success_rate >= 80:
            print("\n🎉 EXCELLENT! Camera + AI pipeline is ready for deployment!")
        elif success_rate >= 60:
            print("\n✅ GOOD! Camera + AI pipeline mostly working. Minor issues to address.")
        else:
            print("\n⚠️  NEEDS WORK! Camera + AI pipeline has significant issues.")
        
        return 0 if success_rate >= 60 else 1
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        tester.cleanup_resources()
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        tester.cleanup_resources()
        return 1

if __name__ == "__main__":
    sys.exit(main())