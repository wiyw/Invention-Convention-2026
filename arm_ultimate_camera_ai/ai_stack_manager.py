#!/usr/bin/env python3
"""
Arduino UNO Q4GB Ultimate AI Stack Manager
Multi-backend AI with intelligent fallbacks for ARM
"""

import os
import sys
import json
import importlib.util
from pathlib import Path

class AIStackManager:
    def __init__(self):
        self.ai_backends = {}
        self.active_backend = None
        self.models = {}
        self.config = {}
        
    def load_config(self):
        """Load configuration"""
        try:
            with open('ultimate_config.json', 'r') as f:
                self.config = json.load(f)
            print("✅ Configuration loaded")
            return True
        except:
            print("⚠️  No configuration found - using defaults")
            self.config = {
                'ai_stack': {
                    'backends': ['ultralytics', 'onnx', 'tflite', 'opencv'],
                    'confidence_threshold': 0.6,
                    'max_detections': 3
                },
                'camera': {
                    'resolution': [320, 240],
                    'fps_target': 8
                }
            }
            return False
    
    def setup_ultralytics_backend(self):
        """Setup YOLO + PyTorch backend"""
        print("🤖 Setting up Ultralytics YOLO Backend")
        print("-" * 40)
        
        try:
            # Test PyTorch import
            torch_result = self._test_import('torch', 'import torch; print(torch.__version__)')
            if not torch_result['success']:
                print(f"❌ PyTorch import failed: {torch_result['error']}")
                return False
            
            print(f"✅ PyTorch {torch_result['output']}")
            
            # Test Ultralytics import
            yolo_result = self._test_import('ultralytics', 'from ultralytics import YOLO; print(YOLO.__version__)')
            if not yolo_result['success']:
                print(f"❌ Ultralytics import failed: {yolo_result['error']}")
                return False
            
            print(f"✅ Ultralytics {yolo_result['output']}")
            
            # Load YOLO model
            model_path = self._find_model('yolo26n.pt')
            if not model_path:
                print("❌ YOLO26n model not found")
                return False
            
            print(f"📥 Loading YOLO model: {model_path}")
            
            from ultralytics import YOLO
            model = YOLO(model_path)
            
            # Test model with small image
            import numpy as np
            test_image = np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8)
            
            try:
                results = model(test_image, verbose=False)
                self.ai_backends['ultralytics'] = {
                    'model': model,
                    'working': True,
                    'confidence': self.config['ai_stack']['confidence_threshold'],
                    'model_path': model_path
                }
                print("✅ Ultralytics backend working")
                return True
                
            except Exception as e:
                print(f"❌ Ultralytics inference failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Ultralytics setup failed: {e}")
            return False
    
    def setup_onnx_backend(self):
        """Setup ONNX Runtime backend"""
        print("🔧 Setting up ONNX Runtime Backend")
        print("-" * 40)
        
        try:
            # Test ONNX Runtime import
            onnx_result = self._test_import('onnxruntime', 'import onnxruntime; print(onnxruntime.get_device())')
            if not onnx_result['success']:
                print(f"❌ ONNX Runtime import failed: {onnx_result['error']}")
                return False
            
            print(f"✅ ONNX Runtime {onnx_result['output']}")
            
            # Check for ONNX model
            onnx_model_path = self._find_model('yolo26n.onnx')
            
            if onnx_model_path:
                # Try to load ONNX model
                try:
                    import onnxruntime as ort
                    sess = ort.InferenceSession(onnx_model_path)
                    
                    self.ai_backends['onnx'] = {
                        'session': sess,
                        'model_path': onnx_model_path,
                        'working': True
                    }
                    print("✅ ONNX backend working")
                    return True
                    
                except Exception as e:
                    print(f"❌ ONNX model loading failed: {e}")
            
            # Create fallback ONNX detector
            self.ai_backends['onnx'] = {
                'working': True,
                'fallback': True,
                'description': 'ONNX Runtime with fallback detection'
            }
            print("✅ ONNX backend (fallback mode)")
            return True
            
        except Exception as e:
            print(f"❌ ONNX setup failed: {e}")
            return False
    
    def setup_tflite_backend(self):
        """Setup TensorFlow Lite backend"""
        print("📱 Setting up TensorFlow Lite Backend")
        print("-" * 40)
        
        try:
            # Test TFLite import
            tflite_result = self._test_import('tflite_runtime', 'import tflite_runtime; print("TFLite working")')
            if not tflite_result['success']:
                print(f"❌ TensorFlow Lite import failed: {tflite_result['error']}")
                return False
            
            print("✅ TensorFlow Lite working")
            
            # Check for TFLite model
            tflite_model_path = self._find_model('yolo26n.tflite')
            
            self.ai_backends['tflite'] = {
                'working': True,
                'fallback': True,
                'description': 'TensorFlow Lite with fallback detection'
            }
            print("✅ TensorFlow Lite backend (fallback mode)")
            return True
            
        except Exception as e:
            print(f"❌ TFLite setup failed: {e}")
            return False
    
    def setup_opencv_backend(self):
        """Setup OpenCV backend (simple detection)"""
        print("📷 Setting up OpenCV Backend")
        print("-" * 40)
        
        try:
            # Test OpenCV import
            cv2_result = self._test_import('cv2', 'import cv2; print(cv2.__version__)')
            if not cv2_result['success']:
                print(f"❌ OpenCV import failed: {cv2_result['error']}")
                return False
            
            print(f"✅ OpenCV {cv2_result['output']}")
            
            # Create OpenCV-based detector
            def simple_detector(frame):
                """Simple detection using OpenCV"""
                import numpy as np
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Threshold for bright areas (mock objects)
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Create mock results
                class MockDetection:
                    def __init__(self, boxes):
                        self.boxes = boxes
                        
                class MockBox:
                    def __init__(self, x, y, w, h):
                        self.xyxy = [x, y, x+w, y+h]
                        self.conf = 0.7
                        self.cls = 0
                
                mock_boxes = []
                for contour in contours[:self.config['ai_stack']['max_detections']]:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 20 and h > 20:  # Minimum size
                        mock_boxes.append(MockBox(x, y, w, h))
                
                return [MockDetection(mock_boxes)]
            
            self.ai_backends['opencv'] = {
                'detector': simple_detector,
                'working': True,
                'description': 'OpenCV simple detection'
            }
            print("✅ OpenCV backend working")
            return True
            
        except Exception as e:
            print(f"❌ OpenCV setup failed: {e}")
            return False
    
    def setup_all_backends(self):
        """Setup all available AI backends"""
        print("🤖 Setting up All AI Backends")
        print("=" * 50)
        
        backends_setup = [
            ('ultralytics', self.setup_ultralytics_backend),
            ('onnx', self.setup_onnx_backend),
            ('tflite', self.setup_tflite_backend),
            ('opencv', self.setup_opencv_backend)
        ]
        
        for backend_name, setup_func in backends_setup:
            try:
                if setup_func():
                    print(f"✅ {backend_name} backend ready")
                else:
                    print(f"❌ {backend_name} backend failed")
            except Exception as e:
                print(f"❌ {backend_name} backend crashed: {e}")
        
        # Select best available backend
        if 'ultralytics' in self.ai_backends:
            self.active_backend = 'ultralytics'
        elif 'onnx' in self.ai_backends:
            self.active_backend = 'onnx'
        elif 'tflite' in self.ai_backends:
            self.active_backend = 'tflite'
        elif 'opencv' in self.ai_backends:
            self.active_backend = 'opencv'
        else:
            self.active_backend = None
            print("❌ No AI backends available")
            return False
        
        print(f"🎯 Selected AI backend: {self.active_backend}")
        return True
    
    def detect_objects(self, frame):
        """Detect objects using active backend"""
        if not self.active_backend:
            return []
        
        backend = self.ai_backends[self.active_backend]
        
        if 'model' in backend:
            # YOLO backend
            results = backend['model'](frame, verbose=False, conf=self.config['ai_stack']['confidence_threshold'])
            return results
        elif 'detector' in backend:
            # OpenCV backend
            return backend['detector'](frame)
        elif 'fallback' in backend:
            # Fallback backend
            return self._fallback_detection(frame)
        else:
            # Generic fallback
            return self._fallback_detection(frame)
    
    def _fallback_detection(self, frame):
        """Fallback object detection"""
        import numpy as np
        
        # Simple motion/bright area detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if 'cv2' in sys.modules else frame.mean(axis=2).astype(np.uint8)
        else:
            gray = frame
        
        # Find bright areas as "objects"
        thresh_val = 200
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY) if 'cv2' in sys.modules else (gray > thresh_val).astype(np.uint8)
        
        # Use numpy operations if OpenCV not available
        if 'cv2' in sys.modules:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            class MockDetection:
                def __init__(self, boxes):
                    self.boxes = boxes
                    
            class MockBox:
                def __init__(self, x, y, w, h):
                    self.xyxy = [x, y, x+w, y+h]
                    self.conf = 0.6
                    self.cls = 0
            
            boxes = []
            for contour in contours[:3]:  # Limit detections
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 20:
                    boxes.append(MockBox(x, y, w, h))
            
            return [MockDetection(boxes)]
        else:
            # Pure numpy fallback
            if 'scipy' in sys.modules:
                from scipy import ndimage
            else:
                ndimage = None
            
            if ndimage:
                labeled_array, num_features = ndimage.label(binary)
                boxes = []
                
                for i in range(1, min(num_features + 1, 4)):
                    positions = np.where(labeled_array == i)
                    if len(positions[0]) > 0:
                        y_min, y_max = positions[0].min(), positions[0].max()
                        x_min, x_max = positions[1].min(), positions[1].max()
                        
                        class MockBox:
                            def __init__(self, x_min, y_min, x_max, y_max):
                                self.xyxy = [x_min, y_min, x_max, y_max]
                                self.conf = 0.6
                                self.cls = 0
                        
                        class MockDetection:
                            def __init__(self, boxes):
                                self.boxes = boxes
                        
                        boxes.append(MockBox(x_min, y_min, x_max, y_max))
                
                class MockDetection:
                    def __init__(self, boxes):
                        self.boxes = boxes
                
                return [MockDetection(boxes)]
            else:
                # Super basic fallback
                return []
    
    def _test_import(self, module_name, test_code):
        """Test Python module import"""
        try:
            result = importlib.util.find_spec(module_name)
            if result is None:
                return {
                    'success': False,
                    'error': f'{module_name} not found'
                }
            
            # Try to execute test code
            import subprocess
            result = subprocess.run(
                [sys.executable, '-c', test_code],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'output': result.stdout.strip()
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr.strip()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _find_model(self, filename):
        """Find AI model file"""
        search_paths = [
            Path('models') / filename,
            Path(filename),
            Path.cwd() / 'models' / filename
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def get_backend_info(self):
        """Get active backend information"""
        if not self.active_backend:
            return None
        
        backend = self.ai_backends[self.active_backend]
        
        info = {
            'name': self.active_backend,
            'working': backend.get('working', False),
            'description': backend.get('description', 'AI detection backend')
        }
        
        if 'model' in backend:
            info['model_type'] = 'YOLO'
            info['model_path'] = backend.get('model_path', 'Unknown')
        elif 'detector' in backend:
            info['model_type'] = 'OpenCV Simple'
        elif 'fallback' in backend:
            info['model_type'] = 'Fallback'
        
        return info
    
    def save_config(self):
        """Save AI stack configuration"""
        ai_config = {
            'active_backend': self.active_backend,
            'available_backends': list(self.ai_backends.keys()),
            'backend_info': self.get_backend_info(),
            'config': self.config['ai_stack']
        }
        
        with open('ai_stack_config.json', 'w') as f:
            json.dump(ai_config, f, indent=2)
        
        print(f"✅ AI stack configuration saved")

def main():
    """Main function"""
    manager = AIStackManager()
    
    try:
        print("🤖 Arduino UNO Q4GB Ultimate AI Stack Manager")
        print("=" * 60)
        
        # Load configuration
        manager.load_config()
        
        # Setup all backends
        if manager.setup_all_backends():
            # Save configuration
            manager.save_config()
            
            # Print summary
            print("\n" + "=" * 60)
            print("🤖 AI STACK MANAGER - SUMMARY")
            print("=" * 60)
            
            print(f"✅ Active Backend: {manager.active_backend}")
            print(f"📊 Available Backends: {list(manager.ai_backends.keys())}")
            
            backend_info = manager.get_backend_info()
            if backend_info:
                print(f"🎯 Model Type: {backend_info['model_type']}")
                print(f"📋 Description: {backend_info['description']}")
            
            print("\n🎉 AI stack setup completed!")
            print("🎯 Backend will be selected automatically during runtime")
            
            return 0
        else:
            print("❌ AI stack setup failed")
            return 1
            
    except Exception as e:
        print(f"❌ AI stack manager crashed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())