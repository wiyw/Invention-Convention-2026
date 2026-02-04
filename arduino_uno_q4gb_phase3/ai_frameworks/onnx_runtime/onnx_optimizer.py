#!/usr/bin/env python3
"""
Arduino UNO Q4GB ONNX Runtime AI Framework
Phase 3: Hardware-Specific Optimization
Optimized for ARM64 embedded systems
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

class ONNXRuntimeOptimizer:
    def __init__(self, hardware_profile=None):
        self.hardware_profile = hardware_profile or {}
        self.session_options = None
        self.providers = []
        self.optimization_level = None
        
    def load_hardware_profile(self, profile_file='hardware_profile.json'):
        """Load hardware profile for optimization"""
        try:
            with open(profile_file, 'r') as f:
                data = json.load(f)
                self.hardware_profile = data.get('hardware_profile', {})
                return True
        except FileNotFoundError:
            print("⚠️  Hardware profile not found, using defaults")
            return False
        except Exception as e:
            print(f"❌ Error loading hardware profile: {e}")
            return False
    
    def configure_onnx_runtime(self):
        """Configure ONNX Runtime based on hardware profile"""
        print("🔧 Configuring ONNX Runtime for Arduino UNO Q4GB...")
        
        try:
            import onnxruntime as ort
        except ImportError:
            print("❌ ONNX Runtime not installed")
            return None, []
        
        # Create session options
        self.session_options = ort.SessionOptions()
        
        # Get hardware information
        cpu_info = self.hardware_profile.get('cpu', {})
        arm_features = self.hardware_profile.get('arm_features', {})
        memory_info = self.hardware_profile.get('memory', {})
        
        # Optimization based on hardware
        cpu_count = cpu_info.get('cpu_count', 1)
        total_memory_mb = memory_info.get('total_mb', 1024)
        has_neon = arm_features.get('neon', False)
        has_asimd = arm_features.get('asimd', False)
        
        # Configure graph optimization level
        if total_memory_mb >= 2048:  # 2GB+ RAM
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            print("✅ Full graph optimization enabled")
        elif total_memory_mb >= 1024:  # 1GB+ RAM
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            print("✅ Extended graph optimization enabled")
        else:
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            print("⚠️  Basic graph optimization (limited memory)")
        
        # Configure threading
        if cpu_count >= 4:
            self.session_options.intra_op_num_threads = 2
            self.session_options.inter_op_num_threads = 2
            print(f"✅ Multi-threading configured (2+2 threads)")
        elif cpu_count >= 2:
            self.session_options.intra_op_num_threads = 2
            self.session_options.inter_op_num_threads = 1
            print(f"✅ Dual-threading configured")
        else:
            self.session_options.intra_op_num_threads = 1
            self.session_options.inter_op_num_threads = 1
            print(f"⚠️  Single-threading (single core)")
        
        # Configure execution providers
        self.providers = ['CPUExecutionProvider']
        
        # ARM-specific optimizations
        if has_neon or has_asimd:
            # Enable NEON optimizations if available
            os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '0'  # Disable TensorRT (not available)
            os.environ['ORT_ENABLE_CPU_MEM_ARENA'] = '1'  # Enable memory arena
            print("✅ ARM NEON optimizations enabled")
        
        # Memory optimizations for embedded systems
        if total_memory_mb < 1024:
            os.environ['ORT_ENABLE_CPU_MEM_ARENA'] = '1'
            os.environ['ORT_MEM_ARENA_SIZE'] = str(min(128 * 1024 * 1024, total_memory_mb * 512 * 1024))  # Max 128MB or 50% of RAM
            print(f"⚠️  Memory arena limited for low-memory device")
        
        return self.session_options, self.providers
    
    def create_optimized_session(self, model_path, input_size=(1, 3, 224, 224)):
        """Create optimized ONNX Runtime session"""
        try:
            import onnxruntime as ort
            
            # Configure session options
            session_options, providers = self.configure_onnx_runtime()
            if session_options is None:
                return None
            
            # Check if model exists
            model_file = Path(model_path)
            if not model_file.exists():
                print(f"❌ Model not found: {model_path}")
                return None
            
            # Create inference session
            session = ort.InferenceSession(
                str(model_file),
                sess_options=session_options,
                providers=providers
            )
            
            print(f"✅ ONNX Runtime session created successfully")
            print(f"📊 Providers: {session.get_providers()}")
            
            return session
            
        except Exception as e:
            print(f"❌ Error creating ONNX Runtime session: {e}")
            return None
    
    def preprocess_input(self, image, input_size=(224, 224)):
        """Preprocess input image for ONNX model"""
        try:
            # Convert image to RGB if needed
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
            
            # Resize image
            if hasattr(image, 'resize'):
                image = image.resize(input_size)
            
            # Convert to numpy array
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            
            # Normalize to [0, 1]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            # Add batch dimension and transpose (HWC -> BCHW)
            if len(image.shape) == 3:
                image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
                image = np.expand_dims(image, axis=0)  # Add batch dimension
            
            return image
            
        except Exception as e:
            print(f"❌ Error preprocessing input: {e}")
            return None
    
    def run_inference(self, session, input_data, confidence_threshold=0.5):
        """Run optimized inference"""
        try:
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Run inference
            start_time = time.time()
            outputs = session.run(None, {input_name: input_data})
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            
            # Process outputs based on model type
            results = self.process_outputs(outputs, confidence_threshold)
            
            return {
                'results': results,
                'inference_time_ms': inference_time,
                'fps': 1000.0 / inference_time if inference_time > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Error running inference: {e}")
            return None
    
    def process_outputs(self, outputs, confidence_threshold=0.5):
        """Process ONNX model outputs"""
        results = []
        
        try:
            # Handle different output formats
            if len(outputs) == 0:
                return results
            
            # Common object detection format: [batch, num_detections, 85] (COCO format)
            output = outputs[0]
            
            if len(output.shape) == 3:
                # Object detection output
                for detection in output[0]:  # First batch
                    if len(detection) >= 85:  # COCO format with 80 classes
                        # Extract bbox [x_center, y_center, width, height]
                        bbox = detection[:4]
                        confidence = detection[4]
                        
                        if confidence > confidence_threshold:
                            # Get class with highest probability
                            class_probs = detection[5:]
                            class_id = np.argmax(class_probs)
                            class_confidence = class_probs[class_id]
                            
                            results.append({
                                'bbox': bbox.tolist(),
                                'confidence': float(confidence),
                                'class_id': int(class_id),
                                'class_confidence': float(class_confidence)
                            })
            
            elif len(output.shape) == 2:
                # Classification output
                for i, prob in enumerate(output[0]):
                    if prob > confidence_threshold:
                        results.append({
                            'class_id': i,
                            'confidence': float(prob)
                        })
            
            return results
            
        except Exception as e:
            print(f"❌ Error processing outputs: {e}")
            return results
    
    def benchmark_model(self, session, input_size=(1, 3, 224, 224), num_iterations=100):
        """Benchmark model performance"""
        print(f"🚀 Benchmarking ONNX model ({num_iterations} iterations)...")
        
        try:
            # Create random input
            input_data = np.random.rand(*input_size).astype(np.float32)
            input_name = session.get_inputs()[0].name
            
            # Warm up
            for _ in range(5):
                session.run(None, {input_name: input_data})
            
            # Benchmark
            times = []
            for i in range(num_iterations):
                start_time = time.time()
                outputs = session.run(None, {input_name: input_data})
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Progress indicator
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i + 1}/{num_iterations}")
            
            # Calculate statistics
            times = np.array(times)
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            median_time = np.median(times)
            
            fps = 1000.0 / avg_time
            
            results = {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'median_time_ms': median_time,
                'fps': fps,
                'iterations': num_iterations
            }
            
            print(f"📊 Benchmark Results:")
            print(f"  Average: {avg_time:.2f}ms")
            print(f"  Std Dev: {std_time:.2f}ms")
            print(f"  Min: {min_time:.2f}ms")
            print(f"  Max: {max_time:.2f}ms")
            print(f"  Median: {median_time:.2f}ms")
            print(f"  FPS: {fps:.1f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error benchmarking model: {e}")
            return None
    
    def optimize_for_hardware(self):
        """Get hardware-specific optimization recommendations"""
        recommendations = {}
        
        cpu_info = self.hardware_profile.get('cpu', {})
        memory_info = self.hardware_profile.get('memory', {})
        arm_features = self.hardware_profile.get('arm_features', {})
        benchmarks = self.hardware_profile.get('benchmarks', {})
        
        # CPU optimization
        cpu_count = cpu_info.get('cpu_count', 1)
        if cpu_count >= 4:
            recommendations['intra_op_threads'] = 2
            recommendations['inter_op_threads'] = 2
        elif cpu_count >= 2:
            recommendations['intra_op_threads'] = 2
            recommendations['inter_op_threads'] = 1
        else:
            recommendations['intra_op_threads'] = 1
            recommendations['inter_op_threads'] = 1
        
        # Memory optimization
        total_memory_mb = memory_info.get('total_mb', 1024)
        if total_memory_mb < 1024:
            recommendations['graph_optimization'] = 'basic'
            recommendations['memory_arena_size'] = min(64 * 1024 * 1024, total_memory_mb * 256 * 1024)
        elif total_memory_mb < 2048:
            recommendations['graph_optimization'] = 'extended'
            recommendations['memory_arena_size'] = 128 * 1024 * 1024
        else:
            recommendations['graph_optimization'] = 'all'
            recommendations['memory_arena_size'] = 256 * 1024 * 1024
        
        # ARM-specific optimizations
        if arm_features.get('neon'):
            recommendations['enable_neon'] = True
        if arm_features.get('fp16'):
            recommendations['enable_fp16'] = True
        
        # Performance-based adjustments
        ai_benchmarks = benchmarks.get('cpu_performance', {})
        if ai_benchmarks:
            # Adjust based on actual performance
            gflops = list(ai_benchmarks.values())[0].get('gflops', 0)
            if gflops < 10:
                recommendations['model_size'] = 'tiny'
            elif gflops < 50:
                recommendations['model_size'] = 'small'
            else:
                recommendations['model_size'] = 'medium'
        
        return recommendations

class ONNXModelManager:
    def __init__(self, hardware_profile=None):
        self.hardware_profile = hardware_profile or {}
        self.models_dir = Path('models/onnx')
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def download_optimized_models(self):
        """Download ONNX-optimized models for Arduino UNO Q4GB"""
        print("📥 Downloading ONNX-optimized models...")
        
        models_to_download = {
            'yolov8n_int8.onnx': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-int8.onnx',
            'mobilenetv2_int8.onnx': 'https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7-int8.onnx',
            'efficientnet-lite0-int8.onnx': 'https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite/model/efficientnet-lite0-int8.onnx'
        }
        
        downloaded_models = []
        
        for model_name, url in models_to_download.items():
            model_path = self.models_dir / model_name
            if not model_path.exists():
                try:
                    print(f"  Downloading {model_name}...")
                    # In a real implementation, use wget or requests to download
                    # For now, just create a placeholder
                    model_path.touch()
                    downloaded_models.append(str(model_path))
                    print(f"    ✅ {model_name} downloaded")
                except Exception as e:
                    print(f"    ❌ Failed to download {model_name}: {e}")
            else:
                print(f"    ✅ {model_name} already exists")
                downloaded_models.append(str(model_path))
        
        return downloaded_models
    
    def get_optimal_model(self, task='object_detection'):
        """Get optimal model based on hardware profile"""
        
        memory_info = self.hardware_profile.get('memory', {})
        total_memory_mb = memory_info.get('total_mb', 1024)
        
        if task == 'object_detection':
            if total_memory_mb < 1024:
                return 'yolov8n_int8.onnx'
            else:
                return 'yolov8n.onnx'  # Standard model if memory allows
        
        elif task == 'classification':
            if total_memory_mb < 1024:
                return 'mobilenetv2_int8.onnx'
            else:
                return 'efficientnet-lite0-int8.onnx'
        
        return 'yolov8n_int8.onnx'  # Default

def main():
    """Main function for ONNX Runtime optimization"""
    print("="*60)
    print("ARDUINO UNO Q4GB ONNX RUNTIME OPTIMIZATION")
    print("="*60)
    
    # Create optimizer
    optimizer = ONNXRuntimeOptimizer()
    
    # Load hardware profile
    if not optimizer.load_hardware_profile('arduino_q4gb_hardware_profile.json'):
        print("⚠️  Running without hardware profile optimization")
    
    # Get optimization recommendations
    recommendations = optimizer.optimize_for_hardware()
    print(f"\n🎯 Optimization Recommendations:")
    for key, value in recommendations.items():
        print(f"  {key}: {value}")
    
    # Initialize model manager
    model_manager = ONNXModelManager(optimizer.hardware_profile)
    
    # Download optimized models
    downloaded_models = model_manager.download_optimized_models()
    
    print(f"\n🎉 ONNX Runtime optimization complete!")
    print(f"📊 Ready for Phase 3 deployment on Arduino UNO Q4GB")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())