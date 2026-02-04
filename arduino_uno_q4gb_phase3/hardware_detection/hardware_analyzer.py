#!/usr/bin/env python3
"""
Arduino UNO Q4GB Hardware Detection and Analysis Tool
Phase 3: Hardware-Specific Optimization
"""

import os
import sys
import platform
import subprocess
import json
import time
from pathlib import Path

class ArduinoQ4GBHardwareAnalyzer:
    def __init__(self):
        self.hardware_profile = {}
        self.benchmark_results = {}
        
    def get_cpu_info(self):
        """Get detailed CPU information"""
        print("🔍 Analyzing CPU architecture...")
        
        cpu_info = {}
        
        try:
            # Basic architecture
            cpu_info['architecture'] = platform.machine()
            cpu_info['processor'] = platform.processor()
            
            # Get detailed CPU info from /proc/cpuinfo
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo_lines = f.readlines()
                    
                for line in cpuinfo_lines:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key == 'model name':
                            cpu_info['model_name'] = value
                        elif key == 'cpu MHz':
                            cpu_info['cpu_mhz'] = float(value)
                        elif key == 'Features':
                            cpu_info['features'] = value.split()
                        elif key == 'CPU implementer':
                            cpu_info['implementer'] = value
                        elif key == 'CPU architecture':
                            cpu_info['cpu_arch'] = value
                        elif key == 'CPU variant':
                            cpu_info['cpu_variant'] = value
                        elif key == 'CPU part':
                            cpu_info['cpu_part'] = value
                        elif key == 'CPU revision':
                            cpu_info['cpu_revision'] = value
            
            # Get lscpu information if available
            try:
                result = subprocess.run(['lscpu'], capture_output=True, text=True)
                if result.returncode == 0:
                    lscpu_lines = result.stdout.split('\n')
                    for line in lscpu_lines:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            if key == 'Model name':
                                cpu_info['lscpu_model'] = value
                            elif key == 'CPU(s)':
                                cpu_info['cpu_count'] = int(value)
                            elif key == 'Thread(s) per core':
                                cpu_info['threads_per_core'] = int(value)
                            elif key == 'Core(s) per socket':
                                cpu_info['cores_per_socket'] = int(value)
                            elif key == 'Socket(s)':
                                cpu_info['sockets'] = int(value)
                            elif key == 'CPU max MHz':
                                cpu_info['max_mhz'] = float(value)
                            elif key == 'CPU min MHz':
                                cpu_info['min_mhz'] = float(value)
                            elif key == 'L1d cache':
                                cpu_info['l1d_cache'] = value
                            elif key == 'L1i cache':
                                cpu_info['l1i_cache'] = value
                            elif key == 'L2 cache':
                                cpu_info['l2_cache'] = value
                            elif key == 'L3 cache':
                                cpu_info['l3_cache'] = value
            except FileNotFoundError:
                pass  # lscpu not available
                
        except Exception as e:
            print(f"Error getting CPU info: {e}")
            
        return cpu_info
    
    def get_memory_info(self):
        """Get memory information"""
        print("🧠 Analyzing memory configuration...")
        
        memory_info = {}
        
        try:
            # Use free command
            result = subprocess.run(['free', '-h'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.startswith('Mem:'):
                        parts = line.split()
                        memory_info['total'] = parts[1]
                        memory_info['used'] = parts[2]
                        memory_info['free'] = parts[3]
                        memory_info['available'] = parts[6]
                        
                        # Convert GB to MB for calculations
                        if 'G' in memory_info['total']:
                            memory_info['total_mb'] = float(memory_info['total'].replace('G', '')) * 1024
                        elif 'M' in memory_info['total']:
                            memory_info['total_mb'] = float(memory_info['total'].replace('M', ''))
            
            # Check for swap
            for line in lines:
                if line.startswith('Swap:'):
                    parts = line.split()
                    memory_info['swap_total'] = parts[1]
                    memory_info['swap_used'] = parts[2]
                    break
                    
        except Exception as e:
            print(f"Error getting memory info: {e}")
            
        return memory_info
    
    def detect_arm_features(self):
        """Detect ARM-specific CPU features"""
        print("🦾 Detecting ARM CPU features...")
        
        arm_features = {}
        
        try:
            cpu_info = self.get_cpu_info()
            
            # Parse ARM features from /proc/cpuinfo
            if 'features' in cpu_info:
                features = cpu_info['features']
                
                arm_features['neon'] = 'asimd' in features or 'neon' in features
                arm_features['asimd'] = 'asimd' in features
                arm_features['fp16'] = 'fp16' in features or 'fphp' in features
                arm_features['asimdhp'] = 'asimdhp' in features
                arm_features['asimddp'] = 'asimddp' in features
                arm_features['asimdfhm'] = 'asimdfhm' in features
                arm_features['crc32'] = 'crc32' in features
                arm_features['sha1'] = 'sha1' in features
                arm_features['sha2'] = 'sha2' in features
                arm_features['aes'] = 'aes' in features
                arm_features['pmull'] = 'pmull' in features
            
            # Detect ARM processor type
            if 'cpu_part' in cpu_info:
                cpu_part = cpu_info['cpu_part']
                arm_features['cortex_type'] = self.decode_cortex_type(cpu_part)
            
            # Get performance counters availability
            try:
                result = subprocess.run(['perf', 'list'], capture_output=True, text=True)
                arm_features['perf_available'] = result.returncode == 0
            except FileNotFoundError:
                arm_features['perf_available'] = False
                
        except Exception as e:
            print(f"Error detecting ARM features: {e}")
            
        return arm_features
    
    def decode_cortex_type(self, cpu_part):
        """Decode ARM CPU part to Cortex type"""
        cortex_map = {
            '0xd03': 'Cortex-A53',
            '0xd07': 'Cortex-A57',
            '0xd08': 'Cortex-A72',
            '0xd09': 'Cortex-A73',
            '0xd0a': 'Cortex-A75',
            '0xd0b': 'Cortex-A76',
            '0xd0c': 'Cortex-A76AE',
            '0xd0d': 'Cortex-A77',
            '0xd0e': 'Cortex-A78',
            '0xd40': 'Neoverse-N1',
            '0xd41': 'Cortex-A78AE',
            '0xd42': 'Cortex-A78C',
            '0xd43': 'Cortex-A710',
            '0xd44': 'Cortex-X2',
            '0xd46': 'Cortex-A715',
            '0xd47': 'Cortex-X3',
            '0xd48': 'Neoverse-V2',
            '0xd49': 'Neoverse-N2',
            '0xd4a': 'Neoverse-E1',
            '0xd4b': 'Cortex-A78C',
        }
        
        return cortex_map.get(cpu_part, f'Unknown ARM CPU (part: {cpu_part})')
    
    def benchmark_memory_performance(self):
        """Benchmark memory performance"""
        print("⚡ Benchmarking memory performance...")
        
        try:
            import numpy as np
            
            # Memory bandwidth test
            sizes = [1024*1024, 10*1024*1024, 100*1024*1024]  # 1MB, 10MB, 100MB
            bandwidth_results = {}
            
            for size in sizes:
                # Create random arrays
                a = np.random.randint(0, 256, size=size//4, dtype=np.uint32)
                b = np.random.randint(0, 256, size=size//4, dtype=np.uint32)
                
                # Benchmark memory copy
                start_time = time.time()
                c = a + b
                end_time = time.time()
                
                duration = end_time - start_time
                bandwidth_mb = (size * 3) / (1024*1024) / duration  # MB/s
                
                bandwidth_results[f'{size//(1024*1024)}MB'] = {
                    'bandwidth_mb_s': bandwidth_mb,
                    'duration_s': duration
                }
                
                print(f"  {size//(1024*1024)}MB: {bandwidth_mb:.1f} MB/s")
            
            return bandwidth_results
            
        except ImportError:
            print("  NumPy not available for memory benchmark")
            return {}
        except Exception as e:
            print(f"  Memory benchmark error: {e}")
            return {}
    
    def benchmark_cpu_performance(self):
        """Benchmark CPU performance"""
        print("🔥 Benchmarking CPU performance...")
        
        try:
            import numpy as np
            
            # Matrix multiplication benchmark
            sizes = [64, 128, 256, 512]
            cpu_results = {}
            
            for size in sizes:
                # Create random matrices
                a = np.random.rand(size, size).astype(np.float32)
                b = np.random.rand(size, size).astype(np.float32)
                
                # Benchmark matrix multiplication
                start_time = time.time()
                c = np.dot(a, b)
                end_time = time.time()
                
                duration = end_time - start_time
                operations = 2 * size ** 3  # FLOPs for matrix multiplication
                gflops = operations / (duration * 1e9)
                
                cpu_results[f'{size}x{size}'] = {
                    'gflops': gflops,
                    'duration_s': duration
                }
                
                print(f"  {size}x{size}: {gflops:.2f} GFLOPS")
            
            return cpu_results
            
        except ImportError:
            print("  NumPy not available for CPU benchmark")
            return {}
        except Exception as e:
            print(f"  CPU benchmark error: {e}")
            return {}
    
    def analyze_linux_environment(self):
        """Analyze Linux environment for AI optimization"""
        print("🐧 Analyzing Linux environment...")
        
        linux_info = {}
        
        try:
            # OS information
            if os.path.exists('/etc/os-release'):
                with open('/etc/os-release', 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"')
                            
                            if key == 'ID':
                                linux_info['os_id'] = value
                            elif key == 'ID_LIKE':
                                linux_info['os_like'] = value
                            elif key == 'VERSION_ID':
                                linux_info['version_id'] = value
                            elif key == 'PRETTY_NAME':
                                linux_info['pretty_name'] = value
            
            # Kernel information
            linux_info['kernel'] = platform.release()
            linux_info['kernel_version'] = platform.version()
            
            # Check for GPU acceleration support
            try:
                result = subprocess.run(['lspci'], capture_output=True, text=True)
                if result.returncode == 0:
                    linux_info['has_gpu'] = any(gpu.lower() in result.stdout.lower() 
                                              for gpu in ['vga', 'display', '3d', 'gpu'])
            except FileNotFoundError:
                linux_info['has_gpu'] = False
            
            # Check for ARM-specific optimizations
            linux_info['arm_optimizations'] = {
                'vfpv4_available': os.path.exists('/proc/cpuinfo'),  # Basic check
                'neon_available': os.path.exists('/proc/cpuinfo'),
                'big_little': False  # Would need more detailed detection
            }
            
        except Exception as e:
            print(f"Error analyzing Linux environment: {e}")
            
        return linux_info
    
    def generate_hardware_profile(self):
        """Generate complete hardware profile"""
        print("\n" + "="*60)
        print("ARDUINO UNO Q4GB HARDWARE ANALYSIS")
        print("="*60)
        
        # Collect all hardware information
        self.hardware_profile = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'cpu': self.get_cpu_info(),
            'memory': self.get_memory_info(),
            'arm_features': self.detect_arm_features(),
            'linux_environment': self.analyze_linux_environment()
        }
        
        # Run benchmarks
        self.benchmark_results = {
            'memory_performance': self.benchmark_memory_performance(),
            'cpu_performance': self.benchmark_cpu_performance()
        }
        
        # Add benchmarks to profile
        self.hardware_profile['benchmarks'] = self.benchmark_results
        
        return self.hardware_profile
    
    def get_optimization_recommendations(self):
        """Get hardware-specific optimization recommendations"""
        print("\n🎯 OPTIMIZATION RECOMMENDATIONS")
        print("-"*40)
        
        recommendations = {
            'ai_framework': 'onnx',
            'model_precision': 'int8',
            'batch_size': 1,
            'inference_threads': 2,
            'memory_optimization': 'aggressive',
            'hardware_acceleration': 'none'
        }
        
        cpu_info = self.hardware_profile.get('cpu', {})
        arm_features = self.hardware_profile.get('arm_features', {})
        memory_info = self.hardware_profile.get('memory', {})
        
        # Framework selection based on hardware
        if arm_features.get('asimd', False):
            print("✅ NEON/ASIMD detected - ONNX Runtime optimized")
            recommendations['ai_framework'] = 'onnx'
        else:
            print("⚠️  Limited SIMD support - TensorFlow Lite recommended")
            recommendations['ai_framework'] = 'tflite'
        
        # Memory optimization based on available RAM
        if memory_info.get('total_mb', 0) < 1024:  # Less than 1GB
            print("⚠️  Low memory detected - aggressive optimization needed")
            recommendations['model_precision'] = 'int8'
            recommendations['memory_optimization'] = 'very_aggressive'
        elif memory_info.get('total_mb', 0) < 2048:  # Less than 2GB
            print("✅ Moderate memory - standard optimization")
            recommendations['model_precision'] = 'int8'
            recommendations['memory_optimization'] = 'standard'
        else:
            print("✅ Good memory available - balanced optimization")
            recommendations['model_precision'] = 'fp16'
        
        # Threading optimization based on CPU count
        cpu_count = cpu_info.get('cpu_count', 1)
        if cpu_count >= 4:
            recommendations['inference_threads'] = 4
        elif cpu_count >= 2:
            recommendations['inference_threads'] = 2
        else:
            recommendations['inference_threads'] = 1
        
        print(f"📊 Recommended AI Framework: {recommendations['ai_framework']}")
        print(f"📊 Model Precision: {recommendations['model_precision']}")
        print(f"📊 Inference Threads: {recommendations['inference_threads']}")
        print(f"📊 Memory Optimization: {recommendations['memory_optimization']}")
        
        return recommendations
    
    def save_profile(self, filename='hardware_profile.json'):
        """Save hardware profile to file"""
        profile_data = {
            'hardware_profile': self.hardware_profile,
            'optimization_recommendations': self.get_optimization_recommendations()
        }
        
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
        
        print(f"\n💾 Hardware profile saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print hardware analysis summary"""
        cpu_info = self.hardware_profile.get('cpu', {})
        memory_info = self.hardware_profile.get('memory', {})
        arm_features = self.hardware_profile.get('arm_features', {})
        
        print("\n" + "="*60)
        print("HARDWARE ANALYSIS SUMMARY")
        print("="*60)
        print(f"CPU: {cpu_info.get('model_name', 'Unknown')}")
        print(f"Architecture: {cpu_info.get('architecture', 'Unknown')}")
        print(f"Cores: {cpu_info.get('cpu_count', 'Unknown')}")
        print(f"Max Frequency: {cpu_info.get('max_mhz', 'Unknown')} MHz")
        print(f"Memory: {memory_info.get('total', 'Unknown')}")
        print(f"NEON Support: {'Yes' if arm_features.get('neon') else 'No'}")
        print(f"ASIMD Support: {'Yes' if arm_features.get('asimd') else 'No'}")
        print(f"FP16 Support: {'Yes' if arm_features.get('fp16') else 'No'}")
        print("="*60)

def main():
    """Main function to run hardware analysis"""
    analyzer = ArduinoQ4GBHardwareAnalyzer()
    
    try:
        # Generate hardware profile
        profile = analyzer.generate_hardware_profile()
        
        # Print summary
        analyzer.print_summary()
        
        # Get optimization recommendations
        recommendations = analyzer.get_optimization_recommendations()
        
        # Save profile
        profile_file = analyzer.save_profile('arduino_q4gb_hardware_profile.json')
        
        print(f"\n🎉 Hardware analysis complete!")
        print(f"📄 Profile saved: {profile_file}")
        print(f"🎯 Use this profile for Phase 3 optimization setup")
        
        return 0
        
    except Exception as e:
        print(f"❌ Hardware analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())