#!/usr/bin/env python3
"""
Arduino UNO Q4GB Performance Benchmark Suite
Phase 3: Hardware-Specific Optimization
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path

class ArduinoQ4GBBenchmarkSuite:
    def __init__(self):
        self.results = {}
        self.hardware_profile = {}
        
    def load_hardware_profile(self, profile_file='hardware_profile.json'):
        """Load hardware profile from previous analysis"""
        try:
            with open(profile_file, 'r') as f:
                data = json.load(f)
                self.hardware_profile = data.get('hardware_profile', {})
                return True
        except FileNotFoundError:
            print("⚠️  Hardware profile not found. Run hardware_analyzer.py first.")
            return False
        except Exception as e:
            print(f"❌ Error loading hardware profile: {e}")
            return False
    
    def benchmark_numpy_operations(self):
        """Benchmark NumPy operations for AI workloads"""
        print("🔢 Benchmarking NumPy operations...")
        
        results = {}
        
        try:
            # Matrix operations
            sizes = [64, 128, 256, 512]
            
            for size in sizes:
                # Matrix multiplication
                a = np.random.rand(size, size).astype(np.float32)
                b = np.random.rand(size, size).astype(np.float32)
                
                start_time = time.time()
                c = np.dot(a, b)
                end_time = time.time()
                
                matmul_time = end_time - start_time
                
                # Element-wise operations
                start_time = time.time()
                d = a * b + a
                end_time = time.time()
                
                elementwise_time = end_time - start_time
                
                # Reduction operations
                start_time = time.time()
                s = np.sum(a, axis=1)
                end_time = time.time()
                
                reduction_time = end_time - start_time
                
                results[f'{size}x{size}'] = {
                    'matmul_time': matmul_time,
                    'elementwise_time': elementwise_time,
                    'reduction_time': reduction_time,
                    'matmul_gflops': (2 * size ** 3) / (matmul_time * 1e9) if matmul_time > 0 else 0
                }
                
                print(f"  {size}x{size}: MatMul {matmul_time:.3f}s, Elementwise {elementwise_time:.3f}s, Reduction {reduction_time:.3f}s")
                
        except Exception as e:
            print(f"  NumPy benchmark error: {e}")
            
        return results
    
    def benchmark_memory_bandwidth(self):
        """Benchmark memory bandwidth with different patterns"""
        print("🧠 Benchmarking memory bandwidth...")
        
        results = {}
        
        try:
            # Different access patterns
            sizes = [10*1024*1024, 50*1024*1024, 100*1024*1024]  # 10MB, 50MB, 100MB
            
            for size in sizes:
                array_size = size // 4  # 32-bit integers
                data = np.random.randint(0, 256, size=array_size, dtype=np.uint32)
                
                # Sequential access
                start_time = time.time()
                total = np.sum(data)
                end_time = time.time()
                sequential_time = end_time - start_time
                sequential_bandwidth = size / (1024*1024) / sequential_time  # MB/s
                
                # Random access simulation
                indices = np.random.permutation(array_size)
                start_time = time.time()
                total = np.sum(data[indices])
                end_time = time.time()
                random_time = end_time - start_time
                random_bandwidth = size / (1024*1024) / random_time  # MB/s
                
                # Memory copy
                copy_data = np.zeros_like(data)
                start_time = time.time()
                np.copyto(copy_data, data)
                end_time = time.time()
                copy_time = end_time - start_time
                copy_bandwidth = (size * 2) / (1024*1024) / copy_time  # MB/s (read+write)
                
                results[f'{size//(1024*1024)}MB'] = {
                    'sequential_bandwidth_mb_s': sequential_bandwidth,
                    'random_bandwidth_mb_s': random_bandwidth,
                    'copy_bandwidth_mb_s': copy_bandwidth,
                    'sequential_time': sequential_time,
                    'random_time': random_time,
                    'copy_time': copy_time
                }
                
                print(f"  {size//(1024*1024)}MB: Sequential {sequential_bandwidth:.1f} MB/s, Random {random_bandwidth:.1f} MB/s, Copy {copy_bandwidth:.1f} MB/s")
                
        except Exception as e:
            print(f"  Memory bandwidth benchmark error: {e}")
            
        return results
    
    def benchmark_ai_inference_simulation(self):
        """Simulate AI inference workloads"""
        print("🤖 Benchmarking AI inference simulation...")
        
        results = {}
        
        try:
            # Simulate different model sizes (based on operations)
            model_configs = {
                'tiny': {'input_size': (64, 64, 3), 'operations': 100000},
                'small': {'input_size': (128, 128, 3), 'operations': 500000},
                'medium': {'input_size': (256, 256, 3), 'operations': 2000000},
                'large': {'input_size': (512, 512, 3), 'operations': 8000000}
            }
            
            for model_name, config in model_configs.items():
                input_size = config['input_size']
                operations = config['operations']
                
                # Create input tensor
                input_data = np.random.rand(*input_size).astype(np.float32)
                
                # Simulate convolutions with matrix operations
                conv_time = 0
                for _ in range(10):  # Multiple conv layers
                    # Simulate convolution with matrix multiplication
                    start_time = time.time()
                    reshaped = input_data.reshape(-1, input_size[-1])
                    weights = np.random.rand(input_size[-1], 32).astype(np.float32)
                    output = np.dot(reshaped, weights)
                    conv_time += time.time() - start_time
                
                # Simulate activation functions
                start_time = time.time()
                activated = np.maximum(output, 0)  # ReLU
                activation_time = time.time() - start_time
                
                # Simulate pooling
                start_time = time.time()
                pooled = activated.reshape(8, -1, 32).max(axis=1)
                pooling_time = time.time() - start_time
                
                total_time = conv_time + activation_time + pooling_time
                ops_per_second = operations / total_time if total_time > 0 else 0
                fps = 1.0 / total_time if total_time > 0 else 0
                
                results[model_name] = {
                    'conv_time': conv_time,
                    'activation_time': activation_time,
                    'pooling_time': pooling_time,
                    'total_time': total_time,
                    'ops_per_second': ops_per_second,
                    'fps': fps
                }
                
                print(f"  {model_name:6s}: {total_time:.3f}s, {fps:.1f} FPS, {ops_per_second/1e6:.1f} MOPS")
                
        except Exception as e:
            print(f"  AI inference benchmark error: {e}")
            
        return results
    
    def benchmark_threading_performance(self):
        """Benchmark multi-threading performance"""
        print("🧵 Benchmarking threading performance...")
        
        results = {}
        
        try:
            # Test different thread counts
            thread_counts = [1, 2, 4]
            matrix_size = 256
            
            for threads in thread_counts:
                os.environ['OMP_NUM_THREADS'] = str(threads)
                os.environ['MKL_NUM_THREADS'] = str(threads)
                os.environ['NUMEXPR_NUM_THREADS'] = str(threads)
                
                # Multiple matrix multiplications
                a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
                b = np.random.rand(matrix_size, matrix_size).astype(np.float32)
                
                start_time = time.time()
                for _ in range(10):
                    c = np.dot(a, b)
                end_time = time.time()
                
                total_time = end_time - start_time
                avg_time = total_time / 10
                
                results[f'{threads}_threads'] = {
                    'total_time': total_time,
                    'avg_time': avg_time,
                    'speedup': 1.0 / avg_time if avg_time > 0 else 0
                }
                
                print(f"  {threads} threads: {avg_time:.3f}s per operation")
            
            # Calculate speedup relative to 1 thread
            if '1_threads' in results:
                baseline = results['1_threads']['avg_time']
                for threads in thread_counts:
                    if f'{threads}_threads' in results:
                        speedup = baseline / results[f'{threads}_threads']['avg_time']
                        results[f'{threads}_threads']['speedup'] = speedup
                        print(f"    Speedup ({threads} threads): {speedup:.2f}x")
                
        except Exception as e:
            print(f"  Threading benchmark error: {e}")
            
        return results
    
    def benchmark_power_consumption(self):
        """Estimate power consumption (basic estimation)"""
        print("⚡ Estimating power consumption...")
        
        results = {}
        
        try:
            # Check for power-related files
            power_info = {}
            
            # Check /sys/class/power_supply if available
            if os.path.exists('/sys/class/power_supply'):
                for supply in os.listdir('/sys/class/power_supply'):
                    supply_path = f'/sys/class/power_supply/{supply}'
                    if os.path.isdir(supply_path):
                        try:
                            with open(f'{supply_path}/type', 'r') as f:
                                supply_type = f.read().strip()
                            power_info[supply] = {'type': supply_type}
                            
                            if supply_type == 'Battery':
                                try:
                                    with open(f'{supply_path}/capacity', 'r') as f:
                                        power_info[supply]['capacity'] = f.read().strip()
                                    with open(f'{supply_path}/status', 'r') as f:
                                        power_info[supply]['status'] = f.read().strip()
                                except:
                                    pass
                        except:
                            pass
            
            # Check thermal information
            thermal_info = {}
            if os.path.exists('/sys/class/thermal'):
                for thermal in os.listdir('/sys/class/thermal'):
                    thermal_path = f'/sys/class/thermal/{thermal}'
                    if os.path.isdir(thermal_path):
                        try:
                            with open(f'{thermal_path}/temp', 'r') as f:
                                temp = f.read().strip()
                                thermal_info[thermal] = {'temp': temp}
                        except:
                            pass
            
            results['power_supply'] = power_info
            results['thermal'] = thermal_info
            results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"  Power supplies found: {len(power_info)}")
            print(f"  Thermal sensors found: {len(thermal_info)}")
            
        except Exception as e:
            print(f"  Power consumption estimation error: {e}")
            
        return results
    
    def run_comprehensive_benchmark(self):
        """Run all benchmark tests"""
        print("\n" + "="*60)
        print("ARDUINO UNO Q4GB PERFORMANCE BENCHMARK SUITE")
        print("="*60)
        
        # Load hardware profile if available
        if not self.load_hardware_profile():
            print("⚠️  Continuing without hardware profile...")
        
        # Run all benchmarks
        benchmark_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'numpy_operations': self.benchmark_numpy_operations(),
            'memory_bandwidth': self.benchmark_memory_bandwidth(),
            'ai_inference_simulation': self.benchmark_ai_inference_simulation(),
            'threading_performance': self.benchmark_threading_performance(),
            'power_consumption': self.benchmark_power_consumption()
        }
        
        self.results = benchmark_results
        
        return benchmark_results
    
    def analyze_performance(self):
        """Analyze performance and provide optimization recommendations"""
        print("\n🎯 PERFORMANCE ANALYSIS")
        print("-"*40)
        
        recommendations = {}
        
        # Analyze AI inference performance
        ai_results = self.results.get('ai_inference_simulation', {})
        if ai_results:
            # Check if medium model can run at reasonable speed
            medium_fps = ai_results.get('medium', {}).get('fps', 0)
            if medium_fps >= 10:
                print("✅ Medium models perform well (10+ FPS)")
                recommendations['recommended_model_size'] = 'medium'
            elif medium_fps >= 5:
                print("⚠️  Medium models borderline (5-10 FPS)")
                recommendations['recommended_model_size'] = 'small'
            else:
                print("❌ Medium models too slow, use tiny/small")
                recommendations['recommended_model_size'] = 'small' if ai_results.get('small', {}).get('fps', 0) >= 10 else 'tiny'
        
        # Analyze threading performance
        threading_results = self.results.get('threading_performance', {})
        if threading_results:
            speedup_2_threads = threading_results.get('2_threads', {}).get('speedup', 0)
            speedup_4_threads = threading_results.get('4_threads', {}).get('speedup', 0)
            
            if speedup_2_threads > 1.5:
                print("✅ Good multi-threading performance")
                recommendations['recommended_threads'] = 4 if speedup_4_threads > 2.5 else 2
            else:
                print("⚠️  Limited multi-threading benefit")
                recommendations['recommended_threads'] = 1
        
        # Analyze memory bandwidth
        memory_results = self.results.get('memory_bandwidth', {})
        if memory_results:
            # Check 100MB sequential bandwidth
            bandwidth_100mb = memory_results.get('100MB', {}).get('sequential_bandwidth_mb_s', 0)
            if bandwidth_100mb > 1000:  # > 1 GB/s
                print("✅ Excellent memory bandwidth")
                recommendations['memory_optimization'] = 'standard'
            elif bandwidth_100mb > 500:  # > 500 MB/s
                print("⚠️  Moderate memory bandwidth")
                recommendations['memory_optimization'] = 'conservative'
            else:
                print("❌ Limited memory bandwidth")
                recommendations['memory_optimization'] = 'aggressive'
        
        return recommendations
    
    def save_benchmark_results(self, filename='benchmark_results.json'):
        """Save benchmark results to file"""
        results_data = {
            'benchmark_results': self.results,
            'performance_recommendations': self.analyze_performance(),
            'hardware_profile': self.hardware_profile
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n💾 Benchmark results saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # AI Performance Summary
        ai_results = self.results.get('ai_inference_simulation', {})
        if ai_results:
            print("🤖 AI Inference Performance:")
            for model, results in ai_results.items():
                fps = results.get('fps', 0)
                ops = results.get('ops_per_second', 0) / 1e6
                print(f"  {model:6s}: {fps:5.1f} FPS, {ops:6.1f} MOPS")
        
        # Memory Performance Summary
        memory_results = self.results.get('memory_bandwidth', {})
        if memory_results:
            print("🧠 Memory Bandwidth:")
            for size, results in memory_results.items():
                bandwidth = results.get('sequential_bandwidth_mb_s', 0)
                print(f"  {size:6s}: {bandwidth:7.1f} MB/s")
        
        # Threading Performance Summary
        threading_results = self.results.get('threading_performance', {})
        if threading_results:
            print("🧵 Threading Performance:")
            for threads, results in threading_results.items():
                speedup = results.get('speedup', 0)
                time_ms = results.get('avg_time', 0) * 1000
                print(f"  {threads:10s}: {time_ms:6.1f}ms, {speedup:4.2f}x speedup")
        
        print("="*60)

def main():
    """Main function to run benchmark suite"""
    benchmark = ArduinoQ4GBBenchmarkSuite()
    
    try:
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark()
        
        # Analyze performance
        recommendations = benchmark.analyze_performance()
        
        # Print summary
        benchmark.print_summary()
        
        # Save results
        results_file = benchmark.save_benchmark_results('arduino_q4gb_benchmark_results.json')
        
        print(f"\n🎉 Benchmark suite complete!")
        print(f"📄 Results saved: {results_file}")
        print(f"🎯 Use these results for Phase 3 optimization")
        
        return 0
        
    except Exception as e:
        print(f"❌ Benchmark suite failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())