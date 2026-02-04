#!/usr/bin/env python3
"""
Arduino UNO Q4GB AI Framework Selector
Phase 3: Hardware-Specific Optimization
Automatically selects optimal AI framework based on hardware capabilities
"""

import os
import sys
import json
import subprocess
from pathlib import Path

class FrameworkSelector:
    def __init__(self):
        self.hardware_profile = {}
        self.framework_scores = {}
        self.recommended_framework = None
        
    def load_hardware_profile(self, profile_file='hardware_profile.json'):
        """Load hardware profile"""
        try:
            with open(profile_file, 'r') as f:
                data = json.load(f)
                self.hardware_profile = data.get('hardware_profile', {})
                return True
        except FileNotFoundError:
            print("❌ Hardware profile not found. Run hardware_analyzer.py first.")
            return False
        except Exception as e:
            print(f"❌ Error loading hardware profile: {e}")
            return False
    
    def evaluate_onnx_runtime(self):
        """Evaluate ONNX Runtime compatibility"""
        print("🔍 Evaluating ONNX Runtime compatibility...")
        
        score = 0
        details = []
        
        try:
            # Check if ONNX Runtime is available
            result = subprocess.run([sys.executable, '-c', 'import onnxruntime'], 
                                  capture_output=True)
            if result.returncode == 0:
                score += 40
                details.append("ONNX Runtime available")
            else:
                details.append("ONNX Runtime not installed")
                return 0, details
        except:
            details.append("ONNX Runtime import failed")
            return 0, details
        
        # ARM-specific optimizations
        cpu_info = self.hardware_profile.get('cpu', {})
        arm_features = self.hardware_profile.get('arm_features', {})
        
        if arm_features.get('neon') or arm_features.get('asimd'):
            score += 20
            details.append("ARM NEON/ASIMD support detected")
        
        if arm_features.get('fp16'):
            score += 10
            details.append("FP16 support available")
        
        # Memory compatibility
        memory_info = self.hardware_profile.get('memory', {})
        total_memory_mb = memory_info.get('total_mb', 0)
        
        if total_memory_mb >= 1024:  # 1GB+
            score += 15
            details.append("Sufficient memory for ONNX Runtime")
        elif total_memory_mb >= 512:  # 512MB+
            score += 10
            details.append("Minimal memory for ONNX Runtime")
        else:
            score -= 10
            details.append("Insufficient memory for ONNX Runtime")
        
        # CPU compatibility
        if cpu_info.get('cpu_count', 1) >= 2:
            score += 15
            details.append("Multi-core CPU detected")
        
        return score, details
    
    def evaluate_tensorflow_lite(self):
        """Evaluate TensorFlow Lite compatibility"""
        print("🔍 Evaluating TensorFlow Lite compatibility...")
        
        score = 0
        details = []
        
        try:
            # Check if TensorFlow Lite is available
            result = subprocess.run([sys.executable, '-c', 'import tflite_runtime'], 
                                  capture_output=True)
            if result.returncode == 0:
                score += 40
                details.append("TensorFlow Lite Runtime available")
            else:
                # Try full TensorFlow
                result = subprocess.run([sys.executable, '-c', 'import tensorflow'], 
                                      capture_output=True)
                if result.returncode == 0:
                    score += 35
                    details.append("TensorFlow available (not TFLite Runtime)")
                else:
                    details.append("TensorFlow Lite not installed")
                    return 0, details
        except:
            details.append("TensorFlow Lite import failed")
            return 0, details
        
        # ARM optimizations
        arm_features = self.hardware_profile.get('arm_features', {})
        
        if arm_features.get('neon'):
            score += 15
            details.append("ARM NEON support detected")
        
        # Memory efficiency (TFLite is very memory efficient)
        memory_info = self.hardware_profile.get('memory', {})
        total_memory_mb = memory_info.get('total_mb', 0)
        
        if total_memory_mb >= 512:  # 512MB+
            score += 20
            details.append("Sufficient memory for TensorFlow Lite")
        elif total_memory_mb >= 256:  # 256MB+
            score += 15
            details.append("Minimal memory for TensorFlow Lite")
        else:
            score += 5  # TFLite can work in very low memory
            details.append("Low memory mode available")
        
        # CPU compatibility
        cpu_info = self.hardware_profile.get('cpu', {})
        if cpu_info.get('cpu_count', 1) >= 1:
            score += 10
            details.append("Single-core compatible")
        
        return score, details
    
    def evaluate_pytorch(self):
        """Evaluate PyTorch compatibility (as fallback)"""
        print("🔍 Evaluating PyTorch compatibility...")
        
        score = 0
        details = []
        
        try:
            # Check if PyTorch is available and works
            result = subprocess.run([sys.executable, '-c', 'import torch; print(torch.__version__)'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                score += 30
                details.append(f"PyTorch {version} available")
                
                # Test if PyTorch actually works (no illegal instruction)
                test_result = subprocess.run([sys.executable, '-c', 'import torch; x = torch.rand(10, 10); print(x.sum())'], 
                                           capture_output=True, text=True)
                if test_result.returncode == 0:
                    score += 20
                    details.append("PyTorch functional")
                else:
                    score -= 30
                    details.append("PyTorch has runtime issues")
                    return score if score > 0 else 0, details
            else:
                details.append("PyTorch not installed")
                return 0, details
        except:
            details.append("PyTorch import failed")
            return 0, details
        
        # ARM compatibility issues (PyTorch often has problems)
        cpu_info = self.hardware_profile.get('cpu', {})
        arm_features = self.hardware_profile.get('arm_features', {})
        
        # PyTorch ARM compatibility is often problematic
        score -= 10
        details.append("PyTorch ARM compatibility concerns")
        
        # Memory requirements (PyTorch is memory hungry)
        memory_info = self.hardware_profile.get('memory', {})
        total_memory_mb = memory_info.get('total_mb', 0)
        
        if total_memory_mb >= 2048:  # 2GB+
            score += 15
            details.append("Sufficient memory for PyTorch")
        elif total_memory_mb >= 1024:  # 1GB+
            score += 5
            details.append("Minimal memory for PyTorch")
        else:
            score -= 15
            details.append("Insufficient memory for PyTorch")
        
        return max(0, score), details
    
    def evaluate_openvino(self):
        """Evaluate OpenVINO compatibility (Intel ARM64)"""
        print("🔍 Evaluating OpenVINO compatibility...")
        
        score = 0
        details = []
        
        try:
            # Check if OpenVINO is available
            result = subprocess.run([sys.executable, '-c', 'import openvino'], 
                                  capture_output=True)
            if result.returncode == 0:
                score += 40
                details.append("OpenVINO available")
            else:
                details.append("OpenVINO not installed")
                return 0, details
        except:
            details.append("OpenVINO import failed")
            return 0, details
        
        # ARM64 support (OpenVINO has limited ARM64 support)
        cpu_info = self.hardware_profile.get('cpu', {})
        architecture = cpu_info.get('architecture', '')
        
        if 'aarch64' in architecture:
            score += 20
            details.append("ARM64 support detected")
        else:
            score -= 10
            details.append("Limited ARM support")
        
        return max(0, score), details
    
    def evaluate_all_frameworks(self):
        """Evaluate all available AI frameworks"""
        print("\n🎯 Evaluating AI Frameworks...")
        print("="*50)
        
        frameworks = {
            'onnx': self.evaluate_onnx_runtime,
            'tflite': self.evaluate_tensorflow_lite,
            'pytorch': self.evaluate_pytorch,
            'openvino': self.evaluate_openvino
        }
        
        for framework_name, evaluator in frameworks.items():
            score, details = evaluator()
            self.framework_scores[framework_name] = {
                'score': score,
                'details': details
            }
            
            print(f"\n{framework_name.upper()}:")
            print(f"  Score: {score}/100")
            for detail in details:
                print(f"    - {detail}")
        
        # Select best framework
        best_framework = max(self.framework_scores.keys(), 
                            key=lambda x: self.framework_scores[x]['score'])
        best_score = self.framework_scores[best_framework]['score']
        
        if best_score < 30:
            print(f"\n⚠️  No suitable framework found (best score: {best_score})")
            self.recommended_framework = None
        else:
            print(f"\n✅ Recommended framework: {best_framework.upper()} (Score: {best_score}/100)")
            self.recommended_framework = best_framework
        
        return self.recommended_framework
    
    def generate_installation_script(self, framework=None):
        """Generate installation script for selected framework"""
        if framework is None:
            framework = self.recommended_framework
        
        if framework is None:
            print("❌ No framework selected for installation")
            return None
        
        script_path = f"install_{framework}.sh"
        
        scripts = {
            'onnx': '''#!/bin/bash
# ONNX Runtime Installation Script for Arduino UNO Q4GB

echo "Installing ONNX Runtime..."

# Install CPU-optimized ONNX Runtime
pip install onnxruntime

# Install additional dependencies
pip install numpy pillow opencv-python

echo "ONNX Runtime installation complete!"
''',
            'tflite': '''#!/bin/bash
# TensorFlow Lite Installation Script for Arduino UNO Q4GB

echo "Installing TensorFlow Lite..."

# Install TensorFlow Lite Runtime (preferred for embedded)
pip install tflite-runtime

# Fallback to full TensorFlow if needed
if [ $? -ne 0 ]; then
    echo "TFLite Runtime failed, installing full TensorFlow..."
    pip install tensorflow
fi

# Install additional dependencies
pip install numpy pillow opencv-python

echo "TensorFlow Lite installation complete!"
''',
            'pytorch': '''#!/bin/bash
# PyTorch Installation Script for Arduino UNO Q4GB

echo "Installing PyTorch..."

# Install CPU-only PyTorch for ARM64
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install additional dependencies
pip install numpy pillow opencv-python

echo "PyTorch installation complete!"
''',
            'openvino': '''#!/bin/bash
# OpenVINO Installation Script for Arduino UNO Q4GB

echo "Installing OpenVINO..."

# Install OpenVINO
pip install openvino

# Install additional dependencies
pip install numpy pillow opencv-python

echo "OpenVINO installation complete!"
'''
        }
        
        with open(script_path, 'w') as f:
            f.write(scripts.get(framework, '# Unknown framework\n'))
        
        os.chmod(script_path, 0o755)
        print(f"📄 Installation script generated: {script_path}")
        
        return script_path
    
    def create_framework_config(self, framework=None):
        """Create configuration file for selected framework"""
        if framework is None:
            framework = self.recommended_framework
        
        if framework is None:
            print("❌ No framework selected for configuration")
            return None
        
        config = {
            'selected_framework': framework,
            'framework_scores': self.framework_scores,
            'hardware_profile': self.hardware_profile,
            'timestamp': subprocess.run(['date'], capture_output=True, text=True).stdout.strip()
        }
        
        config_file = f'framework_config_{framework}.json'
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"📄 Framework configuration saved: {config_file}")
        return config_file
    
    def print_recommendation_summary(self):
        """Print detailed recommendation summary"""
        print("\n" + "="*60)
        print("AI FRAMEWORK RECOMMENDATION SUMMARY")
        print("="*60)
        
        if self.recommended_framework is None:
            print("❌ No suitable AI framework found")
            print("\nTroubleshooting:")
            print("1. Check hardware profile compatibility")
            print("2. Install framework dependencies manually")
            print("3. Consider using lighter frameworks")
            return
        
        print(f"🎯 RECOMMENDED: {self.recommended_framework.upper()}")
        print(f"📊 Score: {self.framework_scores[self.recommended_framework]['score']}/100")
        
        print(f"\n📋 Framework Comparison:")
        for framework, data in self.framework_scores.items():
            status = "✅" if framework == self.recommended_framework else "  "
            print(f"{status} {framework:12s}: {data['score']:3d}/100")
        
        print(f"\n🔧 Installation:")
        print(f"  Run: ./install_{self.recommended_framework}.sh")
        
        print(f"\n📄 Configuration:")
        print(f"  Config file: framework_config_{self.recommended_framework}.json")
        
        print("="*60)

def main():
    """Main function for framework selection"""
    selector = FrameworkSelector()
    
    # Load hardware profile
    if not selector.load_hardware_profile('arduino_q4gb_hardware_profile.json'):
        print("❌ Cannot proceed without hardware profile")
        return 1
    
    # Evaluate all frameworks
    recommended = selector.evaluate_all_frameworks()
    
    # Generate installation script
    if recommended:
        selector.generate_installation_script()
        selector.create_framework_config()
    
    # Print summary
    selector.print_recommendation_summary()
    
    return 0 if recommended else 1

if __name__ == "__main__":
    sys.exit(main())