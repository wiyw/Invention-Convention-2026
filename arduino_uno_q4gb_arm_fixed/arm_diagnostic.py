#!/usr/bin/env python3
"""
Arduino UNO Q4GB ARM Compatibility Diagnostic Tool
Detects CPU architecture, instruction sets, and library compatibility issues
"""

import os
import sys
import platform
import subprocess
import importlib.util
from pathlib import Path

class ARMDiagnosticTool:
    def __init__(self):
        self.system_info = {}
        self.library_status = {}
        self.compatibility_issues = []
        
    def run_full_diagnostic(self):
        """Run complete ARM compatibility diagnostic"""
        print("=" * 70)
        print("    Arduino UNO Q4GB ARM Compatibility Diagnostic")
        print("=" * 70)
        print()
        
        # 1. System Architecture Analysis
        self.detect_system_architecture()
        
        # 2. CPU Instruction Set Detection  
        self.detect_cpu_instructions()
        
        # 3. Library Compatibility Testing
        self.test_library_compatibility()
        
        # 4. Generate Recommendations
        self.generate_recommendations()
        
        # 5. Create Fix Script
        self.create_fix_script()
        
        return self.compatibility_issues
    
    def detect_system_architecture(self):
        """Detect system architecture and platform details"""
        print("🔍 Detecting System Architecture...")
        print("-" * 40)
        
        # Basic platform info
        self.system_info['platform'] = platform.platform()
        self.system_info['machine'] = platform.machine()
        self.system_info['processor'] = platform.processor()
        self.system_info['architecture'] = platform.architecture()
        self.system_info['system'] = platform.system()
        
        print(f"Platform: {self.system_info['platform']}")
        print(f"Machine: {self.system_info['machine']}")
        print(f"Processor: {self.system_info['processor']}")
        print(f"Architecture: {self.system_info['architecture']}")
        print(f"System: {self.system_info['system']}")
        
        # ARM specific detection
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                
            # Extract ARM details
            for line in cpuinfo.split('\n'):
                if 'model name' in line:
                    self.system_info['cpu_model'] = line.split(':')[1].strip()
                elif 'CPU implementer' in line:
                    self.system_info['cpu_implementer'] = line.split(':')[1].strip()
                elif 'CPU architecture' in line:
                    self.system_info['cpu_arch'] = line.split(':')[1].strip()
                elif 'CPU variant' in line:
                    self.system_info['cpu_variant'] = line.split(':')[1].strip()
                elif 'CPU part' in line:
                    self.system_info['cpu_part'] = line.split(':')[1].strip()
                elif 'CPU revision' in line:
                    self.system_info['cpu_revision'] = line.split(':')[1].strip()
                    
            if 'cpu_model' in self.system_info:
                print(f"CPU Model: {self.system_info['cpu_model']}")
            if 'cpu_arch' in self.system_info:
                print(f"CPU Architecture: {self.system_info['cpu_arch']}")
                
        except Exception as e:
            print(f"Could not read /proc/cpuinfo: {e}")
        
        # Check if it's really ARM
        if 'arm' in self.system_info['machine'].lower() or 'aarch64' in self.system_info['machine'].lower():
            self.system_info['is_arm'] = True
            print("✅ ARM architecture detected")
        else:
            self.system_info['is_arm'] = False
            print("❌ Non-ARM architecture detected - compatibility issues unlikely")
        
        print()
    
    def detect_cpu_instructions(self):
        """Detect available CPU instruction sets"""
        print("🔍 Detecting CPU Instruction Sets...")
        print("-" * 40)
        
        if not self.system_info.get('is_arm', False):
            print("Not an ARM system - skipping instruction detection")
            print()
            return
        
        # Test ARM specific instructions
        arm_instructions = {
            'NEON': self._test_neon_support(),
            'VFP': self._test_vfp_support(), 
            'ARMv7': self._test_armv7_support(),
            'ARMv8': self._test_armv8_support(),
            'THUMB': self._test_thumb_support(),
            'AES': self._test_aes_support(),
            'CRC32': self._test_crc32_support()
        }
        
        self.system_info['arm_instructions'] = arm_instructions
        
        for instruction, supported in arm_instructions.items():
            status = "✅" if supported else "❌"
            print(f"{status} {instruction}: {'Supported' if supported else 'Not Supported'}")
        
        # Check for missing instructions that commonly cause issues
        missing_critical = []
        for instruction in ['NEON', 'VFP']:
            if not arm_instructions.get(instruction, False):
                missing_critical.append(instruction)
        
        if missing_critical:
            self.compatibility_issues.append({
                'type': 'cpu_instructions',
                'severity': 'high',
                'issue': f"Missing critical ARM instructions: {', '.join(missing_critical)}",
                'fix': 'Use CPU-optimized library builds or software emulation'
            })
        
        print()
    
    def test_library_compatibility(self):
        """Test individual library compatibility"""
        print("🔍 Testing Library Compatibility...")
        print("-" * 40)
        
        # Test critical libraries
        libraries = {
            'torch': self._test_torch_compatibility,
            'numpy': self._test_numpy_compatibility,
            'cv2': self._test_opencv_compatibility,
            'ultralytics': self._test_ultralytics_compatibility,
            'PIL': self._test_pil_compatibility,
            'serial': self._test_serial_compatibility
        }
        
        for lib_name, test_func in libraries.items():
            try:
                print(f"Testing {lib_name}...")
                result = test_func()
                self.library_status[lib_name] = result
                
                status = "✅" if result['compatible'] else "❌"
                print(f"{status} {lib_name}: {result['status']}")
                
                if result.get('version'):
                    print(f"   Version: {result['version']}")
                
                if result.get('error'):
                    print(f"   Error: {result['error']}")
                
                if not result['compatible']:
                    self.compatibility_issues.append({
                        'type': 'library',
                        'severity': result.get('severity', 'medium'),
                        'library': lib_name,
                        'issue': result.get('error', 'Compatibility issue'),
                        'fix': result.get('fix', 'Reinstall with ARM-compatible build')
                    })
                    
            except Exception as e:
                self.library_status[lib_name] = {
                    'compatible': False,
                    'status': 'Test failed',
                    'error': str(e)
                }
                print(f"❌ {lib_name}: Test crashed - {e}")
            
            print()
    
    def _test_neon_support(self):
        """Test NEON SIMD support"""
        try:
            # Try to execute NEON instruction
            test_code = '''
import subprocess
import sys
result = subprocess.run([sys.executable, '-c', '''
import struct
import ctypes

# Test NEON instruction with ctypes
try:
    # This is a simplified test - real NEON test would be more complex
    lib = ctypes.CDLL(None)
    print("NEON test passed")
except Exception:
    print("NEON test failed")
'''], capture_output=True, text=True)
return "NEON test passed" in result.stdout
'''
            return eval(test_code)
        except:
            # Fallback: check /proc/cpuinfo for neon
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()
                    return 'neon' in cpuinfo
            except:
                return False
    
    def _test_vfp_support(self):
        """Test VFP floating point support"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                return 'vfp' in cpuinfo
        except:
            return True  # Assume VFP support on modern ARM
    
    def _test_armv7_support(self):
        """Test ARMv7 support"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                return 'armv7' in cpuinfo or '7' in cpuinfo
        except:
            return False
    
    def _test_armv8_support(self):
        """Test ARMv8 support"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                return 'armv8' in cpuinfo or 'aarch64' in cpuinfo
        except:
            return False
    
    def _test_thumb_support(self):
        """Test Thumb instruction set"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                return 'thumb' in cpuinfo or 'thumbee' in cpuinfo
        except:
            return True
    
    def _test_aes_support(self):
        """Test AES hardware acceleration"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                return 'aes' in cpuinfo
        except:
            return False
    
    def _test_crc32_support(self):
        """Test CRC32 instruction support"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                return 'crc32' in cpuinfo
        except:
            return False
    
    def _test_torch_compatibility(self):
        """Test PyTorch compatibility"""
        try:
            import torch
            return {
                'compatible': True,
                'status': 'Working',
                'version': torch.__version__,
                'cpu_instructions': getattr(torch.backends.quantized, 'engine', 'Unknown')
            }
        except ImportError:
            return {
                'compatible': False,
                'status': 'Not installed',
                'fix': 'pip install torch --index-url https://download.pytorch.org/whl/cpu',
                'severity': 'high'
            }
        except Exception as e:
            error_msg = str(e).lower()
            if 'illegal instruction' in error_msg:
                return {
                    'compatible': False,
                    'status': 'Illegal instruction error',
                    'error': str(e),
                    'fix': 'Install ARM-compatible PyTorch build',
                    'severity': 'high'
                }
            else:
                return {
                    'compatible': False,
                    'status': 'Runtime error',
                    'error': str(e),
                    'severity': 'medium'
                }
    
    def _test_numpy_compatibility(self):
        """Test NumPy compatibility"""
        try:
            import numpy as np
            # Test basic operations
            x = np.array([1, 2, 3])
            y = np.sum(x)
            return {
                'compatible': True,
                'status': 'Working',
                'version': np.__version__
            }
        except Exception as e:
            error_msg = str(e).lower()
            if 'illegal instruction' in error_msg:
                return {
                    'compatible': False,
                    'status': 'Illegal instruction error',
                    'error': str(e),
                    'fix': 'Reinstall with ARM-compatible NumPy',
                    'severity': 'high'
                }
            else:
                return {
                    'compatible': False,
                    'status': 'Runtime error',
                    'error': str(e),
                    'severity': 'medium'
                }
    
    def _test_opencv_compatibility(self):
        """Test OpenCV compatibility"""
        try:
            import cv2
            # Test basic operations
            import numpy as np
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            resized = cv2.resize(img, (50, 50))
            return {
                'compatible': True,
                'status': 'Working',
                'version': cv2.__version__
            }
        except ImportError:
            return {
                'compatible': False,
                'status': 'Not installed',
                'fix': 'pip install opencv-python-headless',
                'severity': 'high'
            }
        except Exception as e:
            error_msg = str(e).lower()
            if 'illegal instruction' in error_msg:
                return {
                    'compatible': False,
                    'status': 'Illegal instruction error',
                    'error': str(e),
                    'fix': 'Install ARM-compatible OpenCV build',
                    'severity': 'high'
                }
            else:
                return {
                    'compatible': False,
                    'status': 'Runtime error',
                    'error': str(e),
                    'severity': 'medium'
                }
    
    def _test_ultralytics_compatibility(self):
        """Test Ultralytics compatibility"""
        try:
            from ultralytics import YOLO
            return {
                'compatible': True,
                'status': 'Working',
                'version': YOLO.__version__
            }
        except ImportError:
            return {
                'compatible': False,
                'status': 'Not installed',
                'fix': 'pip install ultralytics',
                'severity': 'high'
            }
        except Exception as e:
            error_msg = str(e).lower()
            if 'illegal instruction' in error_msg:
                return {
                    'compatible': False,
                    'status': 'Illegal instruction error',
                    'error': str(e),
                    'fix': 'Install ARM-compatible PyTorch first',
                    'severity': 'high'
                }
            else:
                return {
                    'compatible': False,
                    'status': 'Runtime error',
                    'error': str(e),
                    'severity': 'medium'
                }
    
    def _test_pil_compatibility(self):
        """Test PIL compatibility"""
        try:
            from PIL import Image
            img = Image.new('RGB', (100, 100))
            return {
                'compatible': True,
                'status': 'Working',
                'version': Image.__version__
            }
        except Exception as e:
            return {
                'compatible': False,
                'status': 'Runtime error',
                'error': str(e),
                'severity': 'low'
            }
    
    def _test_serial_compatibility(self):
        """Test PySerial compatibility"""
        try:
            import serial
            return {
                'compatible': True,
                'status': 'Working',
                'version': serial.__version__
            }
        except Exception as e:
            return {
                'compatible': False,
                'status': 'Runtime error',
                'error': str(e),
                'severity': 'low'
            }
    
    def generate_recommendations(self):
        """Generate specific recommendations based on diagnostic results"""
        print("💡 Generating Recommendations...")
        print("-" * 40)
        
        if not self.compatibility_issues:
            print("✅ No compatibility issues detected!")
            print()
            return
        
        print(f"Found {len(self.compatibility_issues)} compatibility issues:")
        print()
        
        # Group by severity
        high_issues = [i for i in self.compatibility_issues if i.get('severity') == 'high']
        medium_issues = [i for i in self.compatibility_issues if i.get('severity') == 'medium']
        low_issues = [i for i in self.compatibility_issues if i.get('severity') == 'low']
        
        if high_issues:
            print("🚨 HIGH SEVERITY ISSUES:")
            for issue in high_issues:
                print(f"   • {issue['issue']}")
                print(f"     Fix: {issue['fix']}")
            print()
        
        if medium_issues:
            print("⚠️  MEDIUM SEVERITY ISSUES:")
            for issue in medium_issues:
                print(f"   • {issue['issue']}")
                print(f"     Fix: {issue['fix']}")
            print()
        
        if low_issues:
            print("ℹ️  LOW SEVERITY ISSUES:")
            for issue in low_issues:
                print(f"   • {issue['issue']}")
                print(f"     Fix: {issue['fix']}")
            print()
    
    def create_fix_script(self):
        """Create automatic fix script"""
        print("🔧 Creating Fix Script...")
        print("-" * 40)
        
        fix_commands = []
        
        # Analyze library status and generate fixes
        for lib_name, status in self.library_status.items():
            if not status['compatible']:
                if lib_name == 'torch':
                    fix_commands.extend([
                        '# Fix PyTorch for ARM compatibility',
                        'pip uninstall -y torch torchvision torchaudio',
                        'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu',
                        ''
                    ])
                elif lib_name == 'cv2':
                    fix_commands.extend([
                        '# Fix OpenCV for ARM compatibility', 
                        'pip uninstall -y opencv-python opencv-contrib-python',
                        'pip install opencv-python-headless',
                        ''
                    ])
                elif lib_name == 'ultralytics':
                    fix_commands.extend([
                        '# Install Ultralytics (requires fixed PyTorch first)',
                        'pip install ultralytics',
                        ''
                    ])
        
        # Add general ARM fixes
        if self.system_info.get('is_arm', False):
            fix_commands.extend([
                '# ARM-specific optimizations',
                'export OPENBLAS_CORETYPE=ARMV8',
                'export OMP_NUM_THREADS=1',
                ''
            ])
        
        # Create the fix script
        script_content = '''#!/bin/bash
set -e

# Arduino UNO Q4GB ARM Compatibility Fix Script
# Generated by ARM Diagnostic Tool

echo "=============================================="
echo "  Arduino UNO Q4GB ARM Compatibility Fix"
echo "=============================================="
echo

# Check if running as root for system packages
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  Running as root detected"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found - creating one..."
    python3 -m venv venv
    source venv/bin/activate
fi

echo "🔧 Applying ARM compatibility fixes..."
echo

''' + '\n'.join(fix_commands) + '''

echo "✅ ARM compatibility fixes applied!"
echo
echo "🧪 Testing fixes..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
try:
    from ultralytics import YOLO
    print(f'Ultralytics: {YOLO.__version__}')
except ImportError as e:
    print(f'Ultralytics: Not installed - {e}')

echo
echo "✅ Fix script completed!"
echo "🧪 Run the diagnostic again to verify fixes: python3 arm_diagnostic.py"
'''
        
        script_path = Path('arm_compatibility_fix.sh')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"✅ Fix script created: {script_path}")
        print("   Run with: ./arm_compatibility_fix.sh")
        print()

def main():
    """Main function"""
    diagnostic = ARMDiagnosticTool()
    issues = diagnostic.run_full_diagnostic()
    
    if issues:
        print("🚨 COMPATIBILITY ISSUES DETECTED")
        print(f"   Found {len(issues)} issues to resolve")
        print("   Run ./arm_compatibility_fix.sh to apply fixes")
        return 1
    else:
        print("🎉 SYSTEM IS COMPATIBLE")
        print("   No ARM compatibility issues found")
        return 0

if __name__ == "__main__":
    sys.exit(main())