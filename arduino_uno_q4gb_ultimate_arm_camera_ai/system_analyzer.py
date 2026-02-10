#!/usr/bin/env python3
"""
Arduino UNO Q4GB Ultimate System Analyzer
Comprehensive system detection for ARM deployment
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

class SystemAnalyzer:
    def __init__(self):
        self.system_info = {}
        self.package_info = {}
        self.recommendations = {}
        
    def analyze_system(self):
        """Complete system analysis"""
        print("🔍 Analyzing System for ARM Deployment")
        print("=" * 50)
        
        # Basic system info
        self.system_info['platform'] = platform.platform()
        self.system_info['system'] = platform.system()
        self.system_info['machine'] = platform.machine()
        self.system_info['architecture'] = platform.architecture()
        self.system_info['python_version'] = sys.version_info[:2]
        
        print(f"Platform: {self.system_info['platform']}")
        print(f"System: {self.system_info['system']}")
        print(f"Architecture: {self.system_info['machine']}")
        print(f"Python: {'.'.join(map(str, self.system_info['python_version']))}")
        
        # Linux specific analysis
        if self.system_info['system'] == 'Linux':
            self._analyze_linux_system()
        
        # ARM specific analysis
        self._analyze_arm_capabilities()
        
        # Python environment analysis
        self._analyze_python_environment()
        
        # Package availability analysis
        self._analyze_package_availability()
        
        # Generate recommendations
        self._generate_recommendations()
        
        return self.system_info, self.recommendations
    
    def _analyze_linux_system(self):
        """Analyze Linux distribution and packages"""
        print("\n🐧 Linux Distribution Analysis")
        print("-" * 30)
        
        # Distribution detection
        dist_info = {}
        
        # Check various distribution files
        if Path('/etc/debian_version').exists():
            try:
                with open('/etc/debian_version', 'r') as f:
                    version = f.read().strip()
                dist_info['type'] = 'debian'
                dist_info['version'] = version
                print(f"Debian version: {version}")
            except:
                pass
        
        if Path('/etc/lsb-release').exists():
            try:
                with open('/etc/lsb-release', 'r') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            dist_info[key.strip()] = value.strip().strip('"')
                if 'DISTRIB_ID' in dist_info:
                    print(f"Distribution: {dist_info['DISTRIB_ID']} {dist_info.get('DISTRIB_RELEASE', '')}")
            except:
                pass
        
        # Raspberry Pi detection
        if Path('/boot/firmware').exists() or 'raspberry' in platform.platform().lower():
            dist_info['type'] = 'raspberry_pi'
            print("Raspberry Pi detected")
        
        # CPU information
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                
            # Count cores
            cores = cpuinfo.count('processor')
            dist_info['cpu_cores'] = cores
            print(f"CPU cores: {cores}")
            
            # CPU model
            for line in cpuinfo.split('\n'):
                if 'model name' in line:
                    model = line.split(':', 1)[1].strip()
                    dist_info['cpu_model'] = model
                    print(f"CPU model: {model}")
                    break
                    
            # ARM features
            features = []
            if 'neon' in cpuinfo.lower():
                features.append('NEON SIMD')
            if 'vfp' in cpuinfo.lower():
                features.append('VFP floating point')
            if 'aes' in cpuinfo.lower():
                features.append('AES hardware')
            
            if features:
                print(f"ARM features: {', '.join(features)}")
            dist_info['arm_features'] = features
                
        except:
            print("Could not read CPU info")
        
        self.system_info['linux_info'] = dist_info
    
    def _analyze_arm_capabilities(self):
        """Analyze ARM-specific capabilities"""
        print("\n💪 ARM Capabilities Analysis")
        print("-" * 30)
        
        machine = self.system_info['machine'].lower()
        
        if 'arm' in machine or 'aarch64' in machine:
            self.system_info['is_arm'] = True
            print("✅ ARM architecture detected")
            
            # ARM version detection
            if 'aarch64' in machine:
                self.system_info['arm_version'] = 'ARMv8 (64-bit)'
                print("ARM version: ARMv8 (64-bit)")
            elif 'armv7' in machine:
                self.system_info['arm_version'] = 'ARMv7 (32-bit)'
                print("ARM version: ARMv7 (32-bit)")
            else:
                self.system_info['arm_version'] = 'Unknown ARM'
                print("ARM version: Unknown ARM")
        else:
            self.system_info['is_arm'] = False
            print("⚠️  Non-ARM architecture detected")
            print("    This deployment is optimized for ARM systems")
    
    def _analyze_python_environment(self):
        """Analyze Python environment setup"""
        print("\n🐍 Python Environment Analysis")
        print("-" * 30)
        
        # Virtual environment detection
        if hasattr(sys, 'real_prefix') and sys.real_prefix:
            self.system_info['in_venv'] = True
            print(f"✅ Virtual environment: {sys.real_prefix}")
        elif hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
            self.system_info['in_venv'] = True
            print(f"✅ Virtual environment: {sys.prefix}")
        else:
            self.system_info['in_venv'] = False
            print("⚠️  No virtual environment detected")
        
        # pip version
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                              capture_output=True, text=True)
            if result.returncode == 0:
                pip_version = result.stdout.strip()
                self.system_info['pip_version'] = pip_version
                print(f"Pip version: {pip_version}")
        except:
            print("Could not get pip version")
        
        # Package manager availability
        managers = []
        if subprocess.run(['which', 'apt-get'], capture_output=True).returncode == 0:
            managers.append('apt-get')
        if subprocess.run(['which', 'apt'], capture_output=True).returncode == 0:
            managers.append('apt')
        if subprocess.run(['which', 'yum'], capture_output=True).returncode == 0:
            managers.append('yum')
        if subprocess.run(['which', 'dnf'], capture_output=True).returncode == 0:
            managers.append('dnf')
        
        if managers:
            print(f"Package managers: {', '.join(managers)}")
            self.system_info['package_managers'] = managers
    
    def _analyze_package_availability(self):
        """Analyze available packages"""
        print("\n📦 Package Availability Analysis")
        print("-" * 30)
        
        packages_to_check = {
            'python3-dev': ['Python development headers'],
            'build-essential': ['Build tools'],
            'cmake': ['CMake build system'],
            'pkg-config': ['Package configuration'],
            'libjpeg-dev': ['JPEG development'],
            'libpng-dev': ['PNG development'],
            'libblas-dev': ['Basic Linear Algebra Subprograms'],
            'libopenblas-dev': ['OpenBLAS optimization'],
            'liblapack-dev': ['Linear Algebra PACKage'],
            'libatlas-base-dev': ['Automatically Tuned Linear Algebra Software'],
            'python3-numpy': ['NumPy (system)'],
            'python3-opencv': ['OpenCV (system)'],
            'python3-pil': ['Pillow (system)'],
            'python3-serial': ['PySerial (system)']
        }
        
        available_packages = {}
        
        if 'apt-get' in self.system_info.get('package_managers', []):
            for package, descriptions in packages_to_check.items():
                try:
                    # Check if package exists
                    result = subprocess.run(
                        ['apt-cache', 'show', package], 
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        available_packages[package] = True
                        print(f"✅ {package}: {' - '.join(descriptions)}")
                    else:
                        available_packages[package] = False
                        print(f"❌ {package}: Not available")
                except subprocess.TimeoutExpired:
                    print(f"⏰ {package}: Check timed out")
                    available_packages[package] = None
                except:
                    print(f"❌ {package}: Check failed")
                    available_packages[package] = None
        
        self.package_info = available_packages
    
    def _generate_recommendations(self):
        """Generate deployment recommendations"""
        print("\n💡 Deployment Recommendations")
        print("-" * 30)
        
        # Installation method recommendation
        if self.system_info.get('is_arm', False):
            if self.system_info.get('python_version', (3, 8)) >= (3, 13):
                if self.package_info.get('python3-numpy', False):
                    self.recommendations['installation_method'] = 'system_packages'
                    print("🎯 Recommended: System packages (Python 3.13 + ARM + good packages)")
                else:
                    self.recommendations['installation_method'] = 'venv_override'
                    print("🎯 Recommended: Virtual environment with PEP 668 override")
            else:
                self.recommendations['installation_method'] = 'standard_venv'
                print("🎯 Recommended: Standard virtual environment")
            
            # Package recommendations based on availability
            if not self.package_info.get('libblas-dev', False) and self.package_info.get('libopenblas-dev', False):
                self.recommendations['blas_package'] = 'libopenblas-dev'
                print("📦 Recommended BLAS: libopenblas-dev (ARM optimized)")
            elif self.package_info.get('libblas-dev', False):
                self.recommendations['blas_package'] = 'libblas-dev'
                print("📦 Recommended BLAS: libblas-dev (basic)")
            else:
                self.recommendations['blas_package'] = 'unavailable'
                print("⚠️  No BLAS packages available - expect CPU-only performance")
        
        # AI stack recommendation
        if self.system_info.get('is_arm', False):
            if 'NEON' in self.system_info.get('linux_info', {}).get('arm_features', []):
                self.recommendations['ai_stack'] = 'full_stack'
                print("🤖 Recommended AI Stack: Full YOLO + PyTorch (NEON available)")
            else:
                self.recommendations['ai_stack'] = 'lightweight_stack'
                print("🤖 Recommended AI Stack: Lightweight models (no NEON optimization)")
        
        # Performance settings
        cores = self.system_info.get('linux_info', {}).get('cpu_cores', 1)
        if cores >= 4:
            self.recommendations['performance'] = 'balanced'
            print("⚡ Recommended Performance: Balanced (4+ cores)")
        else:
            self.recommendations['performance'] = 'conservative'
            print("⚡ Recommended Performance: Conservative (fewer cores)")
    
    def save_analysis(self, filename='system_analysis.json'):
        """Save analysis to file"""
        analysis_data = {
            'timestamp': str(datetime.datetime.now()),
            'system_info': self.system_info,
            'package_info': self.package_info,
            'recommendations': self.recommendations
        }
        
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"\n📄 Analysis saved to: {filename}")
    
    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "=" * 50)
        print("🎯 SYSTEM ANALYSIS SUMMARY")
        print("=" * 50)
        
        print(f"Architecture: {self.system_info.get('machine', 'Unknown')}")
        print(f"ARM System: {'Yes' if self.system_info.get('is_arm', False) else 'No'}")
        print(f"Python Version: {'.'.join(map(str, self.system_info.get('python_version', (0, 0))))}")
        print(f"Installation Method: {self.recommendations.get('installation_method', 'Unknown')}")
        print(f"AI Stack: {self.recommendations.get('ai_stack', 'Unknown')}")
        print(f"Performance Profile: {self.recommendations.get('performance', 'Unknown')}")

def main():
    """Main function"""
    analyzer = SystemAnalyzer()
    
    try:
        system_info, recommendations = analyzer.analyze_system()
        analyzer.save_analysis()
        analyzer.print_summary()
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    import datetime
    sys.exit(main())