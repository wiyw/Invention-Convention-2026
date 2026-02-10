#!/usr/bin/env python3
"""
Arduino UNO Q4GB Ultimate Package Manager
Multi-method installation with intelligent fallbacks
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime

class UltimatePackageManager:
    def __init__(self):
        self.installation_log = []
        self.successful_packages = []
        self.failed_packages = []
        self.used_method = None
        self.system_info = {}
        
    def load_system_info(self):
        """Load system analysis if available"""
        try:
            with open('system_analysis.json', 'r') as f:
                self.system_info = json.load(f)
        except:
            print("⚠️  No system analysis found - will analyze on the fly")
    
    def log_action(self, action, status, details=""):
        """Log installation actions"""
        from datetime import datetime
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'status': status,
            'details': details
        }
        self.installation_log.append(entry)
        
        status_icon = "✅" if status == "success" else "❌" if status == "failed" else "⚠️"
        print(f"{status_icon} {action}: {details}")
    
    def run_command(self, command, description="", timeout=300):
        """Run command with error handling"""
        try:
            if isinstance(command, str):
                command = command.split()
            
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                check=False
            )
            
            success = result.returncode == 0
            details = result.stdout.strip() if result.stdout else result.stderr.strip()
            
            self.log_action(description, "success" if success else "failed", details)
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            self.log_action(description, "failed", "Command timed out")
            return False, "", "Command timed out"
        except Exception as e:
            self.log_action(description, "failed", str(e))
            return False, "", str(e)
    
    def install_system_packages(self):
        """Method 1: System packages (most reliable)"""
        print("\n📦 Method 1: System Packages Installation")
        print("=" * 50)
        
        # Determine system packages based on distribution
        packages = []
        
        # Base development packages
        base_packages = ['python3-dev', 'build-essential', 'cmake', 'pkg-config']
        
        # BLAS packages (try multiple options)
        blas_packages = ['libblas-dev', 'libopenblas-dev', 'liblapack-dev']
        
        # Image processing packages
        image_packages = ['libjpeg-dev', 'libpng-dev', 'libtiff-dev', 'libavcodec-dev']
        
        # Python packages (if available)
        python_packages = []
        if self.system_info.get('recommendations', {}).get('installation_method') == 'system_packages':
            python_packages = ['python3-numpy', 'python3-opencv', 'python3-pil', 'python3-serial']
        
        all_packages = base_packages + blas_packages + image_packages + python_packages
        
        print(f"Installing {len(all_packages)} system packages...")
        
        # Update package list
        success, _, _ = self.run_command(['sudo', 'apt-get', 'update'], "Package list update")
        if not success:
            print("❌ Failed to update package list")
            return False
        
        # Install packages with error tolerance
        for package in all_packages:
            # Check if package exists first
            success, _, _ = self.run_command(
                ['apt-cache', 'show', package], 
                f"Check {package}"
            )
            
            if success:
                success, _, _ = self.run_command(
                    ['sudo', 'apt-get', 'install', '-y', package],
                    f"Install {package}"
                )
                if success:
                    self.successful_packages.append(package)
                else:
                    self.failed_packages.append(package)
            else:
                self.log_action(f"Skip {package}", "skipped", "Package not available")
        
        return len(self.successful_packages) > 0
    
    def install_venv_override(self):
        """Method 2: Virtual environment with PEP 668 override"""
        print("\n🐍 Method 2: Virtual Environment + Override")
        print("=" * 50)
        
        # Create clean virtual environment
        venv_path = Path('venv_ultimate')
        
        if venv_path.exists():
            shutil.rmtree(venv_path)
        
        success, _, _ = self.run_command(
            [sys.executable, '-m', 'venv', '--system-site-packages', str(venv_path)],
            "Create virtual environment"
        )
        
        if not success:
            return False
        
        # Activate virtual environment
        venv_python = venv_path / 'bin' / 'python'
        venv_pip = venv_path / 'bin' / 'pip'
        
        if not venv_python.exists():
            print("❌ Virtual environment creation failed")
            return False
        
        # Install packages with override
        packages_to_install = [
            ('numpy', '2.1.0'),  # Python 3.13 compatible
            ('opencv-python-headless', '4.8.0'),  # Latest stable
            ('pillow', '10.0.0'),  # Latest
            ('pyserial', '3.5'),  # Latest
        ]
        
        for package, version in packages_to_install:
            package_spec = f"{package}=={version}" if version else package
            
            success, _, _ = self.run_command([
                str(venv_pip), 
                'install', '--break-system-packages', 
                package_spec
            ], f"Install {package}")
            
            if success:
                self.successful_packages.append(f"{package} (venv)")
            else:
                # Try fallback version
                fallback_versions = {
                    'numpy': '1.26.0',
                    'opencv-python-headless': '4.7.0',
                    'pillow': '9.5.0'
                }
                
                if package in fallback_versions:
                    fallback_spec = f"{package}=={fallback_versions[package]}"
                    success, _, _ = self.run_command([
                        str(venv_pip), 
                        'install', '--break-system-packages', 
                        fallback_spec
                    ], f"Install {package} (fallback)")
                    
                    if success:
                        self.successful_packages.append(f"{package} (venv fallback)")
        
        return venv_path.exists()
    
    def install_user_packages(self):
        """Method 3: User installation"""
        print("\n👤 Method 3: User Installation")
        print("=" * 50)
        
        packages_to_install = [
            ('numpy', '2.0.0'),
            ('opencv-python-headless', '4.7.0'),
            ('pillow', '9.5.0'),
            ('pyserial', '3.5')
        ]
        
        for package, version in packages_to_install:
            package_spec = f"{package}=={version}" if version else package
            
            success, _, _ = self.run_command([
                sys.executable, '-m', 'pip', 'install', '--user', 
                '--upgrade', '--no-warn-script-location', package_spec
            ], f"Install {package} (user)")
            
            if success:
                self.successful_packages.append(f"{package} (user)")
        
        return True
    
    def install_alternative_python(self):
        """Method 4: Alternative Python version"""
        print("\n🐍 Method 4: Alternative Python Installation")
        print("=" * 50)
        
        # Try to install Python 3.11 (better ML support)
        success, _, _ = self.run_command(
            ['sudo', 'apt-get', 'install', '-y', 'python3.11', 'python3.11-venv', 'python3.11-dev'],
            "Install Python 3.11"
        )
        
        if success:
            # Create venv with Python 3.11
            venv_path = Path('venv_py311')
            
            if venv_path.exists():
                shutil.rmtree(venv_path)
            
            success, _, _ = self.run_command(
                ['python3.11', '-m', 'venv', str(venv_path)],
                "Create Python 3.11 venv"
            )
            
            if success:
                venv_pip = venv_path / 'bin' / 'pip'
                
                # Install packages with Python 3.11
                packages = ['numpy', 'opencv-python-headless', 'pillow', 'pyserial', 'torch', 'ultralytics']
                
                for package in packages:
                    success, _, _ = self.run_command([
                        str(venv_pip), 'install', package
                    ], f"Install {package} (Python 3.11)")
                    
                    if success:
                        self.successful_packages.append(f"{package} (py311)")
                
                return venv_path.exists()
        
        return False
    
    def smart_install(self):
        """Intelligent installation method selection"""
        print("🧠 Smart Installation Method Selection")
        print("=" * 50)
        
        self.load_system_info()
        
        # Determine best method based on system analysis
        installation_method = self.system_info.get('recommendations', {}).get('installation_method', 'venv_override')
        
        if installation_method == 'system_packages':
            methods = [
                ('system_packages', self.install_system_packages),
                ('venv_override', self.install_venv_override),
                ('user_packages', self.install_user_packages),
                ('alternative_python', self.install_alternative_python)
            ]
        elif installation_method == 'venv_override':
            methods = [
                ('venv_override', self.install_venv_override),
                ('user_packages', self.install_user_packages),
                ('alternative_python', self.install_alternative_python)
            ]
        else:
            methods = [
                ('venv_override', self.install_venv_override),
                ('user_packages', self.install_user_packages),
                ('alternative_python', self.install_alternative_python)
            ]
        
        # Try methods in order
        for method_name, method_func in methods:
            print(f"\n🎯 Trying {method_name}...")
            self.used_method = method_name
            
            try:
                if method_func():
                    print(f"✅ {method_name} succeeded!")
                    self.log_action("Installation", "success", f"Method {method_name}")
                    return True
                else:
                    print(f"❌ {method_name} failed")
                    self.log_action("Installation", "failed", f"Method {method_name}")
            except Exception as e:
                print(f"❌ {method_name} crashed: {e}")
                self.log_action("Installation", "crashed", f"Method {method_name}: {e}")
        
        print("❌ All installation methods failed")
        return False
    
    def install_ai_models(self):
        """Install AI models"""
        print("\n🤖 Installing AI Models")
        print("=" * 30)
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # YOLO model download
        yolo_path = models_dir / 'yolo26n.pt'
        
        if not yolo_path.exists():
            print("Downloading YOLO26n model...")
            
            # Try wget first
            success, _, _ = self.run_command([
                'wget', '-O', str(yolo_path),
                'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'
            ], "Download YOLO model (wget)")
            
            # Fallback to curl
            if not success:
                success, _, _ = self.run_command([
                    'curl', '-L', '-o', str(yolo_path),
                    'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt'
                ], "Download YOLO model (curl)")
            
            if yolo_path.exists():
                model_size = yolo_path.stat().st_size / (1024 * 1024)  # MB
                self.log_action("YOLO Model", "success", f"Downloaded {model_size:.1f} MB")
                return True
            else:
                self.log_action("YOLO Model", "failed", "Download failed")
                return False
        else:
            self.log_action("YOLO Model", "skipped", "Already exists")
            return True
    
    def generate_config(self):
        """Generate configuration based on successful installation"""
        print("\n⚙️  Generating Configuration")
        print("=" * 30)
        
        config = {
            'installation_method': self.used_method,
            'successful_packages': self.successful_packages,
            'failed_packages': self.failed_packages,
            'arm_optimized': True,
            'camera': {
                'backends': ['V4L2'],
                'test_indices': list(range(5)),
                'resolution': [320, 240],
                'fps_target': 8
            },
            'ai_stack': {
                'backends': ['ultralytics', 'onnx', 'tflite', 'opencv'],
                'confidence_threshold': 0.6,
                'max_detections': 3
            },
            'performance': {
                'threads': 1,
                'memory_limit': '512MB',
                'optimization_level': 'balanced'
            }
        }
        
        # Determine activation script
        if self.used_method == 'system_packages':
            config['activation'] = {
                'command': 'source /usr/bin/activate_venv.sh',
                'description': 'System-wide activation'
            }
        elif 'venv' in self.used_method:
            config['activation'] = {
                'command': 'source venv_ultimate/bin/activate',
                'description': 'Virtual environment activation'
            }
        elif 'user' in self.used_method:
            config['activation'] = {
                'command': 'export PATH="$HOME/.local/bin:$PATH"',
                'description': 'User packages activation'
            }
        elif '311' in self.used_method:
            config['activation'] = {
                'command': 'source venv_py311/bin/activate',
                'description': 'Python 3.11 activation'
            }
        
        with open('ultimate_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        self.log_action("Configuration", "success", f"Generated for {self.used_method}")
        return config
    
    def save_installation_log(self):
        """Save installation log"""
        log_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'used_method': self.used_method,
            'successful_packages': self.successful_packages,
            'failed_packages': self.failed_packages,
            'installation_log': self.installation_log,
            'success_rate': len(self.successful_packages) / (len(self.successful_packages) + len(self.failed_packages)) * 100 if (len(self.successful_packages) + len(self.failed_packages)) > 0 else 0
        }
        
        with open('installation_log.json', 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n📄 Installation log saved: installation_log.json")
    
    def print_summary(self):
        """Print installation summary"""
        print("\n" + "=" * 60)
        print("🎯 ULTIMATE PACKAGE MANAGER - SUMMARY")
        print("=" * 60)
        
        print(f"✅ Successful Packages: {len(self.successful_packages)}")
        for pkg in self.successful_packages:
            print(f"   • {pkg}")
        
        if self.failed_packages:
            print(f"❌ Failed Packages: {len(self.failed_packages)}")
            for pkg in self.failed_packages:
                print(f"   • {pkg}")
        
        print(f"🎯 Used Method: {self.used_method}")
        print(f"📊 Success Rate: {len(self.successful_packages) / (len(self.successful_packages) + len(self.failed_packages)) * 100:.1f}%")
        
        if len(self.successful_packages) >= 5:
            print("🎉 EXCELLENT! Installation highly successful")
        elif len(self.successful_packages) >= 3:
            print("✅ GOOD! Installation mostly successful")
        else:
            print("⚠️  LIMITED! Installation had issues")

def main():
    """Main function"""
    manager = UltimatePackageManager()
    
    try:
        print("🚀 Arduino UNO Q4GB Ultimate Package Manager")
        print("=" * 60)
        
        # Run smart installation
        if manager.smart_install():
            # Install AI models
            if manager.install_ai_models():
                # Generate configuration
                manager.generate_config()
                
                # Save log
                manager.save_installation_log()
                
                # Print summary
                manager.print_summary()
                
                print("\n✅ Ultimate installation completed!")
                print("🎯 Check ultimate_config.json for activation instructions")
                
                return 0
            else:
                print("❌ AI model installation failed")
                return 1
        else:
            print("❌ All installation methods failed")
            manager.save_installation_log()
            manager.print_summary()
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Installation interrupted")
        return 1
    except Exception as e:
        print(f"❌ Installation crashed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())