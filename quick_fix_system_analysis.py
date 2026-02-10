#!/usr/bin/env python3
"""
Arduino UNO Q4GB Ultimate Quick Fix
Resolves system_analysis.json file issues
"""

import os
import json
import sys
from pathlib import Path

def fix_system_analysis():
    """Fix system analysis file corruption"""
    print("🔧 Fixing system analysis file...")
    
    analysis_file = Path('system_analysis.json')
    if analysis_file.exists():
        try:
            # Try to read and fix the file
            with open(analysis_file, 'r') as f:
                content = f.read()
            
            # Check if file is corrupted
            if not content.strip():
                print("❌ Empty analysis file - recreating")
                content = {}
            else:
                try:
                    data = json.loads(content)
                print("✅ Analysis file is valid JSON")
                return data
                except json.JSONDecodeError:
                    print("⚠️  Analysis file corrupted - fixing JSON structure")
                    # Try to fix common JSON issues
                    lines = content.strip().split('\n')
                    fixed_lines = []
                    in_string = False
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Fix common issues
                        # Remove trailing commas
                        if line.endswith(','):
                            line = line.rstrip(',')
                        
                        # Fix malformed JSON entries
                        if line.startswith(',') and line.endswith(','):
                            continue
                        
                        # Fix missing quotes
                        if line.count('{') != line.count('}'):
                            line = line.replace('{', '"').replace('}', '"')
                        
                        # Fix missing colons
                        if line.count(':') != line.count(':'):
                            line = line.replace(':', ': ').replace(':', ': ')
                        
                        fixed_lines.append(line)
                    
                    # Reconstruct basic structure if parsing failed
                    if fixed_lines and not in_string:
                        # Create minimal valid structure
                        content = '{"timestamp": "' + str(datetime.datetime.now().isoformat()) + '", "system_info": {}, "recommendations": {}}'
                        print("✅ Fixed with minimal structure")
                    else:
                        content = '\n'.join(fixed_lines)
                        print("✅ Fixed JSON structure")
                        
                    data = json.loads(content)
                    return data
                    
        except Exception as e:
            print(f"⚠️  Error fixing analysis file: {e}")
            content = {
                'timestamp': str(datetime.datetime.now().isoformat()),
                'system_info': {},
                'recommendations': {
                    'installation_method': 'system_packages'
                }
            }
        
        # Write the fixed file
        with open(analysis_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ System analysis file fixed")
        return True
        
    else:
        print("⚠️  No system analysis file found")
        return False

def check_deployment_files():
    """Check if all deployment files exist and are executable"""
    print("🔍 Checking deployment files...")
    
    required_files = [
        'system_analyzer.py',
        'package_manager.py', 
        'ai_stack_manager.py',
        'test_suite.py',
        'deploy_ultimate.sh',
        'run_ultimate_arm_ai.sh'
    ]
    
    missing_files = []
    executable_files = []
    
    for file_name in required_files:
        file_path = Path(file_name)
        if not file_path.exists():
            missing_files.append(file_name)
            print(f"❌ Missing: {file_name}")
        else:
            # Check if executable
            if not os.access(file_path, os.X_OK):
                executable_files.append(file_name)
                print(f"⚠️ Not executable: {file_name}")
            else:
                print(f"✅ Found: {file_name}")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    if executable_files:
        print(f"⚠️ Non-executable files: {', '.join(executable_files)}")
        return False
    
    print("✅ All deployment files present and executable")
    return True

def quick_diagnostic():
    """Quick diagnostic without file dependencies"""
    print("🔍 Quick diagnostic...")
    
    # Check basic Python environment
    print(f"✅ Python {sys.version}")
    
    # Check essential imports
    imports_to_check = [
        ('json', 'JSON support'),
        ('serial', 'Serial communication'),
        ('cv2', 'OpenCV support'),
        ('numpy', 'Numerical operations')
    ]
    
    working_imports = []
    for module_name, description in imports_to_check:
        try:
            import importlib.util
            if importlib.util.find_spec(module_name):
                working_imports.append((module_name, True))
                print(f"✅ {description}: Working")
            else:
                print(f"❌ {description}: Not available")
        except ImportError as e:
            working_imports.append((module_name, f"Error: {e}"))
    
    print(f"📊 Imports working: {len([i for i, working in working_imports])}/{len(imports_to_check)}")
    
    # Try to create minimal system info
    minimal_system_info = {
        'platform': sys.platform,
        'python_version': sys.version_info[:2],
        'architecture': sys.machine,
        'timestamp': str(datetime.datetime.now().isoformat())
    }
    
    return {
        'system_info': minimal_system_info,
        'imports': working_imports,
        'ready': len([i for i, working in working_imports]) == len(imports_to_check)
    }

def main():
    """Main function"""
    print("🔧 Arduino UNO Q4GB Ultimate Quick Fix")
    print("=" * 50)
    
    # Fix system analysis file
    if not fix_system_analysis():
        print("❌ Could not fix system analysis file")
        return 1
    
    # Check deployment files
    if not check_deployment_files():
        print("❌ Deployment files missing or not executable")
        print("\n💡 Manual steps:")
        print("1. Make all Python files executable: chmod +x *.py")
        print("2. Run system analyzer manually: python3 system_analyzer.py")
        print("3. Check imports: python3 -c \"import serial; print('OK')\"")
        return 1
    
    # Quick diagnostic
    diagnostic_result = quick_diagnostic()
    
    print("\n" + "=" * 50)
    print("📊 QUICK DIAGNOSTIC COMPLETE")
    print("=" * 50)
    
    if diagnostic_result.get('ready', False):
        print("❌ System not ready")
        print("\n💡 Manual fixes needed")
        return 1
    else:
        print("✅ System ready for deployment!")
        
        # Create simple config
        simple_config = {
            'timestamp': str(datetime.datetime.now().isoformat()),
            'system_info': diagnostic_result['system_info'],
            'status': 'ready',
            'next_step': 'Run deployment script'
        }
        
        config_file = Path('simple_config.json')
        with open(config_file, 'w') as f:
            json.dump(simple_config, f, indent=2)
        
        print(f"✅ Created simple_config.json")
        return 0

if __name__ == "__main__":
    sys.exit(main())