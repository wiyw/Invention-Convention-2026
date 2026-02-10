#!/usr/bin/env python3
"""
Arduino UNO Q4GB Ultimate Quick Fix - Windows Version
Fixes corrupted system analysis file for immediate deployment
"""

import os
import sys
import json
from pathlib import Path

def fix_system_analysis():
    """Fix corrupted system analysis file"""
    print("🔧 Fixing System Analysis File...")
    
    analysis_file = Path('system_analysis.json')
    
    if analysis_file.exists():
        print(f"✅ Found: {analysis_file}")
        
        # Read and validate
        try:
            with open(analysis_file, 'r') as f:
                content = f.read()
                
            # Basic validation
            if not content.strip():
                    print("📄 Empty file detected")
                    return False
                
                # Try to parse JSON
                try:
                    data = json.loads(content)
                    print("✅ JSON structure valid")
                    return data
                except json.JSONDecodeError:
                    print("🛠️  JSON corrupted - attempting fix")
                    
                    # Create minimal valid structure
                    minimal_data = {
                        'timestamp': str(datetime.datetime.now()),
                        'system_info': {
                            'platform': sys.platform(),
                            'python_version': sys.version_info[:2],
                            'architecture': sys.machine()
                        },
                        'recommendations': {
                            'installation_method': 'system_packages'
                        }
                    }
                    
                    json.dump(minimal_data, analysis_file, indent=2)
                    print("✅ Fixed with minimal structure")
                    return minimal_data
                    
        except Exception as e:
            print(f"🛠️ Error fixing file: {e}")
            return None
    
    else:
        print("📄 No system analysis file found - creating new one")
        
        # Create new file with minimal valid structure
        minimal_data = {
            'timestamp': str(datetime.datetime.now()),
            'system_info': {
                'platform': sys.platform(),
                'python_version': sys.version_info[:2],
                'architecture': sys.machine()
            },
            'recommendations': {
                'installation_method': 'system_packages'
            }
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(minimal_data, f, indent=2)
            print("✅ Created new system analysis file")
        
        return minimal_data

def main():
    print("🔧 Arduino UNO Q4GB Ultimate Quick Fix")
    print("=" * 50)
    
    if fix_system_analysis():
        print("✅ System analysis file fixed successfully!")
        print("\n🚀 Quick fix complete!")
        print("📋 You can now run: python3 quick_fix_system_analysis.py")
        print("📋 Or proceed with: ./deploy_ultimate.sh")
        print("\n📋 The deployment will auto-detect and fix any remaining issues")
        return 0
    else:
        print("❌ Failed to fix system analysis")
        return 1

if __name__ == "__main__":
    sys.exit(main())