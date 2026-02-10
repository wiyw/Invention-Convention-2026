#!/usr/bin/env python3
"""
Test Arduino App Labs Structure
Minimal working example
"""

import time

def main():
    print("=== Arduino App Labs Test ===")
    print("Python script working...")
    
    while True:
        print("Test running... Press Ctrl+C to stop")
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest stopped by user")