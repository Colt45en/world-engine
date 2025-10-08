#!/usr/bin/env python3
"""
🔧 DEPENDENCY FIXER AND SYSTEM REPAIR
====================================

Quick script to install missing dependencies and fix common issues.
"""

import subprocess
import sys
import os
import importlib

def install_package(package):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies():
    """Check and install all required dependencies"""
    required_packages = [
        "numpy",
        "pandas", 
        "matplotlib",
        "seaborn",
        "websockets",
        "psutil",
        "requests",
        "aiohttp",
        "jsonschema",
        "scipy",
        "sympy",
        "nltk"
    ]
    
    print("🔍 Checking dependencies...")
    missing = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n🔧 Installing {len(missing)} missing packages...")
        for package in missing:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
    else:
        print("\n✅ All dependencies are installed!")

def fix_file_permissions():
    """Fix file permissions if needed"""
    try:
        # Make Python files executable
        os.chmod("vector_node_network.py", 0o755)
        os.chmod("implement_pain_opportunity_system.py", 0o755)
        os.chmod("analyze_consciousness_patterns.py", 0o755)
        os.chmod("master_nexus_codepad.py", 0o755)
        print("✅ File permissions fixed")
    except:
        print("⚠️ Could not fix file permissions (may not be needed)")

def main():
    print("🔧 SYSTEM REPAIR UTILITY")
    print("========================")
    
    # Check and install dependencies
    check_and_install_dependencies()
    
    # Fix file permissions
    fix_file_permissions()
    
    print("\n✅ System repair complete!")
    print("\nNow you can run:")
    print("python master_nexus_codepad.py")

if __name__ == "__main__":
    main()