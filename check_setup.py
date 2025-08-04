#!/usr/bin/env python3
"""
PP6: Boston Housing Project - Setup Verification Script
Checks if all dependencies and directories are properly configured.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\n📦 Checking dependencies...")
    required_packages = [
        'tensorflow', 'sklearn', 'pandas', 'numpy', 
        'matplotlib', 'seaborn', 'jupyter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_directories():
    """Check if required directories exist."""
    print("\n📁 Checking directories...")
    required_dirs = ['models', 'results', 'visualizations']
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/")
            missing_dirs.append(dir_name)
    
    return len(missing_dirs) == 0, missing_dirs

def check_files():
    """Check if required files exist."""
    print("\n📄 Checking project files...")
    required_files = [
        'requirements.txt', 'setup_project.sh', 'Makefile',
        'boston_housing_improved.py'
    ]
    missing_files = []
    
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists() and file_path.is_file():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name}")
            missing_files.append(file_name)
    
    return len(missing_files) == 0, missing_files

def main():
    """Main diagnostic function."""
    print("🏠 PP6: Boston Housing Project - Setup Diagnostic")
    print("=" * 45)
    
    all_good = True
    
    # Check Python version
    if not check_python_version():
        all_good = False
    
    # Check dependencies
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        all_good = False
    
    # Check directories
    dirs_ok, missing_dirs = check_directories()
    if not dirs_ok:
        all_good = False
    
    # Check files
    files_ok, missing_files = check_files()
    if not files_ok:
        all_good = False
    
    # Summary
    print("\n" + "=" * 45)
    if all_good:
        print("🎉 All checks passed! Project is ready to run.")
        print("\n🚀 To run the project:")
        print("   python boston_housing_improved.py")
    else:
        print("⚠️  Issues detected:")
        if missing_deps:
            print(f"   Missing packages: {', '.join(missing_deps)}")
            print("   Run: pip install -r requirements.txt")
        if missing_dirs:
            print(f"   Missing directories: {', '.join(missing_dirs)}")
            print("   Run: mkdir -p " + " ".join(missing_dirs))
        if missing_files:
            print(f"   Missing files: {', '.join(missing_files)}")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())