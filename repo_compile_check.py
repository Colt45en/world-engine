#!/usr/bin/env python3
"""
🔍 REPOSITORY-WIDE COMPILE CHECK
================================

Scans all Python files in the repository and reports syntax errors.
Uses compile() to check parseability without executing code.

Usage:
    python repo_compile_check.py [--verbose]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

def compile_check_file(filepath: Path) -> Tuple[bool, str]:
    """
    Check if a Python file compiles without syntax errors.
    
    Returns:
        (success: bool, error_message: str)
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        compile(source, str(filepath), 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {e}"

def find_python_files(root_dir: Path, exclude_patterns: List[str] = None) -> List[Path]:
    """Find all Python files in the directory tree."""
    if exclude_patterns is None:
        exclude_patterns = ['__pycache__', '.git', 'node_modules', 'venv', '.venv', 'dist']
    
    python_files = []
    for py_file in root_dir.rglob('*.py'):
        # Skip excluded directories
        if any(pattern in str(py_file) for pattern in exclude_patterns):
            continue
        python_files.append(py_file)
    
    return sorted(python_files)

def main():
    parser = argparse.ArgumentParser(description='Repository-wide Python compile check')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show all files checked (not just errors)')
    parser.add_argument('--root', type=str, default='.',
                       help='Root directory to scan (default: current directory)')
    
    args = parser.parse_args()
    
    root_dir = Path(args.root).resolve()
    
    print("🔍 REPOSITORY-WIDE COMPILE CHECK")
    print("=" * 60)
    print(f"📁 Scanning: {root_dir}")
    print()
    
    # Find all Python files
    python_files = find_python_files(root_dir)
    print(f"📊 Found {len(python_files)} Python files to check")
    print()
    
    # Check each file
    passed = []
    failed = []
    
    for py_file in python_files:
        success, error_msg = compile_check_file(py_file)
        
        relative_path = py_file.relative_to(root_dir)
        
        if success:
            passed.append(relative_path)
            if args.verbose:
                print(f"✓ {relative_path}")
        else:
            failed.append((relative_path, error_msg))
            print(f"✗ {relative_path}")
            print(f"  {error_msg}")
            print()
    
    # Summary
    print()
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"✓ Passed: {len(passed)} files")
    print(f"✗ Failed: {len(failed)} files")
    
    if failed:
        print()
        print("❌ FILES WITH ERRORS:")
        for filepath, error in failed:
            print(f"  • {filepath}")
    
    print()
    
    # Exit code
    if failed:
        print("❌ Compile check FAILED - fix syntax errors above")
        sys.exit(1)
    else:
        print("✅ Compile check PASSED - all files parse successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
