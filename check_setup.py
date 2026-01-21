#!/usr/bin/env python3
"""
Setup Verification Script for Waverider GUI
This script checks if all required files are in the correct locations
"""

import os
import sys

def check_file_structure():
    """Check if folder structure is correct"""
    
    print("="*70)
    print("Waverider GUI - File Structure Check")
    print("="*70)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\nChecking directory: {script_dir}\n")
    
    all_good = True
    
    # Required structure:
    # your_folder/
    # ├── waverider_generator/
    # │   ├── __init__.py
    # │   ├── generator.py
    # │   ├── cad_export.py
    # │   ├── flowfield.py
    # │   └── plotting_tools.py
    # ├── waverider_gui_with_analysis.py
    # ├── reference_area_calculator.py
    # └── check_setup.py (this file)
    
    checks = {
        'GUI file': 'waverider_gui.py',
        'Reference area calculator': 'reference_area_calculator.py',
        'waverider_generator folder': 'waverider_generator',
        'generator.py': 'waverider_generator/generator.py',
        'cad_export.py': 'waverider_generator/cad_export.py',
        'flowfield.py': 'waverider_generator/flowfield.py',
        '__init__.py': 'waverider_generator/__init__.py',
    }
    
    print("Checking required files:\n")
    
    for name, path in checks.items():
        full_path = os.path.join(script_dir, path)
        exists = os.path.exists(full_path)
        
        if exists:
            print(f"  ✓ {name:30s} → {path}")
        else:
            print(f"  ✗ {name:30s} → {path} (MISSING!)")
            all_good = False
    
    print("\n" + "="*70)
    
    if all_good:
        print("✓ All required files found!")
        print("\nYou can now run the GUI:")
        print(f"  cd {script_dir}")
        print(f"  python3 waverider_gui_with_analysis.py")
    else:
        print("✗ Some files are missing!")
        print("\nRequired folder structure:")
        print("""
your_main_folder/
├── waverider_generator/
│   ├── __init__.py
│   ├── generator.py
│   ├── cad_export.py
│   ├── flowfield.py
│   └── plotting_tools.py
├── waverider_gui_with_analysis.py  ← GUI file
├── reference_area_calculator.py    ← For accurate A_ref
└── check_setup.py                  ← This script
        """)
        print("Make sure all files are in the correct locations!")
    
    print("="*70)
    
    return all_good


def test_imports():
    """Try importing the modules"""
    
    print("\n" + "="*70)
    print("Testing Imports")
    print("="*70 + "\n")
    
    all_good = True
    
    # Test waverider_generator
    try:
        from waverider_generator.generator import waverider
        print("✓ waverider_generator imports successfully")
    except ImportError as e:
        print(f"✗ Cannot import waverider_generator: {e}")
        all_good = False
    
    # Test reference_area_calculator
    try:
        from reference_area_calculator import calculate_planform_area_from_waverider
        print("✓ reference_area_calculator imports successfully")
    except ImportError as e:
        print(f"✗ Cannot import reference_area_calculator: {e}")
        all_good = False
    
    # Test optional: PySAGAS
    try:
        from pysagas.cfd import OPM
        print("✓ PySAGAS imports successfully (optional)")
    except ImportError:
        print("⚠️  PySAGAS not installed (optional - for analysis)")
        print("   Install with: pip install pysagas")
    
    print()
    
    return all_good


def show_current_structure():
    """Show what files actually exist in current directory"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*70)
    print("Current Directory Contents")
    print("="*70)
    print(f"\nDirectory: {script_dir}\n")
    
    # List all Python files in current directory
    print("Python files in main folder:")
    for item in sorted(os.listdir(script_dir)):
        full_path = os.path.join(script_dir, item)
        if item.endswith('.py'):
            size = os.path.getsize(full_path)
            print(f"  • {item} ({size:,} bytes)")
    
    # Check for waverider_generator folder
    wg_path = os.path.join(script_dir, 'waverider_generator')
    if os.path.exists(wg_path) and os.path.isdir(wg_path):
        print("\nFiles in waverider_generator/:")
        for item in sorted(os.listdir(wg_path)):
            if item.endswith('.py'):
                full_path = os.path.join(wg_path, item)
                size = os.path.getsize(full_path)
                print(f"  • {item} ({size:,} bytes)")
    else:
        print("\n✗ waverider_generator/ folder not found!")
    
    print()


if __name__ == "__main__":
    print("\n")
    
    # Show current structure
    show_current_structure()
    
    # Check file structure
    structure_ok = check_file_structure()
    
    # Test imports if structure is ok
    if structure_ok:
        imports_ok = test_imports()
        
        if imports_ok:
            print("\n" + "="*70)
            print("✓✓✓ ALL CHECKS PASSED! ✓✓✓")
            print("="*70)
            print("\nYou're ready to run the GUI!")
            print("\nCommand:")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"  cd {script_dir}")
            print(f"  python3 waverider_gui_with_analysis.py")
            print()
        else:
            print("\n" + "="*70)
            print("✗ Import test failed")
            print("="*70)
            print("\nFiles exist but cannot be imported.")
            print("Check for syntax errors or missing dependencies.")
    else:
        print("\nFix the missing files first, then run this script again.")
    
    print()
