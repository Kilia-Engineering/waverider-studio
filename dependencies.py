#!/usr/bin/env python3
"""
Dependency Availability Checker
Checks which optional dependencies are available
"""

# Check PySAGAS availability
try:
    import pysagas
    PYSAGAS_AVAILABLE = True
except ImportError:
    PYSAGAS_AVAILABLE = False

# Check Gmsh availability
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False

# Check reference area calculator availability
try:
    from reference_area_calculator import calculate_planform_area_from_waverider
    AREA_CALC_AVAILABLE = True
except ImportError:
    AREA_CALC_AVAILABLE = False

# Print status when imported
if __name__ == "__main__":
    print("Dependency Status:")
    print(f"  PySAGAS:  {'✓ Available' if PYSAGAS_AVAILABLE else '✗ Not installed'}")
    print(f"  Gmsh:     {'✓ Available' if GMSH_AVAILABLE else '✗ Not installed'}")
    print(f"  Area Calc: {'✓ Available' if AREA_CALC_AVAILABLE else '✗ Not installed'}")
