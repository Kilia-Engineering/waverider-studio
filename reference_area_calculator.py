#!/usr/bin/env python3
"""
Reference Area Calculation Utilities for Waveriders

Provides multiple methods to calculate accurate reference areas:
1. Direct from waverider geometry (most accurate)
2. From STL mesh (projected area)
3. From STEP/CAD solid
4. Simple approximation (width × height)
"""

import numpy as np
from scipy.spatial import ConvexHull


def calculate_planform_area_from_waverider(waverider):
    """
    Calculate accurate planform area from waverider geometry.
    
    This is the MOST ACCURATE method as it uses the actual geometry
    before discretization into STL.
    
    Method: Projects all upper surface points onto X-Z plane and
    computes the area using numerical integration.
    
    Parameters:
    -----------
    waverider : waverider object
        The waverider geometry object
        
    Returns:
    --------
    area : float
        Planform area in m²
    method : str
        Description of calculation method
    """
    # Get upper surface coordinates
    X = waverider.upper_surface_x  # (ny, nx)
    Y = waverider.upper_surface_y
    Z = waverider.upper_surface_z
    
    # Project onto X-Z plane (top view)
    # For accurate area, we need to account for both halves
    
    # Method 1: Numerical integration over surface patches
    total_area = 0.0
    ny, nx = X.shape
    
    # Integrate over each quadrilateral patch
    for i in range(ny - 1):
        for j in range(nx - 1):
            # Get four corners of patch
            p1 = np.array([X[i, j], Z[i, j]])
            p2 = np.array([X[i+1, j], Z[i+1, j]])
            p3 = np.array([X[i+1, j+1], Z[i+1, j+1]])
            p4 = np.array([X[i, j+1], Z[i, j+1]])
            
            # Calculate area of quadrilateral using cross product
            # Split into two triangles
            area1 = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
            area2 = 0.5 * abs(np.cross(p3 - p1, p4 - p1))
            total_area += (area1 + area2)
    
    # This is one half (Z >= 0), so double it for full vehicle
    full_area = 2.0 * total_area
    
    return full_area, "Direct integration over upper surface geometry"


def calculate_wetted_area_from_waverider(waverider):
    """
    Calculate total wetted area (upper + lower surfaces).
    
    Useful for drag estimation and surface area requirements.
    
    Parameters:
    -----------
    waverider : waverider object
        
    Returns:
    --------
    upper_area : float
        Upper surface wetted area in m²
    lower_area : float
        Lower surface wetted area in m²
    total_area : float
        Total wetted area in m²
    """
    # Upper surface area (with 3D curvature)
    X = waverider.upper_surface_x
    Y = waverider.upper_surface_y
    Z = waverider.upper_surface_z
    
    upper_area = 0.0
    ny, nx = X.shape
    
    for i in range(ny - 1):
        for j in range(nx - 1):
            # Four corners in 3D
            p1 = np.array([X[i, j], Y[i, j], Z[i, j]])
            p2 = np.array([X[i+1, j], Y[i+1, j], Z[i+1, j]])
            p3 = np.array([X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]])
            p4 = np.array([X[i, j+1], Y[i, j+1], Z[i, j+1]])
            
            # Triangle 1: p1-p2-p3
            area1 = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
            # Triangle 2: p1-p3-p4
            area2 = 0.5 * np.linalg.norm(np.cross(p3 - p1, p4 - p1))
            upper_area += (area1 + area2)
    
    # Double for both halves
    upper_area *= 2.0
    
    # Lower surface area (from streamlines)
    lower_area = 0.0
    streams = waverider.lower_surface_streams
    n_streams = len(streams)
    
    for i in range(n_streams - 1):
        stream1 = streams[i]
        stream2 = streams[i + 1]
        
        # Resample to same number of points if needed
        n_points = min(len(stream1), len(stream2))
        if len(stream1) != len(stream2):
            from scipy.interpolate import interp1d
            t1 = np.linspace(0, 1, len(stream1))
            t2 = np.linspace(0, 1, len(stream2))
            t_common = np.linspace(0, 1, n_points)
            
            s1 = np.column_stack([
                interp1d(t1, stream1[:, 0])(t_common),
                interp1d(t1, stream1[:, 1])(t_common),
                interp1d(t1, stream1[:, 2])(t_common),
            ])
            s2 = np.column_stack([
                interp1d(t2, stream2[:, 0])(t_common),
                interp1d(t2, stream2[:, 1])(t_common),
                interp1d(t2, stream2[:, 2])(t_common),
            ])
        else:
            s1 = stream1
            s2 = stream2
        
        # Calculate area of strip
        for j in range(n_points - 1):
            p1 = s1[j]
            p2 = s1[j+1]
            p3 = s2[j+1]
            p4 = s2[j]
            
            # Two triangles
            area1 = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
            area2 = 0.5 * np.linalg.norm(np.cross(p3 - p1, p4 - p1))
            lower_area += (area1 + area2)
    
    # Double for both halves
    lower_area *= 2.0
    
    total_area = upper_area + lower_area
    
    return upper_area, lower_area, total_area


def calculate_reference_area_from_stl(stl_filename):
    """
    Calculate projected planform area from STL file.
    
    Projects all triangles onto X-Z plane and sums areas.
    Less accurate than direct geometry method but works with any STL.
    
    Parameters:
    -----------
    stl_filename : str
        Path to STL file
        
    Returns:
    --------
    area : float
        Projected planform area in m²
    method : str
        Description of calculation method
    """
    try:
        from pysagas.geometry.parsers import MeshIO
        cells = MeshIO.load_from_file(stl_filename)
    except:
        # Fallback to manual STL parsing
        return _calculate_area_from_stl_manual(stl_filename)
    
    # Each cell is a triangle with vertices
    total_area = 0.0
    
    for cell in cells:
        # Get vertices (typically cell.verts or similar)
        # Project onto X-Z plane (ignore Y coordinate)
        try:
            v1 = np.array([cell.verts[0].x, cell.verts[0].z])
            v2 = np.array([cell.verts[1].x, cell.verts[1].z])
            v3 = np.array([cell.verts[2].x, cell.verts[2].z])
            
            # Area of triangle in 2D
            area = 0.5 * abs(np.cross(v2 - v1, v3 - v1))
            total_area += area
        except:
            # If structure is different, try alternative
            pass
    
    return total_area, "Projection from STL mesh"


def _calculate_area_from_stl_manual(stl_filename):
    """
    Manual STL parsing as fallback.
    """
    import struct
    
    with open(stl_filename, 'rb') as f:
        # Read header (80 bytes)
        header = f.read(80)
        
        # Read number of triangles
        n_triangles = struct.unpack('I', f.read(4))[0]
        
        total_area = 0.0
        
        for _ in range(n_triangles):
            # Read normal (3 floats, skip)
            f.read(12)
            
            # Read 3 vertices (9 floats)
            v1 = struct.unpack('fff', f.read(12))
            v2 = struct.unpack('fff', f.read(12))
            v3 = struct.unpack('fff', f.read(12))
            
            # Project onto X-Z plane (indices 0 and 2)
            p1 = np.array([v1[0], v1[2]])
            p2 = np.array([v2[0], v2[2]])
            p3 = np.array([v3[0], v3[2]])
            
            # Area of projected triangle
            area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
            total_area += area
            
            # Read attribute byte count (2 bytes, skip)
            f.read(2)
    
    return total_area, "Manual STL parsing"


def calculate_reference_area_convex_hull(waverider):
    """
    Calculate reference area using convex hull of planform.
    
    Conservative estimate (slightly larger than actual).
    
    Parameters:
    -----------
    waverider : waverider object
        
    Returns:
    --------
    area : float
        Convex hull planform area in m²
    method : str
        Description of calculation method
    """
    # Get all upper surface points projected to X-Z plane
    X = waverider.upper_surface_x.flatten()
    Z = waverider.upper_surface_z.flatten()
    
    # Create point cloud (one half only, will double)
    points = np.column_stack([X, Z])
    
    # Compute convex hull
    hull = ConvexHull(points)
    
    # Hull area (one half)
    half_area = hull.volume  # In 2D, volume = area
    
    # Double for full vehicle
    full_area = 2.0 * half_area
    
    return full_area, "Convex hull of planform projection"


def calculate_reference_area_simple(waverider):
    """
    Simple rectangular approximation: width × height.
    
    Fast but least accurate. Useful for quick estimates.
    
    Parameters:
    -----------
    waverider : waverider object
        
    Returns:
    --------
    area : float
        Simple rectangular area in m²
    method : str
        Description of calculation method
    """
    area = waverider.width * waverider.height
    return area, "Simple rectangular approximation (width × height)"


def compare_reference_area_methods(waverider, stl_file=None):
    """
    Compare all available reference area calculation methods.
    
    Parameters:
    -----------
    waverider : waverider object
    stl_file : str, optional
        Path to STL file for STL-based calculation
        
    Returns:
    --------
    results : dict
        Dictionary with all methods and their results
    """
    results = {}
    
    # Method 1: Direct geometry (most accurate)
    try:
        area, method = calculate_planform_area_from_waverider(waverider)
        results['direct_geometry'] = {
            'area': area,
            'method': method,
            'accuracy': '★★★★★ (Most accurate)'
        }
    except Exception as e:
        results['direct_geometry'] = {'error': str(e)}
    
    # Method 2: Convex hull
    try:
        area, method = calculate_reference_area_convex_hull(waverider)
        results['convex_hull'] = {
            'area': area,
            'method': method,
            'accuracy': '★★★★☆ (Conservative)'
        }
    except Exception as e:
        results['convex_hull'] = {'error': str(e)}
    
    # Method 3: Simple approximation
    try:
        area, method = calculate_reference_area_simple(waverider)
        results['simple'] = {
            'area': area,
            'method': method,
            'accuracy': '★★☆☆☆ (Quick estimate)'
        }
    except Exception as e:
        results['simple'] = {'error': str(e)}
    
    # Method 4: From STL (if provided)
    if stl_file:
        try:
            area, method = calculate_reference_area_from_stl(stl_file)
            results['stl_projection'] = {
                'area': area,
                'method': method,
                'accuracy': '★★★★☆ (Good for any mesh)'
            }
        except Exception as e:
            results['stl_projection'] = {'error': str(e)}
    
    # Method 5: Wetted areas
    try:
        upper, lower, total = calculate_wetted_area_from_waverider(waverider)
        results['wetted_areas'] = {
            'upper_area': upper,
            'lower_area': lower,
            'total_area': total,
            'method': 'Full 3D surface area calculation',
            'accuracy': '★★★★★ (For drag estimation)'
        }
    except Exception as e:
        results['wetted_areas'] = {'error': str(e)}
    
    return results


def print_area_comparison(results):
    """Pretty print area comparison results."""
    print("\n" + "="*70)
    print("REFERENCE AREA COMPARISON")
    print("="*70)
    
    for method_name, data in results.items():
        print(f"\n{method_name.upper().replace('_', ' ')}:")
        if 'error' in data:
            print(f"  ❌ Error: {data['error']}")
        elif 'total_area' in data:
            # Wetted areas
            print(f"  Upper surface:  {data['upper_area']:.4f} m²")
            print(f"  Lower surface:  {data['lower_area']:.4f} m²")
            print(f"  Total wetted:   {data['total_area']:.4f} m²")
            print(f"  Accuracy: {data['accuracy']}")
        else:
            print(f"  Area: {data['area']:.4f} m²")
            print(f"  Method: {data['method']}")
            print(f"  Accuracy: {data['accuracy']}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("  Use 'direct_geometry' for most accurate reference area")
    print("  Use 'simple' only for quick estimates")
    print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude')
    from waverider_generator.generator import waverider as wr
    
    print("Generating baseline waverider...")
    waverider = wr(
        M_inf=5.0,
        beta=15.0,
        height=1.34,
        width=3.0,
        dp=[0.11, 0.63, 0.0, 0.46],
        n_upper_surface=10000,
        n_shockwave=10000,
        n_planes=40,
        n_streamwise=30,
        delta_streamwise=0.1
    )
    
    print("\nCalculating reference areas using multiple methods...")
    results = compare_reference_area_methods(waverider)
    
    print_area_comparison(results)
    
    # Show difference from simple approximation
    simple_area = results['simple']['area']
    direct_area = results['direct_geometry']['area']
    
    difference_pct = 100 * (direct_area - simple_area) / simple_area
    
    print(f"\nDifference between accurate and simple methods:")
    print(f"  Simple (width × height): {simple_area:.4f} m²")
    print(f"  Accurate (direct):       {direct_area:.4f} m²")
    print(f"  Difference:              {difference_pct:+.2f}%")
    
    if abs(difference_pct) > 10:
        print(f"\n⚠️  Significant difference! Using width×height may cause {abs(difference_pct):.1f}% error in coefficients!")
    else:
        print(f"\n✓ Difference is small (<10%), simple approximation is acceptable")