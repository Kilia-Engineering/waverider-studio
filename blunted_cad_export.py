"""
CAD Export for Blunted Waveriders

This module provides CAD export functionality specifically designed for 
waveriders with blunted leading edges. Unlike the standard interpPlate 
approach, this uses lofted surfaces through cross-sections which properly
handles the modified stream geometry from blunting.

Author: Claude (Anthropic)
"""

import numpy as np
import cadquery as cq
from cadquery import exporters
from scipy.interpolate import interp1d


def export_blunted_waverider(waverider, filename: str, sides: str = "both", 
                              scale: float = 1.0, n_loft_sections: int = None):
    """
    Export a blunted waverider to STEP format using lofted surfaces.
    
    This function creates cross-section profiles at each spanwise station
    and lofts between them to create smooth surfaces. This approach works
    well with blunted geometry where the point distribution varies along
    each stream.
    
    Parameters:
    -----------
    waverider : waverider or BluntedWaverider
        The waverider object to export
    filename : str
        Output STEP filename
    sides : str
        'left', 'right', or 'both'
    scale : float
        Scale factor (1.0 = meters)
    n_loft_sections : int, optional
        Number of cross-sections for lofting. If None, uses number of streams.
        
    Returns:
    --------
    cq.Workplane or cq.Solid
        The exported CAD geometry
    """
    
    print(f"Exporting blunted waverider using lofted surface approach...")
    
    # Extract streams
    us_streams = waverider.upper_surface_streams
    ls_streams = waverider.lower_surface_streams
    
    n_stations = len(us_streams)
    print(f"  Number of spanwise stations: {n_stations}")
    
    # Determine number of loft sections
    if n_loft_sections is None:
        n_loft_sections = n_stations
    
    # We'll create cross-section wires at each station and loft between them
    # Each cross-section is a closed wire: upper stream -> TE -> lower stream -> LE
    
    cross_sections = []
    
    for i in range(n_stations):
        try:
            upper_stream = us_streams[i]
            lower_stream = ls_streams[i]
            
            # Skip stations with too few points
            if len(upper_stream) < 3 or len(lower_stream) < 3:
                print(f"  Skipping station {i}: too few points")
                continue
            
            # Create cross-section wire
            # The cross-section goes: LE -> along upper -> TE -> along lower (reversed) -> back to LE
            
            # Upper stream points (from LE to TE)
            upper_pts = [tuple(p * scale) for p in upper_stream]
            
            # Lower stream points (from LE to TE, we need TE to LE so reverse)
            lower_pts = [tuple(p * scale) for p in lower_stream[::-1]]
            
            # Create the cross-section as a closed spline
            # Combine: upper (LE->TE) + lower reversed (TE->LE)
            # The loop closes at LE
            
            all_pts = upper_pts + lower_pts[1:]  # Skip first point of lower (it's the TE, same as last of upper... wait no)
            
            # Actually, upper goes from LE to TE, lower also goes from LE to TE
            # So upper[-1] is TE, lower[-1] is also TE (they should be close)
            # And upper[0] is LE, lower[0] is also LE (they should be the same)
            
            # For a closed section: LE_upper -> ... -> TE_upper -> TE_lower -> ... -> LE_lower -> close to LE_upper
            # Since LE_upper == LE_lower, we have a closed loop
            
            # upper_pts: from LE to TE (upper surface)
            # lower_pts_reversed: from TE to LE (lower surface)
            lower_pts_reversed = [tuple(p * scale) for p in lower_stream[::-1]]
            
            # Combine into closed loop (skip duplicate TE point)
            section_pts = list(upper_pts) + list(lower_pts_reversed[1:])
            
            # Close the loop by adding first point at end if not already there
            if np.linalg.norm(np.array(section_pts[0]) - np.array(section_pts[-1])) > 1e-6 * scale:
                section_pts.append(section_pts[0])
            
            # Create spline wire
            if len(section_pts) >= 4:
                try:
                    wire = cq.Workplane("XY").spline(section_pts, periodic=True).close().val()
                    cross_sections.append(wire)
                except Exception as e:
                    print(f"  Station {i}: spline failed ({e}), trying polyline")
                    # Fallback to polyline
                    try:
                        wire = cq.Wire.makePolygon([cq.Vector(*p) for p in section_pts])
                        cross_sections.append(wire)
                    except Exception as e2:
                        print(f"  Station {i}: polyline also failed ({e2})")
                        
        except Exception as e:
            print(f"  Error at station {i}: {e}")
            continue
    
    print(f"  Created {len(cross_sections)} cross-sections")
    
    if len(cross_sections) < 2:
        raise ValueError("Need at least 2 valid cross-sections for lofting")
    
    # Create lofted solid
    try:
        # Use CadQuery's loft functionality
        # First, create a workplane and add the wires
        result = cq.Workplane("XY")
        
        # Try to loft between the cross-sections
        # CadQuery loft expects wires on different planes
        solid = cq.Solid.makeLoft([w for w in cross_sections])
        
        print(f"  Loft successful!")
        
    except Exception as e:
        print(f"  Loft failed: {e}")
        print(f"  Falling back to surface-based approach...")
        
        # Fallback: create surfaces from streams directly
        solid = _create_surfaces_from_streams(us_streams, ls_streams, waverider.length, scale)
    
    # Handle sides
    if sides == "left":
        if filename:
            cq.exporters.export(solid, filename)
        return solid
    
    elif sides == "right":
        right_side = solid.mirror(mirrorPlane='XY')
        if filename:
            cq.exporters.export(right_side, filename)
        return right_side
    
    elif sides == "both":
        right_side = solid.mirror(mirrorPlane='XY')
        try:
            combined = cq.Workplane("XY").newObject([solid]).union(right_side)
            if filename:
                cq.exporters.export(combined, filename)
            return combined
        except:
            # If union fails, export as compound
            compound = cq.Compound.makeCompound([solid, right_side])
            if filename:
                cq.exporters.export(compound, filename)
            return compound
    
    return solid


def _create_surfaces_from_streams(us_streams, ls_streams, length, scale):
    """
    Create surfaces directly from stream data using ruled surfaces between adjacent streams.
    
    This is a fallback method that creates the geometry as a collection of ruled
    surface patches between adjacent streamlines.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.TopoDS import TopoDS_Wire
    from OCP.gp import gp_Pnt
    from OCP.Geom import Geom_BSplineCurve
    from OCP.TColgp import TColgp_Array1OfPnt
    from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    
    print("  Creating surfaces from stream patches...")
    
    faces = []
    n_streams = len(us_streams)
    
    # Create upper surface patches
    for i in range(n_streams - 1):
        try:
            stream1 = us_streams[i] * scale
            stream2 = us_streams[i + 1] * scale
            
            # Resample streams to same length
            n_pts = min(len(stream1), len(stream2), 50)
            
            t1 = np.linspace(0, 1, len(stream1))
            t2 = np.linspace(0, 1, len(stream2))
            t_new = np.linspace(0, 1, n_pts)
            
            s1_resampled = np.zeros((n_pts, 3))
            s2_resampled = np.zeros((n_pts, 3))
            
            for dim in range(3):
                s1_resampled[:, dim] = interp1d(t1, stream1[:, dim])(t_new)
                s2_resampled[:, dim] = interp1d(t2, stream2[:, dim])(t_new)
            
            # Create ruled surface between the two streams
            edge1 = cq.Edge.makeSpline([cq.Vector(*p) for p in s1_resampled])
            edge2 = cq.Edge.makeSpline([cq.Vector(*p) for p in s2_resampled])
            
            # Create a ruled face
            face = cq.Face.makeRuledSurface(edge1, edge2)
            faces.append(face)
            
        except Exception as e:
            print(f"    Upper patch {i}-{i+1} failed: {e}")
    
    # Create lower surface patches
    for i in range(n_streams - 1):
        try:
            stream1 = ls_streams[i] * scale
            stream2 = ls_streams[i + 1] * scale
            
            n_pts = min(len(stream1), len(stream2), 50)
            
            t1 = np.linspace(0, 1, len(stream1))
            t2 = np.linspace(0, 1, len(stream2))
            t_new = np.linspace(0, 1, n_pts)
            
            s1_resampled = np.zeros((n_pts, 3))
            s2_resampled = np.zeros((n_pts, 3))
            
            for dim in range(3):
                s1_resampled[:, dim] = interp1d(t1, stream1[:, dim])(t_new)
                s2_resampled[:, dim] = interp1d(t2, stream2[:, dim])(t_new)
            
            edge1 = cq.Edge.makeSpline([cq.Vector(*p) for p in s1_resampled])
            edge2 = cq.Edge.makeSpline([cq.Vector(*p) for p in s2_resampled])
            
            face = cq.Face.makeRuledSurface(edge1, edge2)
            faces.append(face)
            
        except Exception as e:
            print(f"    Lower patch {i}-{i+1} failed: {e}")
    
    # Create trailing edge face
    try:
        te_upper = np.vstack([s[-1] for s in us_streams]) * scale
        te_lower = np.vstack([s[-1] for s in ls_streams]) * scale
        
        edge1 = cq.Edge.makeSpline([cq.Vector(*p) for p in te_upper])
        edge2 = cq.Edge.makeSpline([cq.Vector(*p) for p in te_lower])
        
        # Add closing edges at symmetry plane
        sym_edge1 = cq.Edge.makeLine(cq.Vector(*te_upper[0]), cq.Vector(*te_lower[0]))
        
        te_face = cq.Face.makeRuledSurface(edge1, edge2)
        faces.append(te_face)
    except Exception as e:
        print(f"    TE face failed: {e}")
    
    # Create symmetry plane face
    try:
        # Symmetry plane is at z=0, bounded by upper and lower centerline streams
        sym_upper = us_streams[0] * scale  # Centerline upper
        sym_lower = ls_streams[0] * scale  # Centerline lower
        
        edge1 = cq.Edge.makeSpline([cq.Vector(*p) for p in sym_upper])
        edge2 = cq.Edge.makeSpline([cq.Vector(*p) for p in sym_lower])
        
        sym_face = cq.Face.makeRuledSurface(edge1, edge2)
        faces.append(sym_face)
    except Exception as e:
        print(f"    Symmetry face failed: {e}")
    
    print(f"  Created {len(faces)} surface patches")
    
    # Try to create a shell/solid from the faces
    if len(faces) > 0:
        try:
            shell = cq.Shell.makeShell(faces)
            solid = cq.Solid.makeSolid(shell)
            return solid
        except Exception as e:
            print(f"  Could not create solid from shell: {e}")
            # Return as compound of faces
            return cq.Compound.makeCompound(faces)
    
    raise ValueError("Could not create any valid geometry")


def export_waverider_stl(waverider, filename: str, scale: float = 1.0, 
                         linear_deflection: float = 0.01):
    """
    Export waverider directly to STL format by triangulating the streams.
    
    This is a more reliable export method that doesn't require complex
    surface fitting - it directly creates triangular facets from the
    stream data.
    
    Parameters:
    -----------
    waverider : waverider or BluntedWaverider
        The waverider object
    filename : str
        Output STL filename
    scale : float
        Scale factor
    linear_deflection : float
        Mesh density control (smaller = finer mesh)
    """
    import struct
    
    us_streams = waverider.upper_surface_streams
    ls_streams = waverider.lower_surface_streams
    
    # Debug: Check if this is a blunted waverider and print LE info
    print(f"\n{'='*60}")
    print(f"STL EXPORT - Geometry verification:")
    print(f"  Waverider type: {type(waverider).__name__}")
    
    if hasattr(waverider, 'params'):
        print(f"  BLUNTED waverider detected!")
        print(f"  Nose radius: {waverider.params.nose_radius * 1000:.2f} mm")
        if hasattr(waverider.params, 'profile_type'):
            print(f"  Profile type: {waverider.params.profile_type}")
        else:
            print(f"  Profile type: simple-ellipse")
    else:
        print(f"  Sharp (original) waverider")
    
    # Check leading edge points
    le_points = np.vstack([s[0] for s in us_streams])
    print(f"  Number of LE points: {len(le_points)}")
    print(f"  First LE point (nose): {le_points[0]}")
    
    # For blunted, the first stream should have nose profile points
    print(f"  Upper stream 0 length: {len(us_streams[0])} points")
    print(f"  Upper stream 0 first 3 points:")
    for i, pt in enumerate(us_streams[0][:3]):
        print(f"    [{i}]: {pt}")
    print(f"{'='*60}\n")
    
    triangles = []
    
    # Triangulate upper surface
    for i in range(len(us_streams) - 1):
        stream1 = us_streams[i] * scale
        stream2 = us_streams[i + 1] * scale
        
        # Resample to same length
        n_pts = min(len(stream1), len(stream2))
        t1 = np.linspace(0, 1, len(stream1))
        t2 = np.linspace(0, 1, len(stream2))
        t_new = np.linspace(0, 1, n_pts)
        
        s1 = np.zeros((n_pts, 3))
        s2 = np.zeros((n_pts, 3))
        for dim in range(3):
            s1[:, dim] = interp1d(t1, stream1[:, dim])(t_new)
            s2[:, dim] = interp1d(t2, stream2[:, dim])(t_new)
        
        # Create triangles
        for j in range(n_pts - 1):
            # Two triangles per quad
            p1, p2, p3, p4 = s1[j], s1[j+1], s2[j+1], s2[j]
            triangles.append((p1, p2, p3))
            triangles.append((p1, p3, p4))
    
    # Triangulate lower surface
    for i in range(len(ls_streams) - 1):
        stream1 = ls_streams[i] * scale
        stream2 = ls_streams[i + 1] * scale
        
        n_pts = min(len(stream1), len(stream2))
        t1 = np.linspace(0, 1, len(stream1))
        t2 = np.linspace(0, 1, len(stream2))
        t_new = np.linspace(0, 1, n_pts)
        
        s1 = np.zeros((n_pts, 3))
        s2 = np.zeros((n_pts, 3))
        for dim in range(3):
            s1[:, dim] = interp1d(t1, stream1[:, dim])(t_new)
            s2[:, dim] = interp1d(t2, stream2[:, dim])(t_new)
        
        for j in range(n_pts - 1):
            p1, p2, p3, p4 = s1[j], s1[j+1], s2[j+1], s2[j]
            # Reverse winding for lower surface
            triangles.append((p1, p4, p3))
            triangles.append((p1, p3, p2))
    
    # Mirror for full vehicle
    mirrored_tris = []
    for tri in triangles:
        p1, p2, p3 = tri
        # Mirror across XY plane (negate Z)
        mp1 = np.array([p1[0], p1[1], -p1[2]])
        mp2 = np.array([p2[0], p2[1], -p2[2]])
        mp3 = np.array([p3[0], p3[1], -p3[2]])
        # Reverse winding for mirrored side
        mirrored_tris.append((mp1, mp3, mp2))
    
    all_triangles = triangles + mirrored_tris
    
    # Write binary STL
    with open(filename, 'wb') as f:
        # Header (80 bytes)
        header = b'Binary STL - Blunted Waverider' + b'\0' * 50
        f.write(header[:80])
        
        # Number of triangles
        f.write(struct.pack('<I', len(all_triangles)))
        
        # Write each triangle
        for tri in all_triangles:
            p1, p2, p3 = tri
            # Compute normal
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p3) - np.array(p1)
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            else:
                normal = np.array([0, 0, 1])
            
            # Write normal
            f.write(struct.pack('<fff', *normal))
            # Write vertices
            f.write(struct.pack('<fff', *p1))
            f.write(struct.pack('<fff', *p2))
            f.write(struct.pack('<fff', *p3))
            # Attribute byte count
            f.write(struct.pack('<H', 0))
    
    print(f"Exported {len(all_triangles)} triangles to {filename}")
    return len(all_triangles)
