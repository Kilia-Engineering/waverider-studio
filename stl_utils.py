#!/usr/bin/env python3
"""
STL Utilities for Waverider GUI
Handles STL mesh generation using Gmsh and STL file I/O
"""

import numpy as np
import os
import struct


def generate_stl_with_gmsh(step_file, stl_output, min_size=0.005, max_size=0.05):
    """
    Generate STL mesh from STEP file using Gmsh
    
    Parameters:
    -----------
    step_file : str
        Path to input STEP file
    stl_output : str
        Path to output STL file
    min_size : float
        Minimum element size in meters (default: 0.005m = 5mm)
    max_size : float
        Maximum element size in meters (default: 0.05m = 50mm)
        
    Returns:
    --------
    success : bool
        True if successful
    message : str
        Status message
    """
    try:
        import gmsh
    except ImportError:
        return False, "Gmsh Python API not installed. Install with: pip install gmsh"
    
    if not os.path.exists(step_file):
        return False, f"STEP file not found: {step_file}"
    
    try:
        # Initialize Gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress terminal output
        
        # Import STEP file
        gmsh.merge(step_file)
        
        # Set mesh parameters (values are already in meters)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.RecombineAll", 0)  # Use triangles
        
        # Generate 2D mesh (surface mesh)
        gmsh.model.mesh.generate(2)
        
        # Write STL
        gmsh.write(stl_output)
        
        # Get mesh statistics
        num_triangles = gmsh.model.mesh.getNodes()[0].size // 3
        
        # Finalize
        gmsh.finalize()
        
        return True, f"Mesh generated successfully: {num_triangles} triangles"
        
    except Exception as e:
        try:
            gmsh.finalize()
        except:
            pass
        return False, f"Gmsh error: {str(e)}"


def load_stl_file(filename):
    """
    Load STL file and return vertices and faces
    
    Parameters:
    -----------
    filename : str
        Path to STL file (binary or ASCII)
        
    Returns:
    --------
    vertices : ndarray (N, 3)
        Vertex coordinates
    faces : ndarray (M, 3)
        Triangle face indices
    info : dict
        Mesh information (num_triangles, bounds, etc.)
    """
    # Try binary first
    try:
        return _load_stl_binary(filename)
    except:
        # Fall back to ASCII
        try:
            return _load_stl_ascii(filename)
        except Exception as e:
            raise ValueError(f"Failed to load STL file: {str(e)}")


def _load_stl_binary(filename):
    """Load binary STL file"""
    with open(filename, 'rb') as f:
        # Read header (80 bytes)
        header = f.read(80)
        
        # Read number of triangles
        n_triangles = struct.unpack('I', f.read(4))[0]
        
        # Read triangles
        vertices_list = []
        faces = []
        vertex_map = {}  # Map vertex tuple to index
        vertex_count = 0
        
        for i in range(n_triangles):
            # Read normal (skip)
            f.read(12)
            
            # Read 3 vertices
            tri_vertices = []
            for j in range(3):
                v = struct.unpack('fff', f.read(12))
                v_tuple = tuple(v)
                
                # Check if vertex already exists
                if v_tuple not in vertex_map:
                    vertex_map[v_tuple] = vertex_count
                    vertices_list.append(v)
                    vertex_count += 1
                
                tri_vertices.append(vertex_map[v_tuple])
            
            faces.append(tri_vertices)
            
            # Read attribute byte count (skip)
            f.read(2)
    
    vertices = np.array(vertices_list)
    faces = np.array(faces, dtype=np.int32)
    
    # Calculate info
    info = {
        'num_triangles': n_triangles,
        'num_vertices': len(vertices),
        'bounds': {
            'x': (vertices[:, 0].min(), vertices[:, 0].max()),
            'y': (vertices[:, 1].min(), vertices[:, 1].max()),
            'z': (vertices[:, 2].min(), vertices[:, 2].max()),
        },
        'format': 'binary'
    }
    
    return vertices, faces, info


def _load_stl_ascii(filename):
    """Load ASCII STL file"""
    vertices_list = []
    faces = []
    vertex_map = {}
    vertex_count = 0
    
    with open(filename, 'r') as f:
        current_vertices = []
        
        for line in f:
            line = line.strip()
            
            if line.startswith('vertex'):
                # Parse vertex coordinates
                parts = line.split()
                v = tuple(float(x) for x in parts[1:4])
                
                if v not in vertex_map:
                    vertex_map[v] = vertex_count
                    vertices_list.append(v)
                    vertex_count += 1
                
                current_vertices.append(vertex_map[v])
                
            elif line.startswith('endfacet'):
                # Complete triangle
                if len(current_vertices) == 3:
                    faces.append(current_vertices)
                current_vertices = []
    
    vertices = np.array(vertices_list)
    faces = np.array(faces, dtype=np.int32)
    
    # Calculate info
    info = {
        'num_triangles': len(faces),
        'num_vertices': len(vertices),
        'bounds': {
            'x': (vertices[:, 0].min(), vertices[:, 0].max()),
            'y': (vertices[:, 1].min(), vertices[:, 1].max()),
            'z': (vertices[:, 2].min(), vertices[:, 2].max()),
        },
        'format': 'ascii'
    }
    
    return vertices, faces, info


def calculate_mesh_quality(vertices, faces):
    """
    Calculate mesh quality metrics
    
    Parameters:
    -----------
    vertices : ndarray
    faces : ndarray
        
    Returns:
    --------
    quality : dict
        Quality metrics
    """
    areas = []
    aspect_ratios = []
    
    for face in faces:
        v0, v1, v2 = vertices[face]
        
        # Calculate edge lengths
        e0 = np.linalg.norm(v1 - v0)
        e1 = np.linalg.norm(v2 - v1)
        e2 = np.linalg.norm(v0 - v2)
        
        # Triangle area
        s = (e0 + e1 + e2) / 2  # Semi-perimeter
        area = np.sqrt(s * (s - e0) * (s - e1) * (s - e2))
        areas.append(area)
        
        # Aspect ratio (longest edge / shortest edge)
        max_edge = max(e0, e1, e2)
        min_edge = min(e0, e1, e2)
        if min_edge > 0:
            aspect_ratios.append(max_edge / min_edge)
    
    areas = np.array(areas)
    aspect_ratios = np.array(aspect_ratios)
    
    quality = {
        'num_triangles': len(faces),
        'num_vertices': len(vertices),
        'min_area': areas.min(),
        'max_area': areas.max(),
        'mean_area': areas.mean(),
        'min_aspect_ratio': aspect_ratios.min(),
        'max_aspect_ratio': aspect_ratios.max(),
        'mean_aspect_ratio': aspect_ratios.mean(),
    }
    
    return quality


def format_mesh_info(info, quality=None):
    """Format mesh information for display"""
    text = f"Triangles: {info['num_triangles']:,}\n"
    text += f"Vertices: {info['num_vertices']:,}\n"
    text += f"Format: {info['format']}\n\n"
    
    text += "Bounds:\n"
    for axis, (vmin, vmax) in info['bounds'].items():
        text += f"  {axis}: [{vmin:.3f}, {vmax:.3f}] m\n"
    
    if quality:
        text += f"\nQuality:\n"
        text += f"  Mean aspect ratio: {quality['mean_aspect_ratio']:.2f}\n"
        text += f"  Max aspect ratio: {quality['max_aspect_ratio']:.2f}\n"
    
    return text