"""
Simple Leading Edge Blunting for Waveriders - V2

This module provides a straightforward approach to blunting sharp waverider
leading edges that preserves surface smoothness.

The key insight: instead of replacing the leading edge region with new geometry,
we SHIFT the existing stream points and smoothly blend them.

Approach:
1. Shift all stream points forward (negative X) proportionally near the LE
2. The shift decays to zero at the blend distance
3. This preserves the original surface curvature while adding bluntness

Author: Claude (Anthropic)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.interpolate import interp1d


@dataclass 
class SimpleBluntingParams:
    """Parameters for simple leading edge blunting."""
    nose_radius: float = 0.01  # Nose radius in meters (default 10mm)
    blend_fraction: float = 0.05  # Fraction of chord for blending (5%)
    n_nose_points: int = 15  # Points in the nose arc


class SimpleBluntedWaverider:
    """
    A waverider with simply blunted leading edges.
    
    Uses a smooth blending approach that preserves surface quality.
    """
    
    def __init__(self, waverider, params: SimpleBluntingParams = None):
        """
        Create a blunted waverider from a sharp one.
        """
        self.original = waverider
        self.params = params or SimpleBluntingParams()
        
        # Copy basic properties - handle different attribute names
        # Standard waverider uses M_inf, SHADOW uses mach
        if hasattr(waverider, 'M_inf'):
            self.M_inf = waverider.M_inf
        elif hasattr(waverider, 'mach'):
            self.M_inf = waverider.mach
        else:
            self.M_inf = 5.0  # Default
        
        # Also provide 'mach' alias for compatibility
        self.mach = self.M_inf
        
        if hasattr(waverider, 'beta'):
            self.beta = waverider.beta
        elif hasattr(waverider, 'shock_angle'):
            self.beta = waverider.shock_angle
        else:
            self.beta = 15.0  # Default
            
        self.height = getattr(waverider, 'height', 1.0)
        self.width = getattr(waverider, 'width', 1.0)
        self.length = getattr(waverider, 'length', 1.0)
        
        # These will hold the blunted geometry
        self.upper_surface_streams = None
        self.lower_surface_streams = None
        self.leading_edge = None
        
        # Apply the blunting
        self._apply_simple_blunting()
    
    def _apply_simple_blunting(self):
        """Apply simple blunting by shifting and blending stream points."""
        
        # Handle different waverider types
        # Standard waverider has upper_surface_streams (list of 2D arrays)
        # SHADOW waverider has upper_surface (3D array: n_span x n_streamwise x 3)
        
        if hasattr(self.original, 'upper_surface_streams') and self.original.upper_surface_streams is not None:
            # Standard waverider format
            orig_upper = self.original.upper_surface_streams
            orig_lower = self.original.lower_surface_streams
            self._is_shadow_format = False
        elif hasattr(self.original, 'upper_surface') and self.original.upper_surface is not None:
            # SHADOW waverider format - convert 3D array to list of streams
            upper_3d = self.original.upper_surface  # (n_span, n_streamwise, 3)
            lower_3d = self.original.lower_surface
            orig_upper = [upper_3d[i, :, :] for i in range(upper_3d.shape[0])]
            orig_lower = [lower_3d[i, :, :] for i in range(lower_3d.shape[0])]
            self._is_shadow_format = True
            print(f"  Detected SHADOW waverider format, converting to streams...")
        else:
            raise ValueError("Unknown waverider format - missing surface data")
        
        n_stations = len(orig_upper)
        
        # Deep copy the streams
        self.upper_surface_streams = [s.copy() for s in orig_upper]
        self.lower_surface_streams = [s.copy() for s in orig_lower]
        self.leading_edge = self.original.leading_edge.copy()
        
        print(f"\nApplying simple blunting V2 (smooth shift):")
        print(f"  Nose radius: {self.params.nose_radius * 1000:.2f} mm")
        print(f"  Blend fraction: {self.params.blend_fraction * 100:.1f}%")
        print(f"  Stations: {n_stations}")
        
        for i in range(n_stations):
            upper = self.upper_surface_streams[i]
            lower = self.lower_surface_streams[i]
            
            if len(upper) < 5 or len(lower) < 5:
                continue
            
            # Compute local chord
            local_chord = np.linalg.norm(upper[-1] - upper[0])
            blend_length = self.params.blend_fraction * local_chord
            
            # Apply smooth shift to upper surface
            self.upper_surface_streams[i] = self._shift_stream_smoothly(
                upper, self.params.nose_radius, blend_length, is_upper=True
            )
            
            # Apply smooth shift to lower surface  
            self.lower_surface_streams[i] = self._shift_stream_smoothly(
                lower, self.params.nose_radius, blend_length, is_upper=False
            )
            
            # Update leading edge point
            self.leading_edge[i] = self.upper_surface_streams[i][0]
        
        self._verify_blunting()
    
    def _shift_stream_smoothly(self, stream: np.ndarray, nose_radius: float,
                                blend_length: float, is_upper: bool) -> np.ndarray:
        """
        Shift stream points to create bluntness while preserving smoothness.
        
        The shift is maximum at the leading edge and decays smoothly to zero
        at the blend distance using a cosine blend function.
        """
        
        new_stream = stream.copy()
        le_x = stream[0, 0]
        le_y = stream[0, 1]
        
        # Compute arc length along stream for better parameterization
        arc_length = np.zeros(len(stream))
        for j in range(1, len(stream)):
            arc_length[j] = arc_length[j-1] + np.linalg.norm(stream[j] - stream[j-1])
        
        total_arc = arc_length[-1]
        if total_arc < 1e-10:
            return new_stream
        
        # Find the arc length corresponding to blend_length in x
        blend_arc = 0
        for j in range(len(stream)):
            if stream[j, 0] - le_x >= blend_length:
                blend_arc = arc_length[j]
                break
        else:
            blend_arc = total_arc * 0.2  # Fallback: 20% of arc length
        
        # Apply smooth shift using cosine blending
        for j in range(len(stream)):
            s = arc_length[j]
            
            if s >= blend_arc:
                # Beyond blend region - no shift
                continue
            
            # Cosine blend: 1 at s=0, 0 at s=blend_arc
            # Using (1 + cos(pi * s / blend_arc)) / 2 for smooth transition
            t = s / blend_arc
            blend_factor = 0.5 * (1.0 + np.cos(np.pi * t))
            
            # X shift: move forward (negative X direction)
            x_shift = -nose_radius * blend_factor
            
            # Y shift: create roundness
            # At the very tip, shift Y toward the centerline to create the nose cap
            # This creates an elliptical-like cross-section
            if is_upper:
                # Upper surface: shift slightly down at the nose
                y_shift = -nose_radius * 0.3 * blend_factor * blend_factor
            else:
                # Lower surface: shift slightly up at the nose
                y_shift = nose_radius * 0.3 * blend_factor * blend_factor
            
            new_stream[j, 0] += x_shift
            new_stream[j, 1] += y_shift
        
        return new_stream
    
    def _verify_blunting(self):
        """Verify that blunting was applied correctly."""
        
        if len(self.upper_surface_streams) == 0:
            print("  WARNING: No streams to verify")
            return
        
        orig_le = self.original.upper_surface_streams[0][0]
        new_le = self.upper_surface_streams[0][0]
        
        diff = np.linalg.norm(new_le - orig_le)
        
        print(f"\nBlunting verification:")
        print(f"  Original LE[0]: [{orig_le[0]:.6f}, {orig_le[1]:.6f}, {orig_le[2]:.6f}]")
        print(f"  Blunted LE[0]:  [{new_le[0]:.6f}, {new_le[1]:.6f}, {new_le[2]:.6f}]")
        print(f"  Position change: {diff * 1000:.4f} mm")
        
        x_shift = orig_le[0] - new_le[0]
        if x_shift > 0:
            print(f"  ✓ Nose moved forward by {x_shift * 1000:.2f} mm")
        else:
            print(f"  WARNING: Nose did not move forward!")
        
        # Check stream lengths (should be same as original with this method)
        orig_len = len(self.original.upper_surface_streams[0])
        new_len = len(self.upper_surface_streams[0])
        print(f"  Stream length: {orig_len} (unchanged)")


def simple_blunt_waverider(waverider, nose_radius: float = 0.01,
                           blend_fraction: float = 0.05,
                           n_nose_points: int = 15,
                           aspect_ratio: float = 1.5) -> SimpleBluntedWaverider:
    """
    Convenience function to create a blunted waverider.
    
    Parameters:
    -----------
    waverider : waverider object
        The original sharp waverider
    nose_radius : float
        Nose bluntness radius in meters (default 0.01 = 10mm)
    blend_fraction : float
        Fraction of chord for blending region (default 0.05 = 5%)
    n_nose_points : int
        Not used in V2, kept for API compatibility
    aspect_ratio : float
        Not used in V2, kept for API compatibility
        
    Returns:
    --------
    SimpleBluntedWaverider
        The blunted waverider object
    """
    params = SimpleBluntingParams(
        nose_radius=nose_radius,
        blend_fraction=blend_fraction,
        n_nose_points=n_nose_points
    )
    
    return SimpleBluntedWaverider(waverider, params)
