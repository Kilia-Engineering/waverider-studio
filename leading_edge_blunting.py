#!/usr/bin/env python3
"""
Leading Edge Blunting Module for Waveriders

Provides post-processing capabilities to convert infinitely sharp waverider
leading edges into manufacturable blunted profiles while maintaining surface
continuity (G1 tangent or G2 curvature).

Two blunting profiles are supported:
1. Ellipse - Classic blunting with controllable aspect ratio
2. Power-law - y ∝ x^n profile common in hypersonic nose designs

The algorithm works in local 2D coordinate systems at each spanwise station,
then transforms back to 3D. This ensures proper geometric handling.

Author: Angelos & Claude
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq
from typing import Tuple, List, Optional, Literal
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class BluntingParameters:
    """Parameters controlling the leading edge blunting."""
    
    # Blunting profile type
    profile_type: Literal["ellipse", "power_law"] = "ellipse"
    
    # Bluntness radius at the nose (m)
    # For ellipse: semi-minor axis (perpendicular to flow)
    # For power-law: characteristic radius of curvature at nose
    nose_radius: float = 0.005  # 5mm default
    
    # Blend length - how far back the modification extends (as fraction of local chord)
    blend_length_fraction: float = 0.05  # 5% of local chord
    
    # For ellipse: aspect ratio (a/b where a is along flow, b is perpendicular)
    ellipse_aspect_ratio: float = 2.0
    
    # For power-law: exponent n (0.5 = parabola, 1.0 = linear)
    # Typical values: 0.5-0.75 for hypersonic
    power_law_exponent: float = 0.5
    
    # Number of points to use for the blunted nose region
    n_nose_points: int = 25
    
    # Number of points for the blend region on each surface
    n_blend_points: int = 15
    
    # Whether to vary the bluntness along the span
    variable_bluntness: bool = False
    
    # If variable_bluntness is True, this is the ratio of tip radius to root radius
    tip_to_root_ratio: float = 0.5


def _compute_arc_length(points: np.ndarray) -> np.ndarray:
    """Compute cumulative arc length along a curve."""
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    arc_length = np.zeros(len(points))
    arc_length[1:] = np.cumsum(segment_lengths)
    return arc_length


def _cubic_hermite_blend(t: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                         m0: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """
    Cubic Hermite spline interpolation for G1 continuity.
    
    Parameters:
    -----------
    t : array of parameter values in [0, 1]
    p0, p1 : start and end points (3D)
    m0, m1 : tangent vectors at start and end
    
    Returns:
    --------
    points : array of interpolated 3D points
    """
    t = np.atleast_1d(t)
    t2 = t * t
    t3 = t2 * t
    
    # Hermite basis functions
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    
    points = np.zeros((len(t), 3))
    for i in range(3):
        points[:, i] = h00 * p0[i] + h10 * m0[i] + h01 * p1[i] + h11 * m1[i]
    
    return points


def _quintic_hermite_blend(t: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                           m0: np.ndarray, m1: np.ndarray,
                           c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """
    Quintic Hermite spline interpolation for G2 continuity.
    
    Parameters:
    -----------
    t : array of parameter values in [0, 1]
    p0, p1 : start and end points (3D)
    m0, m1 : first derivative (tangent) vectors at start and end
    c0, c1 : second derivative (curvature) vectors at start and end
    
    Returns:
    --------
    points : array of interpolated 3D points
    """
    t = np.atleast_1d(t)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    
    # Quintic Hermite basis functions
    h0 = 1 - 10*t3 + 15*t4 - 6*t5
    h1 = t - 6*t3 + 8*t4 - 3*t5
    h2 = 0.5*t2 - 1.5*t3 + 1.5*t4 - 0.5*t5
    h3 = 10*t3 - 15*t4 + 6*t5
    h4 = -4*t3 + 7*t4 - 3*t5
    h5 = 0.5*t3 - t4 + 0.5*t5
    
    points = np.zeros((len(t), 3))
    for i in range(3):
        points[:, i] = (h0 * p0[i] + h1 * m0[i] + h2 * c0[i] +
                        h3 * p1[i] + h4 * m1[i] + h5 * c1[i])
    
    return points


def _estimate_tangent(points: np.ndarray, idx: int, arc_length: np.ndarray) -> np.ndarray:
    """Estimate tangent vector at a point using finite differences."""
    n = len(points)
    if idx == 0:
        # Forward difference
        ds = arc_length[1] - arc_length[0]
        if ds < 1e-12:
            ds = 1e-12
        tangent = (points[1] - points[0]) / ds
    elif idx == n - 1:
        # Backward difference
        ds = arc_length[-1] - arc_length[-2]
        if ds < 1e-12:
            ds = 1e-12
        tangent = (points[-1] - points[-2]) / ds
    else:
        # Central difference
        ds = arc_length[idx+1] - arc_length[idx-1]
        if ds < 1e-12:
            ds = 1e-12
        tangent = (points[idx+1] - points[idx-1]) / ds
    
    return tangent


def _estimate_curvature_vector(points: np.ndarray, idx: int, 
                                arc_length: np.ndarray) -> np.ndarray:
    """Estimate second derivative (curvature direction) at a point."""
    n = len(points)
    if idx == 0 or idx == n - 1:
        # Use one-sided second derivative
        if idx == 0 and n >= 3:
            ds1 = arc_length[1] - arc_length[0]
            ds2 = arc_length[2] - arc_length[1]
            if ds1 < 1e-12 or ds2 < 1e-12:
                return np.zeros(3)
            curv = 2 * ((points[2] - points[1])/ds2 - (points[1] - points[0])/ds1) / (ds1 + ds2)
        elif idx == n - 1 and n >= 3:
            ds1 = arc_length[-2] - arc_length[-3]
            ds2 = arc_length[-1] - arc_length[-2]
            if ds1 < 1e-12 or ds2 < 1e-12:
                return np.zeros(3)
            curv = 2 * ((points[-1] - points[-2])/ds2 - (points[-2] - points[-3])/ds1) / (ds1 + ds2)
        else:
            return np.zeros(3)
    else:
        # Central second derivative
        ds1 = arc_length[idx] - arc_length[idx-1]
        ds2 = arc_length[idx+1] - arc_length[idx]
        if ds1 < 1e-12 or ds2 < 1e-12:
            return np.zeros(3)
        curv = 2 * ((points[idx+1] - points[idx])/ds2 - (points[idx] - points[idx-1])/ds1) / (ds1 + ds2)
    
    return curv


class BluntedWaverider:
    """
    A waverider with blunted leading edges.
    
    This class wraps an existing waverider object and provides modified
    surface streams with realistic leading edge geometry.
    
    The blunting algorithm:
    1. At each spanwise station, extract the local upper/lower surface profiles
    2. Compute a local 2D coordinate system (streamwise, normal)
    3. Generate the nose profile (ellipse or power-law) in 2D
    4. Use Hermite spline blending for smooth G1/G2 continuity
    5. Transform back to 3D and reconstruct the streams
    """
    
    def __init__(self, waverider, params: BluntingParameters):
        """
        Initialize a blunted waverider from a sharp waverider.
        
        Parameters:
        -----------
        waverider : waverider object
            The original sharp-leading-edge waverider
        params : BluntingParameters
            Parameters controlling the blunting
        """
        self.original = waverider
        self.params = params
        
        # Copy basic properties
        self.M_inf = waverider.M_inf
        self.beta = waverider.beta
        self.height = waverider.height
        self.width = waverider.width
        self.length = waverider.length
        
        # These will be populated by the blunting algorithm
        self.upper_surface_streams = None
        self.lower_surface_streams = None
        self.leading_edge = None
        self.nose_profiles = []  # Store the nose profile at each station
        
        # Perform the blunting
        self._apply_blunting()
    
    def _apply_blunting(self):
        """Apply the blunting algorithm to all leading edge stations."""
        
        n_stations = len(self.original.upper_surface_streams)
        
        # Deep copy the original streams
        self.upper_surface_streams = [stream.copy() for stream in self.original.upper_surface_streams]
        self.lower_surface_streams = [stream.copy() for stream in self.original.lower_surface_streams]
        self.leading_edge = self.original.leading_edge.copy()
        
        for i in range(n_stations):
            # Skip the very tip (singular point)
            if i == n_stations - 1:
                continue
                
            # Get local bluntness radius (may vary along span)
            local_radius = self._get_local_radius(i, n_stations)
            
            if local_radius <= 0:
                continue
            
            # Get the streams at this station
            upper_stream = self.upper_surface_streams[i]
            lower_stream = self.lower_surface_streams[i]
            
            # Compute the local chord length for blend calculation
            local_chord = self._compute_local_chord(upper_stream, lower_stream)
            blend_length = self.params.blend_length_fraction * local_chord
            
            # Get the leading edge point
            le_point = upper_stream[0]  # Should match lower_stream[0]
            
            # Compute local coordinate system at leading edge
            tangent_upper, tangent_lower = self._compute_local_tangents(
                upper_stream, lower_stream
            )
            
            # Generate the blunted nose profile
            nose_profile = self._generate_nose_profile(
                le_point, tangent_upper, tangent_lower, 
                local_radius, blend_length
            )
            
            self.nose_profiles.append({
                'station': i,
                'profile': nose_profile,
                'radius': local_radius,
                'blend_length': blend_length
            })
            
            # Modify the streams to incorporate the nose profile
            self._blend_nose_into_streams(i, nose_profile, blend_length)
        
        # Debug: verify that streams were actually modified
        print(f"\n{'='*60}")
        print(f"BLUNTING VERIFICATION:")
        print(f"  Nose radius: {self.params.nose_radius * 1000:.2f} mm")
        print(f"  Stations processed: {len(self.nose_profiles)}")
        
        if len(self.upper_surface_streams) > 0:
            orig_len = len(self.original.upper_surface_streams[0])
            new_len = len(self.upper_surface_streams[0])
            print(f"  Original stream 0 length: {orig_len}")
            print(f"  Blunted stream 0 length: {new_len}")
            
            # Compare first points (should be different if blunted)
            orig_le = self.original.upper_surface_streams[0][0]
            new_le = self.upper_surface_streams[0][0]
            diff = np.linalg.norm(new_le - orig_le)
            print(f"  Original LE point: {orig_le}")
            print(f"  Blunted LE point:  {new_le}")
            print(f"  LE position difference: {diff*1000:.4f} mm")
            
            if diff < 1e-6:
                print(f"  WARNING: Leading edge appears unchanged!")
            else:
                print(f"  ✓ Leading edge successfully modified")
        print(f"{'='*60}\n")
    
    def _get_local_radius(self, station_idx: int, n_stations: int) -> float:
        """Get the bluntness radius at a given spanwise station."""
        
        if not self.params.variable_bluntness:
            return self.params.nose_radius
        
        # Linear variation from root to tip
        t = station_idx / (n_stations - 1)
        root_radius = self.params.nose_radius
        tip_radius = root_radius * self.params.tip_to_root_ratio
        
        return root_radius + t * (tip_radius - root_radius)
    
    def _compute_local_chord(self, upper_stream: np.ndarray, 
                              lower_stream: np.ndarray) -> float:
        """Compute the local chord length at a station."""
        
        # Chord is approximately the x-extent of the streams
        x_le = upper_stream[0, 0]
        x_te = upper_stream[-1, 0]
        
        return x_te - x_le
    
    def _compute_local_tangents(self, upper_stream: np.ndarray,
                                 lower_stream: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the tangent vectors at the leading edge for upper and lower surfaces.
        
        Returns unit vectors pointing downstream (away from LE).
        """
        
        # Use first few points to estimate tangent direction
        if len(upper_stream) >= 3:
            # Quadratic fit for smoother tangent
            tangent_upper = upper_stream[2] - upper_stream[0]
        else:
            tangent_upper = upper_stream[1] - upper_stream[0]
        
        if len(lower_stream) >= 3:
            tangent_lower = lower_stream[2] - lower_stream[0]
        else:
            tangent_lower = lower_stream[1] - lower_stream[0]
        
        # Normalize
        tangent_upper = tangent_upper / np.linalg.norm(tangent_upper)
        tangent_lower = tangent_lower / np.linalg.norm(tangent_lower)
        
        return tangent_upper, tangent_lower
    
    def _generate_nose_profile(self, le_point: np.ndarray,
                                tangent_upper: np.ndarray,
                                tangent_lower: np.ndarray,
                                radius: float,
                                blend_length: float) -> dict:
        """
        Generate the blunted nose profile in 3D.
        
        Returns a dictionary containing:
        - 'upper': points for upper surface blend
        - 'lower': points for lower surface blend
        - 'nose': points for the nose cap itself
        """
        
        if self.params.profile_type == "ellipse":
            return self._generate_ellipse_profile(
                le_point, tangent_upper, tangent_lower, radius, blend_length
            )
        elif self.params.profile_type == "power_law":
            return self._generate_power_law_profile(
                le_point, tangent_upper, tangent_lower, radius, blend_length
            )
        else:
            raise ValueError(f"Unknown profile type: {self.params.profile_type}")
    
    def _generate_ellipse_profile(self, le_point: np.ndarray,
                                   tangent_upper: np.ndarray,
                                   tangent_lower: np.ndarray,
                                   radius: float,
                                   blend_length: float) -> dict:
        """
        Generate an elliptical nose profile that is tangent to both surfaces.
        
        The ellipse is positioned so that:
        1. It passes through (or near) the original leading edge
        2. Its tangent at the upper end matches the upper surface tangent
        3. Its tangent at the lower end matches the lower surface tangent
        """
        
        n_pts = self.params.n_nose_points
        aspect = self.params.ellipse_aspect_ratio
        
        # Semi-axes
        b = radius  # perpendicular to flow (the "bluntness")
        a = b * aspect  # along flow direction
        
        # Compute the bisector direction (average of upper and lower tangents)
        bisector = tangent_upper + tangent_lower
        bisector_norm = np.linalg.norm(bisector)
        if bisector_norm < 1e-10:
            bisector = tangent_upper  # Surfaces parallel, use upper tangent
        else:
            bisector = bisector / bisector_norm
        
        # Compute the opening angle between upper and lower surfaces
        dot = np.dot(tangent_upper, tangent_lower)
        opening_angle = np.arccos(np.clip(dot, -1, 1))
        half_angle = opening_angle / 2
        
        # Compute normal direction (perpendicular to bisector, in the wedge plane)
        # The normal points from lower surface towards upper surface
        # Use Gram-Schmidt to get component of tangent_upper perpendicular to bisector
        normal = tangent_upper - np.dot(tangent_upper, bisector) * bisector
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            # Tangents are parallel to bisector, construct normal from cross product
            # Use a reference direction (try z-axis, then y-axis)
            ref = np.array([0, 0, 1])
            normal = np.cross(bisector, ref)
            if np.linalg.norm(normal) < 1e-10:
                ref = np.array([0, 1, 0])
                normal = np.cross(bisector, ref)
            normal = normal / np.linalg.norm(normal)
        else:
            normal = normal / normal_norm
        
        # For an ellipse x²/a² + y²/b² = 1, the tangent angle at point (x,y) is:
        # dy/dx = -b²x / (a²y)
        # At angle θ (parametric), x = a*cos(θ), y = b*sin(θ)
        # Tangent direction: (-a*sin(θ), b*cos(θ))
        # 
        # We want the ellipse tangent to match the surface tangent at the blend points
        # The angle of the tangent to horizontal is: atan2(b*cos(θ), -a*sin(θ))
        #
        # For the upper surface (positive y), we need to find θ_upper such that
        # the tangent matches tangent_upper projected onto the 2D plane
        
        # Project surface tangents onto the local 2D plane (bisector, normal)
        t_upper_local = np.array([np.dot(tangent_upper, bisector), 
                                  np.dot(tangent_upper, normal)])
        t_lower_local = np.array([np.dot(tangent_lower, bisector),
                                  np.dot(tangent_lower, normal)])
        
        # Normalize 2D tangents
        t_upper_local = t_upper_local / np.linalg.norm(t_upper_local)
        t_lower_local = t_lower_local / np.linalg.norm(t_lower_local)
        
        # Find ellipse parameter where tangent matches
        # Ellipse tangent at θ: (-a*sin(θ), b*cos(θ)) normalized
        # We want this to equal (tx, ty) of the surface tangent
        
        def find_ellipse_angle(tx, ty, a, b):
            """Find θ where ellipse tangent matches (tx, ty)."""
            # Tangent direction: (-a*sin(θ), b*cos(θ))
            # Normalized: (-a*sin(θ), b*cos(θ)) / sqrt(a²sin²θ + b²cos²θ)
            # We want: -a*sin(θ)/norm = tx, b*cos(θ)/norm = ty
            # So: -a*sin(θ) / (b*cos(θ)) = tx/ty (if ty != 0)
            # tan(θ) = -b*tx / (a*ty)
            
            if abs(ty) > 1e-10:
                theta = np.arctan2(-b * tx, a * ty)
            else:
                # ty ≈ 0 means tangent is nearly vertical
                theta = np.pi/2 if tx < 0 else -np.pi/2
            return theta
        
        # Upper surface tangent points downstream (+x direction in local coords)
        # so tx > 0, ty can be positive or negative
        theta_upper = find_ellipse_angle(t_upper_local[0], t_upper_local[1], a, b)
        theta_lower = find_ellipse_angle(t_lower_local[0], t_lower_local[1], a, b)
        
        # Ensure theta_lower < theta_upper (lower is below upper)
        if theta_lower > theta_upper:
            theta_lower, theta_upper = theta_upper, theta_lower
        
        # Clamp to reasonable range (avoid going past ±90°)
        theta_upper = np.clip(theta_upper, -np.pi/2 + 0.1, np.pi/2 - 0.05)
        theta_lower = np.clip(theta_lower, -np.pi/2 + 0.05, np.pi/2 - 0.1)
        
        # If the angle range is too small, expand it
        if theta_upper - theta_lower < 0.2:
            mid = (theta_upper + theta_lower) / 2
            theta_upper = mid + 0.15
            theta_lower = mid - 0.15
        
        # Generate ellipse points from lower to upper
        theta = np.linspace(theta_lower, theta_upper, n_pts)
        
        # Ellipse in local 2D coordinates (origin at ellipse center)
        x_local = a * np.cos(theta)  # along bisector
        y_local = b * np.sin(theta)  # along normal
        
        # The ellipse center needs to be positioned such that:
        # 1. The nose (rightmost point, θ=0) is near the original LE
        # 2. The endpoints connect smoothly to the surfaces
        #
        # Position center so that the ellipse nose (x=a, y=0) is at the original LE
        # adjusted upstream slightly
        center = le_point - bisector * a
        
        # Transform to 3D global coordinates
        nose_points = np.zeros((n_pts, 3))
        for j in range(n_pts):
            nose_points[j] = center + x_local[j] * bisector + y_local[j] * normal
        
        # Compute tangents at endpoints for blend matching
        # Ellipse tangent at θ: (-a*sin(θ), b*cos(θ))
        tangent_at_upper = -a * np.sin(theta_upper) * bisector + b * np.cos(theta_upper) * normal
        tangent_at_upper = tangent_at_upper / np.linalg.norm(tangent_at_upper)
        
        tangent_at_lower = -a * np.sin(theta_lower) * bisector + b * np.cos(theta_lower) * normal
        tangent_at_lower = tangent_at_lower / np.linalg.norm(tangent_at_lower)
        
        return {
            'nose': nose_points,
            'upper_blend_start': nose_points[-1],
            'lower_blend_start': nose_points[0],
            'tangent_upper': tangent_at_upper,
            'tangent_lower': tangent_at_lower,
            'center': center,
            'bisector': bisector,
            'normal': normal,
            'semi_major': a,
            'semi_minor': b,
            'theta_range': (theta_lower, theta_upper)
        }
    
    def _generate_power_law_profile(self, le_point: np.ndarray,
                                     tangent_upper: np.ndarray,
                                     tangent_lower: np.ndarray,
                                     radius: float,
                                     blend_length: float) -> dict:
        """
        Generate a power-law nose profile: r = k * x^n
        
        For a power-law body of revolution, the radius of curvature at the nose
        tip relates to the profile parameters. Common values:
        - n = 0.5 (parabolic): Used for minimum drag at hypersonic speeds
        - n = 0.667: Good compromise between drag and heat transfer
        - n = 0.75: Blunter, better for thermal management
        
        The profile is r = k * x^n, where r is the radial distance from the axis
        and x is the axial distance from the tip.
        
        The radius of curvature at the tip (x=0) for n < 1 is:
        R_n = (1 / (2*k^2))^(1/(2n-1)) for the 2D case
        
        We size k to achieve the desired nose radius.
        """
        
        n_pts = self.params.n_nose_points
        n_exp = self.params.power_law_exponent
        
        # Compute bisector and normal (same approach as ellipse)
        bisector = tangent_upper + tangent_lower
        bisector_norm = np.linalg.norm(bisector)
        if bisector_norm < 1e-10:
            bisector = tangent_upper
        else:
            bisector = bisector / bisector_norm
        
        # Compute normal direction
        normal = tangent_upper - np.dot(tangent_upper, bisector) * bisector
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            ref = np.array([0, 0, 1])
            normal = np.cross(bisector, ref)
            if np.linalg.norm(normal) < 1e-10:
                ref = np.array([0, 1, 0])
                normal = np.cross(bisector, ref)
            normal = normal / np.linalg.norm(normal)
        else:
            normal = normal / normal_norm
        
        # For power law y = k * x^n:
        # - Curvature at origin κ = lim(x→0) of y'' / (1 + y'^2)^(3/2)
        # - For n < 1, y' = k*n*x^(n-1) → ∞ as x → 0
        # - The radius of curvature at the nose is R_n
        #
        # A practical approach: specify k such that at a small reference x,
        # the local radius of curvature matches the desired nose radius.
        #
        # For 2D power-law: R = (1 + y'^2)^(3/2) / |y''|
        # At small x: y ≈ k*x^n, y' ≈ k*n*x^(n-1), y'' ≈ k*n*(n-1)*x^(n-2)
        #
        # Simpler approach: set k so that y(x_ref) = radius at some x_ref
        # x_ref = small fraction of blend_length
        
        x_ref = blend_length * 0.1  # Reference point at 10% of blend
        
        # We want y(x_ref) ≈ radius (approximately)
        # Actually, for proper nose radius, we use the curvature relationship
        # For n = 0.5 (parabola y = k*sqrt(x)), R_n = 1/(2k^2)
        # So k = 1/sqrt(2*R_n) = 1/sqrt(2*radius)
        
        if n_exp == 0.5:
            k = 1.0 / np.sqrt(2.0 * radius)
        else:
            # General case: use empirical sizing
            # k sized so that the profile reaches y = radius at x = some_distance
            # We'll use: y(blend_length/4) = radius
            k = radius / ((blend_length / 4) ** n_exp)
        
        # Compute the opening half-angle of the wedge
        dot = np.dot(tangent_upper, tangent_lower)
        opening_angle = np.arccos(np.clip(dot, -1, 1))
        half_angle = opening_angle / 2
        
        # Generate profile from tip going downstream
        # Use a non-uniform spacing - denser near the tip for better resolution
        # t goes from 0 to 1
        t = np.linspace(0, 1, n_pts)
        # Use sqrt spacing for density near tip
        t_shaped = t ** 0.7
        x_local = t_shaped * blend_length
        
        # Power-law profile
        y_local = k * (x_local ** n_exp)
        
        # Avoid division by zero at tip
        y_local[0] = 0
        
        # Generate upper and lower branches
        nose_points_upper = np.zeros((n_pts, 3))
        nose_points_lower = np.zeros((n_pts, 3))
        
        for j in range(n_pts):
            nose_points_upper[j] = le_point + x_local[j] * bisector + y_local[j] * normal
            nose_points_lower[j] = le_point + x_local[j] * bisector - y_local[j] * normal
        
        # Combine: lower (reversed to go from outboard to tip) + upper (tip to outboard)
        # This creates a continuous curve from lower blend point, through nose, to upper blend point
        nose_points = np.vstack([nose_points_lower[::-1], nose_points_upper[1:]])
        
        # Compute tangents at endpoints for blending
        # dy/dx = k * n * x^(n-1)
        if n_pts > 1:
            x_end = x_local[-1]
            if x_end > 1e-10:
                dydx = k * n_exp * (x_end ** (n_exp - 1))
            else:
                dydx = 0
            
            # Upper tangent: (1, dydx) in local coords, normalized
            t_mag = np.sqrt(1 + dydx**2)
            tangent_at_upper = (bisector + dydx * normal) / t_mag
            tangent_at_lower = (bisector - dydx * normal) / t_mag
        else:
            tangent_at_upper = bisector
            tangent_at_lower = bisector
        
        return {
            'nose': nose_points,
            'upper_blend_start': nose_points_upper[-1],
            'lower_blend_start': nose_points_lower[-1],
            'tangent_upper': tangent_at_upper,
            'tangent_lower': tangent_at_lower,
            'center': le_point,
            'bisector': bisector,
            'normal': normal,
            'k': k,
            'n': n_exp
        }
    
    def _blend_nose_into_streams(self, station_idx: int, nose_profile: dict, 
                                  blend_length: float):
        """
        Modify the surface streams to incorporate the nose profile.
        
        Uses Hermite spline blending for G1 (tangent) continuity.
        The blend connects the nose profile endpoint to a point on the original
        surface, matching tangent directions at both ends.
        """
        
        upper_stream = self.upper_surface_streams[station_idx]
        lower_stream = self.lower_surface_streams[station_idx]
        
        # Check if streams are long enough for blending
        min_stream_length = 8  # Need at least this many points
        if len(upper_stream) < min_stream_length or len(lower_stream) < min_stream_length:
            # Stream too short for blending, skip this station
            print(f"Warning: Station {station_idx} has too few points for blending "
                  f"(upper: {len(upper_stream)}, lower: {len(lower_stream)})")
            return
        
        # Compute arc lengths for tangent estimation
        upper_arc = _compute_arc_length(upper_stream)
        lower_arc = _compute_arc_length(lower_stream)
        
        # Find where to end blending (based on x-distance from LE)
        le_x = upper_stream[0, 0]
        blend_end_x = le_x + blend_length * 1.8  # Extend blend region for smoother transition
        
        # Find indices where x > blend_end_x for upper surface
        # Ensure we leave enough points at both ends
        upper_blend_idx = np.searchsorted(upper_stream[:, 0], blend_end_x)
        min_idx = min(4, len(upper_stream) // 3)  # At least 4 or 1/3 of stream length
        max_idx = max(len(upper_stream) - 4, len(upper_stream) * 2 // 3)
        upper_blend_idx = max(min_idx, min(upper_blend_idx, max_idx))
        
        # Find indices for lower surface  
        lower_blend_idx = np.searchsorted(lower_stream[:, 0], blend_end_x)
        min_idx = min(4, len(lower_stream) // 3)
        max_idx = max(len(lower_stream) - 4, len(lower_stream) * 2 // 3)
        lower_blend_idx = max(min_idx, min(lower_blend_idx, max_idx))
        
        # Get the nose profile data
        nose_pts = nose_profile['nose']
        mid_nose = len(nose_pts) // 2
        
        # Split nose into upper and lower portions
        # Upper: from nose tip (middle) going towards upper surface
        upper_nose = nose_pts[mid_nose:]
        # Lower: from nose tip (middle) going towards lower surface (need to reverse)
        lower_nose = nose_pts[:mid_nose+1][::-1]
        
        # Compute arc lengths for nose portions
        upper_nose_arc = _compute_arc_length(upper_nose)
        lower_nose_arc = _compute_arc_length(lower_nose)
        
        # === UPPER SURFACE BLEND ===
        # Start point: end of upper nose portion
        p0_upper = upper_nose[-1]
        # End point: point on original upper surface
        p1_upper = upper_stream[upper_blend_idx]
        
        # Tangent at start: use the tangent from nose profile if available,
        # otherwise estimate from the nose curve
        if 'tangent_upper' in nose_profile:
            m0_upper = nose_profile['tangent_upper'].copy()
        elif len(upper_nose) >= 2:
            m0_upper = _estimate_tangent(upper_nose, len(upper_nose)-1, upper_nose_arc)
        else:
            m0_upper = nose_profile['bisector'].copy()
        
        # Ensure tangent points in the right direction (downstream, +x)
        if m0_upper[0] < 0:
            m0_upper = -m0_upper
        m0_upper = m0_upper / np.linalg.norm(m0_upper)
        
        # Tangent at end: estimate from original surface
        m1_upper = _estimate_tangent(upper_stream, upper_blend_idx, upper_arc)
        if m1_upper[0] < 0:
            m1_upper = -m1_upper
        m1_upper = m1_upper / np.linalg.norm(m1_upper)
        
        # Compute the chord length for scaling tangents
        chord_upper = np.linalg.norm(p1_upper - p0_upper)
        
        # Scale tangents - the scaling factor affects the "tightness" of the blend
        # Larger values make the curve follow the tangent longer
        scale_factor = 0.5  # Adjust this for tighter/looser blends
        m0_upper_scaled = m0_upper * chord_upper * scale_factor
        m1_upper_scaled = m1_upper * chord_upper * scale_factor
        
        # Generate blend curve using cubic Hermite
        n_blend = self.params.n_blend_points
        t_blend = np.linspace(0, 1, n_blend)
        upper_blend = _cubic_hermite_blend(t_blend, p0_upper, p1_upper, 
                                           m0_upper_scaled, m1_upper_scaled)
        
        # === LOWER SURFACE BLEND ===
        # Start point: end of lower nose portion
        p0_lower = lower_nose[-1]
        # End point: point on original lower surface
        p1_lower = lower_stream[lower_blend_idx]
        
        # Tangent at start
        if 'tangent_lower' in nose_profile:
            m0_lower = nose_profile['tangent_lower'].copy()
        elif len(lower_nose) >= 2:
            m0_lower = _estimate_tangent(lower_nose, len(lower_nose)-1, lower_nose_arc)
        else:
            m0_lower = nose_profile['bisector'].copy()
        
        if m0_lower[0] < 0:
            m0_lower = -m0_lower
        m0_lower = m0_lower / np.linalg.norm(m0_lower)
        
        # Tangent at end: estimate from original surface
        m1_lower = _estimate_tangent(lower_stream, lower_blend_idx, lower_arc)
        if m1_lower[0] < 0:
            m1_lower = -m1_lower
        m1_lower = m1_lower / np.linalg.norm(m1_lower)
        
        # Compute chord length and scale tangents
        chord_lower = np.linalg.norm(p1_lower - p0_lower)
        m0_lower_scaled = m0_lower * chord_lower * scale_factor
        m1_lower_scaled = m1_lower * chord_lower * scale_factor
        
        # Generate blend curve using cubic Hermite
        lower_blend = _cubic_hermite_blend(t_blend, p0_lower, p1_lower,
                                           m0_lower_scaled, m1_lower_scaled)
        
        # === ASSEMBLE NEW STREAMS ===
        # Upper: nose portion + blend + rest of original (skip overlapping points)
        new_upper = np.vstack([
            upper_nose[:-1],           # Nose portion (exclude last to avoid duplicate)
            upper_blend,                # Blend curve
            upper_stream[upper_blend_idx+1:]  # Rest of original
        ])
        
        # Lower: nose portion + blend + rest of original
        new_lower = np.vstack([
            lower_nose[:-1],           # Nose portion (exclude last)
            lower_blend,                # Blend curve
            lower_stream[lower_blend_idx+1:]  # Rest of original
        ])
        
        # Update the streams
        self.upper_surface_streams[station_idx] = new_upper
        self.lower_surface_streams[station_idx] = new_lower
        
        # Update leading edge point to be the nose tip (middle of nose profile)
        self.leading_edge[station_idx] = nose_pts[mid_nose]
        
        # Store blend info for debugging
        nose_profile['upper_blend'] = upper_blend
        nose_profile['lower_blend'] = lower_blend
        nose_profile['upper_blend_idx'] = upper_blend_idx
        nose_profile['lower_blend_idx'] = lower_blend_idx
        nose_profile['m0_upper'] = m0_upper
        nose_profile['m1_upper'] = m1_upper
        nose_profile['m0_lower'] = m0_lower
        nose_profile['m1_lower'] = m1_lower
    
    def get_resampled_streams(self, n_streamwise: int = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get resampled streams with uniform point distribution for CAD export.
        
        The blunting process changes the number of points and their distribution
        along each stream. This method resamples the streams to have a uniform
        distribution suitable for CAD export with interpPlate.
        
        Parameters:
        -----------
        n_streamwise : int, optional
            Number of points along each stream. If None, uses the original count.
            
        Returns:
        --------
        upper_streams : list of np.ndarray
            Resampled upper surface streams
        lower_streams : list of np.ndarray
            Resampled lower surface streams
        """
        from scipy.interpolate import interp1d
        
        # Determine target number of points
        if n_streamwise is None:
            # Use the average of original stream lengths
            orig_lengths = [len(s) for s in self.original.upper_surface_streams]
            n_streamwise = int(np.mean(orig_lengths))
        
        upper_resampled = []
        lower_resampled = []
        
        for i in range(len(self.upper_surface_streams)):
            # Resample upper stream
            upper_stream = self.upper_surface_streams[i]
            if len(upper_stream) > 2:
                # Compute arc length parameterization
                diffs = np.diff(upper_stream, axis=0)
                seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
                arc_length = np.zeros(len(upper_stream))
                arc_length[1:] = np.cumsum(seg_lengths)
                arc_length /= arc_length[-1]  # Normalize to [0, 1]
                
                # Resample uniformly
                t_new = np.linspace(0, 1, n_streamwise)
                new_stream = np.zeros((n_streamwise, 3))
                for dim in range(3):
                    interp_func = interp1d(arc_length, upper_stream[:, dim], 
                                          kind='linear', fill_value='extrapolate')
                    new_stream[:, dim] = interp_func(t_new)
                upper_resampled.append(new_stream)
            else:
                upper_resampled.append(upper_stream.copy())
            
            # Resample lower stream
            lower_stream = self.lower_surface_streams[i]
            if len(lower_stream) > 2:
                diffs = np.diff(lower_stream, axis=0)
                seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
                arc_length = np.zeros(len(lower_stream))
                arc_length[1:] = np.cumsum(seg_lengths)
                arc_length /= arc_length[-1]
                
                t_new = np.linspace(0, 1, n_streamwise)
                new_stream = np.zeros((n_streamwise, 3))
                for dim in range(3):
                    interp_func = interp1d(arc_length, lower_stream[:, dim],
                                          kind='linear', fill_value='extrapolate')
                    new_stream[:, dim] = interp_func(t_new)
                lower_resampled.append(new_stream)
            else:
                lower_resampled.append(lower_stream.copy())
        
        return upper_resampled, lower_resampled
    
    def get_cad_compatible_copy(self):
        """
        Create a copy of this blunted waverider with resampled streams
        suitable for CAD export.
        
        Returns:
        --------
        A modified copy where upper_surface_streams and lower_surface_streams
        have been resampled for better CAD compatibility.
        """
        from copy import deepcopy
        
        # Create a shallow copy
        cad_copy = deepcopy(self)
        
        # Replace streams with resampled versions
        upper_resampled, lower_resampled = self.get_resampled_streams()
        cad_copy.upper_surface_streams = upper_resampled
        cad_copy.lower_surface_streams = lower_resampled
        
        # Update leading edge from resampled streams
        cad_copy.leading_edge = np.vstack([s[0] for s in upper_resampled])
        
        return cad_copy

    def get_nose_radius_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the distribution of nose radius along the span.
        
        Returns:
        --------
        z_coords : np.ndarray
            Spanwise coordinates
        radii : np.ndarray
            Nose radius at each station
        """
        z_coords = []
        radii = []
        
        for profile_data in self.nose_profiles:
            station = profile_data['station']
            z = self.original.leading_edge[station, 2]
            r = profile_data['radius']
            z_coords.append(z)
            radii.append(r)
        
        return np.array(z_coords), np.array(radii)
    
    def compute_leading_edge_length(self) -> float:
        """Compute the total length of the (blunted) leading edge curve."""
        
        total_length = 0.0
        for i in range(len(self.leading_edge) - 1):
            segment = self.leading_edge[i+1] - self.leading_edge[i]
            total_length += np.linalg.norm(segment)
        
        return total_length


def blunt_waverider(waverider, 
                    profile_type: str = "ellipse",
                    nose_radius: float = 0.005,
                    blend_length_fraction: float = 0.05,
                    **kwargs) -> BluntedWaverider:
    """
    Convenience function to create a blunted waverider.
    
    Parameters:
    -----------
    waverider : waverider object
        The original sharp-edged waverider
    profile_type : str
        "ellipse" or "power_law"
    nose_radius : float
        Bluntness radius in meters
    blend_length_fraction : float
        How far back the blend extends (fraction of local chord)
    **kwargs : 
        Additional parameters passed to BluntingParameters
    
    Returns:
    --------
    BluntedWaverider
        A new waverider object with blunted leading edges
    """
    
    params = BluntingParameters(
        profile_type=profile_type,
        nose_radius=nose_radius,
        blend_length_fraction=blend_length_fraction,
        **kwargs
    )
    
    return BluntedWaverider(waverider, params)


def estimate_stagnation_heating(nose_radius: float, velocity: float,
                                 altitude_km: float = 25.0) -> float:
    """
    Estimate stagnation point heating rate using Sutton-Graves correlation.
    
    This gives a rough idea of thermal loads for different nose radii.
    
    Parameters:
    -----------
    nose_radius : float
        Nose radius in meters
    velocity : float  
        Velocity in m/s
    altitude_km : float
        Altitude in km (for density estimation)
    
    Returns:
    --------
    q_dot : float
        Stagnation heating rate in W/m²
    """
    
    # Simplified atmospheric model
    rho_0 = 1.225  # sea level density kg/m³
    H = 8500  # scale height in m
    rho = rho_0 * np.exp(-altitude_km * 1000 / H)
    
    # Sutton-Graves constant for air
    C = 1.83e-4  # (kg^0.5 / m)
    
    # Heating rate: q = C * sqrt(rho/r) * V^3
    q_dot = C * np.sqrt(rho / nose_radius) * velocity**3
    
    return q_dot


def recommend_nose_radius(mach: float, altitude_km: float = 25.0,
                          max_heating_rate: float = 1e6) -> float:
    """
    Recommend a minimum nose radius based on thermal constraints.
    
    Parameters:
    -----------
    mach : float
        Flight Mach number
    altitude_km : float
        Altitude in km
    max_heating_rate : float
        Maximum allowable heating rate in W/m²
    
    Returns:
    --------
    min_radius : float
        Recommended minimum nose radius in meters
    """
    
    # Approximate speed of sound at altitude
    T = 288.15 - 6.5 * altitude_km  # Simple temperature model
    a = np.sqrt(1.4 * 287 * T)
    velocity = mach * a
    
    # Simplified atmospheric density
    rho_0 = 1.225
    H = 8500
    rho = rho_0 * np.exp(-altitude_km * 1000 / H)
    
    # Sutton-Graves
    C = 1.83e-4
    
    # Solve for radius: r = C² * rho * V^6 / q²
    min_radius = (C**2 * rho * velocity**6) / (max_heating_rate**2)
    
    return min_radius


def estimate_stagnation_heating_from_mach(nose_radius: float, mach: float,
                                           altitude_km: float = 25.0) -> float:
    """
    Estimate stagnation point heating rate from Mach number.
    
    Parameters:
    -----------
    nose_radius : float
        Nose radius in meters
    mach : float
        Flight Mach number
    altitude_km : float
        Altitude in km
    
    Returns:
    --------
    q_dot : float
        Stagnation heating rate in W/m²
    """
    # Approximate temperature at altitude (simple ISA model)
    if altitude_km <= 11:
        T = 288.15 - 6.5 * altitude_km
    else:
        T = 216.65  # Isothermal above tropopause (simplified)
    
    # Speed of sound
    gamma = 1.4
    R = 287.0
    a = np.sqrt(gamma * R * T)
    
    # Velocity
    velocity = mach * a
    
    return estimate_stagnation_heating(nose_radius, velocity, altitude_km)


def compute_heating_vs_radius(mach: float, altitude_km: float = 25.0,
                               radii: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute heating rate vs nose radius for plotting.
    
    Parameters:
    -----------
    mach : float
        Flight Mach number
    altitude_km : float
        Altitude in km
    radii : np.ndarray, optional
        Array of radii to evaluate (in meters)
        If None, uses default range from 1mm to 50mm
    
    Returns:
    --------
    radii : np.ndarray
        Nose radii in meters
    heating : np.ndarray
        Heating rates in MW/m²
    """
    if radii is None:
        radii = np.linspace(0.001, 0.05, 50)
    
    heating = np.array([estimate_stagnation_heating_from_mach(r, mach, altitude_km) 
                        for r in radii])
    
    return radii, heating / 1e6  # Convert to MW/m²


def get_atmosphere_properties(altitude_km: float) -> dict:
    """
    Get atmospheric properties at a given altitude (simplified ISA model).
    
    Parameters:
    -----------
    altitude_km : float
        Altitude in km
    
    Returns:
    --------
    dict with keys:
        'temperature': Temperature in K
        'pressure': Pressure in Pa
        'density': Density in kg/m³
        'speed_of_sound': Speed of sound in m/s
    """
    # Constants
    g = 9.81
    R = 287.0
    gamma = 1.4
    
    # Sea level values
    T0 = 288.15
    P0 = 101325.0
    
    # Simplified model
    if altitude_km <= 11:
        # Troposphere (linear temperature decrease)
        lapse_rate = 6.5 / 1000  # K/m
        T = T0 - lapse_rate * altitude_km * 1000
        P = P0 * (T / T0) ** (g / (R * lapse_rate))
    else:
        # Lower stratosphere (isothermal)
        T = 216.65
        T11 = 216.65
        P11 = P0 * (T11 / T0) ** (g / (R * 6.5e-3))
        P = P11 * np.exp(-g * (altitude_km - 11) * 1000 / (R * T))
    
    rho = P / (R * T)
    a = np.sqrt(gamma * R * T)
    
    return {
        'temperature': T,
        'pressure': P,
        'density': rho,
        'speed_of_sound': a
    }


# Example usage and testing
if __name__ == "__main__":
    
    print("Leading Edge Blunting Module")
    print("=" * 50)
    
    # Test thermal calculations
    print("\nThermal Analysis Examples:")
    print("-" * 30)
    
    for mach in [5.0, 7.0, 10.0]:
        rec_radius = recommend_nose_radius(mach, altitude_km=25.0)
        print(f"Mach {mach}: Recommended min radius = {rec_radius*1000:.2f} mm")
    
    print("\nHeating rates for 5mm nose radius at Mach 5, 25km:")
    q = estimate_stagnation_heating(0.005, 5*340, 25)
    print(f"  q_dot = {q/1e6:.2f} MW/m²")
    
    print("\nTo test with actual waverider, use:")
    print("  from waverider_generator.generator import waverider")
    print("  from waverider_generator.leading_edge_blunting import blunt_waverider")
    print("  ")
    print("  wr = waverider(M_inf=5, beta=15, height=1.34, width=3,")
    print("                 dp=[0.2, 0.3, 0.5, 0.5], n_upper_surface=100,")
    print("                 n_shockwave=100, n_planes=30, n_streamwise=50)")
    print("  ")
    print("  blunted = blunt_waverider(wr, profile_type='ellipse',")
    print("                            nose_radius=0.01, blend_length_fraction=0.05)")
