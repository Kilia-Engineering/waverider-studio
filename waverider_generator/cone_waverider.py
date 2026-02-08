"""
Cone-Derived Waverider Generator
================================
Based on Adam Weaver's SHADOW (Stability of Hypersonic Aerodynamic Derivatives of Waveriders)
methodology from his Master's thesis at Utah State University.

This implementation creates waveriders by:
1. Defining a polynomial leading edge curve
2. Projecting it onto a conical shock surface
3. Tracing streamlines through Taylor-Maccoll flow field
4. Creating a freestream upper surface

Author: Adapted for integration with existing waverider_generator package
"""

import numpy as np
import math
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline, interp1d
from typing import Tuple, List, Optional, Union
import warnings


class ConeWaverider:
    """
    Cone-derived waverider using polynomial leading edge parameterization.
    
    This class generates waverider geometries by:
    1. Defining a leading edge via polynomial coefficients
    2. Projecting the leading edge onto a conical shock
    3. Tracing streamlines through the Taylor-Maccoll flow field
    4. Creating a freestream (flat) upper surface
    
    Parameters
    ----------
    mach : float
        Freestream Mach number
    shock_angle : float
        Shock cone angle in degrees
    poly_coeffs : list
        Polynomial coefficients [A_n, A_{n-1}, ..., A_1, A_0] where
        y = A_n*x^n + A_{n-1}*x^{n-1} + ... + A_1*x + A_0
        Note: A_1 (linear term) should be 0 for smooth y-axis reflection
    n_leading_edge : int
        Number of points along the leading edge
    n_streamwise : int
        Number of points along each streamline
    gamma : float, optional
        Ratio of specific heats (default 1.4)
    length : float, optional
        Non-dimensional length of waverider (default 1.0)
    """
    
    def __init__(
        self,
        mach: float,
        shock_angle: float,
        poly_coeffs: List[float],
        n_leading_edge: int = 21,
        n_streamwise: int = 20,
        gamma: float = 1.4,
        length: float = 1.0
    ):
        # Validate inputs
        if mach <= 1.0:
            raise ValueError("Mach number must be greater than 1.0")
        
        # Check shock angle is valid (must be greater than Mach angle)
        mach_angle = np.degrees(np.arcsin(1.0 / mach))
        if shock_angle <= mach_angle:
            raise ValueError(f"Shock angle ({shock_angle}°) must be greater than Mach angle ({mach_angle:.2f}°)")
        
        self.mach = float(mach)
        self.shock_angle = float(shock_angle)
        self.shock_angle_rad = np.radians(shock_angle)
        self.gamma = float(gamma)
        self.length = float(length)
        
        # Process polynomial coefficients
        self.poly_coeffs = list(poly_coeffs)
        self.poly_order = len(self.poly_coeffs) - 1
        
        # Mesh resolution
        self.n_leading_edge = n_leading_edge
        self.n_streamwise = n_streamwise
        
        # Ensure odd number of leading edge points for symmetry
        if self.n_leading_edge % 2 == 0:
            self.n_leading_edge += 1
        
        # Initialize storage
        self.leading_edge = None
        self.lower_surface = None
        self.upper_surface = None
        self.base_surface = None
        
        # Flow field data
        self.vel_field = None
        self.cone_angle = None
        self.deflection_angle = None
        self.post_shock_mach = None
        
        # Geometry metrics
        self.planform_area = None
        self.volume = None
        self.mac = None  # Mean aerodynamic chord
        self.cg = None  # Center of gravity estimate
        
        # Build the waverider
        self._compute_shock_relations()
        self._solve_taylor_maccoll()
        self._generate_leading_edge()
        self._trace_lower_surface()
        self._generate_upper_surface()
        self._compute_geometry_metrics()
    
    def _compute_shock_relations(self):
        """Compute oblique shock relations."""
        M = self.mach
        beta = self.shock_angle_rad
        gamma = self.gamma
        
        # Normal Mach number upstream
        Mn1 = M * np.sin(beta)
        
        # Check for detached shock
        if Mn1 < 1.0:
            raise ValueError("Invalid shock configuration: normal Mach number < 1")
        
        # Flow deflection angle (theta-beta-M relation)
        numerator = 2 * (1/np.tan(beta)) * (M**2 * np.sin(beta)**2 - 1)
        denominator = M**2 * (gamma + np.cos(2*beta)) + 2
        self.deflection_angle = np.arctan(numerator / denominator)
        
        # Normal Mach number downstream
        Mn2_sq = (1 + (gamma-1)/2 * Mn1**2) / (gamma * Mn1**2 - (gamma-1)/2)
        Mn2 = np.sqrt(Mn2_sq)
        
        # Post-shock Mach number
        self.post_shock_mach = Mn2 / np.sin(beta - self.deflection_angle)
        
        # Non-dimensional velocity
        V = 1.0 / np.sqrt(2.0 / ((gamma - 1) * self.post_shock_mach**2) + 1.0)
        
        # Velocity components at shock
        self.Vr_shock = V * np.cos(beta - self.deflection_angle)
        self.Vtheta_shock = -V * np.sin(beta - self.deflection_angle)
    
    def _taylor_maccoll_ode(self, theta, y):
        """
        Taylor-Maccoll ODE system.
        
        Parameters
        ----------
        theta : float
            Polar angle (from cone axis)
        y : array
            [Vr, Vtheta] velocity components
            
        Returns
        -------
        dydt : array
            Derivatives [dVr/dtheta, dVtheta/dtheta]
        """
        Vr, Vtheta = y
        gamma = self.gamma
        
        # Intermediate constant
        A = (gamma - 1) / 2 * (1 - Vr**2 - Vtheta**2)
        
        dVr_dtheta = Vtheta
        
        # Handle singularity at theta = 0
        if abs(theta) < 1e-10:
            dVtheta_dtheta = 0
        else:
            numerator = Vtheta**2 * Vr - A * (2*Vr + Vtheta / np.tan(theta))
            denominator = A - Vtheta**2
            
            if abs(denominator) < 1e-12:
                dVtheta_dtheta = 0
            else:
                dVtheta_dtheta = numerator / denominator
        
        return [dVr_dtheta, dVtheta_dtheta]
    
    def _solve_taylor_maccoll(self):
        """Solve Taylor-Maccoll equations from shock to cone surface."""
        # Initial conditions at shock
        y0 = [self.Vr_shock, self.Vtheta_shock]
        
        # Event function: stop when Vtheta crosses zero
        def vtheta_zero(theta, y):
            return y[1]
        vtheta_zero.terminal = True
        vtheta_zero.direction = 1  # Crossing from negative to positive
        
        # Integrate from shock angle inward toward cone axis
        sol = solve_ivp(
            self._taylor_maccoll_ode,
            (self.shock_angle_rad, 0.01),  # Don't go all the way to 0 to avoid singularity
            y0,
            events=vtheta_zero,
            dense_output=True,
            max_step=0.001
        )
        
        if sol.t_events[0].size > 0:
            self.cone_angle = sol.t_events[0][0]
        else:
            # If event not triggered, use last point where Vtheta is small
            idx = np.argmin(np.abs(sol.y[1]))
            self.cone_angle = sol.t[idx]
        
        # Store velocity field
        # Create arrays from shock to cone
        theta_vals = np.linspace(self.shock_angle_rad, self.cone_angle, 500)
        
        # Interpolate velocity field
        Vr_interp = interp1d(sol.t, sol.y[0], kind='linear', fill_value='extrapolate')
        Vtheta_interp = interp1d(sol.t, sol.y[1], kind='linear', fill_value='extrapolate')
        
        self.vel_field = {
            'theta': theta_vals,
            'Vr': Vr_interp(theta_vals),
            'Vtheta': Vtheta_interp(theta_vals),
            'Vr_interp': Vr_interp,
            'Vtheta_interp': Vtheta_interp
        }
        
        self.cone_angle_deg = np.degrees(self.cone_angle)
    
    def _get_velocity_at_theta(self, theta: float) -> Tuple[float, float]:
        """
        Get interpolated velocity components at given theta.
        
        Parameters
        ----------
        theta : float
            Polar angle in radians
            
        Returns
        -------
        Vr, Vtheta : tuple
            Velocity components
        """
        # Clamp theta to valid range
        theta = np.clip(theta, self.cone_angle, self.shock_angle_rad)
        
        Vr = float(self.vel_field['Vr_interp'](theta))
        Vtheta = float(self.vel_field['Vtheta_interp'](theta))
        
        return Vr, Vtheta
    
    def _polynomial_leading_edge(self, x: float) -> float:
        """
        Evaluate the leading edge polynomial at x.
        
        Parameters
        ----------
        x : float
            x-coordinate (spanwise)
            
        Returns
        -------
        y : float
            y-coordinate (vertical in local frame)
        """
        y = 0
        for i, coeff in enumerate(self.poly_coeffs):
            power = self.poly_order - i
            y += coeff * x**power
        return y
    
    def _project_to_shock_cone(self, x: float, y: float) -> float:
        """
        Project (x, y) point onto the shock cone to get z coordinate.
        
        Given x, y in the local leading edge plane, find z such that
        the point lies on the shock cone.
        
        Parameters
        ----------
        x, y : float
            Local coordinates
            
        Returns
        -------
        z : float
            Streamwise coordinate on shock cone
        """
        cos_beta = np.cos(self.shock_angle_rad)
        sin_beta = np.sin(self.shock_angle_rad)
        
        # From spherical coordinate relation:
        # theta = shock_angle means point is on shock cone
        # cos(theta) = z / sqrt(x^2 + y^2 + z^2)
        # Solving for z:
        numerator = cos_beta**2 * (x**2 + y**2)
        denominator = 1 - cos_beta**2  # = sin^2(beta)
        
        if denominator < 1e-10:
            return 0.0
        
        z = np.sqrt(numerator / denominator)
        return z
    
    def _generate_leading_edge(self):
        """Generate the 3D leading edge by projecting polynomial onto shock cone."""
        # Get y-intercept (point at x=0)
        y_intercept = self.poly_coeffs[-1]  # A_0 term
        
        # Find z coordinate at x=0
        z_start = self._project_to_shock_cone(0, y_intercept)
        z_end = z_start + self.length
        
        # Find x_max by solving where the leading edge reaches z_end
        # This requires finding where the projected curve reaches the base plane
        def find_x_end(x):
            y = self._polynomial_leading_edge(x)
            z = self._project_to_shock_cone(x, y)
            return z - z_end
        
        # Binary search for x_end
        x_low, x_high = 0.0, 10.0
        for _ in range(100):
            x_mid = (x_low + x_high) / 2
            if find_x_end(x_mid) < 0:
                x_low = x_mid
            else:
                x_high = x_mid
        x_end = x_mid
        
        # Generate leading edge points (positive x side)
        n_half = (self.n_leading_edge - 1) // 2
        x_positive = np.linspace(0, x_end, n_half + 1)
        
        # Build full leading edge (symmetric about x=0)
        x_negative = -x_positive[1:][::-1]
        x_full = np.concatenate([x_negative, x_positive])
        
        # Compute y and z for each point
        leading_edge = []
        for x in x_full:
            y = self._polynomial_leading_edge(abs(x)) if x != 0 else self.poly_coeffs[-1]
            z = self._project_to_shock_cone(x, y)
            leading_edge.append([x, y, z])
        
        self.leading_edge = np.array(leading_edge)
        self.z_start = z_start
        self.z_end = z_end
        self.x_end = x_end
    
    def _cart_to_sphere(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert Cartesian to spherical coordinates."""
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-12:
            return 0, 0, 0
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi
    
    def _sphere_to_cart(self, r: float, theta: float, phi: float) -> Tuple[float, float, float]:
        """Convert spherical to Cartesian coordinates."""
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z
    
    def _streamline_ode(self, t, y):
        """
        ODE for streamline tracing in spherical coordinates.
        
        Parameters
        ----------
        t : float
            Integration parameter
        y : array
            [r, theta] position
            
        Returns
        -------
        dydt : array
            [dr/dt, dtheta/dt]
        """
        r, theta = y
        
        # Get velocity at this theta
        Vr, Vtheta = self._get_velocity_at_theta(theta)
        
        dr_dt = Vr
        if r > 1e-10:
            dtheta_dt = Vtheta / r
        else:
            dtheta_dt = 0
        
        return [dr_dt, dtheta_dt]
    
    def _trace_streamline(self, start_point: np.ndarray, target_z: float) -> np.ndarray:
        """
        Trace a streamline from start_point until z reaches target_z.
        
        Parameters
        ----------
        start_point : array
            Starting [x, y, z] position
        target_z : float
            Target z coordinate
            
        Returns
        -------
        streamline : array
            Array of [x, y, z] points along streamline
        """
        x0, y0, z0 = start_point
        r0, theta0, phi0 = self._cart_to_sphere(x0, y0, z0)
        
        # Estimate integration time
        dz = target_z - z0
        estimated_steps = max(100, int(dz * 50))
        
        # Create event to stop at target z
        def reach_target_z(t, y):
            r, theta = y
            _, _, z = self._sphere_to_cart(r, theta, phi0)
            return z - target_z
        reach_target_z.terminal = True
        reach_target_z.direction = 1
        
        # Integrate streamline
        t_span = (0, 100)  # Large enough span
        y0 = [r0, theta0]
        
        sol = solve_ivp(
            self._streamline_ode,
            t_span,
            y0,
            events=reach_target_z,
            dense_output=True,
            max_step=0.01
        )
        
        # Extract points along streamline
        if sol.t_events[0].size > 0:
            t_final = sol.t_events[0][0]
        else:
            t_final = sol.t[-1]
        
        t_eval = np.linspace(0, t_final, self.n_streamwise)
        
        streamline = []
        for t in t_eval:
            r, theta = sol.sol(t)
            x, y, z = self._sphere_to_cart(r, theta, phi0)
            streamline.append([x, y, z])
        
        return np.array(streamline)
    
    def _trace_lower_surface(self):
        """Trace streamlines from each leading edge point to create lower surface."""
        lower_surface = []
        
        for le_point in self.leading_edge:
            streamline = self._trace_streamline(le_point, self.z_end)
            lower_surface.append(streamline)
        
        self.lower_surface = np.array(lower_surface)
    
    def _generate_upper_surface(self):
        """Generate freestream (flat) upper surface."""
        upper_surface = []
        
        for i, le_point in enumerate(self.leading_edge):
            x_start, y_start, z_start = le_point
            
            # Upper surface is a straight line in z direction
            z_vals = np.linspace(z_start, self.z_end, self.n_streamwise)
            
            streamline = []
            for z in z_vals:
                streamline.append([x_start, y_start, z])
            
            upper_surface.append(streamline)
        
        self.upper_surface = np.array(upper_surface)
    
    def _compute_geometry_metrics(self):
        """Compute geometric properties of the waverider."""
        # Planform area (projection onto x-z plane)
        # Use trapezoidal integration
        x_le = self.leading_edge[:, 0]
        z_le = self.leading_edge[:, 2]
        
        # Base width
        base_widths = []
        for i in range(len(self.lower_surface)):
            base_point = self.lower_surface[i, -1, :]
            base_widths.append(abs(base_point[0]))
        
        # Approximate planform area
        dx = np.diff(x_le)
        avg_lengths = np.array([self.length] * (len(x_le) - 1))  # Simplified
        self.planform_area = np.sum(np.abs(dx) * avg_lengths) / 2
        
        # Volume (simplified estimate using cross-sections)
        self.volume = self._estimate_volume()
        
        # Mean aerodynamic chord
        self.mac = self._compute_mac()
        
        # Center of gravity (at 0.75 MAC from nose, typical for hypersonic vehicles)
        self.cg = self._compute_cg()
    
    def _estimate_volume(self) -> float:
        """Estimate waverider volume using cross-sectional integration."""
        n_sections = 20
        z_vals = np.linspace(self.z_start, self.z_end, n_sections)
        
        areas = []
        for z in z_vals:
            # Find cross-section at this z
            area = self._cross_section_area_at_z(z)
            areas.append(area)
        
        # Integrate using trapezoidal rule
        volume = np.trapz(areas, z_vals)
        return volume
    
    def _cross_section_area_at_z(self, z: float) -> float:
        """Compute cross-sectional area at given z location."""
        # Interpolate upper and lower surface points at this z
        # This is a simplified calculation
        
        # Find fraction along length
        frac = (z - self.z_start) / (self.z_end - self.z_start)
        frac = np.clip(frac, 0, 1)
        
        idx = int(frac * (self.n_streamwise - 1))
        idx = min(idx, self.n_streamwise - 1)
        
        # Get upper and lower points at this streamwise location
        upper_points = self.upper_surface[:, idx, :]
        lower_points = self.lower_surface[:, idx, :]
        
        # Compute area using shoelace formula (simplified)
        # Just use trapezoidal approximation between upper and lower
        area = 0
        for i in range(len(upper_points) - 1):
            dx = abs(upper_points[i+1, 0] - upper_points[i, 0])
            dy_upper = abs(upper_points[i+1, 1] - upper_points[i, 1])
            dy_lower = abs(lower_points[i+1, 1] - lower_points[i, 1])
            
            height_avg = abs(lower_points[i, 1] - upper_points[i, 1])
            area += dx * height_avg
        
        return area
    
    def _compute_mac(self) -> float:
        """Compute mean aerodynamic chord."""
        # Simplified: average chord length
        chord_lengths = []
        
        for i in range(len(self.upper_surface)):
            le_z = self.upper_surface[i, 0, 2]
            te_z = self.upper_surface[i, -1, 2]
            chord = te_z - le_z
            chord_lengths.append(chord)
        
        # Weight by local chord
        weights = np.array(chord_lengths)**2
        mac = np.sum(weights) / np.sum(chord_lengths)
        
        return mac
    
    def _compute_cg(self) -> np.ndarray:
        """Compute estimated center of gravity location."""
        # CG at 25% MAC from leading edge, on centerline
        cg_z = self.z_start + 0.25 * self.mac
        
        # Find y at centerline (x=0)
        center_idx = len(self.upper_surface) // 2
        
        # Average y between upper and lower at CG z location
        frac = (cg_z - self.z_start) / (self.z_end - self.z_start)
        frac = np.clip(frac, 0, 1)
        idx = int(frac * (self.n_streamwise - 1))
        
        y_upper = self.upper_surface[center_idx, idx, 1]
        y_lower = self.lower_surface[center_idx, idx, 1]
        cg_y = (y_upper + y_lower) / 2
        
        return np.array([0.0, cg_y, cg_z])
    
    def get_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate triangular mesh for the waverider surface.
        
        Returns
        -------
        vertices : array
            Nx3 array of vertex coordinates
        triangles : array
            Mx3 array of vertex indices forming triangles
        """
        vertices = []
        triangles = []
        
        n_span = len(self.upper_surface)
        n_stream = self.n_streamwise
        
        # Add upper surface vertices
        for i in range(n_span):
            for j in range(n_stream):
                vertices.append(self.upper_surface[i, j, :])
        
        # Add lower surface vertices (excluding leading edge which is shared)
        lower_start_idx = len(vertices)
        for i in range(n_span):
            for j in range(1, n_stream):  # Skip j=0 (leading edge)
                vertices.append(self.lower_surface[i, j, :])
        
        vertices = np.array(vertices)
        
        # Generate triangles for upper surface
        for i in range(n_span - 1):
            for j in range(n_stream - 1):
                # Upper surface indices
                v00 = i * n_stream + j
                v01 = i * n_stream + j + 1
                v10 = (i + 1) * n_stream + j
                v11 = (i + 1) * n_stream + j + 1
                
                # Two triangles per quad (counter-clockwise for outward normal)
                triangles.append([v00, v10, v01])
                triangles.append([v01, v10, v11])
        
        # Generate triangles for lower surface
        lower_n_stream = n_stream - 1  # One less because we skip leading edge
        for i in range(n_span - 1):
            for j in range(lower_n_stream - 1):
                # Lower surface indices
                v00 = lower_start_idx + i * lower_n_stream + j
                v01 = lower_start_idx + i * lower_n_stream + j + 1
                v10 = lower_start_idx + (i + 1) * lower_n_stream + j
                v11 = lower_start_idx + (i + 1) * lower_n_stream + j + 1
                
                # Two triangles per quad (clockwise for inward normal on lower surface)
                triangles.append([v00, v01, v10])
                triangles.append([v01, v11, v10])
        
        # Connect leading edge (upper and lower surfaces share leading edge vertices)
        for i in range(n_span - 1):
            upper_le_0 = i * n_stream
            upper_le_1 = (i + 1) * n_stream
            lower_first_0 = lower_start_idx + i * lower_n_stream
            lower_first_1 = lower_start_idx + (i + 1) * lower_n_stream
            
            triangles.append([upper_le_0, lower_first_0, upper_le_1])
            triangles.append([upper_le_1, lower_first_0, lower_first_1])
        
        # Generate base surface triangles
        base_triangles = self._generate_base_triangles(n_span, n_stream, lower_start_idx, lower_n_stream)
        triangles.extend(base_triangles)
        
        triangles = np.array(triangles)
        
        return vertices, triangles
    
    def _generate_base_triangles(self, n_span, n_stream, lower_start_idx, lower_n_stream):
        """Generate triangles for the base (trailing edge) surface."""
        base_triangles = []
        
        for i in range(n_span - 1):
            # Upper surface trailing edge
            upper_te_0 = i * n_stream + (n_stream - 1)
            upper_te_1 = (i + 1) * n_stream + (n_stream - 1)
            
            # Lower surface trailing edge
            lower_te_0 = lower_start_idx + i * lower_n_stream + (lower_n_stream - 1)
            lower_te_1 = lower_start_idx + (i + 1) * lower_n_stream + (lower_n_stream - 1)
            
            # Two triangles to close the base
            base_triangles.append([upper_te_0, upper_te_1, lower_te_0])
            base_triangles.append([upper_te_1, lower_te_1, lower_te_0])
        
        return base_triangles
    
    def export_tri(self, filename: str):
        """
        Export mesh in NASA Cart3D TRI format.
        
        Parameters
        ----------
        filename : str
            Output filename (should end in .tri)
        """
        vertices, triangles = self.get_mesh()
        
        n_vertices = len(vertices)
        n_triangles = len(triangles)
        
        with open(filename, 'w') as f:
            f.write(f"{n_vertices}\n")
            f.write(f"{n_triangles}\n")
            
            for v in vertices:
                f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
            
            for t in triangles:
                # TRI format uses 1-based indexing
                f.write(f"{t[0]+1} {t[1]+1} {t[2]+1}\n")
    
    def export_stl(self, filename: str):
        """
        Export mesh in STL format.
        
        Parameters
        ----------
        filename : str
            Output filename (should end in .stl)
        """
        vertices, triangles = self.get_mesh()
        
        with open(filename, 'w') as f:
            f.write("solid waverider\n")
            
            for tri in triangles:
                v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
                
                # Compute normal
                e1 = v1 - v0
                e2 = v2 - v0
                normal = np.cross(e1, e2)
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normal = normal / norm
                else:
                    normal = np.array([0, 0, 1])
                
                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            
            f.write("endsolid waverider\n")
    
    def get_reference_values(self, scale: float = 1.0) -> dict:
        """
        Get reference values for aerodynamic analysis.
        
        Parameters
        ----------
        scale : float
            Scale factor to apply (e.g., 60 for 60m waverider)
            
        Returns
        -------
        refs : dict
            Dictionary containing reference length, area, and CG location
        """
        return {
            'length': self.length * scale,
            'area': self.planform_area * scale**2,
            'cg': self.cg * scale,
            'mac': self.mac * scale,
            'cone_angle': self.cone_angle_deg,
            'shock_angle': self.shock_angle,
            'mach': self.mach
        }
    
    def summary(self):
        """Print summary of waverider geometry."""
        print("=" * 50)
        print("Cone-Derived Waverider Summary")
        print("=" * 50)
        print(f"Mach Number:        {self.mach:.2f}")
        print(f"Shock Angle:        {self.shock_angle:.2f}°")
        print(f"Cone Angle:         {self.cone_angle_deg:.2f}°")
        print(f"Deflection Angle:   {np.degrees(self.deflection_angle):.2f}°")
        print(f"Post-shock Mach:    {self.post_shock_mach:.2f}")
        print("-" * 50)
        print(f"Length:             {self.length:.4f}")
        print(f"Planform Area:      {self.planform_area:.4f}")
        print(f"Volume:             {self.volume:.4f}")
        print(f"MAC:                {self.mac:.4f}")
        print(f"CG Location:        [{self.cg[0]:.4f}, {self.cg[1]:.4f}, {self.cg[2]:.4f}]")
        print("-" * 50)
        print(f"Polynomial Order:   {self.poly_order}")
        print(f"Polynomial Coeffs:  {self.poly_coeffs}")
        print(f"Leading Edge Pts:   {self.n_leading_edge}")
        print(f"Streamwise Pts:     {self.n_streamwise}")
        print("=" * 50)


def create_second_order_waverider(
    mach: float,
    shock_angle: float,
    A2: float,
    A0: float,
    **kwargs
) -> ConeWaverider:
    """
    Create a waverider with second-order polynomial leading edge.
    
    y = A2 * x^2 + A0
    
    Parameters
    ----------
    mach : float
        Freestream Mach number
    shock_angle : float
        Shock cone angle in degrees
    A2 : float
        Quadratic coefficient (controls curvature)
    A0 : float
        Y-intercept (vertical offset at centerline)
    **kwargs
        Additional arguments passed to ConeWaverider
        
    Returns
    -------
    waverider : ConeWaverider
        Generated waverider object
    """
    # Second order polynomial: y = A2*x^2 + 0*x + A0
    # Linear term must be 0 for smooth y-axis symmetry
    poly_coeffs = [A2, 0.0, A0]
    
    return ConeWaverider(
        mach=mach,
        shock_angle=shock_angle,
        poly_coeffs=poly_coeffs,
        **kwargs
    )


def create_third_order_waverider(
    mach: float,
    shock_angle: float,
    A3: float,
    A2: float,
    A0: float,
    **kwargs
) -> ConeWaverider:
    """
    Create a waverider with third-order polynomial leading edge.
    
    y = A3 * x^3 + A2 * x^2 + A0
    
    Note: The linear term (A1) is set to 0 for y-axis symmetry.
    
    Parameters
    ----------
    mach : float
        Freestream Mach number
    shock_angle : float
        Shock cone angle in degrees
    A3 : float
        Cubic coefficient
    A2 : float
        Quadratic coefficient
    A0 : float
        Y-intercept
    **kwargs
        Additional arguments passed to ConeWaverider
        
    Returns
    -------
    waverider : ConeWaverider
        Generated waverider object
    """
    # Third order polynomial: y = A3*x^3 + A2*x^2 + 0*x + A0
    poly_coeffs = [A3, A2, 0.0, A0]
    
    return ConeWaverider(
        mach=mach,
        shock_angle=shock_angle,
        poly_coeffs=poly_coeffs,
        **kwargs
    )


# Convenience function to get optimal shock angle for a given Mach number
def optimal_shock_angle(mach: float, gamma: float = 1.4) -> float:
    """
    Estimate optimal shock angle for maximum L/D.
    
    Based on empirical relationship from waverider literature.
    
    Parameters
    ----------
    mach : float
        Freestream Mach number
    gamma : float
        Ratio of specific heats
        
    Returns
    -------
    beta : float
        Recommended shock angle in degrees
    """
    # Mach angle
    mu = np.degrees(np.arcsin(1.0 / mach))
    
    # Empirical relationship: optimal shock is ~1.1-1.3x Mach angle
    # This varies with Mach number
    if mach < 6:
        factor = 1.3
    elif mach < 10:
        factor = 1.25
    elif mach < 15:
        factor = 1.2
    else:
        factor = 1.15
    
    return mu * factor


if __name__ == "__main__":
    # Example usage
    print("Creating Mach 6 cone-derived waverider...")
    
    # Default parameters similar to Adam's thesis
    waverider = create_second_order_waverider(
        mach=6.0,
        shock_angle=12.0,
        A2=-2.0,
        A0=-0.15,
        n_leading_edge=21,
        n_streamwise=20
    )
    
    waverider.summary()
    
    # Export mesh
    waverider.export_stl("waverider_test.stl")
    waverider.export_tri("waverider_test.tri")
    
    print("\nMesh exported to waverider_test.stl and waverider_test.tri")
