"""
Planar Waverider Geometry Engine

Implements the 9-parameter planar waverider definition from:
  Jessen, Larsson, Brehm (2026) — "Comparative optimization of hypersonic
  waveriders using analytical and computational methods"
  Aerospace Science and Technology 172, 111703.

Parameters: f(ℓ, w, n, β, ε, p1, p2, p3, R)
  ℓ  — vehicle length [m]
  w  — vehicle width (full span) [m]
  n  — power-law exponent for LE shape
  β  — planar shock angle [deg]
  ε  — parabolic LE perturbation [-1, 1]
  p1, p2, p3 — Chebyshev perturbation coefficients (1 = no perturbation)
  R  — leading edge nose radius [m]
"""

import numpy as np


class PlanarWaverider:
    """Generates a planar waverider geometry with Chebyshev perturbations."""

    def __init__(self, length=1.0, width=0.3, n=0.5, beta_deg=9.0,
                 epsilon=0.0, p1=1.0, p2=1.0, p3=1.0, R=0.0,
                 M_inf=10.0, gamma=1.4):
        self.length = length          # ℓ
        self.width = width            # w (full span)
        self.n = n                    # power-law exponent
        self.beta_deg = beta_deg      # shock angle [deg]
        self.epsilon = epsilon        # LE perturbation
        self.p1 = p1                  # Chebyshev point at y = w/3
        self.p2 = p2                  # Chebyshev point at y = w/6
        self.p3 = p3                  # Chebyshev point at y = 0
        self.R = R                    # LE radius [m]
        self.M_inf = M_inf
        self.gamma = gamma

        # Computed geometry arrays (populated by generate())
        self.upper_surface_x = None
        self.upper_surface_y = None
        self.upper_surface_z = None
        self.lower_surface_x = None
        self.lower_surface_y = None
        self.lower_surface_z = None
        self.leading_edge = None      # (n_le, 3) array
        self.wedge_angle_deg = None
        self.chebyshev_coeffs = None  # a0..a4

    # ------------------------------------------------------------------
    #  Core geometry equations from the paper
    # ------------------------------------------------------------------

    def _compute_wedge_angle(self, use_finite_mach=None):
        """Compute wedge angle θ from shock angle β.

        When use_finite_mach is None (default), auto-selects:
          - Eq. (22) finite-Mach when p1=p2=p3=1 (on-design, no perturbations)
          - Eq. (4) M→∞ limit when any p_i ≠ 1 (off-design perturbations)
        This matches the paper's convention (Section 3.2).
        """
        beta = np.radians(self.beta_deg)
        g = self.gamma
        M = self.M_inf

        if use_finite_mach is None:
            # Auto-select: Eq. 22 for on-design, Eq. 4 for off-design
            use_finite_mach = (self.p1 == 1.0 and self.p2 == 1.0
                               and self.p3 == 1.0)

        if use_finite_mach:
            # Eq. (22): finite-Mach θ-β-M relation
            num = 2.0 * np.cos(beta) / np.sin(beta) * (M**2 * np.sin(beta)**2 - 1.0)
            den = M**2 * (g + np.cos(2.0 * beta)) + 2.0
            theta = np.arctan(num / den)
        else:
            # Eq. (4): M→∞ limit (for off-design with perturbations)
            num = 2.0 * np.cos(beta) / np.sin(beta) * np.sin(beta)**2
            den = g + np.cos(2.0 * beta)
            theta = np.arctan(num / den)

        self.wedge_angle_deg = np.degrees(theta)
        return theta

    def _leading_edge_x(self, y):
        """LE x-coordinate from power-law (Eq. 1).

        x_LE(y) = ℓ * (2|y|/w)^(1/n)
        """
        y = np.asarray(y, dtype=float)
        ratio = np.clip(2.0 * np.abs(y) / self.width, 0.0, 1.0)
        return self.length * ratio ** (1.0 / self.n)

    def _leading_edge_z(self, x):
        """LE z-coordinate with parabolic perturbation (Eq. 2).

        z_LE(x) = -x * tan(β) * (x/ℓ * ε - ε + 1)
        """
        x = np.asarray(x, dtype=float)
        beta = np.radians(self.beta_deg)
        eps = self.epsilon
        return -x * np.tan(beta) * (x / self.length * eps - eps + 1.0)

    # ------------------------------------------------------------------
    #  Chebyshev perturbation (Eq. 5-7)
    # ------------------------------------------------------------------

    @staticmethod
    def _chebyshev_T(n, eta):
        """Evaluate Chebyshev polynomial T_n(η) of the first kind."""
        eta = np.asarray(eta, dtype=float)
        if n == 0:
            return np.ones_like(eta)
        elif n == 1:
            return eta.copy()
        else:
            T_prev2 = np.ones_like(eta)
            T_prev1 = eta.copy()
            for _ in range(2, n + 1):
                T_curr = 2.0 * eta * T_prev1 - T_prev2
                T_prev2 = T_prev1
                T_prev1 = T_curr
            return T_curr

    @staticmethod
    def _chebyshev_dT(n, eta):
        """Evaluate derivative T'_n(η) of Chebyshev polynomial."""
        eta = np.asarray(eta, dtype=float)
        if n == 0:
            return np.zeros_like(eta)
        elif n == 1:
            return np.ones_like(eta)
        elif n == 2:
            return 4.0 * eta
        elif n == 3:
            return 12.0 * eta**2 - 3.0
        elif n == 4:
            return 32.0 * eta**3 - 16.0 * eta
        else:
            # General: T'_{n+1} = 2*T_n + 2*η*T'_n - T'_{n-1}
            dT_prev2 = np.zeros_like(eta)  # T'_0
            dT_prev1 = np.ones_like(eta)   # T'_1
            T_prev2 = np.ones_like(eta)    # T_0
            T_prev1 = eta.copy()           # T_1
            for k in range(2, n + 1):
                T_curr = 2.0 * eta * T_prev1 - T_prev2
                dT_curr = 2.0 * T_prev1 + 2.0 * eta * dT_prev1 - dT_prev2
                T_prev2, T_prev1 = T_prev1, T_curr
                dT_prev2, dT_prev1 = dT_prev1, dT_curr
            return dT_curr

    def _compute_chebyshev_coefficients(self):
        """Solve 5×5 system (Eq. 7) for Chebyshev coefficients a0..a4.

        Rows:
          0: T*_n at y = w/3   (η = 1/3)   → p1
          1: T*_n at y = w/6   (η = -1/3)  → p2
          2: T*_n at y = 0     (η = -1)    → p3
          3: T*'_n at y = w/2  (η = 1)     → 0  (Neumann at LE)
          4: T*'_n at y = 0    (η = -1)    → 0  (Neumann at centerline)
        """
        w = self.width
        # η values at the control locations
        eta_p1 = 4.0 * (w / 3.0) / w - 1.0   # = 1/3
        eta_p2 = 4.0 * (w / 6.0) / w - 1.0   # = -1/3
        eta_p3 = 4.0 * 0.0 / w - 1.0          # = -1
        eta_le = 4.0 * (w / 2.0) / w - 1.0    # = 1
        eta_cl = -1.0                          # = -1

        A = np.zeros((5, 5))
        rhs = np.array([self.p1, self.p2, self.p3, 0.0, 0.0])

        for k in range(5):
            # Function value rows
            A[0, k] = self._chebyshev_T(k, eta_p1)
            A[1, k] = self._chebyshev_T(k, eta_p2)
            A[2, k] = self._chebyshev_T(k, eta_p3)
            # Derivative rows (dT*/dy = T'_n(η) * 4/w, but 4/w cancels
            # since rhs is 0)
            A[3, k] = self._chebyshev_dT(k, eta_le)
            A[4, k] = self._chebyshev_dT(k, eta_cl)

        self.chebyshev_coeffs = np.linalg.solve(A, rhs)
        return self.chebyshev_coeffs

    def _angle_perturbation(self, y):
        """Evaluate T*(y) = Σ a_n T*_n(y) (Eq. 6).

        Returns the angle multiplier at spanwise stations y.
        """
        y = np.asarray(y, dtype=float)
        eta = 4.0 * np.abs(y) / self.width - 1.0
        eta = np.clip(eta, -1.0, 1.0)
        result = np.zeros_like(eta)
        for k in range(5):
            result += self.chebyshev_coeffs[k] * self._chebyshev_T(k, eta)
        return result

    # ------------------------------------------------------------------
    #  Leading edge rounding (Tincher & Burnett adding-material method)
    # ------------------------------------------------------------------

    def _apply_le_rounding(self, le_pts, theta_local, n_arc=8):
        """Apply circular blunting at each LE station.

        At each spanwise station, inscribe a circle of radius R tangent
        to both upper (horizontal) and lower (angled at θ) surfaces,
        adding material forward of the original sharp LE.

        Returns arrays of rounded LE points (upper arc, center, lower arc).
        """
        if self.R <= 0:
            return le_pts, None

        R = self.R
        n_le = len(le_pts)
        rounded_sections = []

        for i in range(n_le):
            x0, y0, z0 = le_pts[i]
            theta = theta_local[i]

            # Half-angle between upper (horizontal) and lower surface
            half_angle = theta / 2.0

            # Circle center offset from sharp LE
            # Upper surface is horizontal, lower is at angle θ below
            # Center lies on the bisector of the dihedral angle
            dx = R / np.sin(half_angle) * np.cos(half_angle) if half_angle > 1e-8 else R
            dz_up = R  # tangent to horizontal upper surface
            dz_down = R  # tangent to angled lower surface

            # Circle center: shifted forward (negative x) and up
            xc = x0 - R / np.tan(theta) if theta > 1e-8 else x0 - R
            zc = z0 + R

            # Generate arc points from upper tangent to lower tangent
            angle_start = np.pi / 2.0  # upper tangent (horizontal)
            angle_end = np.pi / 2.0 + theta  # lower tangent
            arc_angles = np.linspace(angle_start, angle_end, n_arc)

            arc_pts = np.zeros((n_arc, 3))
            arc_pts[:, 0] = xc - R * np.cos(arc_angles)
            arc_pts[:, 1] = y0
            arc_pts[:, 2] = zc - R * np.sin(arc_angles)

            rounded_sections.append(arc_pts)

        return le_pts, rounded_sections

    # ------------------------------------------------------------------
    #  Surface generation
    # ------------------------------------------------------------------

    def generate(self, nx=60, ny=40):
        """Generate the complete waverider geometry.

        Parameters
        ----------
        nx : int
            Number of streamwise grid points.
        ny : int
            Number of spanwise grid points (half-span, mirrored).

        Returns
        -------
        self : PlanarWaverider
            With populated surface arrays.
        """
        L = self.length
        w = self.width

        # Step 1: Compute wedge angle
        theta = self._compute_wedge_angle()

        # Step 2: Compute Chebyshev coefficients
        self._compute_chebyshev_coefficients()

        # Step 3: Create spanwise stations (half-span: 0 to w/2)
        y_half = np.linspace(0, w / 2.0, ny)

        # Step 4: At each spanwise station, compute LE position
        x_le = self._leading_edge_x(y_half)           # (ny,)
        z_le = self._leading_edge_z(x_le)              # (ny,)

        # Step 5: Compute angle perturbation at each spanwise station
        T_star = self._angle_perturbation(y_half)       # (ny,)

        # Step 6: Create structured grids for upper and lower surfaces
        # Use normalized streamwise coordinate s ∈ [0, 1]
        s = np.linspace(0, 1, nx)

        # Upper surface: z_upper(x, y) = z_LE(x_LE(y)) for all x ≥ x_LE(y)
        # At each y_j, x ranges from x_LE(y_j) to L
        upper_x = np.zeros((ny, nx))
        upper_y = np.zeros((ny, nx))
        upper_z = np.zeros((ny, nx))

        lower_x = np.zeros((ny, nx))
        lower_y = np.zeros((ny, nx))
        lower_z = np.zeros((ny, nx))

        for j in range(ny):
            # Streamwise coordinates from LE to trailing edge
            x_j = x_le[j] + s * (L - x_le[j])

            # Upper surface: flat in streamwise direction at LE height
            z_upper_j = z_le[j]

            # Lower surface: (Eq. 3)
            theta_local = T_star[j] * theta
            z_lower_j = z_upper_j - np.tan(theta_local) * (x_j - x_le[j])

            upper_x[j, :] = x_j
            upper_y[j, :] = y_half[j]
            upper_z[j, :] = z_upper_j

            lower_x[j, :] = x_j
            lower_y[j, :] = y_half[j]
            lower_z[j, :] = z_lower_j

        # Step 7: Mirror to full span (negative y side)
        # Reverse y_half[1:] to avoid duplicating centerline
        upper_x_full = np.vstack([upper_x[::-1], upper_x[1:]])
        upper_y_full = np.vstack([-upper_y[::-1], upper_y[1:]])
        upper_z_full = np.vstack([upper_z[::-1], upper_z[1:]])

        lower_x_full = np.vstack([lower_x[::-1], lower_x[1:]])
        lower_y_full = np.vstack([-lower_y[::-1], lower_y[1:]])
        lower_z_full = np.vstack([lower_z[::-1], lower_z[1:]])

        self.upper_surface_x = upper_x_full
        self.upper_surface_y = upper_y_full
        self.upper_surface_z = upper_z_full
        self.lower_surface_x = lower_x_full
        self.lower_surface_y = lower_y_full
        self.lower_surface_z = lower_z_full

        # Step 8: Leading edge curve (full span)
        le_half = np.column_stack([x_le, y_half, z_le])
        le_mirror = le_half[::-1].copy()
        le_mirror[:, 1] *= -1.0
        self.leading_edge = np.vstack([le_mirror, le_half[1:]])

        # Step 9: Apply LE rounding if R > 0
        if self.R > 0:
            theta_local_half = T_star * theta
            _, self.le_rounded_sections = self._apply_le_rounding(
                le_half, theta_local_half
            )

        return self

    # ------------------------------------------------------------------
    #  Mesh generation (triangle mesh for visualization / export)
    # ------------------------------------------------------------------

    def get_mesh(self):
        """Convert structured surfaces to triangle mesh.

        Returns
        -------
        vertices : ndarray (n_verts, 3)
        faces : ndarray (n_faces, 3) — indices into vertices
        """
        if self.upper_surface_x is None:
            raise RuntimeError("Call generate() first")

        ny_full, nx = self.upper_surface_x.shape
        verts = []
        faces = []
        offset = 0

        # --- Upper surface ---
        for j in range(ny_full):
            for i in range(nx):
                verts.append([
                    self.upper_surface_x[j, i],
                    self.upper_surface_y[j, i],
                    self.upper_surface_z[j, i],
                ])
        for j in range(ny_full - 1):
            for i in range(nx - 1):
                v00 = j * nx + i
                v10 = (j + 1) * nx + i
                v01 = j * nx + (i + 1)
                v11 = (j + 1) * nx + (i + 1)
                # Upper surface normals point up (outward)
                faces.append([v00, v01, v11])
                faces.append([v00, v11, v10])
        offset = len(verts)

        # --- Lower surface ---
        for j in range(ny_full):
            for i in range(nx):
                verts.append([
                    self.lower_surface_x[j, i],
                    self.lower_surface_y[j, i],
                    self.lower_surface_z[j, i],
                ])
        for j in range(ny_full - 1):
            for i in range(nx - 1):
                v00 = offset + j * nx + i
                v10 = offset + (j + 1) * nx + i
                v01 = offset + j * nx + (i + 1)
                v11 = offset + (j + 1) * nx + (i + 1)
                # Lower surface normals point down (outward)
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])
        offset = len(verts)

        # --- Base face (trailing edge, x = L) ---
        # Connect last column of upper to last column of lower
        base_upper_idx = []
        base_lower_idx = []
        for j in range(ny_full):
            # Upper TE vertex index
            base_upper_idx.append(j * nx + (nx - 1))
            # Lower TE vertex index (offset by upper surface vertex count)
            base_lower_idx.append(ny_full * nx + j * nx + (nx - 1))

        for j in range(ny_full - 1):
            u0 = base_upper_idx[j]
            u1 = base_upper_idx[j + 1]
            l0 = base_lower_idx[j]
            l1 = base_lower_idx[j + 1]
            faces.append([u0, u1, l1])
            faces.append([u0, l1, l0])

        return np.array(verts), np.array(faces)

    # ------------------------------------------------------------------
    #  Derived quantities
    # ------------------------------------------------------------------

    def planform_area(self):
        """Compute projected planform area (X-Y plane)."""
        if self.upper_surface_x is None:
            return 0.0
        X = self.upper_surface_x
        Y = self.upper_surface_y
        ny, nx = X.shape
        area = 0.0
        for j in range(ny - 1):
            for i in range(nx - 1):
                p1 = np.array([X[j, i], Y[j, i]])
                p2 = np.array([X[j+1, i], Y[j+1, i]])
                p3 = np.array([X[j+1, i+1], Y[j+1, i+1]])
                p4 = np.array([X[j, i+1], Y[j, i+1]])
                area += 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                area += 0.5 * abs(np.cross(p3 - p1, p4 - p1))
        return area

    def volume(self):
        """Approximate enclosed volume using trapezoidal cross-sections.

        Each streamwise slice has a cross-section in the Y-Z plane.
        We integrate those areas along the X-axis using centerline dx.
        """
        if self.upper_surface_x is None:
            return 0.0
        ny, nx = self.upper_surface_x.shape
        # Use the centerline row (ny//2) for x-spacing, which has full chord
        j_center = ny // 2
        vol = 0.0
        for i in range(nx - 1):
            # Cross-section area at streamwise station i
            area_i = 0.0
            for j in range(ny - 1):
                dz_j = self.upper_surface_z[j, i] - self.lower_surface_z[j, i]
                dz_j1 = (self.upper_surface_z[j+1, i]
                         - self.lower_surface_z[j+1, i])
                dy = abs(self.upper_surface_y[j+1, i]
                         - self.upper_surface_y[j, i])
                area_i += 0.5 * (max(dz_j, 0) + max(dz_j1, 0)) * dy
            dx = abs(self.upper_surface_x[j_center, i+1]
                     - self.upper_surface_x[j_center, i])
            vol += area_i * dx
        return vol

    def base_dimensions(self):
        """Return base (trailing edge) width and max height."""
        if self.upper_surface_x is None:
            return 0.0, 0.0
        ny = self.upper_surface_y.shape[0]
        y_base = self.upper_surface_y[:, -1]
        z_upper_base = self.upper_surface_z[:, -1]
        z_lower_base = self.lower_surface_z[:, -1]
        base_width = y_base.max() - y_base.min()
        base_height = (z_upper_base - z_lower_base).max()
        return base_width, base_height

    def to_dict(self):
        """Serialize parameters to dict for save/load."""
        return {
            'length': self.length,
            'width': self.width,
            'n': self.n,
            'beta_deg': self.beta_deg,
            'epsilon': self.epsilon,
            'p1': self.p1,
            'p2': self.p2,
            'p3': self.p3,
            'R': self.R,
            'M_inf': self.M_inf,
            'gamma': self.gamma,
        }

    @classmethod
    def from_dict(cls, d):
        """Create instance from parameter dict."""
        return cls(**d)
