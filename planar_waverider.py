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

        When use_finite_mach is None (default), uses Eq. (4) — the M→∞
        limit of the θ-β-M relation.  This is the paper's standard formula
        for the base wedge angle, used for both the initial design and the
        full optimisation (Section 3.2).

        Eq. (22) — the finite-Mach relation — is available as an explicit
        override (use_finite_mach=True) for off-design perturbation
        analysis, but is NOT the default.
        """
        beta = np.radians(self.beta_deg)
        g = self.gamma
        M = self.M_inf

        if use_finite_mach is None:
            # Default: Eq. 4 (M→∞ limit) per paper Section 3.2
            use_finite_mach = False

        if use_finite_mach:
            # Eq. (22): finite-Mach θ-β-M relation (for off-design analysis)
            num = 2.0 * np.cos(beta) / np.sin(beta) * (M**2 * np.sin(beta)**2 - 1.0)
            den = M**2 * (g + np.cos(2.0 * beta)) + 2.0
            theta = np.arctan(num / den)
        else:
            # Eq. (4): M→∞ limit — standard formula for base wedge angle
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

    def _blend_le_rounding(self, upper_x, upper_z, lower_x, lower_z,
                           x_le, z_le, T_star, theta, nx, ny):
        """Apply Tincher & Burnett adding-material LE rounding to grids.

        Uses an EXTERIOR circle of radius R that wraps around the outside
        of the sharp LE.  The circle is tangent to both the upper
        (horizontal) and lower (compression) surfaces.  The nose extends
        slightly upstream of x_le, adding material.

        Circle offset is R·tan(θ/2) [tiny], center is ABOVE z_le by R.
        Only **z values** are modified (x stays unchanged) so that the
        spanwise grid structure remains smooth for visualization.

        Parameters
        ----------
        upper_x, upper_z, lower_x, lower_z : ndarray (ny, nx)
            Half-span surface grids — z arrays modified **in-place**.
        x_le, z_le : ndarray (ny,)
            Sharp leading-edge coordinates.
        T_star : ndarray (ny,)
            Chebyshev angle multipliers.
        theta : float
            Base wedge angle [rad].
        nx, ny : int
            Grid dimensions.

        Returns
        -------
        nose_x, nose_z : ndarray (ny,)
            Updated LE coordinates (nose of inscribed circle).
        """
        R = self.R
        L = self.length

        nose_x = np.copy(x_le)
        nose_z = np.copy(z_le)

        for j in range(ny):
            theta_j = T_star[j] * theta
            chord = L - x_le[j]

            # Skip if angle too small or no chord
            if theta_j < np.radians(0.5) or chord < 1e-8:
                continue

            half_theta = theta_j / 2.0

            # Taper R_eff near wingtips: limit nose bump to ≤50% of
            # the local trailing-edge thickness for smooth transition.
            te_thickness = chord * np.tan(theta_j)
            R_max = 0.5 * te_thickness if te_thickness > 1e-9 else 0.0
            R_eff = min(R, R_max)
            if R_eff < 1e-6:
                continue

            # --- T&B adding-material EXTERIOR circle ---
            # Offset = R·tan(θ/2)  [tiny for small θ, NOT R/tan(θ/2)]
            offset = R_eff * np.tan(half_theta)
            xc = x_le[j] + offset                        # center x
            zc = z_le[j] + R_eff                          # center z (ABOVE)

            x_nose_j = xc - R_eff                         # nose (≈ x_le - R)
            x_ut = xc                                      # upper tangent x
            x_lt = xc - R_eff * np.sin(theta_j)           # lower tangent x
            z_lt = zc - R_eff * np.cos(theta_j)           # lower tangent z

            # Modify z only — x values are unchanged
            for i in range(nx):
                x = upper_x[j, i]

                if x < x_nose_j:
                    # Before nose: collapse to nose height (zc = z_le + R)
                    upper_z[j, i] = zc
                    lower_z[j, i] = zc

                elif x <= xc:
                    # Arc region: both surfaces use BOTTOM arc
                    dx_n = (x - xc) / R_eff          # ∈ [-1, 0]
                    dx_n = max(-1.0, min(0.0, dx_n))
                    sq = np.sqrt(1.0 - dx_n * dx_n)
                    z_arc = zc - R_eff * sq           # bottom of circle

                    # Upper: bottom arc until upper tangent (x=xc)
                    upper_z[j, i] = z_arc

                    # Lower: bottom arc until lower tangent, then slope
                    if x <= x_lt:
                        lower_z[j, i] = z_arc  # same as upper (nose region)
                    else:
                        lower_z[j, i] = (z_le[j]
                                         - np.tan(theta_j) * (x - x_le[j]))
                # else: past upper tangent → keep original z

            nose_x[j] = x_nose_j
            nose_z[j] = zc

        return nose_x, nose_z

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
