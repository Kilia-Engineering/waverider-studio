"""
Analytical Aerodynamic Model for Planar Waveriders

Implements the reduced-order model from:
  Jessen, Larsson, Brehm (2026) — Aerospace Science and Technology 172, 111703.

Components:
  - Modified Newtonian Theory at LE (Eq. 8-9)
  - Tangent-wedge / Prandtl-Meyer expansion for surfaces (Eq. 10-11)
  - Maxwell slender-body base pressure (P_base = P_inf / M_inf)
  - Pressure bleed correction for rounded LE (Eq. 12)
  - Eckert reference temperature viscous model (Eq. 15-18)
  - Local Mach / temperature from shock-expansion (Eq. 13-14)
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────
#  Atmosphere model (simple exponential + ISA layers)
# ──────────────────────────────────────────────────────────────────────

def atmosphere(altitude_km):
    """Simple US Standard Atmosphere model.

    Returns (T [K], P [Pa], rho [kg/m³], a [m/s]) at given altitude.
    """
    h = altitude_km * 1000.0  # metres
    g = 9.80665
    R_air = 287.058

    if h <= 11000:
        T = 288.15 - 0.0065 * h
        P = 101325.0 * (T / 288.15) ** (g / (0.0065 * R_air))
    elif h <= 25000:
        T = 216.65
        P = 22632.1 * np.exp(-g * (h - 11000) / (R_air * T))
    elif h <= 47000:
        T = 216.65 + 0.003 * (h - 25000)
        P = 2488.63 * (T / 216.65) ** (-g / (0.003 * R_air))
    elif h <= 53000:
        T = 270.65
        P = 120.446 * np.exp(-g * (h - 47000) / (R_air * T))
    elif h <= 79000:
        T = 270.65 - 0.0045 * (h - 53000)
        P = 58.3105 * (T / 270.65) ** (g / (0.0045 * R_air))
    else:
        T = 180.65
        P = 1.03218 * np.exp(-g * (h - 79000) / (R_air * T))

    rho = P / (R_air * T)
    a = np.sqrt(1.4 * R_air * T)
    return T, P, rho, a


# ──────────────────────────────────────────────────────────────────────
#  Inviscid pressure models (Eq. 8-12)
# ──────────────────────────────────────────────────────────────────────

def cp_max(M, gamma=1.4):
    """Maximum pressure coefficient — Rayleigh pitot formula (Eq. 8)."""
    g = gamma
    term1 = ((g + 1)**2 * M**2 / (4 * g * M**2 - 2 * (g - 1))) ** (g / (g - 1))
    term2 = (1 - g + 2 * g * M**2) / (g + 1)
    p02_p1 = term1 * term2
    return 2.0 / (g * M**2) * (p02_p1 - 1.0)


def modified_newtonian_pressure(theta, M, P_inf, gamma=1.4):
    """Modified Newtonian pressure at LE (Eq. 8-9).

    Parameters
    ----------
    theta : float or ndarray
        Surface inclination angle to freestream [rad].
    M : float
        Freestream Mach number.
    P_inf : float
        Freestream pressure [Pa].

    Returns
    -------
    P : float or ndarray
        Surface pressure [Pa].
    """
    cp = cp_max(M, gamma) * np.sin(theta)**2
    return P_inf * (0.5 * cp * gamma * M**2 + 1.0)


def solve_oblique_beta(theta, M, gamma=1.4, max_iter=30):
    """Solve for weak oblique shock angle β given deflection θ and M.

    Uses Newton's method on the exact θ-β-M relation.
    """
    g = gamma
    mu = np.arcsin(1.0 / M)  # Mach angle (minimum β)

    if theta <= 0 or M <= 1:
        return mu

    # Initial guess: linearized theory β ≈ μ + (γ+1)/2 * θ
    beta = mu + (g + 1) / 2.0 * theta
    beta = min(beta, np.pi / 2.0 - 0.01)

    for _ in range(max_iter):
        sinb = np.sin(beta)
        cosb = np.cos(beta)
        M2sin2b = M**2 * sinb**2

        num = 2.0 * cosb / sinb * (M2sin2b - 1.0)
        den = M**2 * (g + np.cos(2.0 * beta)) + 2.0
        if abs(den) < 1e-12:
            break
        theta_calc = np.arctan(num / den)

        error = theta_calc - theta
        if abs(error) < 1e-12:
            break

        # Numerical derivative dθ/dβ
        db = 1e-8
        beta2 = beta + db
        sinb2 = np.sin(beta2)
        cosb2 = np.cos(beta2)
        num2 = 2.0 * cosb2 / sinb2 * (M**2 * sinb2**2 - 1.0)
        den2 = M**2 * (g + np.cos(2.0 * beta2)) + 2.0
        theta_calc2 = np.arctan(num2 / den2)

        dtheta_dbeta = (theta_calc2 - theta_calc) / db
        if abs(dtheta_dbeta) < 1e-12:
            break

        beta -= error / dtheta_dbeta
        beta = max(beta, mu + 1e-6)
        beta = min(beta, np.pi / 2.0 - 0.01)

    return beta


def oblique_shock_pressure(theta, M, gamma=1.4):
    """Pressure ratio P/P_inf from exact oblique shock (compression).

    Solves the θ-β-M relation exactly, then applies normal shock
    pressure relation at the normal Mach component.
    """
    g = gamma
    beta = solve_oblique_beta(theta, M, g)
    return 1.0 + 2.0 * g / (g + 1.0) * (M**2 * np.sin(beta)**2 - 1.0)


def surface_pressure_ratio(theta, M, gamma=1.4):
    """Surface pressure ratio P/P_inf using exact shock/expansion (Eq. 10).

    Handles compression (θ > 0), expansion (θ < 0), and base (θ extreme).

    Parameters
    ----------
    theta : float
        Local surface angle relative to freestream [rad].
        Positive = compression, negative = expansion.
    M : float
        Freestream Mach number.

    Returns
    -------
    P_ratio : float
        P / P_inf
    """
    g = gamma

    # Expansion limit angle
    theta_limit = -2.0 / ((g - 1) * M) * (1.0 - (1.0 / M) ** ((g - 1) / (2 * g)))

    if theta <= theta_limit:
        # Base / separated: Maxwell slender-body
        return 1.0 / M
    elif theta < -1e-10:
        # Expansion (Prandtl-Meyer-like, Eq. 10 middle branch)
        val = (1.0 + (g - 1) / 2.0 * M * theta) ** (2 * g / (g - 1))
        return max(val, 1.0 / M)
    elif theta > 1e-10:
        # Compression: exact oblique shock
        return oblique_shock_pressure(theta, M, g)
    else:
        return 1.0


def pressure_bleed_correction(M, alpha, Lambda, x_over_R):
    """Pressure increment from LE rounding (Eq. 12).

    Parameters
    ----------
    M : float
        Freestream Mach number.
    alpha : float
        Angle of attack [rad].
    Lambda : float or ndarray
        Local LE sweep angle [rad].
    x_over_R : float or ndarray
        Downstream distance / LE radius.

    Returns
    -------
    delta_P_over_Pinf : float or ndarray
        ΔP / P_inf
    """
    cosL = np.cos(Lambda)
    sinA = np.sin(alpha)
    xR = np.asarray(x_over_R, dtype=float)

    term1 = (0.18 * cosL + 0.37 * sinA) * M**2 / (2.14 + xR)
    term2 = -0.85 * cosL * sinA**2 * M**2 / np.exp(0.41 * sinA**2 * xR)
    term3 = -0.046 * M**2 / np.exp(1.34 * cosL * xR)

    return term1 + term2 + term3


# ──────────────────────────────────────────────────────────────────────
#  Local Mach and temperature (Eq. 13-14)
# ──────────────────────────────────────────────────────────────────────

def local_mach_temperature(theta, M_inf, T_inf, P_ratio, gamma=1.4):
    """Compute local Mach number and temperature behind shock/expansion.

    Uses Eq. 13-14 from the paper.

    Parameters
    ----------
    theta : float
        Surface deflection angle [rad].
    M_inf : float
        Freestream Mach number.
    T_inf : float
        Freestream temperature [K].
    P_ratio : float
        P / P_inf from pressure model.
    gamma : float

    Returns
    -------
    M_local : float
        Local Mach number.
    T_local : float
        Local temperature [K].
    """
    g = gamma

    if theta >= 0:
        # Compression (Eq. 13-14 upper branch)
        if abs(theta) < 1e-10:
            return M_inf, T_inf

        # Exact oblique shock angle
        beta = solve_oblique_beta(theta, M_inf, g)

        sin2b = np.sin(beta)**2

        # Eq. 14 compression: M²_local
        num = (g - 1) * M_inf**2 * sin2b + 2.0
        den = 2.0 * g * M_inf**2 * sin2b - (g - 1)
        if den > 0:
            M_local_sq = num / den / np.sin(beta - theta)**2
            M_local = np.sqrt(max(M_local_sq, 0.01))
        else:
            M_local = 0.1

        # Eq. 13 compression: T/T_inf
        T_ratio_num = (2.0 + (g - 1) * M_inf**2 * sin2b)
        T_ratio_den = (g + 1)**2 * M_inf**2 * sin2b
        if T_ratio_den > 0:
            T_ratio = P_ratio * T_ratio_num / T_ratio_den
        else:
            T_ratio = 1.0
        # Actually Eq. 13 is:
        # T/T_inf = (P/P_inf) * (2 + (g-1)*M²sin²β) / ((g+1)*M²*sin²β)
        # which is just the oblique shock T ratio = (P2/P1) / (rho2/rho1)
        # Let me use the standard: T2/T1 = P_ratio * (2 + (g-1)*M²sin²β) / ((g+1)*M²sin²β)
        rho_ratio = (g + 1) * M_inf**2 * sin2b / (2 + (g - 1) * M_inf**2 * sin2b)
        T_ratio = P_ratio / rho_ratio if rho_ratio > 0 else 1.0
        T_local = T_inf * T_ratio

    else:
        # Expansion (Eq. 13-14 lower branch)
        factor = 1.0 + (g - 1) / 2.0 * M_inf * theta
        if factor > 0:
            T_local = T_inf * factor**2
            M_local = M_inf / factor if factor > 0 else M_inf
        else:
            T_local = T_inf
            M_local = M_inf

    return max(M_local, 0.1), max(T_local, 50.0)


# ──────────────────────────────────────────────────────────────────────
#  Viscous model — Eckert reference temperature (Eq. 15-18)
# ──────────────────────────────────────────────────────────────────────

def eckert_skin_friction(M_local, T_local, P_local, x_from_LE,
                         T_wall=None, gamma=1.4):
    """Compute wall shear stress using Eckert's method (Eq. 15-18).

    Returns τ_w = cf(Re*) * q_ref, where q_ref = (1/2)*ρ*u_e²
    following Anderson's formulation (HATD Sec. 6.8).

    Parameters
    ----------
    M_local : float
        Local Mach number after shock/expansion.
    T_local : float
        Local static temperature [K].
    P_local : float
        Local static pressure [Pa].
    x_from_LE : float
        Streamwise distance from LE [m].
    T_wall : float or None
        Wall temperature [K]. If None, use adiabatic wall.
    gamma : float

    Returns
    -------
    tau_w : float
        Wall shear stress [Pa].
    """
    R_air = 287.058
    g = gamma

    if x_from_LE < 1e-6:
        return 0.0

    # Adiabatic wall temperature if not specified
    if T_wall is None:
        r = 0.89  # recovery factor (turbulent flat plate)
        T_wall = T_local * (1.0 + r * (g - 1) / 2.0 * M_local**2)

    # Reference temperature (Eq. 15)
    T_star = T_local * (1.0 + 0.032 * M_local**2 + 0.58 * (T_wall / T_local - 1.0))
    T_star = max(T_star, 50.0)

    # Reference density (ideal gas)
    rho_star = P_local / (R_air * T_star)

    # Reference viscosity (Sutherland's law, Eq. 16)
    mu_ref = 1.716e-5   # kg/(m·s)
    T_ref = 273.15       # K
    S = 110.4            # K
    mu_star = mu_ref * (T_star / T_ref)**1.5 * (T_ref + S) / (T_star + S)

    # Edge velocity (from local Mach and temperature)
    a_local = np.sqrt(g * R_air * T_local)
    u_e = M_local * a_local

    # Reference Reynolds number (Eq. 16) — for cf evaluation
    Re_star = rho_star * u_e * x_from_LE / mu_star if mu_star > 0 else 1e6
    Re_star = max(Re_star, 1.0)

    # Edge Reynolds number — for transition check (Eq. 17 uses Re_x, not Re*_x)
    mu_e = mu_ref * (T_local / T_ref)**1.5 * (T_ref + S) / (T_local + S)
    rho_e = P_local / (R_air * T_local)
    Re_edge = rho_e * u_e * x_from_LE / mu_e if mu_e > 0 else 1e6
    Re_edge = max(Re_edge, 1.0)

    # Transition Reynolds number (Bowcutt correlation, Eq. 18)
    log_Re_T = 6.421 * np.exp(1.209e-4 * M_local**2.641)
    Re_T = 10.0 ** log_Re_T

    # Skin friction (Eq. 17): cf from Re*, transition from Re_edge
    if Re_edge < Re_T:
        # Laminar
        cf = 0.664 / np.sqrt(Re_star)
    else:
        # Turbulent
        cf = 0.0592 / Re_star**0.2

    # Wall shear stress: τ_w = cf * q_ref = cf * (1/2)*ρ*u_e²
    q_ref = 0.5 * rho_star * u_e**2
    return cf * q_ref


# ──────────────────────────────────────────────────────────────────────
#  Main force computation
# ──────────────────────────────────────────────────────────────────────

class PlanarWaveriderAero:
    """Analytical aerodynamic analysis for a PlanarWaverider geometry."""

    def __init__(self, gamma=1.4):
        self.gamma = gamma

    def compute_forces(self, waverider, M_inf, alpha_deg, altitude_km,
                       T_wall=None):
        """Compute aerodynamic forces using strip-theory analytical model.

        Uses the known surface deflection angles directly from the
        waverider geometry instead of mesh normals.

        Parameters
        ----------
        waverider : PlanarWaverider
            Generated waverider geometry.
        M_inf : float
            Freestream Mach number.
        alpha_deg : float
            Angle of attack [deg].
        altitude_km : float
            Altitude [km].
        T_wall : float or None
            Wall temperature [K]. None = adiabatic.

        Returns
        -------
        results : dict
            CL, CD, L_over_D, L, D, etc.
        """
        g = self.gamma
        alpha = np.radians(alpha_deg)

        # Atmospheric conditions
        T_inf, P_inf, rho_inf, a_inf = atmosphere(altitude_km)
        V_inf = M_inf * a_inf
        q_inf = 0.5 * rho_inf * V_inf**2

        # Reference area (planform)
        S_ref = waverider.planform_area()
        if S_ref < 1e-6:
            return self._empty_results()

        L = waverider.length
        w = waverider.width
        theta_wedge = np.radians(waverider.wedge_angle_deg)
        R_le = waverider.R

        # Use half-span grid directly (then double for full vehicle)
        ny_half = 60   # spanwise stations
        nx_strip = 80  # streamwise stations per strip
        y_half = np.linspace(0, w / 2.0, ny_half)
        dy = y_half[1] - y_half[0] if ny_half > 1 else w / 2.0

        # Chebyshev perturbation at each spanwise station
        T_star = waverider._angle_perturbation(y_half)

        # LE positions
        x_le = waverider._leading_edge_x(y_half)
        z_le = waverider._leading_edge_z(x_le)

        # Accumulate lift and drag
        lift_total = 0.0
        drag_inv_total = 0.0
        drag_visc_total = 0.0

        for j in range(ny_half):
            chord = L - x_le[j]
            if chord < 1e-6:
                continue

            # Local lower surface deflection angle
            theta_lower = T_star[j] * theta_wedge  # always positive
            # Effective angle including AoA:
            #   Lower surface sees: theta_lower + alpha (more compression)
            #   Upper surface sees: -alpha (expansion if alpha > 0)
            theta_lower_eff = theta_lower + alpha
            theta_upper_eff = -alpha  # expansion for positive AoA

            # Streamwise stations
            x_stations = np.linspace(x_le[j], L, nx_strip)
            dx = x_stations[1] - x_stations[0] if nx_strip > 1 else chord

            # --- Lower surface pressure (compression) ---
            P_ratio_lower = surface_pressure_ratio(
                theta_lower_eff, M_inf, g)
            P_lower = P_inf * P_ratio_lower

            # Pressure bleed correction for rounded LE (Eq. 12)
            # Integrate over strip rather than using midpoint
            if R_le > 0:
                sweep = np.arctan2(abs(y_half[j]), max(x_le[j], L * 0.01))
                chord = L - x_le[j]
                if chord > 1e-6:
                    n_bleed = 10  # integration points for bleed
                    x_bleed = np.linspace(0, chord, n_bleed + 1)
                    x_bleed_mid = 0.5 * (x_bleed[:-1] + x_bleed[1:])
                    dx_bleed = chord / n_bleed
                    bleed_avg = 0.0
                    for xb in x_bleed_mid:
                        xR = max(xb, 0.01) / R_le
                        bleed_avg += pressure_bleed_correction(
                            M_inf, alpha, sweep, xR) / n_bleed
                    P_lower += bleed_avg * P_inf

            # Lower surface gauge pressure × projected area
            dP_lower = P_lower - P_inf
            strip_area = chord * dy  # planform area of this strip

            # Force from lower surface pressure:
            #   Acts perpendicular to surface (into body = upward + forward)
            #   Lift component: dP * cos(theta_lower) * strip_area
            #   Drag component: dP * sin(theta_lower) * strip_area
            lift_lower = dP_lower * np.cos(theta_lower_eff) * strip_area
            drag_lower = dP_lower * np.sin(theta_lower_eff) * strip_area

            # --- Upper surface pressure (expansion or neutral) ---
            P_ratio_upper = surface_pressure_ratio(
                theta_upper_eff, M_inf, g)
            P_upper = P_inf * P_ratio_upper
            dP_upper = P_upper - P_inf

            # Upper surface force (pressure acts downward on upper surface)
            lift_upper = -dP_upper * np.cos(theta_upper_eff) * strip_area
            drag_upper = -dP_upper * np.sin(theta_upper_eff) * strip_area

            lift_total += lift_lower + lift_upper
            drag_inv_total += drag_lower + drag_upper

            # --- Viscous drag (both surfaces) ---
            for theta_surf, P_surf, label in [
                (theta_lower_eff, P_lower, 'lower'),
                (theta_upper_eff, P_upper, 'upper'),
            ]:
                P_ratio_s = P_surf / P_inf if P_inf > 0 else 1.0
                M_loc, T_loc = local_mach_temperature(
                    abs(theta_surf), M_inf, T_inf, P_ratio_s, g)

                # Integrate τ_w over the strip streamwise
                visc_drag_strip = 0.0
                for i in range(nx_strip):
                    x_from_le = x_stations[i] - x_le[j]
                    if x_from_le < 1e-6:
                        continue
                    # eckert_skin_friction returns τ_w [Pa]
                    tau_w = eckert_skin_friction(
                        M_loc, T_loc, P_surf, x_from_le, T_wall, g)
                    visc_drag_strip += tau_w * dx * dy

                drag_visc_total += visc_drag_strip

        # Double for full span (symmetric about y=0)
        lift_total *= 2.0
        drag_inv_total *= 2.0
        drag_visc_total *= 2.0

        # --- Base drag ---
        # Base area = integral of (z_upper - z_lower) at x=L over full span
        base_area = 0.0
        for j in range(ny_half):
            theta_local = T_star[j] * theta_wedge
            chord = L - x_le[j]
            height = np.tan(theta_local) * chord
            base_area += height * dy
        base_area *= 2.0  # full span

        P_base = P_inf / M_inf  # Maxwell slender-body
        drag_base = (P_base - P_inf) * base_area  # negative (P_base < P_inf)
        # Actually base pressure reduces drag (acts forward on base)
        # but we want D positive = retarding, so:
        drag_base = -(P_base - P_inf) * base_area  # positive contribution

        # --- LE drag (Modified Newtonian on blunted arc) ---
        drag_le = 0.0
        if R_le > 0:
            Cp_max_val = cp_max(M_inf, g)
            le_pts = np.column_stack([x_le, y_half, z_le])
            for j in range(ny_half - 1):
                ds = np.linalg.norm(le_pts[j+1] - le_pts[j])
                tangent = le_pts[j+1] - le_pts[j]
                tangent /= max(np.linalg.norm(tangent), 1e-12)
                cos_sweep_j = abs(tangent[1])  # y-component = spanwise

                # Local wedge angle determines the LE arc extent
                theta_local = T_star[j] * theta_wedge
                # The LE arc spans angle theta_local (from upper to lower
                # tangent). Drag integral over this arc:
                #   ∫₀^θ sin²(φ) dφ = (θ - sin(θ)cos(θ))/2
                # vs π/2 for a full semicircle.
                if theta_local > 1e-6:
                    arc_factor = (theta_local - np.sin(theta_local)
                                  * np.cos(theta_local)) / 2.0
                else:
                    arc_factor = 0.0
                drag_le += (R_le * Cp_max_val * cos_sweep_j**2
                            * q_inf * arc_factor * ds)
            drag_le *= 2.0  # full span

        # Total forces
        D_total = drag_inv_total + drag_visc_total + drag_base + drag_le
        L_total = lift_total

        CL = L_total / (q_inf * S_ref) if q_inf * S_ref > 0 else 0.0
        CD = D_total / (q_inf * S_ref) if q_inf * S_ref > 0 else 0.0
        L_over_D = L_total / D_total if abs(D_total) > 1e-10 else 0.0

        return {
            'CL': CL,
            'CD': CD,
            'L_over_D': L_over_D,
            'L': L_total,
            'D': D_total,
            'D_inviscid': drag_inv_total,
            'D_viscous': drag_visc_total,
            'D_base': drag_base,
            'D_le': drag_le,
            'S_ref': S_ref,
            'q_inf': q_inf,
            'M_inf': M_inf,
            'alpha_deg': alpha_deg,
            'altitude_km': altitude_km,
            'T_inf': T_inf,
            'P_inf': P_inf,
            'wedge_angle_deg': waverider.wedge_angle_deg,
            'n_strips': ny_half,
        }

    @staticmethod
    def _empty_results():
        return {
            'CL': 0.0, 'CD': 0.0, 'L_over_D': 0.0,
            'L': 0.0, 'D': 0.0, 'D_inviscid': 0.0, 'D_viscous': 0.0,
            'S_ref': 0.0, 'q_inf': 0.0, 'M_inf': 0.0,
            'alpha_deg': 0.0, 'altitude_km': 0.0,
            'T_inf': 0.0, 'P_inf': 0.0,
            'wedge_angle_deg': 0.0, 'n_panels': 0,
        }
