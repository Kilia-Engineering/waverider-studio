"""
Aerodynamic corrections for leading edge blunting.

Implements:
- Modified Newtonian pressure coefficient
- Pressure bleed correction (Guo et al. 2022 / Jessen 2025 empirical model)
- Stagnation-point heating estimate (Fay-Riddell approximation)
- Blunted LE drag increment estimation

References:
  [1] Guo et al. (2022) — "Blunted waverider design for hypersonic flight"
  [2] Jessen (2025) — PhD thesis, Chapter 4: Blunting corrections
  [3] Fay & Riddell (1958) — Stagnation heating correlation
  [4] Anderson (2006) — Hypersonic and High-Temperature Gas Dynamics
"""

import numpy as np


# ---------- constants ----------
GAMMA = 1.4  # ratio of specific heats for air


def _cp_max(M_inf):
    """Maximum pressure coefficient (Rayleigh pitot formula).

    For M >> 1 this approaches (4 / (gamma+1)) * (1 / M^2) ... but we use
    the exact compressible form.
    """
    g = GAMMA
    # Stagnation pressure ratio across normal shock (Rayleigh pitot)
    term1 = ((g + 1) ** 2 * M_inf ** 2 / (4 * g * M_inf ** 2 - 2 * (g - 1))) ** (g / (g - 1))
    term2 = (1 - g + 2 * g * M_inf ** 2) / (g + 1)
    p02_p1 = term1 * term2
    # Cp_max = (2 / (gamma * M^2)) * (p02/p1 - 1)
    return 2.0 / (g * M_inf ** 2) * (p02_p1 - 1.0)


def modified_newtonian_cp(M_inf, theta):
    """Modified Newtonian pressure coefficient.

    Cp = Cp_max * sin^2(theta)

    Parameters
    ----------
    M_inf : float
        Freestream Mach number.
    theta : float or ndarray
        Local surface inclination angle in radians (body angle to freestream).

    Returns
    -------
    float or ndarray
        Pressure coefficient Cp.
    """
    return _cp_max(M_inf) * np.sin(theta) ** 2


def pressure_bleed_correction(M_inf, alpha_deg, sweep_deg, x_over_R):
    """Pressure bleed factor behind a blunted leading edge.

    Based on the empirical model from Guo et al. (2022) and Jessen (2025):
    the blunted LE creates a local bow shock that raises surface pressure
    above the sharp-body inviscid value. This excess decays downstream.

    Delta_Cp / Cp_max ~ A * exp(-B * x/R)

    where A, B depend on Mach, alpha, and sweep.

    Parameters
    ----------
    M_inf : float
        Freestream Mach number.
    alpha_deg : float
        Angle of attack in degrees.
    sweep_deg : float or ndarray
        Local LE sweep angle in degrees.
    x_over_R : float or ndarray
        Downstream distance normalised by local LE radius (x/R).

    Returns
    -------
    float or ndarray
        Fractional pressure increment Delta_Cp / Cp_max (dimensionless).
    """
    alpha = np.radians(alpha_deg)
    sweep = np.radians(np.asarray(sweep_deg, dtype=float))

    # Effective normal Mach component (controls bow shock strength)
    M_n = M_inf * np.cos(sweep) * np.cos(alpha)
    M_n = np.maximum(M_n, 1.01)  # clamp to just above sonic

    # Empirical coefficients (fitted from Guo et al. 2022 Fig. 12)
    A = 0.85 * (1.0 - np.exp(-0.5 * (M_n - 1.0)))
    B = 0.30 + 0.10 * (M_n - 3.0).clip(0, 5)

    x_R = np.asarray(x_over_R, dtype=float)
    return A * np.exp(-B * x_R)


def stagnation_heating(M_inf, R_nose, rho_inf, V_inf, T_wall=300.0):
    """Stagnation-point heat flux using Fay-Riddell correlation.

    q_stag ~ (C / sqrt(R)) * sqrt(rho_inf) * V_inf^3

    Simplified form for engineering estimates (cold wall, equilibrium air).

    Parameters
    ----------
    M_inf : float
        Freestream Mach number (for reference, not directly used).
    R_nose : float
        Nose / LE radius in metres.
    rho_inf : float
        Freestream density in kg/m^3.
    V_inf : float
        Freestream velocity in m/s.
    T_wall : float
        Wall temperature in K (default 300 K cold wall).

    Returns
    -------
    float
        Stagnation-point heat flux in W/m^2.
    """
    # Fay-Riddell constant for equilibrium air (Anderson eqn 6.65 simplified)
    C_FR = 1.83e-4  # (kg^0.5 / m)
    R_nose = max(R_nose, 1e-6)  # prevent division by zero
    q_stag = C_FR / np.sqrt(R_nose) * np.sqrt(rho_inf) * V_inf ** 3
    return q_stag


def blunted_le_drag_increment(M_inf, le_points, radius_dist, sweep_angles_deg,
                               alpha_deg=0.0):
    """Estimate drag increment from blunted leading edge.

    Integrates modified Newtonian pressure over the blunted LE cylindrical
    cross-sections, accounting for sweep.

    Parameters
    ----------
    M_inf : float
        Freestream Mach number.
    le_points : ndarray, shape (n, 3)
        Leading edge point coordinates.
    radius_dist : ndarray, shape (n,)
        Local LE radius at each station.
    sweep_angles_deg : ndarray, shape (n,)
        Local LE sweep angle at each station in degrees.
    alpha_deg : float
        Angle of attack in degrees.

    Returns
    -------
    dict
        'delta_CD' : float — drag coefficient increment (referenced to LE
                     frontal area)
        'drag_force_per_q' : float — drag force / dynamic pressure (m^2),
                     i.e. the drag area increment
        'spanwise_dCD' : ndarray — local drag per unit span at each station
    """
    n = len(le_points)
    if n < 2:
        return {'delta_CD': 0.0, 'drag_force_per_q': 0.0, 'spanwise_dCD': np.zeros(1)}

    Cp_max = _cp_max(M_inf)
    alpha = np.radians(alpha_deg)
    sweep = np.radians(np.asarray(sweep_angles_deg, dtype=float))

    # At each span station, the blunted LE is a swept cylinder
    # Effective normal velocity component
    cos_sweep = np.cos(sweep)

    # Drag per unit span of a swept circular cylinder in modified Newtonian flow:
    # dD/ds = R * Cp_max * cos^2(sweep) * cos(alpha) * q_inf * (pi/2 integration)
    # The pi/2 factor comes from integrating sin^2(theta) over the front face
    # of a cylinder from -pi/2 to pi/2: integral = pi/2
    dD_per_q_per_ds = radius_dist * Cp_max * cos_sweep ** 2 * np.cos(alpha) * (np.pi / 2.0)

    # Compute span-wise arc-length increments
    diffs = np.diff(le_points, axis=0)
    ds = np.linalg.norm(diffs, axis=1)

    # Trapezoidal integration
    dD_avg = 0.5 * (dD_per_q_per_ds[:-1] + dD_per_q_per_ds[1:])
    drag_area = np.sum(dD_avg * ds)

    return {
        'delta_CD': float(drag_area),  # drag area in m^2 (divide by ref area for CD)
        'drag_force_per_q': float(drag_area),
        'spanwise_dCD': dD_per_q_per_ds,
    }


def compute_sweep_angles(le_points):
    """Compute local sweep angle at each LE station.

    Sweep is measured as the angle between the local LE tangent and
    the Z-axis (spanwise direction).

    Parameters
    ----------
    le_points : ndarray, shape (n, 3)
        Leading edge coordinates (X=streamwise, Y=vertical, Z=spanwise).

    Returns
    -------
    ndarray, shape (n,)
        Local sweep angle in degrees at each station.
    """
    n = len(le_points)
    if n < 2:
        return np.zeros(n)

    tangents = np.zeros_like(le_points)
    # Central differences for interior, forward/backward for endpoints
    tangents[1:-1] = le_points[2:] - le_points[:-2]
    tangents[0] = le_points[1] - le_points[0]
    tangents[-1] = le_points[-1] - le_points[-2]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    tangents = tangents / norms

    # Sweep = angle between tangent and Z-axis
    # cos(sweep) = |tangent . z_hat|
    cos_sweep = np.abs(tangents[:, 2])
    cos_sweep = np.clip(cos_sweep, 0.0, 1.0)
    sweep_deg = np.degrees(np.arccos(cos_sweep))

    return sweep_deg
