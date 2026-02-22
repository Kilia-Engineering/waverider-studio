"""
Stability Derivative Analysis for Cone-Derived Waveriders
==========================================================

Implements the perturbation-based stability derivative computation
from Adam Weaver's SHADOW thesis (Utah State, 2025).

Stability derivatives are computed via central finite differencing:
    Cl_beta  = (Cl(+delta_beta) - Cl(-delta_beta)) / (2*delta_beta)
    Cn_beta  = (Cn(+delta_beta) - Cn(-delta_beta)) / (2*delta_beta)
    Cm_alpha = (Cm(+delta_alpha) - Cm(-delta_alpha)) / (2*delta_alpha)

Stability criteria:
    - Pitch stable: Cm_alpha < 0  (negative restoring pitch moment)
    - Yaw stable:   Cn_beta > 0   (positive restoring yaw moment)
    - Roll stable:  Cl_beta < 0   (negative restoring roll moment)

Author: Adapted from Weaver thesis Appendices B & C for PySAGAS integration.
"""

import numpy as np
import os
import tempfile
from typing import Dict, Optional, Tuple

try:
    from pysagas.cfd import OPM
    from pysagas.cfd.solver import FlowSolver
    from pysagas.flow import FlowState
    from pysagas.geometry import Cell, DegenerateCell, Vector
    PYSAGAS_AVAILABLE = True
except ImportError:
    PYSAGAS_AVAILABLE = False


def _create_flow_state(mach: float, pressure: float, temperature: float,
                       aoa_deg: float = 0.0, beta_deg: float = 0.0) -> 'FlowState':
    """
    Create a PySAGAS FlowState with both angle of attack and sideslip.

    Parameters
    ----------
    mach : float
        Freestream Mach number
    pressure : float
        Freestream static pressure (Pa)
    temperature : float
        Freestream static temperature (K)
    aoa_deg : float
        Angle of attack in degrees
    beta_deg : float
        Sideslip angle in degrees

    Returns
    -------
    FlowState
        Flow state with appropriate direction vector
    """
    aoa = np.radians(aoa_deg)
    beta = np.radians(beta_deg)

    # Flow direction with both alpha and beta
    # x = streamwise, y = vertical (lift), z = spanwise
    direction = Vector(
        x=np.cos(aoa) * np.cos(beta),
        y=np.sin(aoa),
        z=np.cos(aoa) * np.sin(beta)
    )

    return FlowState(
        mach=mach,
        pressure=pressure,
        temperature=temperature,
        direction=direction
    )


def _extract_6dof_coefficients(solver, A_ref: float = 1.0,
                                c_ref: float = 1.0) -> Dict[str, float]:
    """
    Extract all 6 aerodynamic coefficients from a solved PySAGAS flow.

    Returns CL, CD, CY (side force), Cl (roll), Cm (pitch), Cn (yaw).

    Parameters
    ----------
    solver : OPM
        Solved PySAGAS OPM solver instance
    A_ref : float
        Reference area for coefficient non-dimensionalization
    c_ref : float
        Reference length for moment non-dimensionalization

    Returns
    -------
    dict
        Aerodynamic coefficients: CL, CD, CY, Cl, Cm, Cn, L/D
    """
    result = solver.flow_result
    q = result.freestream.q

    # Force coefficients via existing body_to_wind transform
    w = FlowSolver.body_to_wind(v=result.net_force, aoa=result.aoa)
    CL = w.y / (q * A_ref)
    CD = w.x / (q * A_ref)
    CY = result.net_force.z / (q * A_ref)  # Side force (body z)

    # Moment coefficients
    # Body-frame moments: x=roll, y=yaw, z=pitch
    net_m = result.net_moment
    Cl = net_m.x / (q * A_ref * c_ref)     # Roll moment
    Cn = net_m.y / (q * A_ref * c_ref)      # Yaw moment

    # Pitch uses wind-frame (existing convention with sign)
    mw = FlowSolver.body_to_wind(v=net_m, aoa=result.aoa)
    Cm = -mw.z / (q * A_ref * c_ref)

    LD = CL / CD if abs(CD) > 1e-10 else 0.0

    return {
        'CL': float(CL), 'CD': float(CD), 'CY': float(CY),
        'Cl': float(Cl), 'Cm': float(Cm), 'Cn': float(Cn),
        'L/D': float(LD)
    }


def _run_pysagas_at_condition(cells, mach: float, pressure: float,
                               temperature: float, aoa_deg: float = 0.0,
                               beta_deg: float = 0.0,
                               A_ref: float = 1.0, c_ref: float = 1.0,
                               save_vtk: str = None) -> Dict[str, float]:
    """
    Run PySAGAS OPM solver at a single flow condition.

    Parameters
    ----------
    cells : list[Cell]
        PySAGAS mesh cells
    mach : float
        Freestream Mach number
    pressure, temperature : float
        Atmospheric conditions (Pa, K)
    aoa_deg : float
        Angle of attack (degrees)
    beta_deg : float
        Sideslip angle (degrees)
    A_ref : float
        Reference area (m^2)
    c_ref : float
        Reference length (m)
    save_vtk : str, optional
        If provided, save VTK pressure distribution to this path prefix

    Returns
    -------
    dict
        6-DOF aerodynamic coefficients
    """
    flow = _create_flow_state(mach, pressure, temperature, aoa_deg, beta_deg)
    solver = OPM(cells=cells, freestream=flow, verbosity=0)
    solver.solve()

    if save_vtk:
        try:
            solver.save(save_vtk)
        except Exception:
            pass  # VTK export is optional, don't fail on it

    return _extract_6dof_coefficients(solver, A_ref=A_ref, c_ref=c_ref)


def compute_stability_derivatives(cells, mach: float, pressure: float,
                                   temperature: float, alpha_deg: float = 0.0,
                                   beta_deg: float = 0.0, delta_deg: float = 5.0,
                                   A_ref: float = 1.0, c_ref: float = 1.0,
                                   save_vtk_prefix: str = None) -> Dict:
    """
    Compute stability derivatives via central finite-difference perturbation.

    Implements the method from Weaver thesis (Ch. 4):
    - Runs 5 PySAGAS evaluations: baseline, alpha+/-delta, beta+/-delta
    - Computes three stability derivatives: Cm_alpha, Cl_beta, Cn_beta
    - Returns full baseline coefficients + derivatives + stability flags

    Parameters
    ----------
    cells : list[Cell]
        PySAGAS mesh cells (from STL or direct mesh)
    mach : float
        Freestream Mach number
    pressure : float
        Freestream static pressure (Pa)
    temperature : float
        Freestream static temperature (K)
    alpha_deg : float
        Baseline angle of attack (degrees)
    beta_deg : float
        Baseline sideslip angle (degrees)
    delta_deg : float
        Perturbation magnitude (degrees). Default: 5 (thesis value)
    A_ref : float
        Reference area (m^2)
    c_ref : float
        Reference length (m) for moment non-dimensionalization
    save_vtk_prefix : str, optional
        If provided, save VTK files for each condition to this directory

    Returns
    -------
    dict with keys:
        Baseline coefficients: CL, CD, CY, Cl, Cm, Cn, L/D
        Stability derivatives: Cm_alpha, Cl_beta, Cn_beta (per radian)
        Stability flags: pitch_stable, yaw_stable, roll_stable, fully_stable
        Perturbed coefficients: alpha_plus, alpha_minus, beta_plus, beta_minus
    """
    if not PYSAGAS_AVAILABLE:
        raise RuntimeError("PySAGAS is not available")

    delta_rad = np.radians(delta_deg)

    # Set up VTK save paths
    def _vtk_path(suffix):
        if save_vtk_prefix:
            return os.path.join(save_vtk_prefix, f"stability_{suffix}")
        return None

    # 1. Baseline condition
    baseline = _run_pysagas_at_condition(
        cells, mach, pressure, temperature, alpha_deg, beta_deg,
        A_ref=A_ref, c_ref=c_ref, save_vtk=_vtk_path("baseline"))

    # 2. Alpha + delta (for dCm/dalpha)
    alpha_plus = _run_pysagas_at_condition(
        cells, mach, pressure, temperature, alpha_deg + delta_deg, beta_deg,
        A_ref=A_ref, c_ref=c_ref, save_vtk=_vtk_path("alpha_plus"))

    # 3. Alpha - delta
    alpha_minus = _run_pysagas_at_condition(
        cells, mach, pressure, temperature, alpha_deg - delta_deg, beta_deg,
        A_ref=A_ref, c_ref=c_ref, save_vtk=_vtk_path("alpha_minus"))

    # 4. Beta + delta (for dCl/dbeta, dCn/dbeta)
    beta_plus = _run_pysagas_at_condition(
        cells, mach, pressure, temperature, alpha_deg, beta_deg + delta_deg,
        A_ref=A_ref, c_ref=c_ref, save_vtk=_vtk_path("beta_plus"))

    # 5. Beta - delta
    beta_minus = _run_pysagas_at_condition(
        cells, mach, pressure, temperature, alpha_deg, beta_deg - delta_deg,
        A_ref=A_ref, c_ref=c_ref, save_vtk=_vtk_path("beta_minus"))

    # Compute stability derivatives (per radian)
    Cm_alpha = (alpha_plus['Cm'] - alpha_minus['Cm']) / (2 * delta_rad)
    Cl_beta = (beta_plus['Cl'] - beta_minus['Cl']) / (2 * delta_rad)
    Cn_beta = (beta_plus['Cn'] - beta_minus['Cn']) / (2 * delta_rad)

    # Stability criteria (Weaver thesis)
    pitch_stable = Cm_alpha < 0
    yaw_stable = Cn_beta > 0
    roll_stable = Cl_beta < 0
    fully_stable = pitch_stable and yaw_stable and roll_stable

    return {
        # Baseline coefficients
        'CL': baseline['CL'],
        'CD': baseline['CD'],
        'CY': baseline['CY'],
        'Cl': baseline['Cl'],
        'Cm': baseline['Cm'],
        'Cn': baseline['Cn'],
        'L/D': baseline['L/D'],

        # Stability derivatives (per radian)
        'Cm_alpha': float(Cm_alpha),
        'Cl_beta': float(Cl_beta),
        'Cn_beta': float(Cn_beta),

        # Stability flags
        'pitch_stable': bool(pitch_stable),
        'yaw_stable': bool(yaw_stable),
        'roll_stable': bool(roll_stable),
        'fully_stable': bool(fully_stable),

        # Perturbed data (for debugging/analysis)
        'alpha_plus': alpha_plus,
        'alpha_minus': alpha_minus,
        'beta_plus': beta_plus,
        'beta_minus': beta_minus,
    }


def cells_from_stl(stl_file: str):
    """
    Load an STL file and create PySAGAS Cell objects.

    Uses meshio for Windows-safe single-threaded loading
    (avoids PySAGAS's multiprocessing STL loader).

    Parameters
    ----------
    stl_file : str
        Path to STL file

    Returns
    -------
    list[Cell]
        PySAGAS Cell objects
    """
    import meshio

    mesh = meshio.read(stl_file)
    points = mesh.points

    triangles = None
    for cell_block in mesh.cells:
        if cell_block.type == 'triangle':
            triangles = cell_block.data
            break

    if triangles is None:
        raise ValueError("No triangles found in STL file")

    cells = []
    for tri in triangles:
        p0, p1, p2 = points[tri[0]], points[tri[1]], points[tri[2]]
        v0 = Vector(x=float(p0[0]), y=float(p0[1]), z=float(p0[2]))
        v1 = Vector(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))
        v2 = Vector(x=float(p2[0]), y=float(p2[1]), z=float(p2[2]))
        try:
            cells.append(Cell.from_points([v0, v1, v2]))
        except DegenerateCell:
            continue

    return cells


def cells_from_waverider(wr) -> list:
    """
    Create PySAGAS Cell objects directly from a ShadowWaverider mesh.

    Parameters
    ----------
    wr : ShadowWaverider
        Waverider object with generated geometry

    Returns
    -------
    list[Cell]
        PySAGAS Cell objects
    """
    verts, tris = wr.get_mesh()

    cells = []
    for tri in tris:
        p0, p1, p2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        v0 = Vector(x=float(p0[0]), y=float(p0[1]), z=float(p0[2]))
        v1 = Vector(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))
        v2 = Vector(x=float(p2[0]), y=float(p2[1]), z=float(p2[2]))
        try:
            cells.append(Cell.from_points([v0, v1, v2]))
        except DegenerateCell:
            continue

    return cells


def waverider_stl_to_temp(wr) -> str:
    """
    Write a ShadowWaverider mesh to a temporary STL file.

    Parameters
    ----------
    wr : ShadowWaverider
        Waverider object

    Returns
    -------
    str
        Path to temporary STL file (caller must clean up)
    """
    verts, tris = wr.get_mesh()
    temp_stl = tempfile.mktemp(suffix='.stl')

    with open(temp_stl, 'w') as f:
        f.write("solid waverider\n")
        for tri in tris:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            n = np.cross(edge1, edge2)
            norm = np.linalg.norm(n)
            if norm > 1e-10:
                n = n / norm
            else:
                n = np.array([0, 0, 1])
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid waverider\n")

    return temp_stl


def analyze_waverider_stability(wr, mach: float, pressure: float = 101325.0,
                                 temperature: float = 288.15,
                                 alpha_deg: float = 0.0, beta_deg: float = 0.0,
                                 delta_deg: float = 5.0,
                                 save_vtk_prefix: str = None) -> Dict:
    """
    High-level function to analyze stability of a ShadowWaverider.

    Generates mesh from waverider, runs PySAGAS at 5 conditions,
    and returns stability derivatives with properly computed A_ref and c_ref.

    Parameters
    ----------
    wr : ShadowWaverider
        Waverider geometry object
    mach : float
        Freestream Mach number
    pressure : float
        Atmospheric pressure (Pa)
    temperature : float
        Atmospheric temperature (K)
    alpha_deg : float
        Angle of attack (degrees)
    beta_deg : float
        Sideslip angle (degrees)
    delta_deg : float
        Perturbation angle for finite differencing (degrees)
    save_vtk_prefix : str, optional
        Directory to save VTK files

    Returns
    -------
    dict
        Full stability analysis results (see compute_stability_derivatives)
    """
    if not PYSAGAS_AVAILABLE:
        raise RuntimeError("PySAGAS is not available")

    # Create cells directly from waverider mesh
    cells = cells_from_waverider(wr)

    # Use waverider's computed reference values
    A_ref = getattr(wr, 'planform_area', 1.0) or 1.0
    c_ref = getattr(wr, 'mac', 1.0) or 1.0

    return compute_stability_derivatives(
        cells=cells,
        mach=mach,
        pressure=pressure,
        temperature=temperature,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        delta_deg=delta_deg,
        A_ref=A_ref,
        c_ref=c_ref,
        save_vtk_prefix=save_vtk_prefix
    )
