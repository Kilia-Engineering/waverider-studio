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


def compute_geometric_sensitivities(
    cells_baseline: list,
    wr_baseline,
    mach: float,
    shock_angle: float,
    poly_order: int,
    x: np.ndarray,
    n_le: int = 15,
    n_stream: int = 15,
    eps: float = 1e-5,
) -> list:
    """
    Compute vertex-to-parameter sensitivities (dvdp) via central finite
    differencing on the waverider geometry, then populate each cell's
    dndp, dAdp, dcdp using PySAGAS's analytic chain rule.

    Parameters
    ----------
    cells_baseline : list[Cell]
        PySAGAS cells at the baseline design (from cells_from_waverider).
    wr_baseline : ShadowWaverider
        Baseline waverider object.
    mach, shock_angle : float
        Flow conditions.
    poly_order : int
        Polynomial order (2 or 3).
    x : ndarray
        Baseline design variable vector [A2, A0] or [A3, A2, A0].
    n_le, n_stream : int
        Mesh resolution (must match baseline).
    eps : float
        Finite-difference step size for each design parameter.

    Returns
    -------
    list[Cell]
        The same cells with geometric sensitivities (dndp, dAdp, dcdp) attached.
    """
    from shadow_waverider import (
        create_second_order_waverider,
        create_third_order_waverider,
    )

    n_params = len(x)
    verts_baseline, tris_baseline = wr_baseline.get_mesh()

    # For each design parameter, compute vertex perturbations
    # dvdp_all[i] has shape (n_verts, 3) = dv/dp_i for parameter i
    dvdp_all = []
    for p_i in range(n_params):
        x_plus = x.copy()
        x_plus[p_i] += eps
        x_minus = x.copy()
        x_minus[p_i] -= eps

        try:
            if poly_order == 2:
                wr_plus = create_second_order_waverider(
                    mach=mach, shock_angle=shock_angle,
                    A2=x_plus[0], A0=x_plus[1],
                    n_leading_edge=n_le, n_streamwise=n_stream)
                wr_minus = create_second_order_waverider(
                    mach=mach, shock_angle=shock_angle,
                    A2=x_minus[0], A0=x_minus[1],
                    n_leading_edge=n_le, n_streamwise=n_stream)
            else:
                wr_plus = create_third_order_waverider(
                    mach=mach, shock_angle=shock_angle,
                    A3=x_plus[0], A2=x_plus[1], A0=x_plus[2],
                    n_leading_edge=n_le, n_streamwise=n_stream)
                wr_minus = create_third_order_waverider(
                    mach=mach, shock_angle=shock_angle,
                    A3=x_minus[0], A2=x_minus[1], A0=x_minus[2],
                    n_leading_edge=n_le, n_streamwise=n_stream)

            verts_plus, _ = wr_plus.get_mesh()
            verts_minus, _ = wr_minus.get_mesh()

            # Central difference: dv/dp = (v+ - v-) / (2*eps)
            dv_dp = (verts_plus - verts_minus) / (2.0 * eps)
            dvdp_all.append(dv_dp)

        except Exception:
            # If perturbation fails, use zero sensitivity
            dvdp_all.append(np.zeros_like(verts_baseline))

    # Now attach per-cell dvdp and compute dndp, dAdp, dcdp via chain rule
    for cell_idx, tri in enumerate(tris_baseline):
        if cell_idx >= len(cells_baseline):
            break

        cell = cells_baseline[cell_idx]

        # Build dvdp matrix for this cell: shape (9, n_params)
        # 9 rows = [p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z]
        dvdp = np.zeros((9, n_params))
        for p_i in range(n_params):
            dv = dvdp_all[p_i]
            # tri[0], tri[1], tri[2] are vertex indices
            dvdp[0:3, p_i] = dv[tri[0]]  # dp0/dp_i
            dvdp[3:6, p_i] = dv[tri[1]]  # dp1/dp_i
            dvdp[6:9, p_i] = dv[tri[2]]  # dp2/dp_i

        # Use PySAGAS's analytic chain rule:
        # dndp = dndv @ dvdp, dAdp = dAdv @ dvdp, dcdp = dcdv @ dvdp
        cell._add_sensitivities(dvdp)

    return cells_baseline


def compute_shape_sensitivities(
    wr,
    mach: float,
    shock_angle: float,
    poly_order: int,
    x: np.ndarray,
    pressure: float = 101325.0,
    temperature: float = 288.15,
    alpha_deg: float = 0.0,
    n_le: int = 15,
    n_stream: int = 15,
    eps: float = 1e-5,
    save_vtk: str = None,
) -> Dict:
    """
    Compute full PySAGAS shape sensitivities (dF/dp, dM/dp) for a waverider
    at the given design point.

    Uses:
    1. Geometric sensitivities computed via finite differencing of the
       waverider parametric geometry.
    2. PySAGAS OPM.solve_sens() for analytic flow sensitivity computation.

    Parameters
    ----------
    wr : ShadowWaverider
        Waverider at the design point.
    mach, shock_angle : float
        Flow conditions.
    poly_order : int
        Polynomial order (2 or 3).
    x : ndarray
        Design variable vector [A2, A0] or [A3, A2, A0].
    pressure, temperature : float
        Freestream conditions.
    alpha_deg : float
        Angle of attack (degrees).
    n_le, n_stream : int
        Mesh resolution.
    eps : float
        FD step for geometric sensitivities.
    save_vtk : str, optional
        If provided, save sensitivity VTK to this path prefix.

    Returns
    -------
    dict with keys:
        f_sens : DataFrame  - dF/dp (3 x n_params)
        m_sens : DataFrame  - dM/dp (3 x n_params)
        cells  : list[Cell] - cells with per-cell sensitivities attached
        parameters : list[str] - parameter names
    """
    if not PYSAGAS_AVAILABLE:
        raise RuntimeError("PySAGAS is not available")

    # Create cells and attach geometric sensitivities
    cells = cells_from_waverider(wr)
    cells = compute_geometric_sensitivities(
        cells_baseline=cells,
        wr_baseline=wr,
        mach=mach,
        shock_angle=shock_angle,
        poly_order=poly_order,
        x=x,
        n_le=n_le,
        n_stream=n_stream,
        eps=eps,
    )

    # Parameter names
    if poly_order == 2:
        param_names = ['A2', 'A0']
    else:
        param_names = ['A3', 'A2', 'A0']

    # Create flow state and solver
    flow = _create_flow_state(mach, pressure, temperature, alpha_deg)
    solver = OPM(cells=cells, freestream=flow, verbosity=0)

    # Solve the nominal flow first
    solver.solve()

    # Compute sensitivities using PySAGAS's solve_sens
    # Since cells already have dndp/dAdp/dcdp, use cells_have_sens_data=True
    sens_result = solver.solve_sens(
        cells_have_sens_data=True,
        parameters=param_names,
    )

    # Export sensitivity VTK if requested
    if save_vtk:
        try:
            _export_sensitivity_vtk(
                cells=cells, param_names=param_names,
                save_path=save_vtk,
            )
        except Exception:
            pass

    return {
        'f_sens': sens_result.f_sens,
        'm_sens': sens_result.m_sens,
        'cells': cells,
        'parameters': param_names,
    }


def _export_sensitivity_vtk(cells: list, param_names: list, save_path: str):
    """
    Export per-cell flow data and force sensitivities to VTK.

    Saves pressure, Mach, temperature, and dFx/dp, dFy/dp, dFz/dp
    for each design parameter as cell data fields.

    Parameters
    ----------
    cells : list[Cell]
        Cells with flowstate and sensitivities computed.
    param_names : list[str]
        Design parameter names.
    save_path : str
        Output file path (without extension, .vtu will be appended).
    """
    import meshio

    # Collect vertices and faces
    vertex_map = {}  # vertex_id -> coordinates
    faces = []

    # Collect cell data
    pressure_data = []
    mach_data = []
    temperature_data = []
    cp_data = []

    # Sensitivity data per parameter
    sens_data = {p: {'dFx': [], 'dFy': [], 'dFz': []} for p in param_names}
    force_mag_sens = {p: [] for p in param_names}

    for cell in cells:
        # Faces
        if cell._face_ids is not None:
            faces.append(cell._face_ids)
            for i, vid in enumerate(cell._face_ids):
                vertex_map[vid] = cell.vertices[i]
        else:
            # No face_ids, use vertex positions as unique keys
            base_id = len(vertex_map)
            face = [base_id, base_id + 1, base_id + 2]
            faces.append(face)
            for i in range(3):
                vertex_map[base_id + i] = cell.vertices[i]

        # Flow data
        if cell.flowstate is not None:
            pressure_data.append(cell.flowstate.P)
            mach_data.append(cell.flowstate.M)
            temperature_data.append(cell.flowstate.T)
            # Cp = (P - P_inf) / q_inf
            q = cell.flowstate.q if hasattr(cell.flowstate, 'q') else 1.0
            cp_data.append(cell.attributes.get('pressure', cell.flowstate.P))
        else:
            pressure_data.append(0.0)
            mach_data.append(0.0)
            temperature_data.append(0.0)
            cp_data.append(0.0)

        # Sensitivity data
        if cell.sensitivities is not None:
            for p_i, p_name in enumerate(param_names):
                if p_i < cell.sensitivities.shape[0]:
                    sens_data[p_name]['dFx'].append(cell.sensitivities[p_i, 0])
                    sens_data[p_name]['dFy'].append(cell.sensitivities[p_i, 1])
                    sens_data[p_name]['dFz'].append(cell.sensitivities[p_i, 2])
                    force_mag_sens[p_name].append(
                        np.linalg.norm(cell.sensitivities[p_i, :]))
                else:
                    sens_data[p_name]['dFx'].append(0.0)
                    sens_data[p_name]['dFy'].append(0.0)
                    sens_data[p_name]['dFz'].append(0.0)
                    force_mag_sens[p_name].append(0.0)
        else:
            for p_name in param_names:
                sens_data[p_name]['dFx'].append(0.0)
                sens_data[p_name]['dFy'].append(0.0)
                sens_data[p_name]['dFz'].append(0.0)
                force_mag_sens[p_name].append(0.0)

    # Build mesh
    sorted_verts = dict(sorted(vertex_map.items()))
    vertices = np.array([v for v in sorted_verts.values()])
    faces_arr = np.array(faces)

    # Build cell_data dictionary
    cell_data = {
        'pressure': [np.array(pressure_data)],
        'Mach': [np.array(mach_data)],
        'temperature': [np.array(temperature_data)],
    }

    for p_name in param_names:
        cell_data[f'dFx_d{p_name}'] = [np.array(sens_data[p_name]['dFx'])]
        cell_data[f'dFy_d{p_name}'] = [np.array(sens_data[p_name]['dFy'])]
        cell_data[f'dFz_d{p_name}'] = [np.array(sens_data[p_name]['dFz'])]
        cell_data[f'dF_mag_d{p_name}'] = [np.array(force_mag_sens[p_name])]

    mesh = meshio.Mesh(
        points=vertices,
        cells=[("triangle", faces_arr)],
        cell_data=cell_data,
    )
    meshio.write(f"{save_path}_sensitivities.vtu", mesh)
