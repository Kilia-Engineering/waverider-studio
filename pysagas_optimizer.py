"""
Gradient-Based Optimizer for SHADOW Waveriders
================================================

Uses finite-difference gradients with PySAGAS OPM solver to perform
gradient-based optimization of cone-derived waverider leading-edge
polynomial coefficients.

Matches the thesis (Weaver, 2025) approach of using scipy.optimize with
stability constraints, but uses PySAGAS instead of HI-Mach.

Supports:
- Maximize L/D (or minimize CD, maximize CL)
- Stability constraints: Cm_alpha < 0, Cn_beta > 0, Cl_beta < 0
- Volume and geometry constraints
- VTK export at each iteration for pressure visualization
- Full convergence history logging

Author: Adapted from Weaver thesis Appendix C for PySAGAS integration.
"""

import os
import time
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Callable
from scipy.optimize import minimize, NonlinearConstraint

from shadow_waverider import (
    ShadowWaverider, create_second_order_waverider,
    create_third_order_waverider
)

try:
    from stability_analysis import (
        compute_stability_derivatives, cells_from_waverider,
        cells_from_stl, _run_pysagas_at_condition
    )
    STABILITY_AVAILABLE = True
except ImportError:
    STABILITY_AVAILABLE = False


class ShadowOptimizer:
    """
    Gradient-based optimizer for SHADOW cone-derived waveriders.

    Uses scipy.optimize.minimize with finite-difference gradients
    computed via PySAGAS OPM solver.

    Parameters
    ----------
    mach : float
        Freestream Mach number
    shock_angle : float
        Shock cone angle (degrees)
    poly_order : int
        Polynomial order (2 or 3)
    pressure : float
        Freestream pressure (Pa)
    temperature : float
        Freestream temperature (K)
    alpha_deg : float
        Angle of attack (degrees)
    objective : str
        Optimization objective: 'L/D', '-CD', 'CL'
    method : str
        scipy.optimize method: 'SLSQP', 'COBYLA', 'Nelder-Mead'
    stability_constrained : bool
        If True, enforce Cm_alpha < 0, Cn_beta > 0, Cl_beta < 0
    volume_min : float
        Minimum volume constraint (0 = no constraint)
    save_vtk : bool
        Save VTK pressure files at each iteration
    output_dir : str
        Output directory for results
    n_le : int
        Number of leading edge points
    n_stream : int
        Number of streamwise points
    verbose : bool
        Print progress during optimization
    """

    def __init__(
        self,
        mach: float = 6.0,
        shock_angle: float = 12.0,
        poly_order: int = 2,
        pressure: float = 101325.0,
        temperature: float = 288.15,
        alpha_deg: float = 0.0,
        objective: str = 'L/D',
        method: str = 'SLSQP',
        stability_constrained: bool = False,
        volume_min: float = 0.0,
        save_vtk: bool = True,
        output_dir: str = 'optimization_results',
        n_le: int = 15,
        n_stream: int = 15,
        verbose: bool = True,
        mesh_min: float = 0.005,
        mesh_max: float = 0.05,
        save_geometry_vtk: bool = True
    ):
        self.mach = mach
        self.shock_angle = shock_angle
        self.poly_order = poly_order
        self.pressure = pressure
        self.temperature = temperature
        self.alpha_deg = alpha_deg
        self.objective = objective
        self.method = method
        self.stability_constrained = stability_constrained
        self.volume_min = volume_min
        self.save_vtk = save_vtk
        self.output_dir = output_dir
        self.n_le = n_le
        self.n_stream = n_stream
        self.verbose = verbose
        self.mesh_min = mesh_min
        self.mesh_max = mesh_max
        self.save_geometry_vtk = save_geometry_vtk

        # Convergence history
        self.history = []
        self.iteration = 0
        self.best_result = None

        # Evaluation cache to avoid redundant PySAGAS calls
        # (SLSQP evaluates constraints at the same point as objective)
        self._eval_cache = {}
        self._last_good_obj = 100.0  # Tracks last successful objective for smooth penalties

        # Callback for GUI progress updates
        self.progress_callback = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def _create_waverider(self, x: np.ndarray) -> ShadowWaverider:
        """Create a ShadowWaverider from design variable vector."""
        if self.poly_order == 2:
            A2, A0 = x
            return create_second_order_waverider(
                mach=self.mach, shock_angle=self.shock_angle,
                A2=A2, A0=A0, n_leading_edge=self.n_le,
                n_streamwise=self.n_stream)
        else:
            A3, A2, A0 = x
            return create_third_order_waverider(
                mach=self.mach, shock_angle=self.shock_angle,
                A3=A3, A2=A2, A0=A0, n_leading_edge=self.n_le,
                n_streamwise=self.n_stream)

    def _evaluate(self, x: np.ndarray, compute_stability: bool = False) -> Dict:
        """
        Evaluate a design point: generate waverider, mesh with Gmsh, run PySAGAS.

        Parameters
        ----------
        x : array
            Design variables [A2, A0] or [A3, A2, A0]
        compute_stability : bool
            If True, compute stability derivatives (5 runs instead of 1)

        Returns
        -------
        dict
            Results including CL, CD, Cm, L/D, and optionally stability derivatives
        """
        # Check evaluation cache to avoid redundant PySAGAS calls
        cache_key = (tuple(np.round(x, 10)), compute_stability)
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        try:
            wr = self._create_waverider(x)
        except Exception as e:
            return {'success': False, 'error': f'Geometry failed: {e}'}

        # Generate cells via STEP → Gmsh → STL pipeline
        try:
            iter_tag = f'iter_{self.iteration:04d}'
            step_path = os.path.join(self.output_dir, f'{iter_tag}.step')
            stl_path = os.path.join(self.output_dir, f'{iter_tag}.stl')

            self._export_step(wr, step_path)
            self._run_gmsh(step_path, stl_path)
            cells = cells_from_stl(stl_path)
        except Exception as e:
            # Fallback to direct triangulation if Gmsh pipeline fails
            if self.verbose:
                print(f"  Gmsh pipeline failed ({e}), falling back to direct mesh")
            cells = cells_from_waverider(wr)

        if len(cells) == 0:
            return {'success': False, 'error': 'All mesh cells are degenerate'}
        A_ref = max(wr.planform_area, 1e-10)
        c_ref = max(wr.mac, 1e-6)

        vtk_prefix = None
        if self.save_vtk:
            vtk_prefix = os.path.join(self.output_dir, f'iter_{self.iteration:04d}')
            os.makedirs(vtk_prefix, exist_ok=True)

        try:
            if compute_stability and STABILITY_AVAILABLE:
                result = compute_stability_derivatives(
                    cells=cells, mach=self.mach, pressure=self.pressure,
                    temperature=self.temperature, alpha_deg=self.alpha_deg,
                    A_ref=A_ref, c_ref=c_ref, save_vtk_prefix=vtk_prefix)
            else:
                aero = _run_pysagas_at_condition(
                    cells, self.mach, self.pressure, self.temperature,
                    aoa_deg=self.alpha_deg, A_ref=A_ref, c_ref=c_ref,
                    save_vtk=os.path.join(vtk_prefix, 'pressure') if vtk_prefix else None)
                result = dict(aero)
                result['Cm_alpha'] = 0.0
                result['Cl_beta'] = 0.0
                result['Cn_beta'] = 0.0

            result['success'] = True
            result['volume'] = wr.volume
            result['planform_area'] = wr.planform_area
            result['vol_efficiency'] = (wr.volume ** (2.0/3.0)) / wr.planform_area if wr.planform_area > 0 else 0.0
            result['mac'] = wr.mac

        except Exception as e:
            result = {'success': False, 'error': f'PySAGAS failed: {e}'}
            self._eval_cache[cache_key] = result
            return result

        # Save geometry VTK with flow data (after PySAGAS run)
        if self.save_geometry_vtk:
            try:
                vtk_path = os.path.join(
                    self.output_dir, f'geometry_{self.iteration:04d}.vtu')
                self._export_geometry_vtk(wr, vtk_path, cells=cells)
            except Exception:
                pass  # Geometry VTK is optional

        self._eval_cache[cache_key] = result
        return result

    def _export_step(self, wr, step_path):
        """Export ShadowWaverider to STEP via NURBS solid."""
        import cadquery as cq
        from waverider_generator.cad_export import build_waverider_solid
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing

        upper = wr.upper_surface
        lower = wr.lower_surface
        n_le = upper.shape[0]
        center = n_le // 2

        upper_half = upper[center:, :, :]
        lower_half = lower[center:, :, :]

        le_curve = upper_half[:, 0, :]
        cl_upper = upper_half[0, :, :]
        cl_lower = lower_half[0, :, :]
        te_upper = upper_half[:, -1, :]
        te_lower = lower_half[:, -1, :]
        upper_streams = [upper_half[i] for i in range(upper_half.shape[0])]
        lower_streams = [lower_half[i] for i in range(lower_half.shape[0])]

        right = build_waverider_solid(
            upper_streams, lower_streams, le_curve,
            cl_upper, cl_lower, te_upper, te_lower)
        right = right.scale(1000.0)  # m -> mm for STEP

        left = right.mirror(mirrorPlane='XY')

        sew = BRepBuilderAPI_Sewing(1e-2)
        sew.Add(right.wrapped)
        sew.Add(left.wrapped)
        sew.Perform()
        result = cq.Shape(sew.SewedShape())
        cq.exporters.export(cq.Workplane("XY").newObject([result]), step_path)

    def _run_gmsh(self, step_path, stl_path):
        """Mesh STEP file with Gmsh -> STL."""
        from optimization_engine import generate_mesh_minmax
        # Convert mesh sizes from meters to mm (STEP is in mm)
        generate_mesh_minmax(
            step_path, stl_path,
            self.mesh_min * 1000.0, self.mesh_max * 1000.0)

    def _export_geometry_vtk(self, wr, vtk_path, cells=None):
        """Save waverider triangular mesh as VTK for ParaView animation.

        If solved PySAGAS cells are provided, includes per-cell flow data
        (pressure, Mach, temperature, Cp) on the geometry mesh.
        """
        import meshio
        verts, tris = wr.get_mesh()

        cell_data = {}
        if cells is not None:
            # Map flow data from PySAGAS cells onto the geometry triangles.
            # cells_from_waverider() produces cells in the same order as
            # get_mesh() triangles, so the indices correspond directly.
            pressure = []
            mach_field = []
            temperature = []
            for i, tri in enumerate(tris):
                if i < len(cells) and cells[i].flowstate is not None:
                    pressure.append(cells[i].flowstate.P)
                    mach_field.append(cells[i].flowstate.M)
                    temperature.append(cells[i].flowstate.T)
                else:
                    pressure.append(0.0)
                    mach_field.append(0.0)
                    temperature.append(0.0)
            cell_data['pressure'] = [np.array(pressure)]
            cell_data['Mach'] = [np.array(mach_field)]
            cell_data['temperature'] = [np.array(temperature)]

        mesh = meshio.Mesh(
            points=verts,
            cells=[("triangle", tris)],
            cell_data=cell_data if cell_data else {},
        )
        meshio.write(vtk_path, mesh)

    def _objective_function(self, x: np.ndarray) -> float:
        """Objective function for scipy.optimize."""
        self.iteration += 1
        start_time = time.time()

        result = self._evaluate(x, compute_stability=self.stability_constrained)

        if not result.get('success', False):
            if self.verbose:
                print(f"  Iter {self.iteration}: FAILED - {result.get('error', 'unknown')}")
            # Use smooth penalty near the last good value to avoid corrupting
            # finite-difference gradient estimates (1e6 creates discontinuities)
            return self._last_good_obj + 10.0

        # Compute objective value (always minimize)
        CL = result.get('CL', 0)
        CD = result.get('CD', 1e-6)

        if self.objective == 'L/D':
            if CL <= 0:
                # Negative CL produces misleading L/D; apply smooth penalty
                # proportional to how negative CL is, staying near the last
                # good objective to preserve gradient continuity
                obj = self._last_good_obj + 5.0 + abs(CL) * 100.0
            else:
                obj = -result.get('L/D', 0)  # Minimize negative L/D = maximize L/D
        elif self.objective == '-CD':
            obj = CD  # Minimize CD
        elif self.objective == 'CL':
            obj = -CL  # Maximize CL
        elif self.objective == 'Vol Efficiency':
            vol_eff = result.get('vol_efficiency', 0.0)
            obj = -vol_eff  # Maximize volumetric efficiency
        else:
            if CL <= 0:
                obj = self._last_good_obj + 5.0 + abs(CL) * 100.0
            else:
                obj = -result.get('L/D', 0)

        # Track last good objective for smooth failure penalties
        if CL > 0 and result.get('success', False):
            self._last_good_obj = obj

        elapsed = time.time() - start_time

        # Log to history
        entry = {
            'iteration': self.iteration,
            'objective': float(obj),
            'time': elapsed,
            **{f'x{i}': float(v) for i, v in enumerate(x)},
            **{k: float(v) for k, v in result.items()
               if isinstance(v, (int, float)) and k != 'success'}
        }
        self.history.append(entry)

        # Track best
        if self.best_result is None or obj < self.best_result['objective']:
            self.best_result = entry.copy()
            self.best_result['x'] = x.copy()

        if self.verbose:
            x_str = ", ".join(f"{v:.4f}" for v in x)
            print(f"  Eval {self.iteration}: obj={obj:.6f}, "
                  f"L/D={result.get('L/D', 0):.4f}, "
                  f"x=[{x_str}] ({elapsed:.1f}s)")

        # GUI callback
        if self.progress_callback:
            self.progress_callback(self.iteration, entry)

        return obj

    def _stability_constraint_pitch(self, x: np.ndarray) -> float:
        """Pitch stability constraint: Cm_alpha < 0 → return -Cm_alpha > 0."""
        result = self._evaluate(x, compute_stability=True)
        if not result.get('success', False):
            return -1.0  # Smooth penalty (not -1e6 which corrupts gradients)
        return -result.get('Cm_alpha', 0)

    def _stability_constraint_yaw(self, x: np.ndarray) -> float:
        """Yaw stability constraint: Cn_beta > 0 → return Cn_beta > 0."""
        result = self._evaluate(x, compute_stability=True)
        if not result.get('success', False):
            return -1.0
        return result.get('Cn_beta', 0)

    def _stability_constraint_roll(self, x: np.ndarray) -> float:
        """Roll stability constraint: Cl_beta < 0 → return -Cl_beta > 0."""
        result = self._evaluate(x, compute_stability=True)
        if not result.get('success', False):
            return -1.0
        return -result.get('Cl_beta', 0)

    def optimize(
        self,
        x0: np.ndarray = None,
        bounds: List[Tuple[float, float]] = None,
        maxiter: int = 50,
        tol: float = 1e-4,
        eps: float = 1e-4
    ) -> Dict:
        """
        Run the gradient-based optimization.

        Parameters
        ----------
        x0 : array, optional
            Initial design variables. Default depends on poly_order.
        bounds : list of tuples, optional
            Bounds for each design variable. Default covers typical range.
        maxiter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        eps : float
            Finite-difference step size for gradient estimation.
            Default 1e-4 (scipy default ~1.5e-8 is too small for noisy aero evaluations).

        Returns
        -------
        dict
            Optimization results including best design, history, and final waverider
        """
        # Default initial point
        if x0 is None:
            if self.poly_order == 2:
                x0 = np.array([-5.0, -0.15])
            else:
                x0 = np.array([0.0, -5.0, -0.15])

        # Default bounds (narrower than full range to avoid degenerate geometries
        # where CL flips sign, which breaks gradient-based methods)
        if bounds is None:
            if self.poly_order == 2:
                bounds = [(-15.0, -0.5), (-0.4, -0.02)]
            else:
                bounds = [(-30.0, 30.0), (-15.0, -0.5), (-0.4, -0.02)]

        # Reset state
        self.history = []
        self.iteration = 0
        self.best_result = None
        self._eval_cache = {}
        self._last_good_obj = 100.0

        n_vars = len(x0)
        if self.verbose:
            print(f"Starting {self.method} optimization")
            print(f"  Mach={self.mach}, shock={self.shock_angle}, order={self.poly_order}")
            print(f"  Objective: maximize {self.objective}")
            print(f"  Stability constrained: {self.stability_constrained}")
            print(f"  x0 = {x0}")
            print(f"  bounds = {bounds}")
            print(f"  maxiter = {maxiter} optimizer iterations")
            if self.method == 'SLSQP':
                print(f"  (SLSQP uses ~{n_vars+1} function evals per iteration "
                      f"for FD gradients, so expect ~{maxiter*(n_vars+1)} evals)")
            print()

        # Build constraints
        constraints = []
        if self.stability_constrained and STABILITY_AVAILABLE:
            if self.method == 'SLSQP':
                constraints.append({
                    'type': 'ineq',
                    'fun': self._stability_constraint_pitch
                })
                constraints.append({
                    'type': 'ineq',
                    'fun': self._stability_constraint_yaw
                })
                constraints.append({
                    'type': 'ineq',
                    'fun': self._stability_constraint_roll
                })

        if self.volume_min > 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: self._evaluate(x).get('volume', 0) - self.volume_min
            })

        # Build optimizer options
        # Note: for SLSQP, 'maxiter' counts optimizer iterations, not function
        # evaluations. Each iteration uses ~(n_vars+1) evaluations for FD gradients.
        # We cap total function evaluations via maxiter * (n_vars+1) equivalent.
        options = {'maxiter': maxiter, 'ftol': tol, 'disp': self.verbose}
        if self.method == 'SLSQP':
            # Larger FD step for noisy aero evaluations (scipy default ~1.5e-8
            # is too small and produces unreliable gradients)
            options['eps'] = eps
        self._maxfev = maxiter  # Track for logging purposes

        # Run optimization
        start_time = time.time()

        try:
            opt_result = minimize(
                self._objective_function,
                x0,
                method=self.method,
                bounds=bounds,
                constraints=constraints if constraints else (),
                options=options
            )
        except Exception as e:
            # Wrap scipy internal errors with helpful advice
            if self.verbose:
                print(f"\nOptimizer raised exception: {e}")
                print("  Tip: Try 'Nelder-Mead' method which is gradient-free and more robust.")
            opt_result = type('Result', (), {
                'success': False,
                'message': f'{e} (try Nelder-Mead for a gradient-free alternative)',
                'x': x0,
                'fun': self._last_good_obj,
            })()

        # Auto-retry with Nelder-Mead if gradient-based method failed
        if not opt_result.success and self.method in ('SLSQP', 'COBYLA'):
            if self.verbose:
                print(f"\n  {self.method} did not converge. Auto-retrying with Nelder-Mead...")

            # Use the best point found so far as starting point for Nelder-Mead
            retry_x0 = self.best_result['x'] if self.best_result is not None else x0
            prev_best_obj = self.best_result['objective'] if self.best_result else None
            prev_history = list(self.history)

            # Reset state for retry (keep best_result for comparison)
            self._eval_cache = {}
            self.iteration = 0
            retry_history_start = len(prev_history)

            try:
                nm_result = minimize(
                    self._objective_function,
                    retry_x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': maxiter,
                        'xatol': 1e-3,
                        'fatol': 1e-4,
                        'adaptive': True,
                        'disp': self.verbose
                    }
                )
                # Use Nelder-Mead result if it's better
                if nm_result.success or (prev_best_obj is not None and nm_result.fun < prev_best_obj):
                    opt_result = nm_result
                    if self.verbose:
                        print(f"  Nelder-Mead improved result: obj={nm_result.fun:.6f}")
                elif self.verbose:
                    print(f"  Nelder-Mead did not improve (obj={nm_result.fun:.6f})")
            except Exception as e:
                if self.verbose:
                    print(f"  Nelder-Mead retry also failed: {e}")

            # Merge histories
            self.history = prev_history + self.history

        total_time = time.time() - start_time

        # Generate final waverider at optimum
        final_x = opt_result.x
        try:
            final_wr = self._create_waverider(final_x)
            final_eval = self._evaluate(final_x, compute_stability=True)

            # Export final geometry
            stl_path = os.path.join(self.output_dir, 'optimized_waverider.stl')
            tri_path = os.path.join(self.output_dir, 'optimized_waverider.tri')
            final_wr.export_stl(stl_path)
            final_wr.export_tri(tri_path)
        except Exception as e:
            final_wr = None
            final_eval = {'error': str(e)}

        # Save convergence history
        self._save_history()

        # Generate shape evolution GIF
        gif_path = None
        try:
            from animation_utils import generate_optimization_gif
            gif_path = os.path.join(self.output_dir, 'waverider_evolution.gif')
            gif_result = generate_optimization_gif(
                history=self.history,
                mach=self.mach,
                shock_angle=self.shock_angle,
                poly_order=self.poly_order,
                output_path=gif_path,
                n_le=self.n_le,
                n_stream=self.n_stream,
            )
            if gif_result and self.verbose:
                print(f"  Animation saved to {gif_result}")
        except Exception as e:
            if self.verbose:
                print(f"  Animation generation skipped: {e}")

        # Compute shape sensitivities on final optimized design
        sensitivity_result = None
        if final_wr is not None:
            try:
                from stability_analysis import compute_shape_sensitivities
                sens_vtk = os.path.join(self.output_dir, 'optimized')
                sensitivity_result = compute_shape_sensitivities(
                    wr=final_wr,
                    mach=self.mach,
                    shock_angle=self.shock_angle,
                    poly_order=self.poly_order,
                    x=final_x,
                    pressure=self.pressure,
                    temperature=self.temperature,
                    alpha_deg=self.alpha_deg,
                    n_le=self.n_le,
                    n_stream=self.n_stream,
                    save_vtk=sens_vtk,
                )
                if self.verbose:
                    print(f"\n  Shape sensitivities (dF/dp):")
                    print(f"  {sensitivity_result['f_sens']}")
                    print(f"  Sensitivity VTK saved to {sens_vtk}_sensitivities.vtu")
            except Exception as e:
                if self.verbose:
                    print(f"  Sensitivity computation skipped: {e}")

        result = {
            'success': opt_result.success,
            'message': opt_result.message,
            'x_optimal': final_x.tolist(),
            'objective_optimal': float(opt_result.fun),
            'n_iterations': self.iteration,
            'total_time': total_time,
            'final_evaluation': final_eval,
            'waverider': final_wr,
            'history': self.history,
            'scipy_result': opt_result,
            'best_found': self.best_result,
            'gif_path': gif_path,
            'sensitivity': sensitivity_result,
        }

        if self.verbose:
            print(f"\nOptimization complete in {total_time:.1f}s ({self.iteration} evaluations)")
            print(f"  Success: {opt_result.success} - {opt_result.message}")
            print(f"  Optimal x: {final_x}")
            if final_eval.get('success', False):
                print(f"  L/D = {final_eval.get('L/D', 0):.4f}")
                print(f"  CL = {final_eval.get('CL', 0):.6f}")
                print(f"  CD = {final_eval.get('CD', 0):.6f}")
                if 'Cm_alpha' in final_eval:
                    print(f"  Cm_alpha = {final_eval.get('Cm_alpha', 0):.6f}")
                    print(f"  Cn_beta  = {final_eval.get('Cn_beta', 0):.6f}")
                    print(f"  Cl_beta  = {final_eval.get('Cl_beta', 0):.6f}")

            if not opt_result.success and self.best_result is not None:
                print(f"\n  Best design found before failure:")
                print(f"    x = {self.best_result.get('x', 'N/A')}")
                print(f"    L/D = {self.best_result.get('L/D', 'N/A')}")
                print(f"  Suggestions:")
                print(f"    - Try 'Nelder-Mead' method (gradient-free, more robust)")
                print(f"    - Narrow the design variable bounds")
                print(f"    - Use the best-found design as a new starting point")

        return result

    def _save_history(self):
        """Save optimization history to JSON."""
        history_file = os.path.join(self.output_dir, 'convergence_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Also save as CSV for easy plotting
        try:
            import pandas as pd
            df = pd.DataFrame(self.history)
            df.to_csv(os.path.join(self.output_dir, 'convergence_history.csv'), index=False)
        except ImportError:
            pass


def run_shadow_optimization(
    mach: float = 6.0,
    shock_angle: float = 12.0,
    poly_order: int = 2,
    x0: np.ndarray = None,
    bounds: List[Tuple[float, float]] = None,
    objective: str = 'L/D',
    method: str = 'SLSQP',
    stability_constrained: bool = False,
    maxiter: int = 50,
    pressure: float = 101325.0,
    temperature: float = 288.15,
    alpha_deg: float = 0.0,
    save_vtk: bool = True,
    output_dir: str = 'optimization_results',
    verbose: bool = True
) -> Dict:
    """
    Convenience function to run SHADOW waverider optimization.

    Parameters match ShadowOptimizer constructor + optimize() method.
    See ShadowOptimizer docstrings for details.

    Returns
    -------
    dict
        Optimization results
    """
    optimizer = ShadowOptimizer(
        mach=mach, shock_angle=shock_angle, poly_order=poly_order,
        pressure=pressure, temperature=temperature, alpha_deg=alpha_deg,
        objective=objective, method=method,
        stability_constrained=stability_constrained,
        save_vtk=save_vtk, output_dir=output_dir, verbose=verbose)

    return optimizer.optimize(x0=x0, bounds=bounds, maxiter=maxiter)


if __name__ == '__main__':
    # Example: optimize a Mach 6 second-order waverider for L/D
    result = run_shadow_optimization(
        mach=6.0,
        shock_angle=12.0,
        poly_order=2,
        x0=np.array([-5.0, -0.15]),
        objective='L/D',
        method='Nelder-Mead',
        maxiter=30,
        save_vtk=False,
        verbose=True
    )

    print(f"\nBest L/D: {result['final_evaluation'].get('L/D', 'N/A')}")
    print(f"Optimal design: {result['x_optimal']}")
