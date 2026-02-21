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
        _run_pysagas_at_condition
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
        verbose: bool = True
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

        # Convergence history
        self.history = []
        self.iteration = 0
        self.best_result = None

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
        Evaluate a design point: generate waverider and run PySAGAS.

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
        try:
            wr = self._create_waverider(x)
        except Exception as e:
            return {'success': False, 'error': f'Geometry failed: {e}'}

        cells = cells_from_waverider(wr)
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
            result['mac'] = wr.mac

        except Exception as e:
            return {'success': False, 'error': f'PySAGAS failed: {e}'}

        return result

    def _objective_function(self, x: np.ndarray) -> float:
        """Objective function for scipy.optimize."""
        self.iteration += 1
        start_time = time.time()

        result = self._evaluate(x, compute_stability=self.stability_constrained)

        if not result.get('success', False):
            if self.verbose:
                print(f"  Iter {self.iteration}: FAILED - {result.get('error', 'unknown')}")
            return 1e6  # Large penalty for failed evaluations

        # Compute objective value (always minimize)
        if self.objective == 'L/D':
            obj = -result.get('L/D', 0)  # Minimize negative L/D = maximize L/D
        elif self.objective == '-CD':
            obj = result.get('CD', 1e6)  # Minimize CD
        elif self.objective == 'CL':
            obj = -result.get('CL', 0)  # Maximize CL
        else:
            obj = -result.get('L/D', 0)

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
            print(f"  Iter {self.iteration}: obj={obj:.6f}, "
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
            return -1e6
        return -result.get('Cm_alpha', 0)

    def _stability_constraint_yaw(self, x: np.ndarray) -> float:
        """Yaw stability constraint: Cn_beta > 0 → return Cn_beta > 0."""
        result = self._evaluate(x, compute_stability=True)
        if not result.get('success', False):
            return -1e6
        return result.get('Cn_beta', 0)

    def _stability_constraint_roll(self, x: np.ndarray) -> float:
        """Roll stability constraint: Cl_beta < 0 → return -Cl_beta > 0."""
        result = self._evaluate(x, compute_stability=True)
        if not result.get('success', False):
            return -1e6
        return -result.get('Cl_beta', 0)

    def optimize(
        self,
        x0: np.ndarray = None,
        bounds: List[Tuple[float, float]] = None,
        maxiter: int = 50,
        tol: float = 1e-6
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

        # Default bounds
        if bounds is None:
            if self.poly_order == 2:
                bounds = [(-20.0, -0.5), (-0.5, -0.01)]
            else:
                bounds = [(-50.0, 50.0), (-20.0, -0.5), (-0.5, -0.01)]

        # Reset state
        self.history = []
        self.iteration = 0
        self.best_result = None

        if self.verbose:
            print(f"Starting {self.method} optimization")
            print(f"  Mach={self.mach}, shock={self.shock_angle}, order={self.poly_order}")
            print(f"  Objective: maximize {self.objective}")
            print(f"  Stability constrained: {self.stability_constrained}")
            print(f"  x0 = {x0}")
            print(f"  bounds = {bounds}")
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

        # Run optimization
        start_time = time.time()

        opt_result = minimize(
            self._objective_function,
            x0,
            method=self.method,
            bounds=bounds,
            constraints=constraints if constraints else (),
            options={'maxiter': maxiter, 'ftol': tol, 'disp': self.verbose}
        )

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
