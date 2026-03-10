"""
Planar Waverider Optimizer

Implements the two-layer optimization framework from:
  Jessen, Larsson, Brehm (2026) - "Comparative optimization of hypersonic
  waveriders using analytical and computational methods"
  Aerospace Science and Technology 172, 111703.

Layer 1: Geometry optimization via scipy differential_evolution (global,
         population-based, derivative-free - analogous to Jessen's MATLAB
         patternsearch over the 7-parameter planar waverider design space).

Layer 2: Analytical aerodynamic model (PlanarWaveriderAero) evaluating
         lift, drag, L/D, and volumetric efficiency at each candidate design.

The 7 optimized variables are:
  x = [width, n, beta_deg, epsilon, p1, p2, p3]

Fixed parameters (flow conditions and vehicle length) are set at evaluator
construction and held constant throughout the optimization.

Classes
-------
PlanarWaveriderEvaluator
    Wraps geometry generation (PlanarWaverider) and aerodynamic analysis
    (PlanarWaveriderAero) into a single cached evaluation call.

PlanarWaveriderOptimizer
    Drives differential_evolution to maximize L/D with an optional
    volumetric efficiency penalty constraint, and provides a Pareto
    sweep method to trace the L/D vs eta_vol trade-off front.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution

from planar_waverider import PlanarWaverider
from planar_waverider_aero import PlanarWaveriderAero


# ---------------------------------------------------------------------------
#  PlanarWaveriderEvaluator
# ---------------------------------------------------------------------------

class PlanarWaveriderEvaluator:
    """Evaluates a planar waverider design point: geometry + aerodynamics.

    Fixed parameters (set at init, held constant during optimization):
        length, R, M_inf, alpha_deg, altitude_km, gamma, T_wall, nx, ny

    Variable parameters (optimized, passed as array x):
        x = [width, n, beta_deg, epsilon, p1, p2, p3]

    All evaluations are cached by rounded parameter tuple to avoid
    redundant geometry regeneration when the optimizer re-visits points.

    References
    ----------
    Jessen, Larsson, Brehm (2026), Eq. 2.5 for volumetric efficiency:
        eta_vol = V^(2/3) / S_plan
    """

    VAR_NAMES = ['width', 'n', 'beta_deg', 'epsilon', 'p1', 'p2', 'p3']

    def __init__(self, length, R, M_inf, alpha_deg, altitude_km,
                 gamma=1.4, T_wall=None, nx=60, ny=40):
        """
        Parameters
        ----------
        length : float
            Vehicle length [m].
        R : float
            Leading edge nose radius [m]. Set 0.0 for sharp LE.
        M_inf : float
            Freestream Mach number.
        alpha_deg : float
            Angle of attack [deg].
        altitude_km : float
            Flight altitude [km].
        gamma : float
            Ratio of specific heats (default 1.4).
        T_wall : float or None
            Wall temperature [K]. None = adiabatic wall.
        nx : int
            Streamwise grid resolution for geometry generation.
        ny : int
            Spanwise (half-span) grid resolution for geometry generation.
        """
        self.length = length
        self.R = R
        self.M_inf = M_inf
        self.alpha_deg = alpha_deg
        self.altitude_km = altitude_km
        self.gamma = gamma
        self.T_wall = T_wall
        self.nx = nx
        self.ny = ny

        self._aero = PlanarWaveriderAero(gamma=gamma)
        self._cache = {}
        self._n_evals = 0
        self._n_cache_hits = 0

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def evaluate(self, x):
        """Evaluate a design point.

        Parameters
        ----------
        x : array-like, length 7
            [width, n, beta_deg, epsilon, p1, p2, p3]

        Returns
        -------
        result : dict
            Keys: 'L_over_D', 'CL', 'CD', 'eta_vol', 'volume',
            'planform_area', 'base_width', 'base_height',
            'D_inviscid', 'D_viscous', 'D_base', 'D_le',
            'success' (bool).
            On failure, 'success' is False and 'L_over_D' is 0.
        """
        x = np.asarray(x, dtype=float)
        cache_key = tuple(np.round(x, 10))

        # --- Cache lookup ---
        self._n_evals += 1
        if cache_key in self._cache:
            self._n_cache_hits += 1
            return self._cache[cache_key]

        # --- Geometry generation ---
        try:
            wr = PlanarWaverider(
                length=self.length,
                width=float(x[0]),
                n=float(x[1]),
                beta_deg=float(x[2]),
                epsilon=float(x[3]),
                p1=float(x[4]),
                p2=float(x[5]),
                p3=float(x[6]),
                R=self.R,
                M_inf=self.M_inf,
                gamma=self.gamma,
            )
            wr.generate(nx=self.nx, ny=self.ny)
        except Exception:
            result = self._penalty_result()
            self._cache[cache_key] = result
            return result

        # --- Derived geometry quantities ---
        try:
            volume = wr.volume()
            planform_area = wr.planform_area()
            base_width, base_height = wr.base_dimensions()

            # Volumetric efficiency (Jessen Eq. 2.5)
            if planform_area > 1e-12 and volume > 0.0:
                eta_vol = volume ** (2.0 / 3.0) / planform_area
            else:
                eta_vol = 0.0
        except Exception:
            result = self._penalty_result()
            self._cache[cache_key] = result
            return result

        # --- Aerodynamic analysis ---
        try:
            forces = self._aero.compute_forces(
                wr,
                M_inf=self.M_inf,
                alpha_deg=self.alpha_deg,
                altitude_km=self.altitude_km,
                T_wall=self.T_wall,
            )
        except Exception:
            result = self._penalty_result()
            self._cache[cache_key] = result
            return result

        # --- Pack result ---
        result = {
            'success': True,
            'L_over_D': forces.get('L_over_D', 0.0),
            'CL': forces.get('CL', 0.0),
            'CD': forces.get('CD', 0.0),
            'eta_vol': eta_vol,
            'volume': volume,
            'planform_area': planform_area,
            'base_width': base_width,
            'base_height': base_height,
            'D_inviscid': forces.get('D_inviscid', 0.0),
            'D_viscous': forces.get('D_viscous', 0.0),
            'D_base': forces.get('D_base', 0.0),
            'D_le': forces.get('D_le', 0.0),
        }

        self._cache[cache_key] = result
        return result

    def cache_stats(self):
        """Return (n_evals, n_cache_hits)."""
        return self._n_evals, self._n_cache_hits

    def clear_cache(self):
        """Clear evaluation cache and reset counters."""
        self._cache.clear()
        self._n_evals = 0
        self._n_cache_hits = 0

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _penalty_result():
        """Return a failed-evaluation penalty dict."""
        return {
            'success': False,
            'L_over_D': 0.0,
            'CL': 0.0,
            'CD': 0.0,
            'eta_vol': 0.0,
            'volume': 0.0,
            'planform_area': 0.0,
            'base_width': 0.0,
            'base_height': 0.0,
            'D_inviscid': 0.0,
            'D_viscous': 0.0,
            'D_base': 0.0,
            'D_le': 0.0,
        }


# ---------------------------------------------------------------------------
#  PlanarWaveriderOptimizer
# ---------------------------------------------------------------------------

class PlanarWaveriderOptimizer:
    """Derivative-free optimizer for planar waverider shape.

    Optimizes 7 variables (width, n, beta_deg, epsilon, p1, p2, p3)
    to maximize L/D, with an optional volumetric efficiency constraint.

    The optimizer uses scipy.optimize.differential_evolution — a global,
    population-based, derivative-free algorithm — analogous to Jessen's
    MATLAB patternsearch over the planar waverider design space.

    The eta_vol constraint is handled via a quadratic penalty added to the
    objective (exterior penalty method), which keeps the problem unconstrained
    from the optimizer's perspective and avoids feasibility-first biases.

    Parameters
    ----------
    evaluator : PlanarWaveriderEvaluator
        Pre-configured evaluator holding fixed flight conditions.
    bounds : list of (lo, hi) tuples, length 7, or None
        Variable bounds in order: [width, n, beta_deg, epsilon, p1, p2, p3].
        If None, DEFAULT_BOUNDS are used, with the width bounds scaled by
        evaluator.length.
    eta_vol_min : float
        Minimum volumetric efficiency (Jessen Eq. 2.5). 0 = unconstrained.
    popsize : int
        DE population size multiplier (total population = popsize * 7).
    maxiter : int
        Maximum number of DE generations.
    tol : float
        Relative convergence tolerance passed to differential_evolution.
    callback : callable or None
        Optional progress callback with signature:
            callback(iteration, maxiter, best_x, best_LD,
                     best_metrics, history)
        Called after each DE generation.
    seed : int or None
        Random seed for reproducibility.

    References
    ----------
    Jessen, Larsson, Brehm (2026), Sections 3.1-3.2 and Table 2.1.
    """

    # Default bounds from Jessen et al. Table 2.1.
    # Width bounds are relative and will be scaled by evaluator.length.
    DEFAULT_BOUNDS = {
        'width':    (0.1, 1.5),    # multiplied by length at init
        'n':        (0.1, 0.9),
        'beta_deg': (1.0, 20.0),
        'epsilon':  (-1.0, 1.0),
        'p1':       (0.5, 3.0),
        'p2':       (0.5, 3.0),
        'p3':       (0.5, 3.0),
    }

    def __init__(self, evaluator, bounds=None, eta_vol_min=0.0,
                 popsize=15, maxiter=100, tol=1e-6,
                 callback=None, seed=None):
        self.evaluator = evaluator
        self.eta_vol_min = eta_vol_min
        self.popsize = popsize
        self.maxiter = maxiter
        self.tol = tol
        self.callback = callback
        self.seed = seed

        # Resolve bounds
        if bounds is not None:
            self.bounds = list(bounds)
        else:
            L = evaluator.length
            self.bounds = [
                (self.DEFAULT_BOUNDS['width'][0] * L,
                 self.DEFAULT_BOUNDS['width'][1] * L),
                self.DEFAULT_BOUNDS['n'],
                self.DEFAULT_BOUNDS['beta_deg'],
                self.DEFAULT_BOUNDS['epsilon'],
                self.DEFAULT_BOUNDS['p1'],
                self.DEFAULT_BOUNDS['p2'],
                self.DEFAULT_BOUNDS['p3'],
            ]

        # State
        self._should_stop = False
        self.history = []
        self.best_x = None
        self.best_metrics = None
        self.best_LD = -np.inf
        self._iteration = 0
        self.result = None

    # ------------------------------------------------------------------
    #  Objective and DE callback
    # ------------------------------------------------------------------

    def _objective(self, x):
        """Objective function for differential_evolution: minimize -L/D.

        Failed evaluations return a large penalty (1e6) so that DE steers
        away from geometry-invalid parameter combinations.  A quadratic
        penalty is added when eta_vol_min > 0 and the constraint is
        violated, following the exterior penalty method.

        Parameters
        ----------
        x : ndarray, shape (7,)
            [width, n, beta_deg, epsilon, p1, p2, p3]

        Returns
        -------
        obj : float
            Scalar objective value to minimise.
        """
        metrics = self.evaluator.evaluate(x)

        # --- Record every function evaluation in history ---
        entry = {'eval': len(self.history) + 1}
        for i, name in enumerate(PlanarWaveriderEvaluator.VAR_NAMES):
            entry[name] = float(x[i])
        entry['L_over_D'] = metrics.get('L_over_D', 0.0)
        entry['CL'] = metrics.get('CL', 0.0)
        entry['CD'] = metrics.get('CD', 0.0)
        entry['eta_vol'] = metrics.get('eta_vol', 0.0)
        entry['volume'] = metrics.get('volume', 0.0)
        entry['success'] = metrics.get('success', False)
        self.history.append(entry)

        # --- Penalty for geometry/aero failure ---
        if not metrics.get('success', False):
            return 1e6

        LD = metrics['L_over_D']

        # --- Update best feasible solution ---
        # Check constraint feasibility before updating best
        eta = metrics.get('eta_vol', 0.0)
        feasible = (self.eta_vol_min <= 0.0) or (eta >= self.eta_vol_min)
        if feasible and LD > self.best_LD:
            self.best_LD = LD
            self.best_x = x.copy()
            self.best_metrics = metrics.copy()

        # --- Base objective: minimise negative L/D ---
        obj = -LD

        # --- Quadratic penalty for eta_vol constraint violation ---
        if self.eta_vol_min > 0.0:
            violation = self.eta_vol_min - eta  # positive when violated
            if violation > 0.0:
                obj += 1e4 * violation

        return obj

    def _de_callback(self, xk, convergence=0):
        """Called by differential_evolution after each generation.

        Returns True to signal early termination if stop() was called.

        Parameters
        ----------
        xk : ndarray
            Best solution vector at current generation.
        convergence : float
            Fractional value of convergence achieved (0..1).
        """
        self._iteration += 1

        if self.callback is not None:
            self.callback(
                self._iteration,
                self.maxiter,
                self.best_x,
                self.best_LD,
                self.best_metrics,
                self.history,
            )

        return self._should_stop

    # ------------------------------------------------------------------
    #  Primary optimization entry point
    # ------------------------------------------------------------------

    def optimize_ld(self):
        """Run single-objective optimization: maximize L/D.

        Resets all state before running.  Uses
        scipy.optimize.differential_evolution with polish=False because
        the waverider objective is non-smooth (discrete Chebyshev system
        solve, piecewise pressure model) and L-BFGS-B polishing would
        consume extra evaluations for negligible gain.

        Returns
        -------
        best_x : ndarray or None
            Best parameter vector [width, n, beta_deg, epsilon, p1, p2, p3],
            or None if no feasible point was found.
        best_metrics : dict or None
            Evaluation result dict at best_x, or None.
        history : list of dict
            One entry per function evaluation.
        """
        # --- Reset state ---
        self._should_stop = False
        self.history = []
        self.best_x = None
        self.best_metrics = None
        self.best_LD = -np.inf
        self._iteration = 0

        # --- Run DE ---
        self.result = differential_evolution(
            self._objective,
            bounds=self.bounds,
            maxiter=self.maxiter,
            popsize=self.popsize,
            tol=self.tol,
            seed=self.seed,
            callback=self._de_callback,
            polish=False,
            disp=False,
        )

        # --- Safety check: if DE's internal best is better than tracked best ---
        # (can happen if a non-feasible point happened to have lower -LD)
        if self.result.fun < -self.best_LD:
            final_metrics = self.evaluator.evaluate(self.result.x)
            if final_metrics.get('success', False):
                eta = final_metrics.get('eta_vol', 0.0)
                feasible = (self.eta_vol_min <= 0.0) or (eta >= self.eta_vol_min)
                if feasible:
                    self.best_x = self.result.x.copy()
                    self.best_metrics = final_metrics
                    self.best_LD = final_metrics['L_over_D']

        return self.best_x, self.best_metrics, self.history

    # ------------------------------------------------------------------
    #  Pareto sweep
    # ------------------------------------------------------------------

    def pareto_sweep(self, n_points=10, callback_point=None):
        """Sweep eta_vol_min to generate the Pareto front: L/D vs eta_vol.

        Algorithm
        ---------
        1. Run unconstrained optimization (eta_vol_min = 0) to find the
           maximum achievable L/D and its associated eta_vol value (eta0).
        2. Uniformly sample n_points values of eta_vol_min between eta0
           and eta0 * 3.0 (a representative upper bound).
        3. At each sampled point, run a full optimization with that
           constraint value active.
        4. Stop early if L/D drops below 50% of the unconstrained maximum
           (infeasible region reached).

        The evaluator cache is cleared between sweep points to prevent
        a warm-started population from masking constraint-driven changes
        in the objective landscape.

        Parameters
        ----------
        n_points : int
            Number of constrained sweep points (total = n_points + 1
            including the unconstrained baseline).
        callback_point : callable or None
            Progress callback with signature:
                callback_point(point_index, total_points, eta_min, best_LD)
            Called after each complete single-point optimization.

        Returns
        -------
        pareto : list of dict
            Each entry has keys:
                'eta_vol_min', 'L_over_D', 'eta_vol', 'best_x', 'metrics'
            Ordered by increasing eta_vol_min (first entry is unconstrained).
        """
        pareto = []

        # --- Step 1: unconstrained baseline ---
        self.eta_vol_min = 0.0
        x0, m0, _ = self.optimize_ld()

        if m0 is None:
            # Optimization produced no feasible point
            return pareto

        eta0 = m0.get('eta_vol', 0.0)
        pareto.append({
            'eta_vol_min': 0.0,
            'L_over_D': m0['L_over_D'],
            'eta_vol': eta0,
            'best_x': x0.copy(),
            'metrics': m0.copy(),
        })

        if callback_point is not None:
            callback_point(1, n_points + 1, 0.0, m0['L_over_D'])

        # --- Step 2: sweep upward from unconstrained eta0 ---
        eta_max = eta0 * 3.0  # upper exploration limit
        eta_values = np.linspace(eta0, eta_max, n_points + 1)[1:]

        for i, eta_min in enumerate(eta_values):
            if self._should_stop:
                break

            self.eta_vol_min = eta_min
            self.evaluator.clear_cache()  # fresh landscape for each sweep point

            xi, mi, _ = self.optimize_ld()

            if mi is None or mi.get('L_over_D', 0.0) < 0.1:
                # Infeasible region: stop sweep
                break

            pareto.append({
                'eta_vol_min': float(eta_min),
                'L_over_D': mi['L_over_D'],
                'eta_vol': mi.get('eta_vol', 0.0),
                'best_x': xi.copy(),
                'metrics': mi.copy(),
            })

            if callback_point is not None:
                callback_point(i + 2, n_points + 1, eta_min, mi['L_over_D'])

            # Early termination: degradation below 50% of unconstrained max
            if mi['L_over_D'] < 0.5 * m0['L_over_D']:
                break

        return pareto

    # ------------------------------------------------------------------
    #  Control
    # ------------------------------------------------------------------

    def stop(self):
        """Signal the optimizer to stop after the current generation.

        The DE callback returns True at the next generation boundary,
        causing differential_evolution to terminate cleanly.
        """
        self._should_stop = True

    # ------------------------------------------------------------------
    #  Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def save_history_csv(history, filepath):
        """Save optimization history to a CSV file.

        All function evaluations are written, including failed ones
        (success=False rows have zero aero metrics).

        Parameters
        ----------
        history : list of dict
            As returned by optimize_ld() or stored on the instance.
        filepath : str
            Destination file path. Created or overwritten.
        """
        if not history:
            return

        keys = list(history[0].keys())

        with open(filepath, 'w') as f:
            f.write(','.join(keys) + '\n')
            for entry in history:
                vals = []
                for k in keys:
                    v = entry.get(k, '')
                    if isinstance(v, bool):
                        vals.append(str(v))
                    elif isinstance(v, str):
                        vals.append(v)
                    else:
                        vals.append(f'{v:.6g}')
                f.write(','.join(vals) + '\n')
