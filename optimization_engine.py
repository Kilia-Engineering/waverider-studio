#!/usr/bin/env python3
"""
Waverider Optimization Engine

This module implements the core optimization problem for waverider design
using pymoo's multi-objective optimization framework.

Key components:
- WaveriderProblem: Pymoo Problem class for waverider optimization
- Design space constraint validation (discriminant formula)
- Parallel evaluation of designs
- Integration with waverider generator, mesh generation, and PySAGAS analysis
"""

import numpy as np
import multiprocessing as mp
import os
import sys
import tempfile
import shutil
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from pymoo.core.problem import Problem

# Add parent directory to path to import waverider modules
sys.path.insert(0, str(Path(__file__).parent))

from waverider_generator.generator import waverider as WaveriderGenerator
from waverider_generator.cad_export import to_CAD
from reference_area_calculator import calculate_reference_area_simple as calculate_reference_area

# Try to import gmsh and pysagas (might not be available in all environments)
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    logging.warning("gmsh not available - mesh generation will fail")

try:
    from pysagas.cfd import OPM
    from pysagas.flow import FlowState
    from pysagas.geometry.parsers import MeshIO
    PYSAGAS_AVAILABLE = True
except ImportError:
    PYSAGAS_AVAILABLE = False
    logging.warning("PySAGAS not available - aerodynamic analysis will fail")


class WaveriderProblem(Problem):
    """
    Pymoo optimization problem for waverider design.
    
    Design Variables:
        X1: Flat region of shockwave (0-1, normalized to width)
        X2: Height of shockwave (0-1, normalized to height)
        X3: Upper surface central control point (0-1)
        X4: Upper surface side control point (0-1)
    
    Objectives (user-selectable):
        - CL: Lift coefficient (maximize/minimize/target)
        - CD: Drag coefficient (maximize/minimize/target)
        - Cm: Moment coefficient (maximize/minimize/|Cm|)
        - Volume: Internal volume (maximize/minimize/target)
        - CL/CD: Lift-to-drag ratio (computed from CL/CD)
    
    Constraints:
        - Design space constraint (discriminant formula from paper)
        - Optional: CL >= min, CD <= max, |Cm| <= max, Volume >= min
    """
    
    def __init__(self,
                 # Flow conditions
                 M_inf: float = 5.0,
                 beta: float = 15.0,
                 altitude: float = 25000.0,  # meters
                 aoa: float = 0.0,  # degrees
                 
                 # Direct atmospheric conditions (override altitude calculation)
                 pressure: float = None,  # Pa - if provided, overrides altitude
                 temperature: float = None,  # K - if provided, overrides altitude
                 
                 # Geometry
                 height: float = 1.34,
                 width: float = 3.0,
                 A_ref: float = None,  # Reference area - if None, uses width*height
                 
                 # Mesh parameters
                 mesh_size: float = 0.2,  # Controls mesh density
                 
                 # Optimization settings
                 objectives: List[Dict] = None,
                 constraints: List[Dict] = None,
                 n_cores: int = 1,
                 
                 # Advanced settings
                 n_planes: int = 40,
                 n_streamwise: int = 30,
                 delta_streamwise: float = 0.1,
                 
                 # Debug
                 verbose: bool = False):
        """
        Initialize waverider optimization problem.
        
        Parameters
        ----------
        objectives : List[Dict]
            List of objective dictionaries with keys:
            - 'name': 'CL', 'CD', 'Cm', 'Volume', or 'LD'
            - 'mode': 'maximize', 'minimize', or 'target'
            - 'target': target value (only for mode='target')
            Example: [{'name': 'CD', 'mode': 'minimize'},
                     {'name': 'Volume', 'mode': 'maximize'}]
        
        constraints : List[Dict]
            List of constraint dictionaries with keys:
            - 'name': 'design_space', 'CL_min', 'CD_max', 'Cm_max', 'Volume_min'
            - 'value': constraint value (not needed for 'design_space')
            - 'active': True/False
            Example: [{'name': 'design_space', 'active': True},
                     {'name': 'CL_min', 'value': 1.0, 'active': True}]
        """
        
        # Store flow conditions
        self.M_inf = M_inf
        self.beta = beta
        self.altitude = altitude
        self.aoa = aoa
        
        # Store geometry
        self.height = height
        self.width = width
        
        # Store mesh parameters
        self.mesh_size = mesh_size
        
        # Store waverider generation parameters
        self.n_planes = n_planes
        self.n_streamwise = n_streamwise
        self.delta_streamwise = delta_streamwise
        
        # Parallelization
        self.n_cores = n_cores
        
        # Verbose mode
        self.verbose = verbose
        
        # Calculate or use provided atmospheric properties
        if pressure is not None and temperature is not None:
            # Use directly provided values
            self.pressure = pressure
            self.temperature = temperature
        else:
            # Calculate from altitude
            self.pressure, self.temperature = self._calculate_atmosphere(altitude)
        
        # Set reference area
        if A_ref is not None:
            self.A_ref = A_ref
        else:
            # Use simple approximation: width * height
            self.A_ref = width * height
        
        # Process objectives
        if objectives is None:
            # Default: minimize drag, maximize volume
            objectives = [
                {'name': 'CD', 'mode': 'minimize'},
                {'name': 'Volume', 'mode': 'maximize'}
            ]
        
        self.objectives = objectives
        n_obj = len(objectives)
        
        # Process constraints
        if constraints is None:
            # Default: only design space constraint
            constraints = [{'name': 'design_space', 'active': True}]
        
        self.constraints = {c['name']: c for c in constraints if c.get('active', True)}
        
        # Count active constraints (excluding design_space which is handled separately)
        perf_constraints = [c for c in self.constraints.values() 
                           if c['name'] != 'design_space']
        n_constr = len(perf_constraints)
        
        # Initialize pymoo Problem
        super().__init__(
            n_var=4,  # X1, X2, X3, X4
            n_obj=n_obj,
            n_constr=n_constr,
            xl=np.array([0.0, 0.0, 0.0, 0.0]),  # Lower bounds
            xu=np.array([1.0, 1.0, 1.0, 1.0])   # Upper bounds
        )
        
        # Evaluation counter
        self.n_eval = 0
        
        if self.verbose:
            print(f"WaveriderProblem initialized:")
            print(f"  Flow: M={M_inf}, β={beta}°, h={altitude}m, α={aoa}°")
            print(f"  Geometry: height={height}m, width={width}m")
            print(f"  Objectives: {[obj['name'] for obj in objectives]}")
            print(f"  Constraints: {list(self.constraints.keys())}")
            print(f"  Parallel cores: {n_cores}")
    
    def _calculate_atmosphere(self, altitude: float) -> Tuple[float, float]:
        """
        Calculate atmospheric properties at given altitude using US Standard Atmosphere.
        
        Parameters
        ----------
        altitude : float
            Altitude in meters
        
        Returns
        -------
        pressure : float
            Static pressure in Pa
        temperature : float
            Static temperature in K
        """
        # US Standard Atmosphere (simplified)
        # Sea level conditions
        P0 = 101325.0  # Pa
        T0 = 288.15    # K
        g = 9.80665    # m/s²
        R = 287.05     # J/(kg·K)
        L = 0.0065     # K/m (temperature lapse rate in troposphere)
        
        if altitude < 11000:  # Troposphere
            T = T0 - L * altitude
            P = P0 * (T / T0) ** (g / (R * L))
        else:  # Stratosphere (isothermal approximation)
            T = 216.65  # K
            P = 22632.1 * np.exp(-g * (altitude - 11000) / (R * T))
        
        return P, T
    
    def calculate_max_x2(self, x1: float, safety_margin: float = 0.90) -> float:
        """
        Calculate maximum allowed X2 given X1, based on paper constraint.
        
        Constraint: X2 / (1 - X1)^4 < (7/64) * (width/height)^4
        Rearranged: X2 < (7/64) * (width/height)^4 * (1 - X1)^4
        
        Parameters
        ----------
        x1 : float
            X1 value
        safety_margin : float
            Safety margin (default 97%)
            
        Returns
        -------
        float
            Maximum allowed X2 value with safety margin
        """
        one_minus_x1 = max(1.0 - x1, 0.001)
        rhs = (7.0 / 64.0) * (self.width / self.height) ** 4
        max_x2 = rhs * (one_minus_x1 ** 4) * safety_margin
        return min(max(max_x2, 0.0), 1.0)
    
    def repair_design(self, X: np.ndarray, safety_margin: float = 0.90) -> np.ndarray:
        """
        Repair invalid designs by reducing X2 to maximum valid value.
        
        Parameters
        ----------
        X : np.ndarray
            Design variables [X1, X2, X3, X4] - can be 1D or 2D
        safety_margin : float
            Safety margin for constraint (default 97%)
            
        Returns
        -------
        X_repaired : np.ndarray
            Repaired design variables
        """
        X_repaired = X.copy()
        
        # Handle both 1D and 2D arrays
        if X.ndim == 1:
            X_repaired = X_repaired.reshape(1, -1)
        
        for i in range(len(X_repaired)):
            x1 = X_repaired[i, 0]
            x2 = X_repaired[i, 1]
            
            # Calculate max X2 for this X1
            max_x2 = self.calculate_max_x2(x1, safety_margin)
            
            # If X2 exceeds max, clamp it
            if x2 > max_x2:
                X_repaired[i, 1] = max_x2
        
        # Return in original shape
        if X.ndim == 1:
            return X_repaired.flatten()
        return X_repaired
    
    def check_design_space_constraint(self, X: np.ndarray, safety_margin: float = 0.90) -> np.ndarray:
        """
        Check design space constraint using discriminant formula.
        
        From the paper (Equation 8):
        X2 / (1-X1)^4 < (7/64) * (width/height)^4
        
        Parameters
        ----------
        X : np.ndarray
            Design variables [X1, X2, X3, X4]
        safety_margin : float
            Safety margin for constraint (default 97%)
        
        Returns
        -------
        valid : np.ndarray
            Boolean array indicating valid designs
        """
        X1 = X[:, 0]
        X2 = X[:, 1]
        
        # Left side of inequality
        lhs = X2 / ((1 - X1) ** 4 + 1e-10)  # Add small epsilon to avoid division by zero
        
        # Right side of inequality
        rhs = (7.0 / 64.0) * (self.width / self.height) ** 4
        
        # Apply safety margin
        rhs_safe = safety_margin * rhs
        
        # Design is valid if lhs < rhs
        valid = lhs < rhs_safe
        
        return valid
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate population of designs.
        
        This is called by pymoo's algorithm to evaluate a batch of designs.
        """
        n_designs = len(X)
        
        if self.verbose:
            print(f"\nEvaluating {n_designs} designs...")
        
        # First, repair any invalid designs (reduce X2 to max valid value)
        X = self.repair_design(X)
        
        # Check design space constraints (should all be valid after repair)
        valid = self.check_design_space_constraint(X)
        n_valid = np.sum(valid)
        
        if self.verbose:
            print(f"  Design space: {n_valid}/{n_designs} valid (after repair)")
        
        # Evaluate designs
        if self.n_cores > 1 and n_designs > 1:
            # Parallel evaluation
            results = self._evaluate_parallel(X, valid)
        else:
            # Sequential evaluation
            results = self._evaluate_sequential(X, valid)
        
        # Extract objectives and constraints
        F = np.zeros((n_designs, self.n_obj))
        G = np.zeros((n_designs, self.n_constr))
        
        # Track which designs succeeded and their errors (for logging purposes)
        self.last_eval_success = np.array([r['success'] for r in results])
        self.last_eval_errors = [r.get('error', '') if not r['success'] else '' for r in results]
        # Store actual result values for callback to access
        self.last_eval_results = results
        
        # Count successes and collect error types
        n_success = sum(1 for r in results if r['success'])
        error_counts = {}
        for r in results:
            if not r['success']:
                err = r.get('error', 'Unknown error')
                # Truncate long error messages
                err_key = err[:50] if len(err) > 50 else err
                error_counts[err_key] = error_counts.get(err_key, 0) + 1
        
        if self.verbose:
            print(f"  Results: {n_success}/{n_designs} successful")
            if error_counts:
                print(f"  Error summary:")
                for err, count in error_counts.items():
                    print(f"    - {err}: {count}")
        
        for i, result in enumerate(results):
            if result['success']:
                # Extract objectives
                for j, obj in enumerate(self.objectives):
                    obj_name = obj['name']
                    obj_mode = obj['mode']
                    
                    if obj_name in ['LD', 'CL/CD', 'L/D']:  # Accept all L/D variants
                        value = result['CL'] / (result['CD'] + 1e-10)
                    else:
                        value = result[obj_name]
                    
                    # Handle maximize vs minimize
                    if obj_mode == 'maximize':
                        F[i, j] = -value  # Pymoo minimizes, so negate
                    elif obj_mode == 'minimize':
                        F[i, j] = value
                    elif obj_mode == 'target':
                        target = obj.get('target', 0.0)
                        F[i, j] = abs(value - target)
                
                # Extract constraints (inequality constraints: g(x) <= 0)
                constraint_idx = 0
                for c_name, c_info in self.constraints.items():
                    if c_name == 'design_space':
                        continue  # Already handled
                    
                    if c_name == 'CL_min':
                        G[i, constraint_idx] = c_info['value'] - result['CL']  # CL >= value -> value - CL <= 0
                    elif c_name == 'CD_max':
                        G[i, constraint_idx] = result['CD'] - c_info['value']  # CD <= value
                    elif c_name == 'Cm_max':
                        G[i, constraint_idx] = abs(result['Cm']) - c_info['value']  # |Cm| <= value
                    elif c_name == 'Volume_min':
                        G[i, constraint_idx] = c_info['value'] - result['Volume']  # Volume >= value
                    
                    constraint_idx += 1
            else:
                # Design failed - assign bad objectives and violated constraints
                F[i, :] = 1e6  # Large penalty
                G[i, :] = 1e6  # Violated constraints
        
        out["F"] = F
        if self.n_constr > 0:
            out["G"] = G
        
        self.n_eval += n_designs
        
        if self.verbose:
            print(f"  Total evaluations: {self.n_eval}")
    
    def _evaluate_sequential(self, X: np.ndarray, valid: np.ndarray) -> List[Dict]:
        """Evaluate designs sequentially (single core)"""
        results = []
        for i, x in enumerate(X):
            if not valid[i]:
                if self.verbose:
                    print(f"  Design {i}: INVALID (design space constraint violated)")
                results.append({'success': False, 'error': 'Design space constraint violated'})
            else:
                result = evaluate_single_design(
                    x, 
                    self.M_inf, self.beta, self.pressure, self.temperature, self.aoa,
                    self.height, self.width, self.A_ref,
                    self.mesh_size, self.n_planes, self.n_streamwise, self.delta_streamwise,
                    worker_id=i, verbose=self.verbose
                )
                if self.verbose:
                    if result['success']:
                        print(f"  Design {i}: SUCCESS - CL={result.get('CL', 'N/A'):.4f}, CD={result.get('CD', 'N/A'):.4f}")
                    else:
                        print(f"  Design {i}: FAILED - {result.get('error', 'Unknown error')}")
                results.append(result)
        return results
    
    def _evaluate_parallel(self, X: np.ndarray, valid: np.ndarray) -> List[Dict]:
        """Evaluate designs in parallel (multi-core) - Windows compatible"""
        import multiprocessing as mp_module
        
        # Use 'spawn' context for Windows compatibility
        # 'spawn' works on all platforms (Windows, Linux, Mac)
        ctx = mp_module.get_context('spawn')
        
        # Prepare arguments for each worker
        args_list = []
        for i, x in enumerate(X):
            if not valid[i]:
                args_list.append(None)  # Skip invalid designs
            else:
                args_list.append((
                    x, 
                    self.M_inf, self.beta, self.pressure, self.temperature, self.aoa,
                    self.height, self.width, self.A_ref,
                    self.mesh_size, self.n_planes, self.n_streamwise, self.delta_streamwise,
                    i  # worker_id
                ))
        
        # Evaluate in parallel using spawn context (Windows-safe)
        with ctx.Pool(processes=self.n_cores) as pool:
            results = pool.map(evaluate_single_design_wrapper, args_list)
        
        return results


def evaluate_single_design_wrapper(args):
    """
    Wrapper for parallel evaluation.
    Handles None (invalid designs) and unpacks arguments.
    """
    if args is None:
        return {'success': False, 'error': 'Design space constraint violated'}
    
    return evaluate_single_design(*args)


# Import reference area calculator if available
try:
    from reference_area_calculator import calculate_planform_area_from_waverider
    AREA_CALC_AVAILABLE = True
except ImportError:
    AREA_CALC_AVAILABLE = False
    
    def calculate_planform_area_from_waverider(waverider):
        """Fallback if reference_area_calculator not available."""
        # Simple approximation using upper surface
        X = waverider.upper_surface_x
        Z = waverider.upper_surface_z
        
        total_area = 0.0
        ny, nx = X.shape
        
        for i in range(ny - 1):
            for j in range(nx - 1):
                p1 = np.array([X[i, j], Z[i, j]])
                p2 = np.array([X[i+1, j], Z[i+1, j]])
                p3 = np.array([X[i+1, j+1], Z[i+1, j+1]])
                p4 = np.array([X[i, j+1], Z[i, j+1]])
                
                area1 = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                area2 = 0.5 * abs(np.cross(p3 - p1, p4 - p1))
                total_area += (area1 + area2)
        
        return 2.0 * total_area, "Direct integration"


def evaluate_single_design(
    x: np.ndarray,
    M_inf: float,
    beta: float, 
    pressure: float,
    temperature: float,
    aoa: float,
    height: float,
    width: float,
    A_ref: float,
    mesh_size: float,
    n_planes: int,
    n_streamwise: int,
    delta_streamwise: float,
    worker_id: int = 0,
    verbose: bool = False
) -> Dict:
    """
    Evaluate a single waverider design.
    
    This function:
    1. Generates waverider geometry
    2. Creates mesh
    3. Runs PySAGAS analysis
    4. Calculates volume
    5. Returns results
    
    Returns
    -------
    result : Dict
        Dictionary with keys:
        - 'success': True if evaluation succeeded
        - 'CL', 'CD', 'Cm': Aerodynamic coefficients
        - 'Volume': Internal volume (m³)
        - 'error': Error message if failed
        - 'time': Evaluation time (seconds)
        
    Warning
    -------
    PySAGAS internally uses multiprocessing. On Windows, this causes issues when
    called from within a multiprocessing Pool (nested spawn). 
    Solution: Set n_cores=1 in WaveriderProblem to disable parallel evaluation.
    """
    start_time = time.time()
    
    # Create isolated temporary directory for this worker
    temp_dir = tempfile.mkdtemp(prefix=f"waverider_worker_{worker_id}_")
    
    try:
        # Extract design variables
        X1, X2, X3, X4 = x
        
        if verbose:
            print(f"  Worker {worker_id}: X=[{X1:.3f}, {X2:.3f}, {X3:.3f}, {X4:.3f}]")
        
        # Step 1: Generate waverider geometry
        try:
            wr = WaveriderGenerator(
                M_inf=M_inf,
                beta=beta,
                height=height,
                width=width,
                dp=[X1, X2, X3, X4],
                n_upper_surface=10000,
                n_shockwave=10000,
                n_planes=n_planes,
                n_streamwise=n_streamwise,
                delta_streamwise=delta_streamwise
            )
        except Exception as e:
            return {
                'success': False,
                'error': f'Waverider generation failed: {str(e)}',
                'time': time.time() - start_time
            }
        
        # Step 2: Calculate volume
        try:
            volume = calculate_volume(wr)
        except Exception as e:
            volume = 0.0
            if verbose:
                print(f"  Worker {worker_id}: Volume calculation failed: {e}")
        
        # Step 2b: Calculate reference area (planform area) for this specific design
        try:
            A_ref_calc, method = calculate_planform_area_from_waverider(wr)
            if verbose:
                print(f"  Worker {worker_id}: A_ref={A_ref_calc:.4f} m² ({method})")
        except Exception as e:
            A_ref_calc = A_ref  # Fallback to passed value
            if verbose:
                print(f"  Worker {worker_id}: A_ref calculation failed, using {A_ref:.4f} m²")
        
        # Step 3: Export CAD and generate mesh
        step_file = os.path.join(temp_dir, 'waverider.step')
        stl_file = os.path.join(temp_dir, 'waverider.stl')
        
        try:
            to_CAD(waverider=wr, sides='both', export=True, filename=step_file, scale=1.0)
        except Exception as e:
            return {
                'success': False,
                'error': f'CAD export failed: {str(e)}',
                'time': time.time() - start_time
            }
        
        # Step 4: Generate mesh
        if not GMSH_AVAILABLE:
            return {
                'success': False,
                'error': 'Gmsh not available',
                'time': time.time() - start_time
            }
        
        try:
            generate_mesh(step_file, stl_file, mesh_size)
        except Exception as e:
            return {
                'success': False,
                'error': f'Mesh generation failed: {str(e)}',
                'time': time.time() - start_time
            }
        
        # Step 5: Run PySAGAS analysis
        if not PYSAGAS_AVAILABLE:
            return {
                'success': False,
                'error': 'PySAGAS not available',
                'time': time.time() - start_time
            }
        
        try:
            CL, CD, Cm = run_pysagas_analysis(
                stl_file, M_inf, pressure, temperature, aoa, A_ref_calc, temp_dir
            )
        except Exception as e:
            return {
                'success': False,
                'error': f'PySAGAS analysis failed: {str(e)}',
                'time': time.time() - start_time
            }
        
        # Success!
        eval_time = time.time() - start_time
        
        if verbose:
            print(f"  Worker {worker_id}: CL={CL:.4f}, CD={CD:.4f}, Vol={volume:.3f} ({eval_time:.1f}s)")
        
        return {
            'success': True,
            'CL': CL,
            'CD': CD,
            'Cm': Cm,
            'Volume': volume,
            'time': eval_time
        }
        
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            if verbose:
                print(f"  Worker {worker_id}: Failed to clean up temp dir: {e}")


def calculate_volume(wr: WaveriderGenerator) -> float:
    """
    Calculate internal volume of waverider.
    
    Uses trapezoidal rule integration over cross-sections.
    
    Parameters
    ----------
    wr : WaveriderGenerator
        Waverider geometry object
        
    Returns
    -------
    volume : float
        Internal volume in m³
        
    Notes
    -----
    The waverider is symmetric, so we calculate the full volume by:
    1. Taking cross-sections along the streamwise direction
    2. Computing the area of each cross-section using shoelace formula
    3. Integrating areas using trapezoidal rule
    """
    # Get streamlines
    upper_streams = wr.upper_surface_streams
    lower_streams = wr.lower_surface_streams
    
    if len(upper_streams) == 0 or len(lower_streams) == 0:
        return 0.0
    
    n_streamwise = upper_streams[0].shape[0]  # Number of points along each stream
    
    # Calculate area of each cross-section at different x-locations
    areas = []
    x_positions = []
    
    for i in range(n_streamwise):
        # Collect all points at this streamwise index
        y_upper = []
        z_upper = []
        y_lower = []
        z_lower = []
        
        for stream in upper_streams:
            if i < stream.shape[0]:  # Safety check
                y_upper.append(stream[i, 1])
                z_upper.append(stream[i, 2])
        
        for stream in lower_streams:
            if i < stream.shape[0]:  # Safety check
                y_lower.append(stream[i, 1])
                z_lower.append(stream[i, 2])
        
        if len(y_upper) == 0 or len(y_lower) == 0:
            continue
        
        # x position (should be same for all streams at this index)
        x_pos = upper_streams[0][i, 0]
        x_positions.append(x_pos)
        
        # Create closed polygon: lower surface + upper surface (reversed)
        z_points = np.concatenate([z_lower, z_upper[::-1]])
        y_points = np.concatenate([y_lower, y_upper[::-1]])
        
        # Shoelace formula for polygon area
        area = 0.5 * abs(np.dot(z_points, np.roll(y_points, 1)) - 
                         np.dot(y_points, np.roll(z_points, 1)))
        
        areas.append(area)
    
    if len(areas) < 2:
        return 0.0
    
    # Integrate using trapezoidal rule
    volume = np.trapz(areas, x_positions)
    
    return abs(volume)  # Ensure positive


def generate_mesh(step_file: str, stl_file: str, mesh_size: float):
    """
    Generate STL mesh from STEP file using gmsh.
    
    Parameters
    ----------
    step_file : str
        Path to STEP file
    stl_file : str
        Path to output STL file
    mesh_size : float
        Mesh size parameter (smaller = finer mesh)
        
    Notes
    -----
    Uses subprocess to avoid thread/signal issues with Gmsh on Windows.
    """
    import subprocess
    import sys
    
    # Create a small Python script to run gmsh
    mesh_script = f'''
import gmsh
import sys

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("General.Verbosity", 0)

try:
    gmsh.model.add("waverider")
    gmsh.model.occ.importShapes(r"{step_file}")
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
    
    gmsh.option.setNumber("Mesh.MeshSizeMin", {mesh_size * 0.5})
    gmsh.option.setNumber("Mesh.MeshSizeMax", {mesh_size * 2.0})
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    
    gmsh.model.mesh.generate(2)
    gmsh.write(r"{stl_file}")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
finally:
    gmsh.finalize()
'''
    
    # Write script to temp file
    script_file = os.path.join(os.path.dirname(step_file), "mesh_script.py")
    with open(script_file, 'w') as f:
        f.write(mesh_script)
    
    try:
        # Run in subprocess
        result = subprocess.run(
            [sys.executable, script_file],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode != 0 or "ERROR" in result.stdout:
            error_msg = result.stderr or result.stdout
            raise RuntimeError(f"Gmsh subprocess failed: {error_msg}")
        
        if not os.path.exists(stl_file):
            raise RuntimeError("STL file was not created")
            
    finally:
        # Clean up script file
        try:
            os.remove(script_file)
        except:
            pass


def run_pysagas_analysis(
    stl_file: str,
    M_inf: float,
    pressure: float,
    temperature: float,
    aoa: float,
    A_ref: float,
    work_dir: str
) -> Tuple[float, float, float]:
    """
    Run PySAGAS aerodynamic analysis.
    
    Parameters
    ----------
    stl_file : str
        Path to STL mesh file
    M_inf : float
        Freestream Mach number
    pressure : float
        Freestream static pressure (Pa)
    temperature : float
        Freestream static temperature (K)
    aoa : float
        Angle of attack (degrees)
    A_ref : float
        Reference area (m²) - not used by PySAGAS coefficients() but kept for API consistency
    work_dir : str
        Working directory for PySAGAS temp files
    
    Returns
    -------
    CL : float
        Lift coefficient
    CD : float
        Drag coefficient  
    Cm : float
        Moment coefficient
        
    Notes
    -----
    This version bypasses PySAGAS's multiprocessing STL loader which causes
    issues on Windows. Instead, we load the STL with meshio and create
    Cell objects manually.
    """
    # Import PySAGAS modules
    from pysagas.cfd import OPM
    from pysagas.flow import FlowState
    from pysagas.geometry import Cell, Vector
    import meshio
    
    # Change to work directory (PySAGAS creates temp files)
    original_dir = os.getcwd()
    os.chdir(work_dir)
    
    try:
        # Load STL using meshio directly (single-threaded, Windows-safe)
        mesh = meshio.read(stl_file)
        
        # Extract vertices and faces
        points = mesh.points  # Nx3 array of vertices
        
        # Get triangle cells - meshio stores them differently depending on format
        triangles = None
        for cell_block in mesh.cells:
            if cell_block.type == 'triangle':
                triangles = cell_block.data
                break
        
        if triangles is None:
            raise ValueError("No triangles found in STL file")
        
        # Create PySAGAS Cell objects manually
        cells = []
        for tri in triangles:
            # Get the three vertices of this triangle
            p0 = points[tri[0]]
            p1 = points[tri[1]]
            p2 = points[tri[2]]
            
            # Create Vector objects for each vertex
            v0 = Vector(x=float(p0[0]), y=float(p0[1]), z=float(p0[2]))
            v1 = Vector(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))
            v2 = Vector(x=float(p2[0]), y=float(p2[1]), z=float(p2[2]))
            
            # Create Cell from vertices
            cell = Cell.from_points([v0, v1, v2])
            cells.append(cell)
        
        # Create freestream flow state
        freestream = FlowState(mach=M_inf, pressure=pressure, temperature=temperature)
        
        # Instantiate solver
        solver = OPM(cells, freestream)
        
        # Run solver at specified AoA
        result = solver.solve(aoa=aoa)
        
        # Get aerodynamic coefficients from flow_result
        CL, CD, Cm = solver.flow_result.coefficients()
        
        # Optionally save results for debugging
        # solver.save("waverider")
        
        return CL, CD, Cm
        
    finally:
        os.chdir(original_dir)


if __name__ == '__main__':
    # Test the optimization problem
    print("Testing WaveriderProblem...")
    
    # Create problem
    problem = WaveriderProblem(
        M_inf=5.0,
        beta=15.0,
        altitude=25000.0,
        height=1.34,
        width=3.0,
        mesh_size=0.3,  # Coarse mesh for testing
        objectives=[
            {'name': 'CD', 'mode': 'minimize'},
            {'name': 'Volume', 'mode': 'maximize'}
        ],
        constraints=[
            {'name': 'design_space', 'active': True},
            {'name': 'CL_min', 'value': 0.8, 'active': True}
        ],
        n_cores=1,
        verbose=True
    )
    
    print(f"\nProblem created:")
    print(f"  Variables: {problem.n_var}")
    print(f"  Objectives: {problem.n_obj}")
    print(f"  Constraints: {problem.n_constr}")
    
    # Test design space constraint
    print("\nTesting design space constraint...")
    X_test = np.array([
        [0.11, 0.63, 0.0, 0.46],   # Valid design from paper
        [0.5, 0.9, 0.5, 0.5],      # Invalid (too much X2)
        [0.0, 0.1, 0.5, 0.5],      # Valid
    ])
    
    valid = problem.check_design_space_constraint(X_test)
    for i, (x, v) in enumerate(zip(X_test, valid)):
        print(f"  Design {i+1}: X={x} -> {'Valid' if v else 'Invalid'}")
    
    print("\n✓ WaveriderProblem test complete!")
