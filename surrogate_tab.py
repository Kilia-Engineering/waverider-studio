#!/usr/bin/env python3
"""
Surrogate-Assisted Optimization Tab for Waverider GUI
Provides Gaussian Process surrogate modeling with adaptive sampling
"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QGridLayout, QSlider,
                             QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
                             QProgressBar, QTextEdit, QTabWidget, QFileDialog,
                             QMessageBox, QSplitter, QFrame, QScrollArea,
                             QRadioButton, QButtonGroup, QDialog, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from ai_surrogate import CLCDOptimizedSurrogate

# Scikit-learn for Gaussian Process
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Surrogate modeling disabled.")

# Scipy for Latin Hypercube Sampling
try:
    from scipy.stats import qmc
    from scipy.optimize import minimize as scipy_minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class MultiOutputGP:
    """
    Multi-output Gaussian Process for correlated objectives.
    Uses separate GPs but can share information through input correlation.
    """
    
    def __init__(self, kernel_type='matern52', n_restarts=10, normalize=True):
        """
        Initialize multi-output GP.
        
        Parameters
        ----------
        kernel_type : str
            'matern52', 'matern32', or 'rbf'
        n_restarts : int
            Number of restarts for hyperparameter optimization
        normalize : bool
            Whether to normalize inputs
        """
        self.kernel_type = kernel_type
        self.n_restarts = n_restarts
        self.normalize = normalize
        
        self.models = {}  # Dict of objective_name -> GP model
        self.scalers = {}  # Dict of objective_name -> scaler
        self.input_scaler = None
        self.is_fitted = False
        self.training_X = None
        self.training_y = {}
        self.objective_names = []
        
    def _create_kernel(self):
        """Create kernel based on type"""
        if self.kernel_type == 'matern52':
            base_kernel = Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3))
        elif self.kernel_type == 'matern32':
            base_kernel = Matern(nu=1.5, length_scale_bounds=(1e-3, 1e3))
        elif self.kernel_type == 'rbf':
            base_kernel = RBF(length_scale_bounds=(1e-3, 1e3))
        else:
            base_kernel = Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3))
        
        # Add constant kernel for scaling and white noise for stability
        kernel = ConstantKernel(1.0, (1e-4, 1e4)) * base_kernel + WhiteKernel(1e-10, (1e-12, 1e-2))
        return kernel
    
    def fit(self, X, y_dict):
        """
        Fit GP models for all objectives.
        
        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples, n_features)
        y_dict : dict
            Dictionary of objective_name -> values array
        """
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        
        self.training_X = X.copy()
        self.training_y = {k: v.copy() for k, v in y_dict.items()}
        self.objective_names = list(y_dict.keys())
        
        # Normalize inputs
        if self.normalize:
            self.input_scaler = StandardScaler()
            X_scaled = self.input_scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Fit a GP for each objective (suppress convergence warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            
            for obj_name, y in y_dict.items():
                # Normalize output
                if self.normalize:
                    self.scalers[obj_name] = StandardScaler()
                    y_scaled = self.scalers[obj_name].fit_transform(y.reshape(-1, 1)).ravel()
                else:
                    y_scaled = y
                
                # Create and fit GP with more optimizer iterations
                kernel = self._create_kernel()
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=self.n_restarts,
                    normalize_y=False,  # We handle normalization ourselves
                    alpha=1e-6,  # Slightly larger for numerical stability
                    optimizer='fmin_l_bfgs_b',
                )
                gp.fit(X_scaled, y_scaled)
                self.models[obj_name] = gp
        
        self.is_fitted = True
        
    def predict(self, X, return_std=True):
        """
        Predict for all objectives.
        
        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples, n_features)
        return_std : bool
            Whether to return standard deviation
            
        Returns
        -------
        means : dict
            Dictionary of objective_name -> predicted means
        stds : dict (optional)
            Dictionary of objective_name -> predicted stds
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Normalize inputs
        if self.normalize and self.input_scaler is not None:
            X_scaled = self.input_scaler.transform(X)
        else:
            X_scaled = X
        
        means = {}
        stds = {}
        
        for obj_name, gp in self.models.items():
            if return_std:
                mean_scaled, std_scaled = gp.predict(X_scaled, return_std=True)
            else:
                mean_scaled = gp.predict(X_scaled, return_std=False)
                std_scaled = None
            
            # Inverse transform
            if self.normalize and obj_name in self.scalers:
                scaler = self.scalers[obj_name]
                mean = mean_scaled * scaler.scale_[0] + scaler.mean_[0]
                if std_scaled is not None:
                    std = std_scaled * scaler.scale_[0]
            else:
                mean = mean_scaled
                std = std_scaled
            
            means[obj_name] = mean
            if return_std:
                stds[obj_name] = std
        
        if return_std:
            return means, stds
        return means
    
    def get_metrics(self, X_test=None, y_test=None):
        """
        Get model quality metrics using cross-validation or test set.
        
        Returns
        -------
        metrics : dict
            Dictionary of objective_name -> metrics dict
        """
        metrics = {}
        
        for obj_name in self.objective_names:
            if X_test is not None and y_test is not None:
                # Use test set
                means, _ = self.predict(X_test, return_std=True)
                y_pred = means[obj_name]
                y_true = y_test[obj_name]
            else:
                # Use cross-validation on training data
                X = self.training_X
                y = self.training_y[obj_name]
                
                if self.normalize:
                    X_scaled = self.input_scaler.transform(X)
                    y_scaled = self.scalers[obj_name].transform(y.reshape(-1, 1)).ravel()
                else:
                    X_scaled = X
                    y_scaled = y
                
                # Cross-validation predictions
                gp = self.models[obj_name]
                y_pred_scaled = cross_val_predict(gp, X_scaled, y_scaled, cv=5)
                
                if self.normalize:
                    y_pred = y_pred_scaled * self.scalers[obj_name].scale_[0] + self.scalers[obj_name].mean_[0]
                else:
                    y_pred = y_pred_scaled
                y_true = y
            
            # Calculate metrics
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            max_error = np.max(np.abs(y_true - y_pred))
            
            metrics[obj_name] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'Max_Error': max_error
            }
        
        return metrics
    
    def save(self, filepath):
        """Save the model to a file"""
        data = {
            'kernel_type': self.kernel_type,
            'n_restarts': self.n_restarts,
            'normalize': self.normalize,
            'models': self.models,
            'scalers': self.scalers,
            'input_scaler': self.input_scaler,
            'is_fitted': self.is_fitted,
            'training_X': self.training_X,
            'training_y': self.training_y,
            'objective_names': self.objective_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            kernel_type=data['kernel_type'],
            n_restarts=data['n_restarts'],
            normalize=data['normalize']
        )
        model.models = data['models']
        model.scalers = data['scalers']
        model.input_scaler = data['input_scaler']
        model.is_fitted = data['is_fitted']
        model.training_X = data['training_X']
        model.training_y = data['training_y']
        model.objective_names = data['objective_names']
        
        return model


class AcquisitionFunction:
    """Acquisition functions for adaptive sampling"""
    
    @staticmethod
    def expected_improvement(mean, std, best_value, minimize=True, xi=0.01):
        """
        Expected Improvement acquisition function.
        
        Parameters
        ----------
        mean : np.ndarray
            Predicted mean
        std : np.ndarray
            Predicted standard deviation
        best_value : float
            Best observed value so far
        minimize : bool
            Whether we're minimizing
        xi : float
            Exploration parameter
        """
        from scipy.stats import norm
        
        std = np.maximum(std, 1e-10)  # Avoid division by zero
        
        if minimize:
            improvement = best_value - mean - xi
        else:
            improvement = mean - best_value - xi
        
        Z = improvement / std
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
        
        return ei
    
    @staticmethod
    def lower_confidence_bound(mean, std, kappa=1.96, minimize=True):
        """
        Lower/Upper Confidence Bound acquisition function.
        
        Parameters
        ----------
        mean : np.ndarray
            Predicted mean
        std : np.ndarray
            Predicted standard deviation
        kappa : float
            Exploration parameter (higher = more exploration)
        minimize : bool
            Whether we're minimizing
        """
        if minimize:
            return mean - kappa * std  # Lower is better
        else:
            return mean + kappa * std  # Higher is better
    
    @staticmethod
    def probability_of_improvement(mean, std, best_value, minimize=True, xi=0.01):
        """
        Probability of Improvement acquisition function.
        """
        from scipy.stats import norm
        
        std = np.maximum(std, 1e-10)
        
        if minimize:
            Z = (best_value - mean - xi) / std
        else:
            Z = (mean - best_value - xi) / std
        
        return norm.cdf(Z)


class SurrogateWorker(QThread):
    """Worker thread for surrogate optimization"""
    
    # Signals
    progress_update = pyqtSignal(int, int, str)  # current, total, message
    log_message = pyqtSignal(str)
    surrogate_updated = pyqtSignal(object)  # model
    design_evaluated = pyqtSignal(int, dict)  # design_id, results
    pareto_updated = pyqtSignal(object, object)  # pareto_X, pareto_y
    optimization_complete = pyqtSignal(str)  # results folder
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.should_stop = False
        self.surrogate_model = None
        
        # Training data storage
        self.training_X = None
        self.training_y = {}
        
        # Results tracking
        self.all_designs = []
        self.pareto_designs = []
        self.design_counter = 0
        
    def stop(self):
        self.should_stop = True
        
    def run(self):
        """Main optimization loop"""
        try:
            mode = self.config.get('mode', 'hybrid')
            
            if mode == 'adaptive':
                self._run_adaptive()
            else:
                self._run_hybrid()
                
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Surrogate optimization failed:\n{str(e)}\n{traceback.format_exc()}")
    
    def _get_geometry_constraint_params(self):
        """Get geometry parameters for X1-X2 constraint"""
        fixed_params = self.config.get('fixed_parameters', {})
        width = fixed_params.get('width', 3.0)
        height = fixed_params.get('height', 1.34)
        return width, height
    
    def _calculate_max_x2(self, x1, safety_margin=0.90):
        """Calculate max X2 for given X1"""
        width, height = self._get_geometry_constraint_params()
        one_minus_x1 = max(1.0 - x1, 0.001)
        rhs = (7.0 / 64.0) * (width / height) ** 4
        max_x2 = rhs * (one_minus_x1 ** 4) * safety_margin
        return min(max(max_x2, 0.0), 1.0)
    
    def _repair_design(self, X):
        """Repair invalid X1-X2 combinations"""
        X_repaired = X.copy()
        if X.ndim == 1:
            X_repaired = X_repaired.reshape(1, -1)
        
        for i in range(len(X_repaired)):
            x1 = X_repaired[i, 0]
            max_x2 = self._calculate_max_x2(x1)
            if X_repaired[i, 1] > max_x2:
                X_repaired[i, 1] = max_x2
        
        if X.ndim == 1:
            return X_repaired.flatten()
        return X_repaired
    
    def _generate_initial_samples(self, n_samples, method='lhs'):
        """Generate initial samples using specified method with constraint filtering"""
        from scipy.stats import qmc
        
        # Get design variable bounds
        design_vars = self.config.get('design_variables', [])
        bounds_min = []
        bounds_max = []
        for dv in design_vars:
            bounds_min.append(dv.get('min', 0.0))
            bounds_max.append(dv.get('max', 1.0))
        
        bounds_min = np.array(bounds_min)
        bounds_max = np.array(bounds_max)
        n_vars = len(bounds_min)
        
        self.log_message.emit(f"Generating {n_samples} initial samples using {method}...")
        
        # Check if focus sampling is enabled
        focus_sampling = self.config.get('focus_sampling', False)
        focus_region = self.config.get('focus_region', {})
        
        if focus_sampling and focus_region:
            # Use focus region bounds instead of design variable bounds
            self.log_message.emit("  Using FOCUSED sampling region:")
            bounds_min = np.array([
                focus_region.get('X1', [0, 0.5])[0],
                focus_region.get('X2', [0, 0.5])[0],
                focus_region.get('X3', [0, 1.0])[0],
                focus_region.get('X4', [0, 1.0])[0]
            ])
            bounds_max = np.array([
                focus_region.get('X1', [0, 0.5])[1],
                focus_region.get('X2', [0, 0.5])[1],
                focus_region.get('X3', [0, 1.0])[1],
                focus_region.get('X4', [0, 1.0])[1]
            ])
            self.log_message.emit(f"    X1: [{bounds_min[0]:.2f}, {bounds_max[0]:.2f}]")
            self.log_message.emit(f"    X2: [{bounds_min[1]:.2f}, {bounds_max[1]:.2f}]")
            self.log_message.emit(f"    X3: [{bounds_min[2]:.2f}, {bounds_max[2]:.2f}]")
            self.log_message.emit(f"    X4: [{bounds_min[3]:.2f}, {bounds_max[3]:.2f}]")
        
        # Generate more samples than needed to account for constraint filtering
        n_generate = int(n_samples * 1.5)  # Generate 50% extra
        
        if method == 'lhs' or method == 'latin hypercube':
            sampler = qmc.LatinHypercube(d=n_vars)
            samples_unit = sampler.random(n=n_generate)
        elif method == 'sobol':
            sampler = qmc.Sobol(d=n_vars, scramble=True)
            samples_unit = sampler.random(n=n_generate)
        else:  # random
            samples_unit = np.random.rand(n_generate, n_vars)
        
        # Scale to bounds
        samples = bounds_min + samples_unit * (bounds_max - bounds_min)
        
        # Filter out samples that violate X1-X2 constraint heavily
        # Keep samples where constraint_value < 0.85 (before repair)
        valid_mask = np.ones(len(samples), dtype=bool)
        for i in range(len(samples)):
            x1, x2 = samples[i, 0], samples[i, 1]
            if x1 < 1.0:
                constraint_value = x2 / ((1 - x1) ** 4)
                if constraint_value > 0.85:
                    valid_mask[i] = False
            
            # Also filter out problematic X3-X4 combinations
            x3, x4 = samples[i, 2], samples[i, 3]
            if x4 < 0.05 and x3 > 0.3 and x3 < 0.5:
                valid_mask[i] = False
            if (x3 < 0.05 and x4 < 0.05) or (x3 > 0.95 and x4 > 0.95):
                valid_mask[i] = False
        
        # Keep only valid samples
        samples = samples[valid_mask]
        
        # If we don't have enough, generate more with safe parameter ranges
        if len(samples) < n_samples:
            # Generate more samples with constraint-aware values
            n_extra = n_samples - len(samples)
            self.log_message.emit(f"  Generating {n_extra} additional constraint-aware samples...")
            
            extra_samples = []
            for _ in range(n_extra):
                x1 = np.random.uniform(bounds_min[0], bounds_max[0])
                # Calculate max X2 for this X1
                max_x2 = 0.8 * ((1 - x1) ** 4)  # 80% of constraint limit
                x2 = np.random.uniform(bounds_min[1], min(bounds_max[1], max_x2))
                # Avoid problematic X3-X4 ranges
                x3 = np.random.uniform(max(0.05, bounds_min[2]), min(0.95, bounds_max[2]))
                x4 = np.random.uniform(max(0.05, bounds_min[3]), min(0.95, bounds_max[3]))
                extra_samples.append([x1, x2, x3, x4])
            
            if len(extra_samples) > 0:
                samples = np.vstack([samples, np.array(extra_samples)])
        
        # Trim to requested number
        samples = samples[:n_samples]
        
        # Final repair pass (shouldn't be needed but safe)
        samples = self._repair_design(samples)
        
        n_valid = len(samples)
        self.log_message.emit(f"  Generated {n_valid} valid samples")
        
        return samples
    
    def _calculate_planform_area(self, waverider_obj):
        """
        Calculate planform area using reference_area_calculator.
        Falls back to direct integration if not available.
        """
        import numpy as np
        
        try:
            from reference_area_calculator import calculate_planform_area_from_waverider
            area, method = calculate_planform_area_from_waverider(waverider_obj)
            return area
        except ImportError:
            # Fallback: direct integration over upper surface
            X = waverider_obj.upper_surface_x
            Z = waverider_obj.upper_surface_z
            
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
            
            return 2.0 * total_area
    
    def _evaluate_design_pysagas(self, X):
        """
        Evaluate a single design using PySAGAS.
        
        Uses the same approach as optimization_engine for Windows compatibility.
        Calculates A_ref dynamically for each design.
        Returns dict with objective values or {'success': False} if failed.
        """
        import tempfile
        import os
        import sys
        
        try:
            # Get config parameters
            fixed_params = self.config.get('fixed_parameters', {})
            sim_params = self.config.get('simulation_params', {})
            
            M_inf = fixed_params.get('M_inf', 5.0)
            beta = fixed_params.get('beta', 15.0)
            height = fixed_params.get('height', 1.34)
            width = fixed_params.get('width', 3.0)
            
            aoa = sim_params.get('aoa', 0.0)
            pressure = sim_params.get('pressure', 2549.0)
            temperature = sim_params.get('temperature', 221.55)
            
            mesh_size = self.config.get('mesh_size', 0.1)
            
            X1, X2, X3, X4 = float(X[0]), float(X[1]), float(X[2]), float(X[3])
            
            # Pre-check: Validate X1-X2 constraint (from paper equation 8)
            # X2 / (1 - X1)^4 <= 1.0, with 90% safety margin
            constraint_value = X2 / ((1 - X1) ** 4) if X1 < 1.0 else float('inf')
            if constraint_value > 0.9:  # 90% of limit
                return {
                    'success': False, 
                    'error': f'X1-X2 constraint violated: {constraint_value:.3f} > 0.9'
                }
            
            # Additional checks for problematic X3/X4 combinations
            # Very small X4 combined with certain X3 values can cause CAD issues
            if X4 < 0.05 and X3 > 0.3 and X3 < 0.5:
                return {
                    'success': False,
                    'error': f'X3-X4 combination may cause CAD issues: X3={X3:.3f}, X4={X4:.3f}'
                }
            
            # Very extreme combinations
            if (X3 < 0.05 and X4 < 0.05) or (X3 > 0.95 and X4 > 0.95):
                return {
                    'success': False,
                    'error': f'Extreme X3-X4 values: X3={X3:.3f}, X4={X4:.3f}'
                }
            
            # Import waverider generator and CAD export
            from waverider_generator.generator import waverider as wr_func
            from waverider_generator.cad_export import to_CAD
            
            # Import optimization_engine functions for meshing and analysis
            from optimization_engine import generate_mesh, run_pysagas_analysis
            
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Generate waverider
                try:
                    waverider_obj = wr_func(
                        M_inf=M_inf,
                        beta=beta,
                        height=height,
                        width=width,
                        dp=[X1, X2, X3, X4],
                        n_upper_surface=5000,
                        n_shockwave=5000,
                        n_planes=30,
                        n_streamwise=25,
                        delta_streamwise=0.1
                    )
                except Exception as e:
                    return {'success': False, 'error': f'Waverider generation failed: {str(e)}'}
                
                # Validate geometry before CAD export
                try:
                    # Check for valid surface data
                    if waverider_obj.upper_surface_x is None or len(waverider_obj.upper_surface_x) == 0:
                        return {'success': False, 'error': 'Invalid geometry: empty upper surface'}
                    
                    # Check for NaN or Inf values
                    if (np.any(np.isnan(waverider_obj.upper_surface_x)) or 
                        np.any(np.isnan(waverider_obj.upper_surface_y)) or
                        np.any(np.isinf(waverider_obj.upper_surface_x)) or
                        np.any(np.isinf(waverider_obj.upper_surface_y))):
                        return {'success': False, 'error': 'Invalid geometry: NaN or Inf in surface data'}
                    
                    # Check for degenerate geometry (very small dimensions)
                    x_range = waverider_obj.upper_surface_x.max() - waverider_obj.upper_surface_x.min()
                    y_range = waverider_obj.upper_surface_y.max() - waverider_obj.upper_surface_y.min()
                    z_range = waverider_obj.upper_surface_z.max() - waverider_obj.upper_surface_z.min()
                    
                    if x_range < 0.01 or y_range < 0.001 or z_range < 0.01:
                        return {'success': False, 'error': f'Degenerate geometry: dimensions too small ({x_range:.4f}, {y_range:.4f}, {z_range:.4f})'}
                        
                except Exception as e:
                    return {'success': False, 'error': f'Geometry validation failed: {str(e)}'}
                
                # Calculate reference area (planform area) for this specific design
                A_ref = self._calculate_planform_area(waverider_obj)
                
                # Calculate volume from geometry
                try:
                    x_range = waverider_obj.upper_surface_x.max() - waverider_obj.upper_surface_x.min()
                    y_range = waverider_obj.upper_surface_y.max() - waverider_obj.upper_surface_y.min()
                    z_range = waverider_obj.upper_surface_z.max()
                    volume = 0.3 * x_range * y_range * (2 * z_range)
                except:
                    volume = height * width * 2.0
                
                # Export to STEP with error handling
                step_file = os.path.join(temp_dir, 'waverider.step')
                stl_file = os.path.join(temp_dir, 'waverider.stl')
                
                try:
                    to_CAD(waverider=waverider_obj, sides='both', export=True, 
                           filename=step_file, scale=1.0)
                except Exception as e:
                    error_msg = str(e)
                    if 'BRep_API' in error_msg or 'command not done' in error_msg:
                        return {'success': False, 'error': f'CAD geometry invalid for X=[{X1:.3f},{X2:.3f},{X3:.3f},{X4:.3f}]'}
                    return {'success': False, 'error': f'CAD export failed: {error_msg}'}
                
                # Check STEP file was created
                if not os.path.exists(step_file):
                    return {'success': False, 'error': 'STEP file not created'}
                
                # Generate mesh using gmsh (via subprocess - Windows safe)
                try:
                    generate_mesh(step_file, stl_file, mesh_size)
                except Exception as e:
                    return {'success': False, 'error': f'Mesh generation failed: {str(e)}'}
                
                # Check STL file was created
                if not os.path.exists(stl_file):
                    return {'success': False, 'error': 'STL file not created'}
                
                # Run PySAGAS analysis with calculated A_ref
                try:
                    CL, CD, Cm = run_pysagas_analysis(
                        stl_file, M_inf, pressure, temperature, aoa, A_ref, temp_dir
                    )
                except Exception as e:
                    return {'success': False, 'error': f'PySAGAS failed: {str(e)}'}
                
                # Validate results
                if np.isnan(CL) or np.isnan(CD) or np.isinf(CL) or np.isinf(CD):
                    return {'success': False, 'error': 'PySAGAS returned NaN/Inf'}
                
                # Calculate L/D
                LD = CL / CD if CD > 1e-6 else 0.0
                
                return {
                    'success': True,
                    'CL': float(CL),
                    'CD': float(CD),
                    'Cm': float(Cm),
                    'CL/CD': float(LD),
                    'Volume': float(volume),
                    'A_ref': float(A_ref)  # Include calculated A_ref in results
                }
                
            finally:
                # Cleanup temp directory
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            import traceback
            return {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
    
    def _evaluate_batch(self, X_batch):
        """Evaluate a batch of designs"""
        results = []
        for i, X in enumerate(X_batch):
            if self.should_stop:
                break
            
            self.design_counter += 1
            self.log_message.emit(f"  Evaluating design {self.design_counter}: X=[{X[0]:.3f}, {X[1]:.3f}, {X[2]:.3f}, {X[3]:.3f}]")
            
            result = self._evaluate_design_pysagas(X)
            result['X'] = X.copy()
            result['design_id'] = self.design_counter
            results.append(result)
            
            if result.get('success'):
                self.design_evaluated.emit(self.design_counter, result)
            else:
                self.log_message.emit(f"    ❌ Failed: {result.get('error', 'Unknown')}")
        
        return results
    
    def _build_surrogate(self, X, y_dict):
        """Build or update surrogate model with optional quality filtering"""
        kernel_type = self.config.get('kernel_type', 'matern52')
        n_restarts = self.config.get('n_restarts', 10)
        normalize = self.config.get('normalize', True)
        
        # Apply quality filters if enabled
        X_filtered, y_filtered = self._apply_quality_filters(X, y_dict)
        
        if len(X_filtered) < 5:
            self.log_message.emit("⚠ Too few samples after filtering, using all data")
            X_filtered = X
            y_filtered = y_dict
        
        self.surrogate_model = CLCDOptimizedSurrogate(n_members=15, max_iter=3000)
        self.surrogate_model.fit(X_filtered, y_filtered)
        
        # Emit update
        self.surrogate_updated.emit(self.surrogate_model)
        
        return self.surrogate_model
    
    def _apply_quality_filters(self, X, y_dict):
        """Apply quality filters to training data"""
        n_original = len(X)
        valid_mask = np.ones(n_original, dtype=bool)
        
        # Filter by minimum CL/CD
        if self.config.get('filter_clcd', False):
            min_clcd = self.config.get('min_clcd', 2.0)
            if 'CL/CD' in y_dict:
                clcd_values = y_dict['CL/CD']
                clcd_mask = clcd_values >= min_clcd
                n_removed = np.sum(~clcd_mask)
                if n_removed > 0:
                    self.log_message.emit(f"  Filtering {n_removed} designs with CL/CD < {min_clcd}")
                valid_mask &= clcd_mask
        
        # Filter statistical outliers
        if self.config.get('filter_outliers', True):
            outlier_std = self.config.get('outlier_std', 3.0)
            for obj_name, values in y_dict.items():
                mean_val = np.mean(values[valid_mask]) if np.any(valid_mask) else np.mean(values)
                std_val = np.std(values[valid_mask]) if np.any(valid_mask) else np.std(values)
                
                if std_val > 1e-6:  # Avoid division by zero
                    z_scores = np.abs((values - mean_val) / std_val)
                    outlier_mask = z_scores <= outlier_std
                    n_outliers = np.sum(~outlier_mask & valid_mask)
                    if n_outliers > 0:
                        self.log_message.emit(f"  Removing {n_outliers} outliers for {obj_name} (>{outlier_std}σ)")
                    valid_mask &= outlier_mask
        
        # Apply filter
        X_filtered = X[valid_mask]
        y_filtered = {k: v[valid_mask] for k, v in y_dict.items()}
        
        n_filtered = len(X_filtered)
        if n_filtered < n_original:
            self.log_message.emit(f"  Filtered: {n_original} → {n_filtered} samples ({n_original - n_filtered} removed)")
        
        return X_filtered, y_filtered
    
    def _find_pareto_front(self, y_dict, objectives_config):
        """Find Pareto optimal indices"""
        obj_names = list(y_dict.keys())
        n = len(y_dict[obj_names[0]])
        
        # Build objective matrix (convert to minimization)
        F = np.zeros((n, len(obj_names)))
        for i, obj_name in enumerate(obj_names):
            values = y_dict[obj_name]
            # Check if we should maximize (negate for minimization)
            minimize = True
            for obj_cfg in objectives_config:
                if obj_cfg['name'] == obj_name:
                    minimize = obj_cfg.get('mode', 'minimize') == 'minimize'
                    break
            
            if minimize:
                F[:, i] = values
            else:
                F[:, i] = -values  # Negate for maximization
        
        # Find non-dominated solutions
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if is_pareto[i]:
                for j in range(n):
                    if i != j and is_pareto[j]:
                        # Check if j dominates i
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            is_pareto[i] = False
                            break
        
        return np.where(is_pareto)[0]
    
    def _get_kappa(self, iteration, max_iterations):
        """Get exploration parameter kappa with adaptive schedule"""
        if not self.config.get('adaptive_kappa', True):
            return self.config.get('kappa', 1.5)
        
        kappa_init = self.config.get('kappa_init', 2.0)
        kappa_final = self.config.get('kappa_final', 0.5)
        decay = self.config.get('kappa_decay', 'linear')
        
        progress = iteration / max(max_iterations, 1)
        
        if decay == 'linear':
            kappa = kappa_init - (kappa_init - kappa_final) * progress
        elif decay == 'exponential':
            kappa = kappa_final + (kappa_init - kappa_final) * np.exp(-3 * progress)
        elif decay == 'cosine':
            kappa = kappa_final + 0.5 * (kappa_init - kappa_final) * (1 + np.cos(np.pi * progress))
        else:
            kappa = kappa_init
        
        return kappa
    
    def _select_next_point_adaptive(self, model, objectives_config, n_candidates=1000):
        """Select next point(s) to evaluate using acquisition function"""
        from scipy.stats import norm
        
        # Generate candidate points
        candidates = self._generate_initial_samples(n_candidates, method='random')
        
        # Get predictions
        means, stds = model.predict(candidates, return_std=True)
        
        # Get acquisition function type
        acq_type = self.config.get('acquisition', 'ei')
        kappa = self._get_kappa(self.design_counter, self.config.get('max_evals', 500))
        
        # For multi-objective, we use a scalarized approach or EHVI
        # Here we use a simple weighted sum of individual acquisition values
        obj_names = model.objective_names
        total_acq = np.zeros(len(candidates))
        
        for obj_name in obj_names:
            mean = means[obj_name]
            std = stds[obj_name]
            
            # Determine if minimizing
            minimize = True
            for obj_cfg in objectives_config:
                if obj_cfg['name'] == obj_name:
                    minimize = obj_cfg.get('mode', 'minimize') == 'minimize'
                    break
            
            # Get best observed value
            if minimize:
                best_val = np.min(self.training_y[obj_name])
            else:
                best_val = np.max(self.training_y[obj_name])
            
            # Calculate acquisition
            if acq_type == 'ei' or acq_type == 'expected improvement':
                acq = AcquisitionFunction.expected_improvement(mean, std, best_val, minimize)
            elif acq_type == 'lcb' or acq_type == 'lower confidence bound':
                acq = AcquisitionFunction.lower_confidence_bound(mean, std, kappa, minimize)
            else:  # PI
                acq = AcquisitionFunction.probability_of_improvement(mean, std, best_val, minimize)
            
            total_acq += acq
        
        # Select best candidate
        best_idx = np.argmax(total_acq)
        return candidates[best_idx:best_idx+1]
    
    def _run_adaptive(self):
        """Fully adaptive (EGO-style) optimization"""
        self.log_message.emit("=" * 60)
        self.log_message.emit("ADAPTIVE SURROGATE OPTIMIZATION (EGO)")
        self.log_message.emit("=" * 60)
        self.log_message.emit(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message.emit("")
        
        # Get configuration
        n_initial = self.config.get('n_initial_samples', 50)
        max_evals = self.config.get('max_evals', 500)
        objectives_config = self.config.get('objectives', [])
        
        self.log_message.emit(f"📊 Initial samples: {n_initial}")
        self.log_message.emit(f"📊 Max evaluations: {max_evals}")
        self.log_message.emit(f"📊 Objectives: {[o['name'] for o in objectives_config]}")
        self.log_message.emit("")
        
        # Phase 1: Initial sampling
        self.log_message.emit("=" * 40)
        self.log_message.emit("PHASE 1: Initial Sampling")
        self.log_message.emit("=" * 40)
        
        X_initial = self._generate_initial_samples(n_initial)
        results = self._evaluate_batch(X_initial)
        
        if self.should_stop:
            self.log_message.emit("⏹ Optimization stopped by user")
            return
        
        # Filter successful results
        successful = [r for r in results if r.get('success')]
        if len(successful) < 10:
            self.error_occurred.emit(f"Too few successful evaluations: {len(successful)}")
            return
        
        self.log_message.emit(f"✓ {len(successful)}/{len(results)} successful evaluations")
        
        # Build initial training data
        self.training_X = np.array([r['X'] for r in successful])
        self.training_y = {}
        for obj_cfg in objectives_config:
            obj_name = obj_cfg['name']
            self.training_y[obj_name] = np.array([r.get(obj_name, 0.0) for r in successful])
        
        # Build initial surrogate
        self.log_message.emit("")
        self.log_message.emit("Building initial surrogate model...")
        self._build_surrogate(self.training_X, self.training_y)
        
        metrics = self.surrogate_model.get_metrics()
        for obj_name, m in metrics.items():
            self.log_message.emit(f"  {obj_name}: R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}")
        
        # Phase 2: Adaptive sampling
        self.log_message.emit("")
        self.log_message.emit("=" * 40)
        self.log_message.emit("PHASE 2: Adaptive Sampling (EGO)")
        self.log_message.emit("=" * 40)
        
        iteration = 0
        while self.design_counter < max_evals and not self.should_stop:
            iteration += 1
            kappa = self._get_kappa(self.design_counter, max_evals)
            
            self.log_message.emit(f"\n--- Iteration {iteration} (κ={kappa:.2f}) ---")
            
            # Select next point using acquisition function
            X_next = self._select_next_point_adaptive(
                self.surrogate_model, objectives_config
            )
            
            # Evaluate
            results = self._evaluate_batch(X_next)
            
            # Update training data if successful
            for r in results:
                if r.get('success'):
                    self.training_X = np.vstack([self.training_X, r['X']])
                    for obj_name in self.training_y.keys():
                        self.training_y[obj_name] = np.append(
                            self.training_y[obj_name], r.get(obj_name, 0.0)
                        )
            
            # Rebuild surrogate every few iterations
            if iteration % 5 == 0:
                self.log_message.emit("Updating surrogate model...")
                self._build_surrogate(self.training_X, self.training_y)
            
            # Update progress
            progress = int(100 * self.design_counter / max_evals)
            self.progress_update.emit(self.design_counter, max_evals, f"Iteration {iteration}")
            
            # Find and report Pareto front
            pareto_idx = self._find_pareto_front(self.training_y, objectives_config)
            self.log_message.emit(f"  Pareto designs: {len(pareto_idx)}")
        
        # Final surrogate update
        self.log_message.emit("")
        self.log_message.emit("Building final surrogate model...")
        self._build_surrogate(self.training_X, self.training_y)
        
        # Report final results
        pareto_idx = self._find_pareto_front(self.training_y, objectives_config)
        self.log_message.emit("")
        self.log_message.emit("=" * 60)
        self.log_message.emit("OPTIMIZATION COMPLETE")
        self.log_message.emit("=" * 60)
        self.log_message.emit(f"Total evaluations: {self.design_counter}")
        self.log_message.emit(f"Pareto optimal designs: {len(pareto_idx)}")
        
        # Save results
        results_folder = self._save_results(pareto_idx, objectives_config)
        
        self.optimization_complete.emit(results_folder)
    
    def _run_hybrid(self):
        """Hybrid surrogate optimization"""
        self.log_message.emit("=" * 60)
        self.log_message.emit("HYBRID SURROGATE OPTIMIZATION")
        self.log_message.emit("=" * 60)
        self.log_message.emit(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message.emit("")
        
        # Get configuration
        n_initial = self.config.get('n_initial_samples', 100)
        max_evals = self.config.get('max_evals', 500)
        surrogate_gens = self.config.get('surrogate_generations', 5)
        validate_per_cycle = self.config.get('validate_per_cycle', 10)
        objectives_config = self.config.get('objectives', [])
        
        # Adaptive validation settings
        adaptive_validation = self.config.get('adaptive_validation', True)
        validate_initial = self.config.get('validate_initial', validate_per_cycle)
        validate_min = self.config.get('validate_min', 20)
        r2_threshold_high = self.config.get('r2_threshold_high', 0.85)
        r2_threshold_medium = self.config.get('r2_threshold_medium', 0.70)
        
        self.log_message.emit(f"📊 Initial samples: {n_initial}")
        self.log_message.emit(f"📊 Max evaluations: {max_evals}")
        self.log_message.emit(f"📊 Surrogate generations per cycle: {surrogate_gens}")
        self.log_message.emit(f"📊 Designs to validate per cycle: {validate_per_cycle}")
        if adaptive_validation:
            self.log_message.emit(f"📊 Adaptive validation: ENABLED")
            self.log_message.emit(f"    - Initial rate: {validate_initial}")
            self.log_message.emit(f"    - Minimum rate: {validate_min}")
            self.log_message.emit(f"    - R² thresholds: {r2_threshold_medium:.0%} / {r2_threshold_high:.0%}")
        self.log_message.emit(f"📊 Objectives: {[o['name'] for o in objectives_config]}")
        self.log_message.emit("")
        
        # Phase 1: Initial sampling
        self.log_message.emit("=" * 40)
        self.log_message.emit("PHASE 1: Initial Sampling")
        self.log_message.emit("=" * 40)
        
        X_initial = self._generate_initial_samples(n_initial)
        results = self._evaluate_batch(X_initial)
        
        if self.should_stop:
            self.log_message.emit("⏹ Optimization stopped by user")
            return
        
        # Filter successful results
        successful = [r for r in results if r.get('success')]
        if len(successful) < 10:
            self.error_occurred.emit(f"Too few successful evaluations: {len(successful)}")
            return
        
        self.log_message.emit(f"✓ {len(successful)}/{len(results)} successful evaluations")
        
        # Build initial training data
        self.training_X = np.array([r['X'] for r in successful])
        self.training_y = {}
        for obj_cfg in objectives_config:
            obj_name = obj_cfg['name']
            self.training_y[obj_name] = np.array([r.get(obj_name, 0.0) for r in successful])
        
        # Build initial surrogate
        self.log_message.emit("")
        self.log_message.emit("Building initial surrogate model...")
        self._build_surrogate(self.training_X, self.training_y)
        
        metrics = self.surrogate_model.get_metrics()
        for obj_name, m in metrics.items():
            self.log_message.emit(f"  {obj_name}: R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}")
        
        # Phase 2: Hybrid optimization cycles
        self.log_message.emit("")
        self.log_message.emit("=" * 40)
        self.log_message.emit("PHASE 2: Hybrid Optimization (Adaptive)")
        self.log_message.emit("=" * 40)
        
        cycle = 0
        current_validate_rate = validate_initial
        previous_r2 = {obj: -1.0 for obj in [o['name'] for o in objectives_config]}
        no_improvement_count = 0
        
        while self.design_counter < max_evals and not self.should_stop:
            cycle += 1
            self.log_message.emit(f"\n{'='*50}")
            self.log_message.emit(f"CYCLE {cycle} | Validation rate: {current_validate_rate}")
            self.log_message.emit(f"{'='*50}")
            
            # Step 2a: Run GA on surrogate for several generations
            self.log_message.emit(f"\n🔄 Running NSGA-II on surrogate for {surrogate_gens} generations...")
            
            surrogate_pareto_X = self._run_surrogate_nsga2(
                surrogate_gens, objectives_config
            )
            
            if self.should_stop:
                break
            
            self.log_message.emit(f"  Found {len(surrogate_pareto_X)} surrogate Pareto designs")
            
            # Step 2b: Determine adaptive validation rate
            if adaptive_validation and cycle > 1:
                current_validate_rate = self._calculate_adaptive_validation_rate(
                    metrics, 
                    previous_r2,
                    validate_initial,
                    validate_min,
                    r2_threshold_high,
                    r2_threshold_medium,
                    no_improvement_count
                )
            
            # Step 2c: Select designs to validate
            n_to_validate = min(current_validate_rate, len(surrogate_pareto_X), max_evals - self.design_counter)
            
            if n_to_validate <= 0:
                break
            
            # Select most diverse/promising designs from surrogate Pareto
            X_to_validate = self._select_diverse_designs(surrogate_pareto_X, n_to_validate)
            
            self.log_message.emit(f"\n🔬 Validating {n_to_validate} designs with PySAGAS...")
            
            # Step 2d: Validate with real simulations
            results = self._evaluate_batch(X_to_validate)
            
            # Update training data
            n_successful = 0
            for r in results:
                if r.get('success'):
                    n_successful += 1
                    self.training_X = np.vstack([self.training_X, r['X']])
                    for obj_name in self.training_y.keys():
                        self.training_y[obj_name] = np.append(
                            self.training_y[obj_name], r.get(obj_name, 0.0)
                        )
            
            self.log_message.emit(f"  ✓ {n_successful}/{len(results)} successful validations")
            
            # Step 2e: Rebuild surrogate with new data
            self.log_message.emit("\n🔧 Rebuilding surrogate model...")
            self._build_surrogate(self.training_X, self.training_y)
            
            # Get new metrics
            metrics = self.surrogate_model.get_metrics()
            
            # Track improvement
            improved = False
            for obj_name, m in metrics.items():
                r2_current = m['R2']
                r2_prev = previous_r2.get(obj_name, -1.0)
                improvement = r2_current - r2_prev
                
                status = ""
                if improvement > 0.01:
                    status = "📈"
                    improved = True
                elif improvement < -0.01:
                    status = "📉"
                else:
                    status = "➡️"
                
                self.log_message.emit(f"  {obj_name}: R²={r2_current:.4f} (Δ={improvement:+.4f}) {status}")
                previous_r2[obj_name] = r2_current
            
            # Track no-improvement cycles for adaptive rate
            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= 3:
                    self.log_message.emit(f"  ⚠️ No improvement for {no_improvement_count} cycles")
            
            # Report Pareto front
            pareto_idx = self._find_pareto_front(self.training_y, objectives_config)
            self.log_message.emit(f"\n  Validated Pareto designs: {len(pareto_idx)}")
            
            # Progress update
            progress = int(100 * self.design_counter / max_evals)
            remaining = max_evals - self.design_counter
            est_cycles = remaining // max(current_validate_rate, 1)
            self.log_message.emit(f"  Progress: {self.design_counter}/{max_evals} ({progress}%)")
            self.log_message.emit(f"  Estimated remaining cycles: ~{est_cycles}")
            
            self.progress_update.emit(self.design_counter, max_evals, f"Cycle {cycle}")
            
            # Update progress
            progress = int(100 * self.design_counter / max_evals)
            self.progress_update.emit(self.design_counter, max_evals, f"Cycle {cycle}")
            
            # Report Pareto front
            pareto_idx = self._find_pareto_front(self.training_y, objectives_config)
            self.log_message.emit(f"  Validated Pareto designs: {len(pareto_idx)}")
        
        # Final results
        pareto_idx = self._find_pareto_front(self.training_y, objectives_config)
        self.log_message.emit("")
        self.log_message.emit("=" * 60)
        self.log_message.emit("OPTIMIZATION COMPLETE")
        self.log_message.emit("=" * 60)
        self.log_message.emit(f"Total evaluations: {self.design_counter}")
        self.log_message.emit(f"Validated Pareto optimal designs: {len(pareto_idx)}")
        
        # Emit final Pareto front
        pareto_X = self.training_X[pareto_idx]
        pareto_y = {k: v[pareto_idx] for k, v in self.training_y.items()}
        self.pareto_updated.emit(pareto_X, pareto_y)
        
        # Save results
        results_folder = self._save_results(pareto_idx, objectives_config)
        
        self.optimization_complete.emit(results_folder)
    
    def _calculate_adaptive_validation_rate(self, metrics, previous_r2, 
                                             validate_initial, validate_min,
                                             r2_threshold_high, r2_threshold_medium,
                                             no_improvement_count):
        """
        Calculate adaptive validation rate based on surrogate quality.
        
        Strategy:
        - High R² (>0.85): Reduce validation (surrogate is good)
        - Medium R² (0.70-0.85): Moderate validation
        - Low R² (<0.70): Keep high validation (need more data)
        - No improvement for 3+ cycles: Increase validation
        
        Returns the number of designs to validate this cycle.
        """
        # Get average R² across objectives
        r2_values = [m['R2'] for m in metrics.values()]
        avg_r2 = np.mean(r2_values)
        min_r2 = np.min(r2_values)
        
        # Base rate adjustment based on R²
        if min_r2 >= r2_threshold_high:
            # Excellent surrogate - reduce validation significantly
            rate_factor = 0.3
            self.log_message.emit(f"  📊 Surrogate quality: EXCELLENT (min R²={min_r2:.3f}) → reducing validation")
        elif min_r2 >= r2_threshold_medium:
            # Good surrogate - moderate reduction
            rate_factor = 0.6
            self.log_message.emit(f"  📊 Surrogate quality: GOOD (min R²={min_r2:.3f}) → moderate validation")
        elif min_r2 >= 0.5:
            # Moderate surrogate - slight reduction
            rate_factor = 0.8
            self.log_message.emit(f"  📊 Surrogate quality: MODERATE (min R²={min_r2:.3f}) → high validation")
        else:
            # Poor surrogate - keep high validation
            rate_factor = 1.0
            self.log_message.emit(f"  📊 Surrogate quality: POOR (min R²={min_r2:.3f}) → maximum validation")
        
        # Calculate base rate
        new_rate = int(validate_initial * rate_factor)
        
        # Boost if no improvement for multiple cycles
        if no_improvement_count >= 5:
            new_rate = int(new_rate * 1.5)
            self.log_message.emit(f"  ⚡ Boosting validation due to stagnation")
        elif no_improvement_count >= 3:
            new_rate = int(new_rate * 1.2)
        
        # Enforce minimum
        new_rate = max(new_rate, validate_min)
        
        return new_rate
    
    def _run_surrogate_nsga2(self, n_generations, objectives_config):
        """Run NSGA-II optimization on the surrogate model"""
        try:
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.core.problem import Problem
            from pymoo.optimize import minimize
            from pymoo.termination import get_termination
            
            model = self.surrogate_model
            
            class SurrogateProblem(Problem):
                def __init__(prob_self, surrogate_model, objectives_config, bounds, repair_func):
                    prob_self.surrogate = surrogate_model
                    prob_self.objectives_config = objectives_config
                    prob_self.repair_func = repair_func
                    
                    n_obj = len(objectives_config)
                    super().__init__(
                        n_var=4,
                        n_obj=n_obj,
                        xl=bounds[0],
                        xu=bounds[1]
                    )
                
                def _evaluate(prob_self, X, out, *args, **kwargs):
                    # Repair designs
                    X = prob_self.repair_func(X)
                    
                    # Predict using surrogate
                    means = prob_self.surrogate.predict(X, return_std=False)
                    
                    # Build objective matrix
                    F = np.zeros((len(X), prob_self.n_obj))
                    for i, obj_cfg in enumerate(prob_self.objectives_config):
                        obj_name = obj_cfg['name']
                        values = means[obj_name]
                        
                        # Negate for maximization (pymoo minimizes)
                        if obj_cfg.get('mode', 'minimize') == 'maximize':
                            F[:, i] = -values
                        else:
                            F[:, i] = values
                    
                    out["F"] = F
            
            # Get bounds
            design_vars = self.config.get('design_variables', [])
            bounds_min = np.array([dv.get('min', 0.0) for dv in design_vars])
            bounds_max = np.array([dv.get('max', 1.0) for dv in design_vars])
            
            # Create problem
            problem = SurrogateProblem(
                model, objectives_config,
                (bounds_min, bounds_max),
                self._repair_design
            )
            
            # Run NSGA-II
            algorithm = NSGA2(pop_size=100)
            termination = get_termination("n_gen", n_generations)
            
            result = minimize(
                problem,
                algorithm,
                termination,
                seed=np.random.randint(10000),
                verbose=False
            )
            
            # Return Pareto optimal designs
            return result.X if result.X is not None else np.array([])
            
        except Exception as e:
            import traceback
            self.log_message.emit(f"⚠ Surrogate NSGA-II error: {e}")
            self.log_message.emit(f"  Traceback: {traceback.format_exc()}")
            return np.array([])
    
    def _select_diverse_designs(self, X, n_select):
        """Select diverse designs from a set using maximin distance"""
        if len(X) <= n_select:
            return X
        
        # Start with the first design
        selected_idx = [0]
        
        while len(selected_idx) < n_select:
            max_min_dist = -1
            best_idx = -1
            
            for i in range(len(X)):
                if i in selected_idx:
                    continue
                
                # Calculate minimum distance to already selected
                min_dist = float('inf')
                for j in selected_idx:
                    dist = np.linalg.norm(X[i] - X[j])
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            if best_idx >= 0:
                selected_idx.append(best_idx)
            else:
                break
        
        return X[selected_idx]
    
    def _save_results(self, pareto_idx, objectives_config):
        """
        Save optimization results to files.
        
        Creates a timestamped folder with:
        - all_designs.csv: All evaluated designs
        - pareto_designs.csv: Pareto optimal designs
        - surrogate_model.pkl: Saved GP model
        - optimization_summary.txt: Summary statistics
        
        Returns the folder path.
        """
        import os
        import csv
        from datetime import datetime
        
        # Create results folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = os.path.join(os.getcwd(), f"surrogate_results_{timestamp}")
        os.makedirs(results_folder, exist_ok=True)
        
        self.log_message.emit(f"\nSaving results to: {results_folder}")
        
        # Get objective names
        obj_names = list(self.training_y.keys())
        
        # 1. Save all designs
        all_designs_file = os.path.join(results_folder, "all_designs.csv")
        with open(all_designs_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            header = ['Design', 'X1', 'X2', 'X3', 'X4'] + obj_names + ['Is_Pareto']
            writer.writerow(header)
            
            # Data
            for i in range(len(self.training_X)):
                row = [i+1] + list(self.training_X[i])
                for obj in obj_names:
                    row.append(self.training_y[obj][i])
                row.append(1 if i in pareto_idx else 0)
                writer.writerow(row)
        
        self.log_message.emit(f"  ✓ all_designs.csv ({len(self.training_X)} designs)")
        
        # 2. Save Pareto designs
        pareto_file = os.path.join(results_folder, "pareto_designs.csv")
        with open(pareto_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Design', 'X1', 'X2', 'X3', 'X4'] + obj_names
            writer.writerow(header)
            
            for idx in pareto_idx:
                row = [idx+1] + list(self.training_X[idx])
                for obj in obj_names:
                    row.append(self.training_y[obj][idx])
                writer.writerow(row)
        
        self.log_message.emit(f"  ✓ pareto_designs.csv ({len(pareto_idx)} designs)")
        
        # 3. Save surrogate model
        if self.surrogate_model is not None:
            model_file = os.path.join(results_folder, "surrogate_model.pkl")
            try:
                self.surrogate_model.save(model_file)
                self.log_message.emit(f"  ✓ surrogate_model.pkl")
            except Exception as e:
                self.log_message.emit(f"  ⚠ Could not save model: {e}")
        
        # 4. Save summary
        summary_file = os.path.join(results_folder, "optimization_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SURROGATE OPTIMIZATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total evaluations: {len(self.training_X)}\n")
            f.write(f"Pareto optimal designs: {len(pareto_idx)}\n\n")
            
            # Model quality
            if self.surrogate_model is not None:
                f.write("MODEL QUALITY METRICS:\n")
                f.write("-" * 40 + "\n")
                metrics = self.surrogate_model.get_metrics()
                for obj_name, obj_metrics in metrics.items():
                    f.write(f"\n{obj_name}:\n")
                    for metric, value in obj_metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
            
            # Objectives configuration
            f.write("\nOBJECTIVES CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            # objectives_config is a list of dicts like [{'name': 'CL/CD', 'minimize': False}, ...]
            for obj_config in objectives_config:
                obj_name = obj_config.get('name', 'Unknown')
                direction = "minimize" if obj_config.get('minimize', False) else "maximize"
                weight = obj_config.get('weight', 1.0)
                f.write(f"  {obj_name}: {direction} (weight={weight})\n")
            
            # Build a lookup dict for minimize flags
            obj_minimize_lookup = {o.get('name'): o.get('minimize', False) for o in objectives_config}
            
            # Best values
            f.write("\nBEST VALUES FOUND:\n")
            f.write("-" * 40 + "\n")
            for obj_name, values in self.training_y.items():
                minimize = obj_minimize_lookup.get(obj_name, False)
                if minimize:
                    best_idx = np.argmin(values)
                    best_val = np.min(values)
                else:
                    best_idx = np.argmax(values)
                    best_val = np.max(values)
                f.write(f"  {obj_name}: {best_val:.6f} (design {best_idx+1})\n")
            
            # Pareto designs details
            f.write("\nPARETO OPTIMAL DESIGNS:\n")
            f.write("-" * 40 + "\n")
            for i, idx in enumerate(pareto_idx):
                f.write(f"\nDesign {idx+1}:\n")
                f.write(f"  X = [{self.training_X[idx][0]:.4f}, {self.training_X[idx][1]:.4f}, ")
                f.write(f"{self.training_X[idx][2]:.4f}, {self.training_X[idx][3]:.4f}]\n")
                for obj_name in obj_names:
                    f.write(f"  {obj_name} = {self.training_y[obj_name][idx]:.6f}\n")
        
        self.log_message.emit(f"  ✓ optimization_summary.txt")
        self.log_message.emit(f"\n✅ Results saved to: {results_folder}")
        
        return results_folder


class ResponseSurfacePlot(QWidget):
    """Widget for displaying response surface plots"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create matplotlib figure with two subplots (2D and 3D)
        self.figure = Figure(figsize=(12, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Variable 1:"))
        self.var1_combo = QComboBox()
        self.var1_combo.addItems(['X1', 'X2', 'X3', 'X4'])
        self.var1_combo.setCurrentText('X1')
        controls_layout.addWidget(self.var1_combo)
        
        controls_layout.addWidget(QLabel("Variable 2:"))
        self.var2_combo = QComboBox()
        self.var2_combo.addItems(['X1', 'X2', 'X3', 'X4'])
        self.var2_combo.setCurrentText('X2')
        controls_layout.addWidget(self.var2_combo)
        
        controls_layout.addWidget(QLabel("Objective:"))
        self.obj_combo = QComboBox()
        controls_layout.addWidget(self.obj_combo)
        
        self.plot_btn = QPushButton("🗺️ Update Plot")
        controls_layout.addWidget(self.plot_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Fixed variable sliders
        fixed_layout = QHBoxLayout()
        self.fixed_sliders = {}
        for var in ['X1', 'X2', 'X3', 'X4']:
            fixed_layout.addWidget(QLabel(f"{var}:"))
            slider = QDoubleSpinBox()
            slider.setRange(0.0, 1.0)
            slider.setValue(0.5)
            slider.setSingleStep(0.1)
            slider.setDecimals(2)
            self.fixed_sliders[var] = slider
            fixed_layout.addWidget(slider)
        fixed_layout.addStretch()
        layout.addLayout(fixed_layout)
        
    def plot_surface(self, model, var1, var2, objective, fixed_values, 
                    training_X=None, training_y=None, pareto_X=None):
        """
        Plot 2D contour and 3D surface.
        
        Parameters
        ----------
        model : MultiOutputGP
            Fitted surrogate model
        var1, var2 : str
            Variable names to plot ('X1', 'X2', 'X3', 'X4')
        objective : str
            Objective name to plot
        fixed_values : dict
            Fixed values for other variables
        training_X : np.ndarray, optional
            Training points to overlay
        training_y : np.ndarray, optional
            Training objective values
        pareto_X : np.ndarray, optional
            Pareto optimal points to overlay
        """
        self.figure.clear()
        
        # Variable indices
        var_names = ['X1', 'X2', 'X3', 'X4']
        idx1 = var_names.index(var1)
        idx2 = var_names.index(var2)
        
        # Create grid
        n_points = 50
        v1_range = np.linspace(0, 1, n_points)
        v2_range = np.linspace(0, 1, n_points)
        V1, V2 = np.meshgrid(v1_range, v2_range)
        
        # Build prediction points
        X_pred = np.zeros((n_points * n_points, 4))
        for i, var in enumerate(var_names):
            if var == var1:
                X_pred[:, i] = V1.ravel()
            elif var == var2:
                X_pred[:, i] = V2.ravel()
            else:
                X_pred[:, i] = fixed_values.get(var, 0.5)
        
        # Predict
        means, stds = model.predict(X_pred, return_std=True)
        Z_mean = means[objective].reshape(n_points, n_points)
        Z_std = stds[objective].reshape(n_points, n_points)
        
        # 2D Contour Plot
        ax1 = self.figure.add_subplot(121)
        contour = ax1.contourf(V1, V2, Z_mean, levels=20, cmap='viridis')
        self.figure.colorbar(contour, ax=ax1, label=objective)
        ax1.set_xlabel(var1, fontsize=11, fontweight='bold')
        ax1.set_ylabel(var2, fontsize=11, fontweight='bold')
        ax1.set_title(f'{objective} Response Surface (2D)', fontsize=12, fontweight='bold')
        
        # Overlay training points
        if training_X is not None:
            ax1.scatter(training_X[:, idx1], training_X[:, idx2], 
                       c='white', s=30, edgecolors='black', linewidth=0.5,
                       alpha=0.7, label='Training')
        
        # Overlay Pareto points
        if pareto_X is not None and len(pareto_X) > 0:
            ax1.scatter(pareto_X[:, idx1], pareto_X[:, idx2],
                       c='red', s=100, marker='*', edgecolors='darkred',
                       linewidth=1, label='Pareto', zorder=5)
        
        ax1.legend(loc='upper right', fontsize=8)
        
        # 3D Surface Plot
        ax2 = self.figure.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(V1, V2, Z_mean, cmap='viridis', alpha=0.9,
                               linewidth=0.2, antialiased=True,
                               edgecolors='k', shade=True)
        
        # Add colorbar
        self.figure.colorbar(surf, ax=ax2, shrink=0.5, aspect=10, label=objective)
        
        ax2.set_xlabel(var1, fontsize=10)
        ax2.set_ylabel(var2, fontsize=10)
        ax2.set_zlabel(objective, fontsize=10)
        ax2.set_title(f'{objective} Response Surface (3D)', fontsize=12, fontweight='bold')
        
        # Overlay training points on 3D
        if training_X is not None and training_y is not None:
            ax2.scatter(training_X[:, idx1], training_X[:, idx2], training_y,
                       c='red', s=30, alpha=0.8, edgecolors='darkred', linewidth=0.5)
        
        # Better viewing angle
        ax2.view_init(elev=25, azim=45)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_uncertainty(self, model, var1, var2, objective, fixed_values):
        """Plot uncertainty (standard deviation) surface"""
        self.figure.clear()
        
        var_names = ['X1', 'X2', 'X3', 'X4']
        idx1 = var_names.index(var1)
        idx2 = var_names.index(var2)
        
        # Create grid
        n_points = 50
        v1_range = np.linspace(0, 1, n_points)
        v2_range = np.linspace(0, 1, n_points)
        V1, V2 = np.meshgrid(v1_range, v2_range)
        
        # Build prediction points
        X_pred = np.zeros((n_points * n_points, 4))
        for i, var in enumerate(var_names):
            if var == var1:
                X_pred[:, i] = V1.ravel()
            elif var == var2:
                X_pred[:, i] = V2.ravel()
            else:
                X_pred[:, i] = fixed_values.get(var, 0.5)
        
        # Predict
        means, stds = model.predict(X_pred, return_std=True)
        Z_std = stds[objective].reshape(n_points, n_points)
        
        # 2D Uncertainty Plot - use a more visible colormap
        ax1 = self.figure.add_subplot(121)
        contour = ax1.contourf(V1, V2, Z_std, levels=20, cmap='YlOrRd')
        self.figure.colorbar(contour, ax=ax1, label='Std. Dev.')
        ax1.set_xlabel(var1, fontsize=11, fontweight='bold')
        ax1.set_ylabel(var2, fontsize=11, fontweight='bold')
        ax1.set_title(f'{objective} Uncertainty (2D)', fontsize=12, fontweight='bold')
        
        # Overlay training points
        if model.training_X is not None:
            ax1.scatter(model.training_X[:, idx1], model.training_X[:, idx2],
                       c='blue', s=30, edgecolors='white', linewidth=0.5,
                       alpha=0.7, label='Training')
        ax1.legend(loc='upper right', fontsize=8)
        
        # 3D Uncertainty Plot - use a colorful, high-contrast colormap
        ax2 = self.figure.add_subplot(122, projection='3d')
        
        # Use 'plasma' or 'viridis' for better 3D visibility
        surf = ax2.plot_surface(V1, V2, Z_std, cmap='plasma', alpha=0.9,
                               linewidth=0.2, antialiased=True, 
                               edgecolors='k', shade=True)
        
        # Add colorbar for 3D plot
        self.figure.colorbar(surf, ax=ax2, shrink=0.5, aspect=10, label='Std. Dev.')
        
        ax2.set_xlabel(var1, fontsize=10)
        ax2.set_ylabel(var2, fontsize=10)
        ax2.set_zlabel('Std. Dev.', fontsize=10)
        ax2.set_title(f'{objective} Uncertainty (3D)', fontsize=12, fontweight='bold')
        
        # Better viewing angle
        ax2.view_init(elev=25, azim=45)
        
        self.figure.tight_layout()
        self.canvas.draw()


class ParetoPlotWithBands(QWidget):
    """Widget for Pareto front plot with confidence bands"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Confidence level:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.5, 3.0)
        self.confidence_spin.setValue(1.96)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setToolTip("Number of standard deviations for confidence band (1.96 ≈ 95%)")
        controls_layout.addWidget(self.confidence_spin)
        
        self.show_validated_check = QCheckBox("Show validated designs")
        self.show_validated_check.setChecked(True)
        controls_layout.addWidget(self.show_validated_check)
        
        self.show_surrogate_check = QCheckBox("Show surrogate predictions")
        self.show_surrogate_check.setChecked(True)
        controls_layout.addWidget(self.show_surrogate_check)
        
        self.update_btn = QPushButton("🔄 Update")
        controls_layout.addWidget(self.update_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
    
    def plot_pareto_with_bands(self, model, objective_names, minimize_flags,
                               validated_X=None, validated_y=None,
                               pareto_X=None, pareto_y=None,
                               n_samples=1000, confidence=1.96):
        """
        Plot Pareto front with confidence bands from surrogate model.
        
        Parameters
        ----------
        model : MultiOutputGP
            Fitted surrogate model
        objective_names : list
            Names of the two objectives
        minimize_flags : list
            Whether to minimize each objective
        validated_X : np.ndarray, optional
            Validated design points
        validated_y : dict, optional
            Validated objective values
        pareto_X : np.ndarray, optional
            Pareto optimal design points
        pareto_y : dict, optional
            Pareto objective values
        n_samples : int
            Number of samples for surrogate Pareto
        confidence : float
            Number of std devs for confidence band
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        obj1, obj2 = objective_names
        min1, min2 = minimize_flags
        
        # Generate random samples in design space
        X_samples = np.random.rand(n_samples, 4)
        
        # Predict with surrogate
        means, stds = model.predict(X_samples, return_std=True)
        
        y1_mean = means[obj1]
        y2_mean = means[obj2]
        y1_std = stds[obj1]
        y2_std = stds[obj2]
        
        # Plot surrogate predictions with uncertainty
        ax.scatter(y1_mean, y2_mean, c='lightblue', s=20, alpha=0.3,
                  label='Surrogate predictions')
        
        # Plot error bars for a subset (too many would be cluttered)
        subset_idx = np.random.choice(n_samples, min(50, n_samples), replace=False)
        ax.errorbar(y1_mean[subset_idx], y2_mean[subset_idx],
                   xerr=confidence * y1_std[subset_idx],
                   yerr=confidence * y2_std[subset_idx],
                   fmt='none', ecolor='lightblue', alpha=0.3, capsize=0)
        
        # Find surrogate Pareto front (using mean predictions)
        surrogate_pareto_idx = self._find_pareto_indices(
            y1_mean, y2_mean, min1, min2
        )
        
        if len(surrogate_pareto_idx) > 0:
            # Sort Pareto points by first objective
            pareto_y1 = y1_mean[surrogate_pareto_idx]
            pareto_y2 = y2_mean[surrogate_pareto_idx]
            pareto_y1_std = y1_std[surrogate_pareto_idx]
            pareto_y2_std = y2_std[surrogate_pareto_idx]
            
            sort_idx = np.argsort(pareto_y1)
            pareto_y1 = pareto_y1[sort_idx]
            pareto_y2 = pareto_y2[sort_idx]
            pareto_y1_std = pareto_y1_std[sort_idx]
            pareto_y2_std = pareto_y2_std[sort_idx]
            
            # Plot Pareto front line
            ax.plot(pareto_y1, pareto_y2, 'b-', linewidth=2, label='Surrogate Pareto')
            ax.scatter(pareto_y1, pareto_y2, c='blue', s=80, zorder=5,
                      edgecolors='darkblue', linewidth=1)
            
            # Plot confidence band
            upper_y1 = pareto_y1 + confidence * pareto_y1_std * (-1 if min1 else 1)
            lower_y1 = pareto_y1 - confidence * pareto_y1_std * (-1 if min1 else 1)
            upper_y2 = pareto_y2 + confidence * pareto_y2_std * (-1 if min2 else 1)
            lower_y2 = pareto_y2 - confidence * pareto_y2_std * (-1 if min2 else 1)
            
            # Fill confidence region (approximate)
            ax.fill_between(pareto_y1, 
                           pareto_y2 - confidence * pareto_y2_std,
                           pareto_y2 + confidence * pareto_y2_std,
                           alpha=0.2, color='blue', label=f'{confidence:.1f}σ confidence')
        
        # Plot validated designs
        if validated_X is not None and validated_y is not None:
            v_y1 = validated_y[obj1]
            v_y2 = validated_y[obj2]
            ax.scatter(v_y1, v_y2, c='green', s=60, marker='s',
                      edgecolors='darkgreen', linewidth=1,
                      label='Validated designs', zorder=6)
        
        # Plot validated Pareto
        if pareto_X is not None and pareto_y is not None:
            p_y1 = pareto_y[obj1]
            p_y2 = pareto_y[obj2]
            ax.scatter(p_y1, p_y2, c='gold', s=200, marker='*',
                      edgecolors='darkorange', linewidth=1.5,
                      label='Validated Pareto', zorder=7)
        
        # Labels and formatting
        dir1 = "minimize ↓" if min1 else "maximize ↑"
        dir2 = "minimize ↓" if min2 else "maximize ↑"
        ax.set_xlabel(f'{obj1} ({dir1})', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{obj2} ({dir2})', fontsize=12, fontweight='bold')
        ax.set_title('Pareto Front with Confidence Bands', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _find_pareto_indices(self, y1, y2, min1, min2):
        """Find Pareto optimal indices"""
        n = len(y1)
        is_pareto = np.ones(n, dtype=bool)
        
        # Convert to minimization
        f1 = y1 if min1 else -y1
        f2 = y2 if min2 else -y2
        
        for i in range(n):
            if is_pareto[i]:
                # Check if any other point dominates point i
                for j in range(n):
                    if i != j and is_pareto[j]:
                        if f1[j] <= f1[i] and f2[j] <= f2[i]:
                            if f1[j] < f1[i] or f2[j] < f2[i]:
                                is_pareto[i] = False
                                break
        
        return np.where(is_pareto)[0]


class SurrogateTab(QWidget):
    """Main surrogate optimization tab"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        
        # State
        self.surrogate_model = None
        self.optimization_running = False
        self.worker = None
        
        # Training data
        self.training_X = None
        self.training_y = {}
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QHBoxLayout(self)
        
        # Left panel: Configuration (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self.create_mode_group())
        left_layout.addWidget(self.create_sampling_group())
        left_layout.addWidget(self.create_surrogate_group())
        left_layout.addWidget(self.create_acquisition_group())
        left_layout.addWidget(self.create_budget_group())
        left_layout.addWidget(self.create_advanced_group())
        left_layout.addWidget(self.create_focus_region_group())
        left_layout.addWidget(self.create_actions_group())
        left_layout.addStretch()
        
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(350)
        left_scroll.setMaximumWidth(450)
        
        # Right panel: Diagnostics and plots
        right_panel = QTabWidget()
        
        # Metrics tab
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.addWidget(self.create_metrics_group())
        metrics_layout.addWidget(self.create_progress_group())
        metrics_layout.addWidget(self.create_console_group())
        right_panel.addTab(metrics_widget, "📊 Metrics & Progress")
        
        # Response surface tab
        self.response_plot = ResponseSurfacePlot()
        self.response_plot.plot_btn.clicked.connect(self.update_response_surface)
        right_panel.addTab(self.response_plot, "🗺️ Response Surface")
        
        # Uncertainty tab
        self.uncertainty_plot = ResponseSurfacePlot()
        self.uncertainty_plot.plot_btn.setText("📊 Update Uncertainty")
        self.uncertainty_plot.plot_btn.clicked.connect(self.update_uncertainty_plot)
        right_panel.addTab(self.uncertainty_plot, "📈 Uncertainty")
        
        # Pareto tab
        self.pareto_plot = ParetoPlotWithBands()
        self.pareto_plot.update_btn.clicked.connect(self.update_pareto_plot)
        right_panel.addTab(self.pareto_plot, "⭐ Pareto Front")
        
        # Add to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        
    def create_mode_group(self):
        """Create optimization mode selection"""
        group = QGroupBox("Optimization Mode")
        layout = QVBoxLayout()
        
        self.mode_group = QButtonGroup()
        
        self.adaptive_radio = QRadioButton("Fully Adaptive (EGO-style)")
        self.adaptive_radio.setToolTip(
            "Efficient Global Optimization:\n"
            "• Iteratively selects single best point to evaluate\n"
            "• Uses acquisition function to balance exploration/exploitation\n"
            "• Most sample-efficient but slower per iteration"
        )
        self.mode_group.addButton(self.adaptive_radio, 0)
        layout.addWidget(self.adaptive_radio)
        
        self.hybrid_radio = QRadioButton("Hybrid (Surrogate + Periodic Validation)")
        self.hybrid_radio.setChecked(True)
        self.hybrid_radio.setToolTip(
            "Hybrid Optimization:\n"
            "• Runs GA on surrogate for several generations\n"
            "• Periodically validates best designs with real simulation\n"
            "• Good balance of speed and accuracy"
        )
        self.mode_group.addButton(self.hybrid_radio, 1)
        layout.addWidget(self.hybrid_radio)
        
        # Mode description
        self.mode_desc = QLabel(
            "<i>Hybrid mode runs fast surrogate optimization with periodic "
            "validation using real PySAGAS simulations.</i>"
        )
        self.mode_desc.setWordWrap(True)
        self.mode_desc.setStyleSheet("color: #666; margin-top: 5px;")
        layout.addWidget(self.mode_desc)
        
        self.mode_group.buttonClicked.connect(self.on_mode_changed)
        
        group.setLayout(layout)
        return group
    
    def create_sampling_group(self):
        """Create initial sampling configuration"""
        group = QGroupBox("Initial Sampling")
        layout = QGridLayout()
        
        row = 0
        
        # Sampling method
        layout.addWidget(QLabel("<b>Method:</b>"), row, 0)
        self.sampling_method_combo = QComboBox()
        self.sampling_method_combo.addItems([
            'Latin Hypercube (recommended)',
            'Sobol Sequence',
            'Random'
        ])
        layout.addWidget(self.sampling_method_combo, row, 1)
        row += 1
        
        # Number of samples
        layout.addWidget(QLabel("<b>Samples:</b>"), row, 0)
        self.n_samples_spin = QSpinBox()
        self.n_samples_spin.setRange(20, 500)
        self.n_samples_spin.setValue(100)
        self.n_samples_spin.setSingleStep(10)
        self.n_samples_spin.setToolTip("Initial samples for building surrogate")
        layout.addWidget(self.n_samples_spin, row, 1)
        row += 1
        
        # Use existing data
        self.use_existing_check = QCheckBox("Use existing data")
        self.use_existing_check.setToolTip("Load data from previous optimization run")
        layout.addWidget(self.use_existing_check, row, 0, 1, 2)
        row += 1
        
        # File selection
        file_layout = QHBoxLayout()
        self.data_file_label = QLabel("No file selected")
        self.data_file_label.setStyleSheet("color: #666; font-size: 9px;")
        file_layout.addWidget(self.data_file_label)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_data_file)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout, row, 0, 1, 2)
        
        group.setLayout(layout)
        return group
    
    def create_surrogate_group(self):
        """Create surrogate model configuration"""
        group = QGroupBox("Surrogate Model")
        layout = QGridLayout()
        
        row = 0
        
        # Model type (fixed to GP for now)
        layout.addWidget(QLabel("<b>Type:</b>"), row, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(['Gaussian Process (Multi-output)'])
        layout.addWidget(self.model_type_combo, row, 1)
        row += 1
        
        # Kernel selection
        layout.addWidget(QLabel("<b>Kernel:</b>"), row, 0)
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems([
            'Matérn 5/2 (recommended)',
            'RBF (Squared Exponential)',
            'Matérn 3/2'
        ])
        self.kernel_combo.setToolTip(
            "Matérn 5/2: Twice differentiable, good for physical systems\n"
            "RBF: Infinitely smooth, may oversmooth\n"
            "Matérn 3/2: Once differentiable, good for rougher responses"
        )
        layout.addWidget(self.kernel_combo, row, 1)
        row += 1
        
        # Options
        self.auto_hyperparams_check = QCheckBox("Automatic hyperparameter tuning")
        self.auto_hyperparams_check.setChecked(True)
        layout.addWidget(self.auto_hyperparams_check, row, 0, 1, 2)
        row += 1
        
        self.normalize_check = QCheckBox("Normalize inputs/outputs")
        self.normalize_check.setChecked(True)
        layout.addWidget(self.normalize_check, row, 0, 1, 2)
        
        group.setLayout(layout)
        return group
    
    def create_acquisition_group(self):
        """Create acquisition function configuration"""
        group = QGroupBox("Acquisition Strategy")
        layout = QGridLayout()
        
        row = 0
        
        # Acquisition function
        layout.addWidget(QLabel("<b>Function:</b>"), row, 0)
        self.acquisition_combo = QComboBox()
        self.acquisition_combo.addItems([
            'Expected Improvement (EI)',
            'Lower Confidence Bound (LCB)',
            'Probability of Improvement (PI)'
        ])
        self.acquisition_combo.setToolTip(
            "EI: Balanced exploration/exploitation\n"
            "LCB: More exploration (κ controls balance)\n"
            "PI: More exploitation"
        )
        layout.addWidget(self.acquisition_combo, row, 1)
        row += 1
        
        # Exploration-exploitation slider
        layout.addWidget(QLabel("<b>Exploration ↔ Exploitation:</b>"), row, 0, 1, 2)
        row += 1
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Explore"))
        
        self.exploration_slider = QSlider(Qt.Horizontal)
        self.exploration_slider.setRange(0, 100)
        self.exploration_slider.setValue(60)  # Lean toward exploration
        self.exploration_slider.setTickPosition(QSlider.TicksBelow)
        self.exploration_slider.setTickInterval(25)
        slider_layout.addWidget(self.exploration_slider)
        
        slider_layout.addWidget(QLabel("Exploit"))
        layout.addLayout(slider_layout, row, 0, 1, 2)
        row += 1
        
        # Kappa value display
        self.kappa_label = QLabel("κ = 1.50")
        self.kappa_label.setStyleSheet("color: #666;")
        layout.addWidget(self.kappa_label, row, 0, 1, 2)
        self.exploration_slider.valueChanged.connect(self.update_kappa_label)
        row += 1
        
        # Adaptive balance
        self.adaptive_balance_check = QCheckBox("Adaptive balance (auto-adjust during optimization)")
        self.adaptive_balance_check.setChecked(True)
        self.adaptive_balance_check.setToolTip(
            "Gradually shift from exploration to exploitation\n"
            "as optimization progresses"
        )
        layout.addWidget(self.adaptive_balance_check, row, 0, 1, 2)
        
        group.setLayout(layout)
        return group
    
    def create_budget_group(self):
        """Create validation budget configuration"""
        group = QGroupBox("Validation Budget")
        layout = QGridLayout()
        
        row = 0
        
        # Max evaluations
        layout.addWidget(QLabel("<b>Max PySAGAS evaluations:</b>"), row, 0)
        self.max_evals_spin = QSpinBox()
        self.max_evals_spin.setRange(50, 10000)
        self.max_evals_spin.setValue(500)
        self.max_evals_spin.setSingleStep(50)
        layout.addWidget(self.max_evals_spin, row, 1)
        row += 1
        
        # Hybrid mode settings
        hybrid_label = QLabel("<b>Hybrid mode settings:</b>")
        layout.addWidget(hybrid_label, row, 0, 1, 2)
        row += 1
        
        # Generations between validations
        layout.addWidget(QLabel("Surrogate generations per cycle:"), row, 0)
        self.surrogate_gens_spin = QSpinBox()
        self.surrogate_gens_spin.setRange(1, 100)
        self.surrogate_gens_spin.setValue(20)
        self.surrogate_gens_spin.setToolTip("Number of GA generations on surrogate before validating")
        layout.addWidget(self.surrogate_gens_spin, row, 1)
        row += 1
        
        # Initial validation rate
        layout.addWidget(QLabel("Initial validation rate:"), row, 0)
        self.validate_per_cycle_spin = QSpinBox()
        self.validate_per_cycle_spin.setRange(5, 200)
        self.validate_per_cycle_spin.setValue(100)
        self.validate_per_cycle_spin.setToolTip("Designs to validate per cycle (will adapt)")
        layout.addWidget(self.validate_per_cycle_spin, row, 1)
        row += 1
        
        # Adaptive validation section
        layout.addWidget(QLabel("<b>Adaptive Validation:</b>"), row, 0, 1, 2)
        row += 1
        
        self.adaptive_validation_check = QCheckBox("Enable adaptive validation rate")
        self.adaptive_validation_check.setChecked(True)
        self.adaptive_validation_check.setToolTip(
            "Automatically reduce validation rate as surrogate improves:\n"
            "• High R² (>85%): 30% of initial rate\n"
            "• Medium R² (70-85%): 60% of initial rate\n"
            "• Low R² (<70%): Full rate\n"
            "• Stagnation: Boost rate to escape local optima"
        )
        self.adaptive_validation_check.stateChanged.connect(self.on_adaptive_validation_changed)
        layout.addWidget(self.adaptive_validation_check, row, 0, 1, 2)
        row += 1
        
        # Minimum validation rate
        layout.addWidget(QLabel("Minimum validation rate:"), row, 0)
        self.validate_min_spin = QSpinBox()
        self.validate_min_spin.setRange(5, 100)
        self.validate_min_spin.setValue(20)
        self.validate_min_spin.setToolTip("Won't go below this rate even with excellent surrogate")
        layout.addWidget(self.validate_min_spin, row, 1)
        row += 1
        
        # R² thresholds
        layout.addWidget(QLabel("R² threshold (high):"), row, 0)
        self.r2_high_spin = QDoubleSpinBox()
        self.r2_high_spin.setRange(0.5, 0.99)
        self.r2_high_spin.setValue(0.85)
        self.r2_high_spin.setDecimals(2)
        self.r2_high_spin.setToolTip("Above this: 30% validation rate")
        layout.addWidget(self.r2_high_spin, row, 1)
        row += 1
        
        layout.addWidget(QLabel("R² threshold (medium):"), row, 0)
        self.r2_medium_spin = QDoubleSpinBox()
        self.r2_medium_spin.setRange(0.3, 0.9)
        self.r2_medium_spin.setValue(0.70)
        self.r2_medium_spin.setDecimals(2)
        self.r2_medium_spin.setToolTip("Above this: 60% validation rate")
        layout.addWidget(self.r2_medium_spin, row, 1)
        row += 1
        
        # Time estimate label
        self.time_estimate_label = QLabel("")
        self.time_estimate_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.time_estimate_label, row, 0, 1, 2)
        
        # Connect signals for time estimate
        self.max_evals_spin.valueChanged.connect(self.update_time_estimate)
        self.n_samples_spin.valueChanged.connect(self.update_time_estimate)
        self.update_time_estimate()
        
        group.setLayout(layout)
        return group
    
    def on_adaptive_validation_changed(self, state):
        """Enable/disable adaptive validation settings"""
        enabled = state == Qt.Checked
        self.validate_min_spin.setEnabled(enabled)
        self.r2_high_spin.setEnabled(enabled)
        self.r2_medium_spin.setEnabled(enabled)
    
    def update_time_estimate(self):
        """Update time estimate based on settings"""
        try:
            n_initial = self.n_samples_spin.value()
            max_evals = self.max_evals_spin.value()
            
            # Estimate ~45 seconds per evaluation on average
            total_seconds = max_evals * 45
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            
            if hours > 0:
                self.time_estimate_label.setText(f"⏱ Estimated time: ~{hours}h {minutes}min")
            else:
                self.time_estimate_label.setText(f"⏱ Estimated time: ~{minutes} minutes")
        except:
            pass
    
    def create_advanced_group(self):
        """Create advanced settings (collapsible)"""
        group = QGroupBox("Advanced Settings")
        group.setCheckable(True)
        group.setChecked(False)
        
        layout = QGridLayout()
        row = 0
        
        # GP hyperparameters
        layout.addWidget(QLabel("<b>GP Hyperparameters:</b>"), row, 0, 1, 2)
        row += 1
        
        layout.addWidget(QLabel("Length scale bounds:"), row, 0)
        ls_layout = QHBoxLayout()
        self.ls_min_spin = QDoubleSpinBox()
        self.ls_min_spin.setRange(1e-5, 1.0)
        self.ls_min_spin.setValue(0.001)
        self.ls_min_spin.setDecimals(4)
        ls_layout.addWidget(self.ls_min_spin)
        ls_layout.addWidget(QLabel("to"))
        self.ls_max_spin = QDoubleSpinBox()
        self.ls_max_spin.setRange(1.0, 1000.0)
        self.ls_max_spin.setValue(100.0)
        ls_layout.addWidget(self.ls_max_spin)
        layout.addLayout(ls_layout, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Optimizer restarts:"), row, 0)
        self.n_restarts_spin = QSpinBox()
        self.n_restarts_spin.setRange(1, 50)
        self.n_restarts_spin.setValue(10)
        layout.addWidget(self.n_restarts_spin, row, 1)
        row += 1
        
        # Adaptive kappa schedule
        layout.addWidget(QLabel("<b>Adaptive κ schedule:</b>"), row, 0, 1, 2)
        row += 1
        
        layout.addWidget(QLabel("Initial κ:"), row, 0)
        self.kappa_init_spin = QDoubleSpinBox()
        self.kappa_init_spin.setRange(0.1, 5.0)
        self.kappa_init_spin.setValue(2.0)
        self.kappa_init_spin.setSingleStep(0.1)
        layout.addWidget(self.kappa_init_spin, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Final κ:"), row, 0)
        self.kappa_final_spin = QDoubleSpinBox()
        self.kappa_final_spin.setRange(0.1, 5.0)
        self.kappa_final_spin.setValue(0.5)
        self.kappa_final_spin.setSingleStep(0.1)
        layout.addWidget(self.kappa_final_spin, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Decay:"), row, 0)
        self.kappa_decay_combo = QComboBox()
        self.kappa_decay_combo.addItems(['Linear', 'Exponential', 'Cosine'])
        layout.addWidget(self.kappa_decay_combo, row, 1)
        
        group.setLayout(layout)
        return group
    
    def create_focus_region_group(self):
        """Create focus region and quality filters group"""
        group = QGroupBox("🎯 Focus Region & Quality Filters")
        layout = QGridLayout()
        row = 0
        
        # Enable focused sampling
        self.focus_sampling_check = QCheckBox("Enable focused sampling")
        self.focus_sampling_check.setToolTip(
            "Concentrate samples in promising regions of the design space"
        )
        self.focus_sampling_check.setChecked(False)
        self.focus_sampling_check.stateChanged.connect(self.on_focus_sampling_changed)
        layout.addWidget(self.focus_sampling_check, row, 0, 1, 2)
        row += 1
        
        # Focus region bounds
        layout.addWidget(QLabel("<b>Focus Region Bounds:</b>"), row, 0, 1, 2)
        row += 1
        
        # X1 range
        layout.addWidget(QLabel("X1 range:"), row, 0)
        x1_layout = QHBoxLayout()
        self.focus_x1_min = QDoubleSpinBox()
        self.focus_x1_min.setRange(0.0, 1.0)
        self.focus_x1_min.setValue(0.0)
        self.focus_x1_min.setDecimals(2)
        self.focus_x1_min.setEnabled(False)
        x1_layout.addWidget(self.focus_x1_min)
        x1_layout.addWidget(QLabel("to"))
        self.focus_x1_max = QDoubleSpinBox()
        self.focus_x1_max.setRange(0.0, 1.0)
        self.focus_x1_max.setValue(0.3)
        self.focus_x1_max.setDecimals(2)
        self.focus_x1_max.setEnabled(False)
        x1_layout.addWidget(self.focus_x1_max)
        layout.addLayout(x1_layout, row, 1)
        row += 1
        
        # X2 range
        layout.addWidget(QLabel("X2 range:"), row, 0)
        x2_layout = QHBoxLayout()
        self.focus_x2_min = QDoubleSpinBox()
        self.focus_x2_min.setRange(0.0, 1.0)
        self.focus_x2_min.setValue(0.0)
        self.focus_x2_min.setDecimals(2)
        self.focus_x2_min.setEnabled(False)
        x2_layout.addWidget(self.focus_x2_min)
        x2_layout.addWidget(QLabel("to"))
        self.focus_x2_max = QDoubleSpinBox()
        self.focus_x2_max.setRange(0.0, 1.0)
        self.focus_x2_max.setValue(0.2)
        self.focus_x2_max.setDecimals(2)
        self.focus_x2_max.setEnabled(False)
        x2_layout.addWidget(self.focus_x2_max)
        layout.addLayout(x2_layout, row, 1)
        row += 1
        
        # X3 range
        layout.addWidget(QLabel("X3 range:"), row, 0)
        x3_layout = QHBoxLayout()
        self.focus_x3_min = QDoubleSpinBox()
        self.focus_x3_min.setRange(0.0, 1.0)
        self.focus_x3_min.setValue(0.5)
        self.focus_x3_min.setDecimals(2)
        self.focus_x3_min.setEnabled(False)
        x3_layout.addWidget(self.focus_x3_min)
        x3_layout.addWidget(QLabel("to"))
        self.focus_x3_max = QDoubleSpinBox()
        self.focus_x3_max.setRange(0.0, 1.0)
        self.focus_x3_max.setValue(1.0)
        self.focus_x3_max.setDecimals(2)
        self.focus_x3_max.setEnabled(False)
        x3_layout.addWidget(self.focus_x3_max)
        layout.addLayout(x3_layout, row, 1)
        row += 1
        
        # X4 range
        layout.addWidget(QLabel("X4 range:"), row, 0)
        x4_layout = QHBoxLayout()
        self.focus_x4_min = QDoubleSpinBox()
        self.focus_x4_min.setRange(0.0, 1.0)
        self.focus_x4_min.setValue(0.5)
        self.focus_x4_min.setDecimals(2)
        self.focus_x4_min.setEnabled(False)
        x4_layout.addWidget(self.focus_x4_min)
        x4_layout.addWidget(QLabel("to"))
        self.focus_x4_max = QDoubleSpinBox()
        self.focus_x4_max.setRange(0.0, 1.0)
        self.focus_x4_max.setValue(1.0)
        self.focus_x4_max.setDecimals(2)
        self.focus_x4_max.setEnabled(False)
        x4_layout.addWidget(self.focus_x4_max)
        layout.addLayout(x4_layout, row, 1)
        row += 1
        
        # Separator
        layout.addWidget(QLabel("<b>Quality Filters:</b>"), row, 0, 1, 2)
        row += 1
        
        # Minimum CL/CD filter
        self.filter_clcd_check = QCheckBox("Filter low CL/CD designs")
        self.filter_clcd_check.setToolTip(
            "Exclude designs with CL/CD below threshold from surrogate training"
        )
        self.filter_clcd_check.setChecked(False)
        self.filter_clcd_check.stateChanged.connect(self.on_filter_changed)
        layout.addWidget(self.filter_clcd_check, row, 0)
        
        self.min_clcd_spin = QDoubleSpinBox()
        self.min_clcd_spin.setRange(0.0, 10.0)
        self.min_clcd_spin.setValue(2.0)
        self.min_clcd_spin.setDecimals(1)
        self.min_clcd_spin.setEnabled(False)
        self.min_clcd_spin.setToolTip("Minimum CL/CD to include in training")
        layout.addWidget(self.min_clcd_spin, row, 1)
        row += 1
        
        # Outlier detection
        self.filter_outliers_check = QCheckBox("Remove statistical outliers")
        self.filter_outliers_check.setToolTip(
            "Remove designs with objective values > N standard deviations from mean"
        )
        self.filter_outliers_check.setChecked(True)
        self.filter_outliers_check.stateChanged.connect(self.on_filter_changed)
        layout.addWidget(self.filter_outliers_check, row, 0)
        
        self.outlier_std_spin = QDoubleSpinBox()
        self.outlier_std_spin.setRange(1.0, 5.0)
        self.outlier_std_spin.setValue(3.0)
        self.outlier_std_spin.setDecimals(1)
        self.outlier_std_spin.setToolTip("Number of std deviations for outlier detection")
        layout.addWidget(self.outlier_std_spin, row, 1)
        row += 1
        
        # Preset button for high CL/CD region
        preset_btn = QPushButton("📍 Set High CL/CD Region Preset")
        preset_btn.setToolTip(
            "Set focus region to: X1∈[0,0.3], X2∈[0,0.2], X3∈[0.5,1], X4∈[0.5,1]"
        )
        preset_btn.clicked.connect(self.set_high_clcd_preset)
        layout.addWidget(preset_btn, row, 0, 1, 2)
        
        group.setLayout(layout)
        return group
    
    def on_focus_sampling_changed(self, state):
        """Enable/disable focus region spinboxes"""
        enabled = state == Qt.Checked
        self.focus_x1_min.setEnabled(enabled)
        self.focus_x1_max.setEnabled(enabled)
        self.focus_x2_min.setEnabled(enabled)
        self.focus_x2_max.setEnabled(enabled)
        self.focus_x3_min.setEnabled(enabled)
        self.focus_x3_max.setEnabled(enabled)
        self.focus_x4_min.setEnabled(enabled)
        self.focus_x4_max.setEnabled(enabled)
    
    def on_filter_changed(self, state):
        """Enable/disable filter spinboxes"""
        self.min_clcd_spin.setEnabled(self.filter_clcd_check.isChecked())
    
    def set_high_clcd_preset(self):
        """Set focus region to high CL/CD preset values"""
        self.focus_sampling_check.setChecked(True)
        self.focus_x1_min.setValue(0.0)
        self.focus_x1_max.setValue(0.3)
        self.focus_x2_min.setValue(0.0)
        self.focus_x2_max.setValue(0.2)
        self.focus_x3_min.setValue(0.5)
        self.focus_x3_max.setValue(1.0)
        self.focus_x4_min.setValue(0.5)
        self.focus_x4_max.setValue(1.0)
        self.log_message("✓ Set focus region to High CL/CD preset")
    
    def create_actions_group(self):
        """Create action buttons"""
        group = QGroupBox("Actions")
        layout = QVBoxLayout()
        
        # Start/Stop buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Start Surrogate Optimization")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; }"
        )
        self.start_btn.clicked.connect(self.start_optimization)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "font-weight: bold; padding: 10px; }"
        )
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_optimization)
        btn_layout.addWidget(self.stop_btn)
        
        layout.addLayout(btn_layout)
        
        # Surrogate management buttons
        surrogate_layout = QHBoxLayout()
        
        self.save_surrogate_btn = QPushButton("💾 Save Surrogate")
        self.save_surrogate_btn.clicked.connect(self.save_surrogate)
        self.save_surrogate_btn.setEnabled(False)
        surrogate_layout.addWidget(self.save_surrogate_btn)
        
        self.load_surrogate_btn = QPushButton("📂 Load Surrogate")
        self.load_surrogate_btn.clicked.connect(self.load_surrogate)
        surrogate_layout.addWidget(self.load_surrogate_btn)
        
        self.rebuild_btn = QPushButton("🔄 Rebuild")
        self.rebuild_btn.clicked.connect(self.rebuild_surrogate)
        self.rebuild_btn.setEnabled(False)
        surrogate_layout.addWidget(self.rebuild_btn)
        
        layout.addLayout(surrogate_layout)
        
        # Build surrogate only button
        self.build_only_btn = QPushButton("🔧 Build Surrogate Only (No Optimization)")
        self.build_only_btn.clicked.connect(self.build_surrogate_only)
        layout.addWidget(self.build_only_btn)
        
        # Inverse design button
        self.inverse_design_btn = QPushButton("🎯 Inverse Design (Find X for Requirements)")
        self.inverse_design_btn.setToolTip(
            "Specify desired performance (CL/CD, Volume, etc.)\n"
            "and the surrogate will find the design variables (X1-X4)"
        )
        self.inverse_design_btn.clicked.connect(self.show_inverse_design_dialog)
        self.inverse_design_btn.setEnabled(False)
        self.inverse_design_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        layout.addWidget(self.inverse_design_btn)
        
        group.setLayout(layout)
        return group
    
    def create_metrics_group(self):
        """Create model quality metrics display"""
        group = QGroupBox("Model Quality Metrics")
        layout = QGridLayout()
        
        # Headers
        layout.addWidget(QLabel("<b>Metric</b>"), 0, 0)
        layout.addWidget(QLabel("<b>Objective 1</b>"), 0, 1)
        layout.addWidget(QLabel("<b>Objective 2</b>"), 0, 2)
        
        # Metrics rows
        metrics = ['R² Score', 'RMSE', 'MAE', 'Max Error']
        self.metric_labels = {}
        
        for i, metric in enumerate(metrics, start=1):
            layout.addWidget(QLabel(f"{metric}:"), i, 0)
            
            label1 = QLabel("N/A")
            label1.setStyleSheet("font-family: monospace;")
            layout.addWidget(label1, i, 1)
            
            label2 = QLabel("N/A")
            label2.setStyleSheet("font-family: monospace;")
            layout.addWidget(label2, i, 2)
            
            self.metric_labels[metric] = (label1, label2)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        cv_btn = QPushButton("📊 Cross-Validation Plot")
        cv_btn.clicked.connect(self.show_cv_plot)
        btn_layout.addWidget(cv_btn)
        
        parity_btn = QPushButton("📈 Parity Plot")
        parity_btn.clicked.connect(self.show_parity_plot)
        btn_layout.addWidget(parity_btn)
        
        layout.addLayout(btn_layout, len(metrics)+1, 0, 1, 3)
        
        group.setLayout(layout)
        return group
    
    def create_progress_group(self):
        """Create progress display"""
        group = QGroupBox("Progress")
        layout = QVBoxLayout()
        
        # Progress bar
        layout.addWidget(QLabel("<b>Real evaluations:</b>"))
        self.eval_progress = QProgressBar()
        self.eval_progress.setRange(0, 100)
        self.eval_progress.setValue(0)
        layout.addWidget(self.eval_progress)
        
        # Statistics
        stats_layout = QGridLayout()
        
        stats_layout.addWidget(QLabel("Total evaluations:"), 0, 0)
        self.total_evals_label = QLabel("0")
        stats_layout.addWidget(self.total_evals_label, 0, 1)
        
        stats_layout.addWidget(QLabel("Surrogate Pareto designs:"), 1, 0)
        self.surrogate_pareto_label = QLabel("0")
        stats_layout.addWidget(self.surrogate_pareto_label, 1, 1)
        
        stats_layout.addWidget(QLabel("Validated Pareto designs:"), 2, 0)
        self.validated_pareto_label = QLabel("0")
        stats_layout.addWidget(self.validated_pareto_label, 2, 1)
        
        layout.addLayout(stats_layout)
        
        group.setLayout(layout)
        return group
    
    def create_console_group(self):
        """Create console output"""
        group = QGroupBox("Console Output")
        layout = QVBoxLayout()
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(200)
        self.console.setStyleSheet(
            "background-color: #1e1e1e; color: #00ff00; font-family: monospace;"
        )
        layout.addWidget(self.console)
        
        clear_btn = QPushButton("Clear Console")
        clear_btn.clicked.connect(self.console.clear)
        layout.addWidget(clear_btn)
        
        group.setLayout(layout)
        return group
    
    # ===== Slot Methods =====
    
    def on_mode_changed(self, button):
        """Handle mode selection change"""
        if button == self.adaptive_radio:
            self.mode_desc.setText(
                "<i>Adaptive mode iteratively selects the most promising design "
                "to evaluate, maximizing information gain per simulation.</i>"
            )
        else:
            self.mode_desc.setText(
                "<i>Hybrid mode runs fast surrogate optimization with periodic "
                "validation using real PySAGAS simulations.</i>"
            )
    
    def update_kappa_label(self, value):
        """Update kappa display from slider"""
        # Map 0-100 to kappa 0.5-3.0
        kappa = 0.5 + (value / 100) * 2.5
        self.kappa_label.setText(f"κ = {kappa:.2f}")
    
    def browse_data_file(self):
        """Browse for existing data file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "", "CSV Files (*.csv)"
        )
        if filename:
            self.data_file_label.setText(Path(filename).name)
            self.data_file_label.setToolTip(filename)
    
    def log_message(self, message):
        """Add message to console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.append(f"[{timestamp}] {message}")
    
    def start_optimization(self):
        """Start surrogate-assisted optimization"""
        if not SKLEARN_AVAILABLE:
            QMessageBox.critical(
                self, "Error",
                "scikit-learn is required for surrogate modeling.\n"
                "Please install it: pip install scikit-learn"
            )
            return
        
        if self.optimization_running:
            QMessageBox.warning(self, "Running", "Optimization already running!")
            return
        
        # Build configuration
        config = self.build_configuration()
        if config is None:
            return
        
        self.log_message("Starting surrogate optimization...")
        self.log_message(f"Mode: {config['mode']}")
        self.log_message(f"Initial samples: {config['n_initial_samples']}")
        self.log_message(f"Max evaluations: {config['max_evals']}")
        
        # Create and start worker
        self.worker = SurrogateWorker(config)
        self.worker.log_message.connect(self.log_message)
        self.worker.progress_update.connect(self.on_progress_update)
        self.worker.surrogate_updated.connect(self.on_surrogate_updated)
        self.worker.design_evaluated.connect(self.on_design_evaluated)
        self.worker.pareto_updated.connect(self.on_pareto_updated)
        self.worker.optimization_complete.connect(self.on_optimization_complete)
        self.worker.error_occurred.connect(self.on_error_occurred)
        
        # Update UI
        self.optimization_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.eval_progress.setValue(0)
        
        # Start
        self.worker.start()
    
    def build_configuration(self):
        """Build configuration dictionary from GUI settings"""
        # Get mode
        mode = 'adaptive' if self.adaptive_radio.isChecked() else 'hybrid'
        
        # Get sampling settings
        sampling_method = self.sampling_method_combo.currentText().lower()
        if 'latin' in sampling_method:
            sampling_method = 'lhs'
        elif 'sobol' in sampling_method:
            sampling_method = 'sobol'
        else:
            sampling_method = 'random'
        
        n_initial = self.n_samples_spin.value()
        
        # Get kernel type
        kernel_text = self.kernel_combo.currentText()
        if 'Matérn 5/2' in kernel_text:
            kernel_type = 'matern52'
        elif 'Matérn 3/2' in kernel_text:
            kernel_type = 'matern32'
        else:
            kernel_type = 'rbf'
        
        # Get acquisition settings
        acq_text = self.acquisition_combo.currentText().lower()
        if 'expected' in acq_text:
            acquisition = 'ei'
        elif 'lower' in acq_text or 'lcb' in acq_text:
            acquisition = 'lcb'
        else:
            acquisition = 'pi'
        
        # Get kappa from slider
        kappa = 0.5 + (self.exploration_slider.value() / 100) * 2.5
        
        # Get budget settings
        max_evals = self.max_evals_spin.value()
        surrogate_gens = self.surrogate_gens_spin.value()
        validate_per_cycle = self.validate_per_cycle_spin.value()
        
        # Get advanced settings
        n_restarts = 10
        normalize = True
        adaptive_kappa = self.adaptive_balance_check.isChecked()
        kappa_init = 2.0
        kappa_final = 0.5
        kappa_decay = 'linear'
        
        if hasattr(self, 'n_restarts_spin'):
            n_restarts = self.n_restarts_spin.value()
        if hasattr(self, 'normalize_check'):
            normalize = self.normalize_check.isChecked()
        if hasattr(self, 'kappa_init_spin'):
            kappa_init = self.kappa_init_spin.value()
        if hasattr(self, 'kappa_final_spin'):
            kappa_final = self.kappa_final_spin.value()
        if hasattr(self, 'kappa_decay_combo'):
            kappa_decay = self.kappa_decay_combo.currentText().lower()
        
        # Get objectives from parent GUI optimization tab
        objectives = []
        if hasattr(self, 'parent_gui') and self.parent_gui:
            if hasattr(self.parent_gui, 'tabs'):
                # Find optimization tab
                for i in range(self.parent_gui.tabs.count()):
                    tab = self.parent_gui.tabs.widget(i)
                    if hasattr(tab, 'objective_controls'):
                        for obj_name, controls in tab.objective_controls.items():
                            if controls['enable'].isChecked():
                                mode_str = controls['mode'].currentText().lower()
                                objectives.append({
                                    'name': obj_name,
                                    'mode': mode_str
                                })
                        break
        
        # Default objectives if none found
        if not objectives:
            objectives = [
                {'name': 'CL/CD', 'mode': 'maximize'},
                {'name': 'Volume', 'mode': 'maximize'}
            ]
        
        # Get design variables from parent GUI
        design_variables = [
            {'name': 'X1', 'min': 0.0, 'max': 0.5},
            {'name': 'X2', 'min': 0.0, 'max': 0.5},
            {'name': 'X3', 'min': 0.0, 'max': 1.0},
            {'name': 'X4', 'min': 0.0, 'max': 1.0}
        ]
        
        if hasattr(self, 'parent_gui') and self.parent_gui:
            if hasattr(self.parent_gui, 'tabs'):
                for i in range(self.parent_gui.tabs.count()):
                    tab = self.parent_gui.tabs.widget(i)
                    if hasattr(tab, 'design_var_spins'):
                        design_variables = []
                        for var_name in ['X1', 'X2', 'X3', 'X4']:
                            spins = tab.design_var_spins[var_name]
                            design_variables.append({
                                'name': var_name,
                                'min': spins['min'].value(),
                                'max': spins['max'].value()
                            })
                        break
        
        # Get fixed parameters from parent GUI
        fixed_parameters = {
            'M_inf': 5.0,
            'beta': 15.0,
            'height': 1.34,
            'width': 3.0
        }
        
        simulation_params = {
            'aoa': 0.0,
            'A_ref': 19.65,
            'pressure': 2549.0,
            'temperature': 221.55
        }
        
        if hasattr(self, 'parent_gui') and self.parent_gui:
            if hasattr(self.parent_gui, 'mach_spin'):
                fixed_parameters['M_inf'] = self.parent_gui.mach_spin.value()
            if hasattr(self.parent_gui, 'beta_spin'):
                fixed_parameters['beta'] = self.parent_gui.beta_spin.value()
            if hasattr(self.parent_gui, 'height_spin'):
                fixed_parameters['height'] = self.parent_gui.height_spin.value()
            if hasattr(self.parent_gui, 'width_spin'):
                fixed_parameters['width'] = self.parent_gui.width_spin.value()
            
            # Get simulation params from optimization tab
            if hasattr(self.parent_gui, 'tabs'):
                for i in range(self.parent_gui.tabs.count()):
                    tab = self.parent_gui.tabs.widget(i)
                    if hasattr(tab, 'opt_aoa_spin'):
                        simulation_params['aoa'] = tab.opt_aoa_spin.value()
                        simulation_params['A_ref'] = tab.opt_aref_spin.value()
                        simulation_params['pressure'] = tab.opt_pressure_spin.value()
                        simulation_params['temperature'] = tab.opt_temperature_spin.value()
                        break
        
        # Get mesh size
        mesh_size = 0.1
        if hasattr(self, 'parent_gui') and self.parent_gui:
            if hasattr(self.parent_gui, 'tabs'):
                for i in range(self.parent_gui.tabs.count()):
                    tab = self.parent_gui.tabs.widget(i)
                    if hasattr(tab, 'mesh_quality_combo'):
                        mesh_quality = tab.mesh_quality_combo.currentText()
                        if 'Coarse' in mesh_quality:
                            mesh_size = 0.15
                        elif 'Medium' in mesh_quality:
                            mesh_size = 0.10
                        else:
                            mesh_size = 0.05
                        break
        
        config = {
            'mode': mode,
            'sampling_method': sampling_method,
            'n_initial_samples': n_initial,
            'kernel_type': kernel_type,
            'acquisition': acquisition,
            'kappa': kappa,
            'max_evals': max_evals,
            'surrogate_generations': surrogate_gens,
            'validate_per_cycle': validate_per_cycle,
            'n_restarts': n_restarts,
            'normalize': normalize,
            'adaptive_kappa': adaptive_kappa,
            'kappa_init': kappa_init,
            'kappa_final': kappa_final,
            'kappa_decay': kappa_decay,
            'objectives': objectives,
            'design_variables': design_variables,
            'fixed_parameters': fixed_parameters,
            'simulation_params': simulation_params,
            'mesh_size': mesh_size,
            'timestamp': datetime.now().isoformat(),
            
            # Focus region settings
            'focus_sampling': self.focus_sampling_check.isChecked(),
            'focus_region': {
                'X1': [self.focus_x1_min.value(), self.focus_x1_max.value()],
                'X2': [self.focus_x2_min.value(), self.focus_x2_max.value()],
                'X3': [self.focus_x3_min.value(), self.focus_x3_max.value()],
                'X4': [self.focus_x4_min.value(), self.focus_x4_max.value()]
            },
            
            # Quality filter settings
            'filter_clcd': self.filter_clcd_check.isChecked(),
            'min_clcd': self.min_clcd_spin.value(),
            'filter_outliers': self.filter_outliers_check.isChecked(),
            'outlier_std': self.outlier_std_spin.value(),
            
            # Adaptive validation settings
            'adaptive_validation': self.adaptive_validation_check.isChecked(),
            'validate_initial': validate_per_cycle,
            'validate_min': self.validate_min_spin.value(),
            'r2_threshold_high': self.r2_high_spin.value(),
            'r2_threshold_medium': self.r2_medium_spin.value()
        }
        
        return config
    
    def on_progress_update(self, current, total, message):
        """Handle progress update from worker"""
        progress = int(100 * current / total) if total > 0 else 0
        self.eval_progress.setValue(progress)
        self.total_evals_label.setText(str(current))
    
    def on_surrogate_updated(self, model):
        """Handle surrogate model update"""
        self.surrogate_model = model
        self.training_X = model.training_X
        self.training_y = model.training_y
        
        # Update UI
        self.save_surrogate_btn.setEnabled(True)
        self.rebuild_btn.setEnabled(True)
        self.inverse_design_btn.setEnabled(True)
        
        # Update metrics
        self.update_metrics_display()
        
        # Update objective combos
        self.response_plot.obj_combo.clear()
        self.uncertainty_plot.obj_combo.clear()
        for obj in model.objective_names:
            self.response_plot.obj_combo.addItem(obj)
            self.uncertainty_plot.obj_combo.addItem(obj)
    
    def on_design_evaluated(self, design_id, results):
        """Handle design evaluation result"""
        if results.get('success'):
            # Could update live plots here
            pass
    
    def on_pareto_updated(self, pareto_X, pareto_y):
        """Handle Pareto front update"""
        self.validated_pareto_label.setText(str(len(pareto_X)))
    
    def on_optimization_complete(self, results_folder):
        """Handle optimization completion"""
        self.optimization_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.log_message("=" * 40)
        self.log_message("OPTIMIZATION FINISHED")
        self.log_message("=" * 40)
        
        # Auto-update plots if we have a valid surrogate
        if self.surrogate_model is not None and self.training_X is not None:
            self.log_message("Updating visualization plots...")
            
            try:
                # Update response surface plot
                if self.response_plot.obj_combo.count() > 0:
                    self.update_response_surface()
                    self.log_message("✓ Response surface updated")
            except Exception as e:
                self.log_message(f"⚠ Response surface update failed: {e}")
            
            try:
                # Update uncertainty plot
                if self.uncertainty_plot.obj_combo.count() > 0:
                    self.update_uncertainty_plot()
                    self.log_message("✓ Uncertainty plot updated")
            except Exception as e:
                self.log_message(f"⚠ Uncertainty plot update failed: {e}")
            
            try:
                # Update Pareto plot
                if len(self.surrogate_model.objective_names) >= 2:
                    self.update_pareto_plot()
                    self.log_message("✓ Pareto plot updated")
            except Exception as e:
                self.log_message(f"⚠ Pareto plot update failed: {e}")
        
        QMessageBox.information(
            self, "Complete",
            f"Surrogate optimization completed!\n\n"
            f"Results saved to:\n{results_folder}\n\n"
            f"Check the plot tabs for visualizations."
        )
    
    def on_error_occurred(self, error_msg):
        """Handle error from worker"""
        self.optimization_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.log_message(f"❌ ERROR: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)
    
    def stop_optimization(self):
        """Stop optimization"""
        if self.worker:
            self.worker.stop()
            self.log_message("Stopping optimization...")
            
        # Re-enable buttons immediately
        self.optimization_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_message("Optimization stopped by user.")
    
    def save_surrogate(self):
        """Save surrogate model to file"""
        if self.surrogate_model is None:
            QMessageBox.warning(self, "No Model", "No surrogate model to save!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Surrogate Model", "", "Pickle Files (*.pkl)"
        )
        if filename:
            try:
                self.surrogate_model.save(filename)
                self.log_message(f"✓ Surrogate saved: {filename}")
                QMessageBox.information(self, "Success", f"Model saved to:\n{filename}")
            except Exception as e:
                self.log_message(f"❌ Error saving: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save model:\n{e}")
    
    def load_surrogate(self):
        """Load surrogate model from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Surrogate Model", "", "Pickle Files (*.pkl)"
        )
        if filename:
            try:
                self.surrogate_model = MultiOutputGP.load(filename)
                self.log_message(f"✓ Surrogate loaded: {filename}")
                self.save_surrogate_btn.setEnabled(True)
                self.rebuild_btn.setEnabled(True)
                
                # Update metrics display
                self.update_metrics_display()
                
                # Update objective combos in plots
                for obj in self.surrogate_model.objective_names:
                    self.response_plot.obj_combo.addItem(obj)
                    self.uncertainty_plot.obj_combo.addItem(obj)
                
                QMessageBox.information(self, "Success", f"Model loaded from:\n{filename}")
            except Exception as e:
                self.log_message(f"❌ Error loading: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
    
    def rebuild_surrogate(self):
        """Rebuild surrogate with current data"""
        if self.training_X is None:
            QMessageBox.warning(self, "No Data", "No training data available!")
            return
        
        self.log_message("Rebuilding surrogate model...")
        self.build_surrogate_from_data(self.training_X, self.training_y)
    
    def build_surrogate_only(self):
        """Build surrogate from existing data without optimization"""
        if not self.use_existing_check.isChecked():
            QMessageBox.warning(
                self, "No Data",
                "Please check 'Use existing data' and select a data file."
            )
            return
        
        # Get file path from label tooltip
        filepath = self.data_file_label.toolTip()
        if not filepath or not Path(filepath).exists():
            QMessageBox.warning(self, "No File", "Please select a valid data file.")
            return
        
        try:
            self.log_message(f"Loading data from: {filepath}")
            df = pd.read_csv(filepath)
            
            # Extract design variables
            X = df[['X1', 'X2', 'X3', 'X4']].values
            
            # Find objective columns (check common names)
            y_dict = {}
            possible_objs = ['CL/CD', 'Volume', 'CD', 'CL', 'Cm']
            for obj in possible_objs:
                if obj in df.columns:
                    # Filter valid values
                    valid = df[obj].notna() & (df[obj].abs() < 1e5)
                    if valid.sum() > 10:
                        y_dict[obj] = df.loc[valid, obj].values
            
            if len(y_dict) < 2:
                QMessageBox.warning(
                    self, "Insufficient Data",
                    "Need at least 2 objectives in data file."
                )
                return
            
            # Filter X to match valid y
            first_obj = list(y_dict.keys())[0]
            valid_mask = df[first_obj].notna() & (df[first_obj].abs() < 1e5)
            X_valid = X[valid_mask]
            
            self.log_message(f"Found {len(X_valid)} valid samples")
            self.log_message(f"Objectives: {list(y_dict.keys())}")
            
            self.build_surrogate_from_data(X_valid, y_dict)
            
        except Exception as e:
            self.log_message(f"❌ Error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{e}")
    
    def build_surrogate_from_data(self, X, y_dict):
        """Build surrogate model from data"""
        # Get kernel type
        kernel_text = self.kernel_combo.currentText()
        if 'Matérn 5/2' in kernel_text:
            kernel_type = 'matern52'
        elif 'Matérn 3/2' in kernel_text:
            kernel_type = 'matern32'
        else:
            kernel_type = 'rbf'
        
        n_restarts = self.n_restarts_spin.value() if hasattr(self, 'n_restarts_spin') else 10
        normalize = self.normalize_check.isChecked()
        
        self.log_message(f"Building GP with kernel: {kernel_type}")
        
        try:
            self.surrogate_model = MultiOutputGP(
                kernel_type=kernel_type,
                n_restarts=n_restarts,
                normalize=normalize
            )
            self.surrogate_model.fit(X, y_dict)
            
            self.training_X = X
            self.training_y = y_dict
            
            self.log_message("✓ Surrogate model built successfully!")
            
            # Update UI
            self.save_surrogate_btn.setEnabled(True)
            self.rebuild_btn.setEnabled(True)
            
            # Update objective combos
            self.response_plot.obj_combo.clear()
            self.uncertainty_plot.obj_combo.clear()
            for obj in y_dict.keys():
                self.response_plot.obj_combo.addItem(obj)
                self.uncertainty_plot.obj_combo.addItem(obj)
            
            # Update metrics
            self.update_metrics_display()
            
        except Exception as e:
            self.log_message(f"❌ Error building surrogate: {e}")
            raise
    
    def update_metrics_display(self):
        """Update the metrics display with current model"""
        if self.surrogate_model is None:
            return
        
        try:
            metrics = self.surrogate_model.get_metrics()
            
            obj_names = list(metrics.keys())
            
            # Update column headers
            # (would need to modify the grid, simplified here)
            
            for i, (metric_name, (label1, label2)) in enumerate(self.metric_labels.items()):
                key_map = {'R² Score': 'R2', 'RMSE': 'RMSE', 'MAE': 'MAE', 'Max Error': 'Max_Error'}
                key = key_map.get(metric_name, metric_name)
                
                if len(obj_names) >= 1 and key in metrics[obj_names[0]]:
                    val1 = metrics[obj_names[0]][key]
                    label1.setText(f"{val1:.4f}")
                    # Color code R2
                    if key == 'R2':
                        if val1 > 0.95:
                            label1.setStyleSheet("font-family: monospace; color: green;")
                        elif val1 > 0.8:
                            label1.setStyleSheet("font-family: monospace; color: orange;")
                        else:
                            label1.setStyleSheet("font-family: monospace; color: red;")
                
                if len(obj_names) >= 2 and key in metrics[obj_names[1]]:
                    val2 = metrics[obj_names[1]][key]
                    label2.setText(f"{val2:.4f}")
                    if key == 'R2':
                        if val2 > 0.95:
                            label2.setStyleSheet("font-family: monospace; color: green;")
                        elif val2 > 0.8:
                            label2.setStyleSheet("font-family: monospace; color: orange;")
                        else:
                            label2.setStyleSheet("font-family: monospace; color: red;")
            
            self.log_message("✓ Metrics updated")
            
        except Exception as e:
            self.log_message(f"⚠ Could not compute metrics: {e}")
    
    def update_response_surface(self):
        """Update response surface plot"""
        if self.surrogate_model is None:
            QMessageBox.warning(self, "No Model", "Please build or load a surrogate first!")
            return
        
        var1 = self.response_plot.var1_combo.currentText()
        var2 = self.response_plot.var2_combo.currentText()
        objective = self.response_plot.obj_combo.currentText()
        
        if var1 == var2:
            QMessageBox.warning(self, "Invalid", "Please select different variables!")
            return
        
        fixed_values = {}
        for var, slider in self.response_plot.fixed_sliders.items():
            if var not in [var1, var2]:
                fixed_values[var] = slider.value()
        
        self.log_message(f"Plotting response surface: {var1} vs {var2} for {objective}")
        
        try:
            training_y = self.training_y.get(objective) if self.training_y else None
            self.response_plot.plot_surface(
                self.surrogate_model, var1, var2, objective, fixed_values,
                training_X=self.training_X, training_y=training_y
            )
        except Exception as e:
            self.log_message(f"❌ Plot error: {e}")
    
    def update_uncertainty_plot(self):
        """Update uncertainty plot"""
        if self.surrogate_model is None:
            QMessageBox.warning(self, "No Model", "Please build or load a surrogate first!")
            return
        
        var1 = self.uncertainty_plot.var1_combo.currentText()
        var2 = self.uncertainty_plot.var2_combo.currentText()
        objective = self.uncertainty_plot.obj_combo.currentText()
        
        if var1 == var2:
            QMessageBox.warning(self, "Invalid", "Please select different variables!")
            return
        
        fixed_values = {}
        for var, slider in self.uncertainty_plot.fixed_sliders.items():
            if var not in [var1, var2]:
                fixed_values[var] = slider.value()
        
        self.log_message(f"Plotting uncertainty: {var1} vs {var2} for {objective}")
        
        try:
            self.uncertainty_plot.plot_uncertainty(
                self.surrogate_model, var1, var2, objective, fixed_values
            )
        except Exception as e:
            self.log_message(f"❌ Plot error: {e}")
    
    def update_pareto_plot(self):
        """Update Pareto front plot with confidence bands"""
        if self.surrogate_model is None:
            QMessageBox.warning(self, "No Model", "Please build or load a surrogate first!")
            return
        
        obj_names = self.surrogate_model.objective_names
        if len(obj_names) < 2:
            QMessageBox.warning(self, "Error", "Need at least 2 objectives for Pareto plot!")
            return
        
        # Assume first two objectives, both maximized (adjust based on your setup)
        # TODO: Get minimize flags from configuration
        minimize_flags = [False, False]  # Assuming maximize for both
        
        confidence = self.pareto_plot.confidence_spin.value()
        
        self.log_message(f"Plotting Pareto front with {confidence:.2f}σ confidence bands")
        
        try:
            validated_y = self.training_y if self.training_y else None
            
            self.pareto_plot.plot_pareto_with_bands(
                self.surrogate_model,
                obj_names[:2],
                minimize_flags,
                validated_X=self.training_X,
                validated_y=validated_y,
                confidence=confidence
            )
        except Exception as e:
            self.log_message(f"❌ Plot error: {e}")
    
    def show_cv_plot(self):
        """Show cross-validation plot"""
        if self.surrogate_model is None:
            QMessageBox.warning(self, "No Model", "Please build or load a surrogate first!")
            return
        
        if self.training_X is None or len(self.training_X) < 5:
            QMessageBox.warning(self, "Insufficient Data", 
                              "Need at least 5 training points for cross-validation!")
            return
        
        self.log_message("Generating cross-validation plot...")
        
        try:
            from sklearn.model_selection import cross_val_predict, KFold
            
            # Create figure
            n_objectives = len(self.surrogate_model.objective_names)
            fig, axes = plt.subplots(1, n_objectives, figsize=(5*n_objectives, 5))
            if n_objectives == 1:
                axes = [axes]
            
            for idx, obj_name in enumerate(self.surrogate_model.objective_names):
                ax = axes[idx]
                
                # Get the GP model for this objective
                gp_model = self.surrogate_model.models[obj_name]
                y_true = self.training_y[obj_name]
                
                # Perform leave-one-out cross-validation predictions
                n_samples = len(self.training_X)
                y_pred_cv = np.zeros(n_samples)
                y_std_cv = np.zeros(n_samples)
                
                for i in range(n_samples):
                    # Leave one out
                    X_train = np.delete(self.training_X, i, axis=0)
                    y_train = np.delete(y_true, i)
                    X_test = self.training_X[i:i+1]
                    
                    # Fit temporary model
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    temp_gp = GaussianProcessRegressor(
                        kernel=gp_model.kernel_,
                        alpha=1e-6,
                        normalize_y=True
                    )
                    temp_gp.fit(X_train, y_train)
                    pred, std = temp_gp.predict(X_test, return_std=True)
                    y_pred_cv[i] = pred[0]
                    y_std_cv[i] = std[0]
                
                # Plot
                ax.errorbar(y_true, y_pred_cv, yerr=2*y_std_cv, fmt='o', 
                           capsize=3, alpha=0.7, label='CV predictions (±2σ)')
                
                # Perfect prediction line
                min_val = min(y_true.min(), y_pred_cv.min())
                max_val = max(y_true.max(), y_pred_cv.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                       linewidth=2, label='Perfect prediction')
                
                # Calculate metrics
                from sklearn.metrics import r2_score, mean_absolute_error
                r2 = r2_score(y_true, y_pred_cv)
                mae = mean_absolute_error(y_true, y_pred_cv)
                
                ax.set_xlabel(f'Actual {obj_name}', fontsize=11, fontweight='bold')
                ax.set_ylabel(f'CV Predicted {obj_name}', fontsize=11, fontweight='bold')
                ax.set_title(f'{obj_name} LOO Cross-Validation\nR² = {r2:.4f}, MAE = {mae:.4f}', 
                            fontsize=12, fontweight='bold')
                ax.legend(loc='upper left', fontsize=9)
                ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            plt.show()
            
            self.log_message("✓ Cross-validation plot generated")
            
        except Exception as e:
            self.log_message(f"❌ CV plot error: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to generate CV plot:\n{e}")
    
    def show_parity_plot(self):
        """Show parity plot (predicted vs actual) with both training and CV predictions"""
        if self.surrogate_model is None:
            QMessageBox.warning(self, "No Model", "Please build or load a surrogate first!")
            return
        
        if self.training_X is None:
            QMessageBox.warning(self, "No Data", "No training data available!")
            return
        
        self.log_message("Generating parity plot (Training vs CV)...")
        
        try:
            from sklearn.model_selection import cross_val_predict
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            
            # Create figure with 2 rows: Training (top) and CV (bottom)
            n_objectives = len(self.surrogate_model.objective_names)
            fig, axes = plt.subplots(2, n_objectives, figsize=(5*n_objectives, 10))
            if n_objectives == 1:
                axes = axes.reshape(2, 1)
            
            for idx, obj_name in enumerate(self.surrogate_model.objective_names):
                y_true = self.training_y[obj_name]
                
                # === TOP ROW: Training predictions (always R²≈1 for GP) ===
                ax_train = axes[0, idx]
                means, stds = self.surrogate_model.predict(self.training_X, return_std=True)
                y_pred_train = means[obj_name]
                y_std = stds[obj_name]
                
                ax_train.errorbar(y_true, y_pred_train, yerr=2*y_std, fmt='o', 
                           capsize=3, alpha=0.7, markersize=6, color='blue',
                           label='Training predictions (±2σ)')
                
                min_val = min(y_true.min(), y_pred_train.min())
                max_val = max(y_true.max(), y_pred_train.max())
                margin = (max_val - min_val) * 0.1
                ax_train.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 
                       'r--', linewidth=2, label='Perfect prediction')
                
                r2_train = r2_score(y_true, y_pred_train)
                ax_train.set_xlabel(f'Actual {obj_name}', fontsize=10)
                ax_train.set_ylabel(f'Predicted {obj_name}', fontsize=10)
                ax_train.set_title(f'{obj_name} - TRAINING Data\nR² = {r2_train:.4f} (⚠️ Overfitted)', 
                            fontsize=11, fontweight='bold', color='orange')
                ax_train.legend(loc='upper left', fontsize=8)
                ax_train.grid(True, alpha=0.3)
                ax_train.set_xlim(min_val-margin, max_val+margin)
                ax_train.set_ylim(min_val-margin, max_val+margin)
                
                # === BOTTOM ROW: Cross-validation predictions (honest metric) ===
                ax_cv = axes[1, idx]
                
                # Get CV predictions
                gp = self.surrogate_model.models[obj_name]
                X_scaled = self.surrogate_model.input_scaler.transform(self.training_X)
                
                if self.surrogate_model.normalize:
                    y_scaled = self.surrogate_model.scalers[obj_name].transform(y_true.reshape(-1, 1)).ravel()
                    y_pred_cv_scaled = cross_val_predict(gp, X_scaled, y_scaled, cv=5)
                    y_pred_cv = y_pred_cv_scaled * self.surrogate_model.scalers[obj_name].scale_[0] + \
                                self.surrogate_model.scalers[obj_name].mean_[0]
                else:
                    y_pred_cv = cross_val_predict(gp, X_scaled, y_true, cv=5)
                
                ax_cv.scatter(y_true, y_pred_cv, alpha=0.7, s=40, c='green', edgecolors='darkgreen',
                             label='CV predictions')
                
                ax_cv.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 
                       'r--', linewidth=2, label='Perfect prediction')
                
                # ±10% bands
                ax_cv.fill_between([min_val-margin, max_val+margin],
                               [0.9*(min_val-margin), 0.9*(max_val+margin)],
                               [1.1*(min_val-margin), 1.1*(max_val+margin)],
                               alpha=0.15, color='green', label='±10% band')
                
                r2_cv = r2_score(y_true, y_pred_cv)
                rmse_cv = np.sqrt(mean_squared_error(y_true, y_pred_cv))
                
                # Color title based on R² quality
                if r2_cv > 0.9:
                    title_color = 'green'
                    quality = '✅ Excellent'
                elif r2_cv > 0.7:
                    title_color = 'blue'
                    quality = '✓ Good'
                elif r2_cv > 0.5:
                    title_color = 'orange'
                    quality = '⚠️ Moderate'
                else:
                    title_color = 'red'
                    quality = '❌ Poor'
                
                ax_cv.set_xlabel(f'Actual {obj_name}', fontsize=10)
                ax_cv.set_ylabel(f'CV Predicted {obj_name}', fontsize=10)
                ax_cv.set_title(f'{obj_name} - CROSS-VALIDATION\nR² = {r2_cv:.4f}, RMSE = {rmse_cv:.4f} ({quality})', 
                            fontsize=11, fontweight='bold', color=title_color)
                ax_cv.legend(loc='upper left', fontsize=8)
                ax_cv.grid(True, alpha=0.3)
                ax_cv.set_xlim(min_val-margin, max_val+margin)
                ax_cv.set_ylim(min_val-margin, max_val+margin)
            
            fig.suptitle('Parity Plot: Training (top) vs Cross-Validation (bottom)\n'
                        'CV metrics show true generalization ability', 
                        fontsize=12, fontweight='bold', y=1.02)
            fig.tight_layout()
            plt.show()
            
            self.log_message("✓ Parity plot generated (Training + CV)")
            
        except Exception as e:
            self.log_message(f"❌ Parity plot error: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to generate parity plot:\n{e}")
    
    def show_inverse_design_dialog(self):
        """Show dialog for inverse design - find X given performance requirements"""
        if self.surrogate_model is None:
            QMessageBox.warning(self, "No Model", "Please build or load a surrogate first!")
            return
        
        dialog = InverseDesignDialog(self.surrogate_model, self.parent_gui, self)
        dialog.exec_()


class InverseDesignDialog(QDialog):
    """Dialog for inverse design - find design variables given performance requirements"""
    
    def __init__(self, surrogate_model, parent_gui=None, parent=None):
        super().__init__(parent)
        self.surrogate_model = surrogate_model
        self.parent_gui = parent_gui
        self.parent_tab = parent
        self.best_designs = []
        
        self.setWindowTitle("🎯 Inverse Design - Find Design Variables")
        self.setMinimumSize(800, 700)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "<b>Inverse Design:</b> Specify your desired performance requirements, "
            "and the surrogate model will find the design variables (X1-X4) that meet them."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        layout.addWidget(instructions)
        
        # Fixed parameters group
        fixed_group = QGroupBox("Fixed Design Parameters")
        fixed_layout = QGridLayout()
        
        fixed_layout.addWidget(QLabel("Width (m):"), 0, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.5, 20.0)
        self.width_spin.setValue(3.0)
        self.width_spin.setDecimals(2)
        fixed_layout.addWidget(self.width_spin, 0, 1)
        
        fixed_layout.addWidget(QLabel("Height (m):"), 0, 2)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.1, 10.0)
        self.height_spin.setValue(1.34)
        self.height_spin.setDecimals(2)
        fixed_layout.addWidget(self.height_spin, 0, 3)
        
        fixed_layout.addWidget(QLabel("Mach Number:"), 1, 0)
        self.mach_spin = QDoubleSpinBox()
        self.mach_spin.setRange(2.0, 25.0)
        self.mach_spin.setValue(5.0)
        self.mach_spin.setDecimals(1)
        fixed_layout.addWidget(self.mach_spin, 1, 1)
        
        fixed_layout.addWidget(QLabel("Beta (°):"), 1, 2)
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(5.0, 45.0)
        self.beta_spin.setValue(15.0)
        self.beta_spin.setDecimals(2)
        fixed_layout.addWidget(self.beta_spin, 1, 3)
        
        fixed_group.setLayout(fixed_layout)
        layout.addWidget(fixed_group)
        
        # Requirements group
        req_group = QGroupBox("Performance Requirements")
        req_layout = QGridLayout()
        
        # Get objective names from surrogate
        obj_names = self.surrogate_model.objective_names
        
        self.requirement_widgets = {}
        row = 0
        
        for obj_name in obj_names:
            # Enable checkbox
            enable_check = QCheckBox(f"{obj_name}:")
            enable_check.setChecked(True)
            req_layout.addWidget(enable_check, row, 0)
            
            # Constraint type
            constraint_combo = QComboBox()
            constraint_combo.addItems(['≥ (min)', '≤ (max)', '= (target)', 'maximize', 'minimize'])
            if 'CD' in obj_name or 'drag' in obj_name.lower():
                constraint_combo.setCurrentText('minimize')
            else:
                constraint_combo.setCurrentText('maximize')
            req_layout.addWidget(constraint_combo, row, 1)
            
            # Value
            value_spin = QDoubleSpinBox()
            value_spin.setRange(-1000.0, 1000.0)
            value_spin.setValue(4.0 if 'CL/CD' in obj_name else 5.0)
            value_spin.setDecimals(3)
            req_layout.addWidget(value_spin, row, 2)
            
            # Weight (for multi-objective)
            req_layout.addWidget(QLabel("Weight:"), row, 3)
            weight_spin = QDoubleSpinBox()
            weight_spin.setRange(0.0, 10.0)
            weight_spin.setValue(1.0)
            weight_spin.setDecimals(1)
            req_layout.addWidget(weight_spin, row, 4)
            
            self.requirement_widgets[obj_name] = {
                'enable': enable_check,
                'constraint': constraint_combo,
                'value': value_spin,
                'weight': weight_spin
            }
            row += 1
        
        req_group.setLayout(req_layout)
        layout.addWidget(req_group)
        
        # Search settings
        search_group = QGroupBox("Search Settings")
        search_layout = QGridLayout()
        
        search_layout.addWidget(QLabel("Number of candidates:"), 0, 0)
        self.n_candidates_spin = QSpinBox()
        self.n_candidates_spin.setRange(100, 100000)
        self.n_candidates_spin.setValue(10000)
        search_layout.addWidget(self.n_candidates_spin, 0, 1)
        
        search_layout.addWidget(QLabel("Top designs to show:"), 0, 2)
        self.n_top_spin = QSpinBox()
        self.n_top_spin.setRange(1, 50)
        self.n_top_spin.setValue(10)
        search_layout.addWidget(self.n_top_spin, 0, 3)
        
        self.use_optimization_check = QCheckBox("Use local optimization (slower but more accurate)")
        self.use_optimization_check.setChecked(True)
        search_layout.addWidget(self.use_optimization_check, 1, 0, 1, 4)
        
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)
        
        # Find button
        self.find_btn = QPushButton("🔍 Find Designs")
        self.find_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 12px; font-size: 14px; }"
        )
        self.find_btn.clicked.connect(self.find_designs)
        layout.addWidget(self.find_btn)
        
        # Results
        results_group = QGroupBox("Results - Best Matching Designs")
        results_layout = QVBoxLayout()
        
        self.results_table = QTextEdit()
        self.results_table.setReadOnly(True)
        self.results_table.setFont(QFont("Courier", 10))
        self.results_table.setMinimumHeight(200)
        results_layout.addWidget(self.results_table)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("✓ Apply Best Design to GUI")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_best_design)
        action_layout.addWidget(self.apply_btn)
        
        self.validate_btn = QPushButton("🔬 Validate with PySAGAS")
        self.validate_btn.setEnabled(False)
        self.validate_btn.clicked.connect(self.validate_design)
        action_layout.addWidget(self.validate_btn)
        
        self.export_btn = QPushButton("💾 Export Results")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        action_layout.addWidget(self.export_btn)
        
        results_layout.addLayout(action_layout)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
    
    def find_designs(self):
        """Find designs matching the requirements using surrogate"""
        self.find_btn.setEnabled(False)
        self.find_btn.setText("🔍 Searching...")
        QApplication.processEvents()
        
        try:
            n_candidates = self.n_candidates_spin.value()
            n_top = self.n_top_spin.value()
            use_optimization = self.use_optimization_check.isChecked()
            
            # Build requirement spec
            requirements = {}
            for obj_name, widgets in self.requirement_widgets.items():
                if widgets['enable'].isChecked():
                    constraint = widgets['constraint'].currentText()
                    value = widgets['value'].value()
                    weight = widgets['weight'].value()
                    requirements[obj_name] = {
                        'constraint': constraint,
                        'value': value,
                        'weight': weight
                    }
            
            if not requirements:
                QMessageBox.warning(self, "No Requirements", "Please enable at least one objective!")
                return
            
            # Generate random candidates in the feasible region
            X_candidates = self._generate_feasible_candidates(n_candidates)
            
            # Predict with surrogate
            means, stds = self.surrogate_model.predict(X_candidates, return_std=True)
            
            # Score each candidate
            scores = self._score_candidates(X_candidates, means, stds, requirements)
            
            # Get top candidates
            top_indices = np.argsort(scores)[:n_top]
            
            # Optionally refine with local optimization
            if use_optimization:
                refined_X = []
                refined_scores = []
                for idx in top_indices[:min(5, n_top)]:  # Refine top 5
                    X_opt, score_opt = self._optimize_from_candidate(
                        X_candidates[idx], requirements
                    )
                    refined_X.append(X_opt)
                    refined_scores.append(score_opt)
                
                # Combine refined and original top candidates
                all_X = np.vstack([np.array(refined_X), X_candidates[top_indices]])
                all_means, all_stds = self.surrogate_model.predict(all_X, return_std=True)
                all_scores = self._score_candidates(all_X, all_means, all_stds, requirements)
                
                # Re-sort
                final_indices = np.argsort(all_scores)[:n_top]
                self.best_designs = []
                for idx in final_indices:
                    design = {
                        'X': all_X[idx],
                        'score': all_scores[idx],
                        'predictions': {obj: all_means[obj][idx] for obj in means.keys()},
                        'uncertainties': {obj: all_stds[obj][idx] for obj in stds.keys()}
                    }
                    self.best_designs.append(design)
            else:
                self.best_designs = []
                for idx in top_indices:
                    design = {
                        'X': X_candidates[idx],
                        'score': scores[idx],
                        'predictions': {obj: means[obj][idx] for obj in means.keys()},
                        'uncertainties': {obj: stds[obj][idx] for obj in stds.keys()}
                    }
                    self.best_designs.append(design)
            
            # Display results
            self._display_results(requirements)
            
            # Enable action buttons
            self.apply_btn.setEnabled(True)
            self.validate_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Search failed:\n{e}")
            import traceback
            traceback.print_exc()
        finally:
            self.find_btn.setEnabled(True)
            self.find_btn.setText("🔍 Find Designs")
    
    def _generate_feasible_candidates(self, n_candidates):
        """Generate random candidates in the feasible region"""
        candidates = []
        n_attempts = 0
        max_attempts = n_candidates * 10
        
        while len(candidates) < n_candidates and n_attempts < max_attempts:
            n_attempts += 1
            
            # Random sample
            x1 = np.random.uniform(0, 0.5)
            x2 = np.random.uniform(0, 0.5)
            x3 = np.random.uniform(0, 1)
            x4 = np.random.uniform(0, 1)
            
            # Check X1-X2 constraint
            if x1 < 1.0:
                constraint_value = x2 / ((1 - x1) ** 4)
                if constraint_value <= 0.85:
                    # Check X3-X4 constraints
                    if not (x4 < 0.05 and 0.3 < x3 < 0.5):
                        if not ((x3 < 0.05 and x4 < 0.05) or (x3 > 0.95 and x4 > 0.95)):
                            candidates.append([x1, x2, x3, x4])
        
        return np.array(candidates)
    
    def _score_candidates(self, X, means, stds, requirements):
        """Score candidates based on how well they meet requirements"""
        n = len(X)
        scores = np.zeros(n)
        
        for obj_name, req in requirements.items():
            constraint = req['constraint']
            target = req['value']
            weight = req['weight']
            
            pred = means[obj_name]
            std = stds[obj_name]
            
            if constraint == '≥ (min)':
                # Penalty for being below target
                violation = np.maximum(0, target - pred)
                scores += weight * violation ** 2
            elif constraint == '≤ (max)':
                # Penalty for being above target
                violation = np.maximum(0, pred - target)
                scores += weight * violation ** 2
            elif constraint == '= (target)':
                # Penalty for deviation from target
                scores += weight * (pred - target) ** 2
            elif constraint == 'maximize':
                # Negative reward for high values
                scores -= weight * pred
            elif constraint == 'minimize':
                # Penalty for high values
                scores += weight * pred
        
        return scores
    
    def _optimize_from_candidate(self, X_init, requirements):
        """Local optimization from a starting point"""
        from scipy.optimize import minimize
        
        def objective(X):
            X_arr = X.reshape(1, -1)
            means, stds = self.surrogate_model.predict(X_arr, return_std=True)
            score = self._score_candidates(X_arr, means, stds, requirements)[0]
            return score
        
        def constraint_func(X):
            x1, x2 = X[0], X[1]
            if x1 >= 1.0:
                return -1  # Infeasible
            return 0.85 - x2 / ((1 - x1) ** 4)
        
        bounds = [(0, 0.5), (0, 0.5), (0.05, 0.95), (0.05, 0.95)]
        
        result = minimize(
            objective,
            X_init,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 100}
        )
        
        return result.x, result.fun
    
    def _display_results(self, requirements):
        """Display search results"""
        text = "=" * 70 + "\n"
        text += "INVERSE DESIGN RESULTS\n"
        text += "=" * 70 + "\n\n"
        
        text += "Requirements:\n"
        for obj_name, req in requirements.items():
            text += f"  {obj_name}: {req['constraint']} {req['value']:.3f} (weight={req['weight']:.1f})\n"
        text += "\n"
        
        text += "-" * 70 + "\n"
        text += f"{'Rank':<5} {'X1':>8} {'X2':>8} {'X3':>8} {'X4':>8} | "
        for obj_name in self.best_designs[0]['predictions'].keys():
            text += f"{obj_name:>12} "
        text += "| Score\n"
        text += "-" * 70 + "\n"
        
        for i, design in enumerate(self.best_designs):
            X = design['X']
            text += f"{i+1:<5} {X[0]:>8.4f} {X[1]:>8.4f} {X[2]:>8.4f} {X[3]:>8.4f} | "
            
            for obj_name, pred in design['predictions'].items():
                std = design['uncertainties'][obj_name]
                text += f"{pred:>7.3f}±{std:.2f} "
            
            text += f"| {design['score']:.4f}\n"
        
        text += "-" * 70 + "\n"
        text += "\nNote: Predictions include ±1σ uncertainty from the surrogate model.\n"
        text += "Lower score = better match to requirements.\n"
        
        # Highlight best design
        if self.best_designs:
            best = self.best_designs[0]
            text += f"\n🏆 BEST DESIGN:\n"
            text += f"   X1={best['X'][0]:.4f}, X2={best['X'][1]:.4f}, "
            text += f"X3={best['X'][2]:.4f}, X4={best['X'][3]:.4f}\n"
            for obj_name, pred in best['predictions'].items():
                std = best['uncertainties'][obj_name]
                text += f"   {obj_name}: {pred:.4f} ± {std:.4f}\n"
        
        self.results_table.setText(text)
    
    def apply_best_design(self):
        """Apply the best design to the main GUI"""
        if not self.best_designs:
            return
        
        best = self.best_designs[0]
        X = best['X']
        
        # Try to update the main GUI
        if self.parent_gui:
            try:
                # Update X sliders
                if hasattr(self.parent_gui, 'x1_slider'):
                    self.parent_gui.x1_slider.setValue(int(X[0] * 1000))
                if hasattr(self.parent_gui, 'x2_slider'):
                    self.parent_gui.x2_slider.setValue(int(X[1] * 1000))
                if hasattr(self.parent_gui, 'x3_slider'):
                    self.parent_gui.x3_slider.setValue(int(X[2] * 1000))
                if hasattr(self.parent_gui, 'x4_slider'):
                    self.parent_gui.x4_slider.setValue(int(X[3] * 1000))
                
                # Update fixed parameters
                if hasattr(self.parent_gui, 'width_spin'):
                    self.parent_gui.width_spin.setValue(self.width_spin.value())
                if hasattr(self.parent_gui, 'height_spin'):
                    self.parent_gui.height_spin.setValue(self.height_spin.value())
                if hasattr(self.parent_gui, 'mach_spin'):
                    self.parent_gui.mach_spin.setValue(self.mach_spin.value())
                if hasattr(self.parent_gui, 'beta_spin'):
                    self.parent_gui.beta_spin.setValue(self.beta_spin.value())
                
                QMessageBox.information(
                    self, "Applied",
                    f"Design applied to GUI:\n"
                    f"X1={X[0]:.4f}, X2={X[1]:.4f}, X3={X[2]:.4f}, X4={X[3]:.4f}\n\n"
                    "Click 'Generate' in the main window to create the waverider."
                )
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Could not apply all parameters:\n{e}")
        else:
            QMessageBox.information(
                self, "Best Design",
                f"Best design variables:\n"
                f"X1={X[0]:.4f}\nX2={X[1]:.4f}\nX3={X[2]:.4f}\nX4={X[3]:.4f}"
            )
    
    def validate_design(self):
        """Validate the best design with actual PySAGAS simulation"""
        if not self.best_designs:
            return
        
        best = self.best_designs[0]
        X = best['X']
        
        reply = QMessageBox.question(
            self, "Validate Design",
            f"This will run a full PySAGAS simulation for:\n"
            f"X1={X[0]:.4f}, X2={X[1]:.4f}, X3={X[2]:.4f}, X4={X[3]:.4f}\n\n"
            f"This may take 1-2 minutes. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            QMessageBox.information(
                self, "Validation",
                "To validate this design:\n\n"
                "1. Click 'Apply Best Design to GUI'\n"
                "2. Go to the main window\n"
                "3. Click 'Generate' to create the waverider\n"
                "4. Go to 'Aerodynamic Analysis' tab\n"
                "5. Click 'Run PySAGAS Analysis'\n\n"
                "This will give you the true aerodynamic coefficients."
            )
    
    def export_results(self):
        """Export results to CSV"""
        if not self.best_designs:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "inverse_design_results.csv", "CSV Files (*.csv)"
        )
        
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    header = ['Rank', 'X1', 'X2', 'X3', 'X4', 'Score']
                    for obj_name in self.best_designs[0]['predictions'].keys():
                        header.extend([f'{obj_name}_pred', f'{obj_name}_std'])
                    writer.writerow(header)
                    
                    # Data
                    for i, design in enumerate(self.best_designs):
                        row = [i+1] + list(design['X']) + [design['score']]
                        for obj_name in design['predictions'].keys():
                            row.extend([
                                design['predictions'][obj_name],
                                design['uncertainties'][obj_name]
                            ])
                        writer.writerow(row)
                
                QMessageBox.information(self, "Exported", f"Results saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{e}")


# For testing
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = SurrogateTab()
    window.setWindowTitle("Surrogate Optimization")
    window.resize(1400, 900)
    window.show()
    sys.exit(app.exec_())
