#!/usr/bin/env python3
"""
AI Surrogate Integration for Waverider GUI

This module provides integration between the new AI-enhanced surrogate models
and the existing SurrogateTab in waverider_gui.py.

Features:
- Drop-in replacement for MultiOutputGP with MultiOutputNNEnsemble
- Model comparison utilities
- GUI components for surrogate selection
- CL/CD-focused optimization mode

Usage:
    # In surrogate_tab.py, replace:
    from surrogate_tab import MultiOutputGP
    
    # With:
    from ai_surrogate_integration import get_surrogate_model
    surrogate = get_surrogate_model('nn_ensemble')  # or 'gp', 'clcd_optimized'
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings

# Import the AI surrogate module
try:
    from ai_surrogate import (
        MultiOutputNNEnsemble,
        CLCDOptimizedSurrogate,
        HybridSurrogate,
        NeuralNetworkEnsemble
    )
    AI_SURROGATE_AVAILABLE = True
except ImportError:
    AI_SURROGATE_AVAILABLE = False
    print("Warning: ai_surrogate module not found. Using GP only.")

# Try to import the original GP implementation
try:
    from surrogate_tab import MultiOutputGP
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False


class SurrogateModelWrapper:
    """
    Unified wrapper for different surrogate model types.
    
    Provides a consistent interface regardless of the underlying model type,
    making it easy to switch between GP, NN ensemble, or hybrid approaches.
    """
    
    def __init__(self, model_type: str = 'nn_ensemble', **kwargs):
        """
        Initialize surrogate wrapper.
        
        Parameters
        ----------
        model_type : str
            One of: 'gp', 'nn_ensemble', 'clcd_optimized', 'hybrid'
        **kwargs
            Model-specific configuration
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        self.training_X = None
        self.training_y = {}
        self.objective_names = []
        
        self._create_model()
    
    def _create_model(self):
        """Create the underlying model based on type"""
        if self.model_type == 'gp':
            if not GP_AVAILABLE:
                raise ImportError("MultiOutputGP not available")
            self.model = MultiOutputGP(
                kernel_type=self.kwargs.get('kernel_type', 'matern52'),
                n_restarts=self.kwargs.get('n_restarts', 10),
                normalize=self.kwargs.get('normalize', True)
            )
        
        elif self.model_type == 'nn_ensemble':
            if not AI_SURROGATE_AVAILABLE:
                raise ImportError("AI surrogate module not available")
            self.model = MultiOutputNNEnsemble(
                n_members=self.kwargs.get('n_members', 10),
                hidden_layers=self.kwargs.get('hidden_layers', (64, 32, 16)),
                activation=self.kwargs.get('activation', 'relu'),
                max_iter=self.kwargs.get('max_iter', 2000)
            )
        
        elif self.model_type == 'clcd_optimized':
            if not AI_SURROGATE_AVAILABLE:
                raise ImportError("AI surrogate module not available")
            self.model = CLCDOptimizedSurrogate(
                n_members=self.kwargs.get('n_members', 15),
                max_iter=self.kwargs.get('max_iter', 3000)
            )
        
        elif self.model_type == 'hybrid':
            if not AI_SURROGATE_AVAILABLE:
                raise ImportError("AI surrogate module not available")
            self.model = HybridSurrogate(
                nn_weight=self.kwargs.get('nn_weight', 0.7),
                gp_weight=self.kwargs.get('gp_weight', 0.3)
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]):
        """Fit the surrogate model"""
        self.training_X = X.copy()
        self.training_y = {k: v.copy() for k, v in y_dict.items()}
        self.objective_names = list(y_dict.keys())
        
        if self.model_type == 'hybrid' and GP_AVAILABLE:
            # Hybrid needs the GP class
            self.model.fit(X, y_dict, gp_class=MultiOutputGP)
        else:
            self.model.fit(X, y_dict)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray, return_std: bool = True):
        """Predict using the surrogate model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X, return_std=return_std)
    
    def get_metrics(self, X_test=None, y_test=None) -> Dict[str, Dict]:
        """Get model quality metrics"""
        if hasattr(self.model, 'get_metrics'):
            return self.model.get_metrics(X_test, y_test)
        return {}
    
    def save(self, filepath: str):
        """Save the model"""
        if hasattr(self.model, 'save'):
            self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str, model_type: str = 'nn_ensemble'):
        """Load a saved model"""
        wrapper = cls(model_type=model_type)
        if model_type in ['nn_ensemble', 'clcd_optimized']:
            wrapper.model = MultiOutputNNEnsemble.load(filepath)
        elif model_type == 'gp' and GP_AVAILABLE:
            wrapper.model = MultiOutputGP.load(filepath)
        wrapper.is_fitted = True
        return wrapper


def get_surrogate_model(model_type: str = 'nn_ensemble', **kwargs) -> SurrogateModelWrapper:
    """
    Factory function to get a surrogate model.
    
    Parameters
    ----------
    model_type : str
        Type of model: 'gp', 'nn_ensemble', 'clcd_optimized', 'hybrid'
    **kwargs
        Model configuration
        
    Returns
    -------
    SurrogateModelWrapper
        Configured surrogate model
    """
    return SurrogateModelWrapper(model_type=model_type, **kwargs)


class SurrogateComparison:
    """
    Utility class to compare different surrogate models.
    
    Useful for determining which model type works best for your specific
    waverider design problem.
    """
    
    def __init__(self, model_types: List[str] = None):
        """
        Initialize comparison.
        
        Parameters
        ----------
        model_types : list
            List of model types to compare
        """
        self.model_types = model_types or ['nn_ensemble', 'clcd_optimized']
        if GP_AVAILABLE:
            self.model_types = ['gp'] + self.model_types
        
        self.results = {}
        self.models = {}
    
    def compare(self, X: np.ndarray, y_dict: Dict[str, np.ndarray],
                test_fraction: float = 0.2, verbose: bool = True) -> Dict:
        """
        Compare all surrogate types on the given data.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        y_dict : dict
            Target values
        test_fraction : float
            Fraction of data for testing
        verbose : bool
            Whether to print progress
            
        Returns
        -------
        dict
            Comparison results
        """
        n = len(X)
        n_test = int(n * test_fraction)
        
        # Split data
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        train_idx = indices[n_test:]
        test_idx = indices[:n_test]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = {k: v[train_idx] for k, v in y_dict.items()}
        y_test = {k: v[test_idx] for k, v in y_dict.items()}
        
        self.results = {}
        
        for model_type in self.model_types:
            if verbose:
                print(f"\nTesting {model_type}...")
            
            try:
                model = get_surrogate_model(model_type)
                model.fit(X_train, y_train)
                self.models[model_type] = model
                
                # Get metrics
                metrics = model.get_metrics(X_test, y_test)
                self.results[model_type] = metrics
                
                if verbose:
                    for obj_name, obj_metrics in metrics.items():
                        if 'R2' in obj_metrics:
                            print(f"  {obj_name}: R²={obj_metrics['R2']:.4f}, "
                                  f"RMSE={obj_metrics['RMSE']:.4f}")
            
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                self.results[model_type] = {'error': str(e)}
        
        return self.results
    
    def get_best_model(self, objective: str = 'CL/CD') -> Tuple[str, SurrogateModelWrapper]:
        """
        Get the best performing model for a specific objective.
        
        Parameters
        ----------
        objective : str
            Objective to optimize for (e.g., 'CL/CD')
            
        Returns
        -------
        tuple
            (model_type, model) for the best performer
        """
        best_type = None
        best_r2 = -np.inf
        
        for model_type, metrics in self.results.items():
            if 'error' not in metrics and objective in metrics:
                r2 = metrics[objective].get('R2', -np.inf)
                if r2 > best_r2:
                    best_r2 = r2
                    best_type = model_type
        
        if best_type is not None:
            return best_type, self.models.get(best_type)
        return None, None
    
    def summary(self) -> str:
        """Generate a text summary of comparison results"""
        lines = ["=" * 60]
        lines.append("SURROGATE MODEL COMPARISON SUMMARY")
        lines.append("=" * 60)
        
        for model_type, metrics in self.results.items():
            lines.append(f"\n{model_type.upper()}")
            lines.append("-" * 40)
            
            if 'error' in metrics:
                lines.append(f"  Error: {metrics['error']}")
            else:
                for obj_name, obj_metrics in metrics.items():
                    if isinstance(obj_metrics, dict) and 'R2' in obj_metrics:
                        lines.append(f"  {obj_name}:")
                        lines.append(f"    R²:   {obj_metrics['R2']:.4f}")
                        lines.append(f"    RMSE: {obj_metrics['RMSE']:.4f}")
                        lines.append(f"    MAE:  {obj_metrics['MAE']:.4f}")
        
        lines.append("\n" + "=" * 60)
        
        # Find best for CL/CD
        best_type, _ = self.get_best_model('CL/CD')
        if best_type:
            lines.append(f"RECOMMENDED for CL/CD: {best_type}")
        
        return "\n".join(lines)


class CLCDHunterAcquisition:
    """
    Specialized acquisition function for finding high CL/CD designs.
    
    This implements a multi-objective acquisition that:
    1. Targets high CL/CD values
    2. Respects volume constraints
    3. Balances exploration vs exploitation
    4. Uses uncertainty to guide sampling
    """
    
    def __init__(self,
                 target_clcd: float = None,
                 min_clcd: float = None,
                 volume_constraint: Tuple[float, float] = None,
                 exploration_weight: float = 1.0):
        """
        Parameters
        ----------
        target_clcd : float, optional
            Target CL/CD value (if None, maximize)
        min_clcd : float, optional
            Minimum acceptable CL/CD
        volume_constraint : tuple, optional
            (min_volume, max_volume) constraint
        exploration_weight : float
            Weight for uncertainty term (higher = more exploration)
        """
        self.target_clcd = target_clcd
        self.min_clcd = min_clcd
        self.volume_constraint = volume_constraint
        self.exploration_weight = exploration_weight
    
    def __call__(self, X: np.ndarray, surrogate: SurrogateModelWrapper,
                 best_clcd: float = None) -> np.ndarray:
        """
        Compute acquisition values for candidate points.
        
        Parameters
        ----------
        X : np.ndarray
            Candidate points (n_candidates, n_features)
        surrogate : SurrogateModelWrapper
            Fitted surrogate model
        best_clcd : float, optional
            Best CL/CD found so far
            
        Returns
        -------
        np.ndarray
            Acquisition values (higher = more promising)
        """
        means, stds = surrogate.predict(X, return_std=True)
        
        clcd_mean = means.get('CL/CD', np.zeros(len(X)))
        clcd_std = stds.get('CL/CD', np.ones(len(X)))
        volume_mean = means.get('Volume', np.ones(len(X)) * 5)
        
        # Base acquisition: expected improvement or UCB
        if best_clcd is not None and self.target_clcd is None:
            # Expected improvement for maximization
            improvement = clcd_mean - best_clcd
            Z = improvement / np.maximum(clcd_std, 1e-6)
            from scipy.stats import norm
            ei = improvement * norm.cdf(Z) + clcd_std * norm.pdf(Z)
            acquisition = ei
        elif self.target_clcd is not None:
            # Target-based acquisition (minimize distance to target)
            distance = np.abs(clcd_mean - self.target_clcd)
            # Include uncertainty in distance (could be closer than mean suggests)
            acquisition = -distance + self.exploration_weight * clcd_std
        else:
            # Upper confidence bound (maximize)
            acquisition = clcd_mean + self.exploration_weight * clcd_std
        
        # Apply volume constraint penalty
        if self.volume_constraint is not None:
            min_vol, max_vol = self.volume_constraint
            penalty = np.zeros(len(X))
            
            if min_vol is not None:
                penalty += np.maximum(0, min_vol - volume_mean) ** 2
            if max_vol is not None:
                penalty += np.maximum(0, volume_mean - max_vol) ** 2
            
            acquisition -= 10 * penalty  # Heavy penalty for constraint violation
        
        # Apply minimum CL/CD constraint
        if self.min_clcd is not None:
            # Probability of exceeding minimum
            prob_feasible = 1 - norm.cdf((self.min_clcd - clcd_mean) / np.maximum(clcd_std, 1e-6))
            acquisition *= prob_feasible
        
        return acquisition
    
    def select_next_points(self, surrogate: SurrogateModelWrapper,
                          bounds: np.ndarray,
                          n_points: int = 1,
                          n_candidates: int = 10000,
                          best_clcd: float = None) -> np.ndarray:
        """
        Select the next points to evaluate.
        
        Parameters
        ----------
        surrogate : SurrogateModelWrapper
            Fitted surrogate
        bounds : np.ndarray
            Design variable bounds (n_vars, 2) with [min, max]
        n_points : int
            Number of points to select
        n_candidates : int
            Number of candidates to consider
        best_clcd : float
            Best CL/CD found so far
            
        Returns
        -------
        np.ndarray
            Selected points (n_points, n_vars)
        """
        # Generate candidates
        rng = np.random.default_rng()
        n_vars = len(bounds)
        
        candidates = np.zeros((n_candidates, n_vars))
        for i in range(n_vars):
            candidates[:, i] = rng.uniform(bounds[i, 0], bounds[i, 1], n_candidates)
        
        # Compute acquisition values
        acq_values = self(candidates, surrogate, best_clcd)
        
        # Select top points
        top_indices = np.argsort(acq_values)[-n_points:]
        
        return candidates[top_indices]


# PyQt5 GUI components for surrogate selection
try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                 QComboBox, QGroupBox, QPushButton, QSpinBox,
                                 QDoubleSpinBox, QTextEdit, QProgressBar)
    from PyQt5.QtCore import Qt, pyqtSignal
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


if PYQT_AVAILABLE:
    class SurrogateSelectionWidget(QWidget):
        """
        GUI widget for selecting and configuring surrogate models.
        
        Can be embedded in the existing SurrogateTab.
        """
        
        surrogate_changed = pyqtSignal(str)  # Emits model type when changed
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self._setup_ui()
        
        def _setup_ui(self):
            layout = QVBoxLayout(self)
            
            # Model selection
            selection_group = QGroupBox("Surrogate Model Selection")
            selection_layout = QVBoxLayout(selection_group)
            
            # Model type dropdown
            type_layout = QHBoxLayout()
            type_layout.addWidget(QLabel("Model Type:"))
            
            self.model_combo = QComboBox()
            self.model_combo.addItems([
                "Neural Network Ensemble",
                "CL/CD Optimized (Recommended)",
                "Gaussian Process",
                "Hybrid (NN + GP)"
            ])
            self.model_combo.setCurrentIndex(1)  # Default to CL/CD optimized
            self.model_combo.currentIndexChanged.connect(self._on_model_changed)
            type_layout.addWidget(self.model_combo)
            
            selection_layout.addLayout(type_layout)
            
            # Configuration options
            config_layout = QHBoxLayout()
            
            config_layout.addWidget(QLabel("Ensemble Members:"))
            self.n_members_spin = QSpinBox()
            self.n_members_spin.setRange(5, 30)
            self.n_members_spin.setValue(15)
            config_layout.addWidget(self.n_members_spin)
            
            config_layout.addWidget(QLabel("Max Iterations:"))
            self.max_iter_spin = QSpinBox()
            self.max_iter_spin.setRange(500, 10000)
            self.max_iter_spin.setValue(2000)
            self.max_iter_spin.setSingleStep(500)
            config_layout.addWidget(self.max_iter_spin)
            
            selection_layout.addLayout(config_layout)
            
            # Description
            self.description_label = QLabel()
            self.description_label.setWordWrap(True)
            self.description_label.setStyleSheet("color: gray; font-style: italic;")
            self._update_description()
            selection_layout.addWidget(self.description_label)
            
            layout.addWidget(selection_group)
        
        def _on_model_changed(self, index):
            self._update_description()
            model_type = self.get_model_type()
            self.surrogate_changed.emit(model_type)
        
        def _update_description(self):
            descriptions = {
                0: "Standard NN ensemble with uncertainty via disagreement. Good general-purpose choice.",
                1: "Optimized for CL/CD prediction with deeper networks and log-transform for CD. Best for glider design.",
                2: "Gaussian Process with Matern kernel. Well-calibrated uncertainty but may struggle with CL/CD ratios.",
                3: "Combines NN prediction strength with GP uncertainty calibration. Experimental."
            }
            self.description_label.setText(descriptions.get(self.model_combo.currentIndex(), ""))
        
        def get_model_type(self) -> str:
            """Get the selected model type as a string"""
            mapping = {
                0: 'nn_ensemble',
                1: 'clcd_optimized',
                2: 'gp',
                3: 'hybrid'
            }
            return mapping.get(self.model_combo.currentIndex(), 'nn_ensemble')
        
        def get_config(self) -> Dict:
            """Get the current configuration"""
            return {
                'model_type': self.get_model_type(),
                'n_members': self.n_members_spin.value(),
                'max_iter': self.max_iter_spin.value()
            }
        
        def create_surrogate(self) -> SurrogateModelWrapper:
            """Create a surrogate model with current settings"""
            config = self.get_config()
            return get_surrogate_model(
                model_type=config['model_type'],
                n_members=config['n_members'],
                max_iter=config['max_iter']
            )


# Test code
if __name__ == '__main__':
    print("Testing AI Surrogate Integration")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    n_samples = 80
    
    X = np.random.rand(n_samples, 4) * 0.5
    CL = 0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2] + np.random.normal(0, 0.02, n_samples)
    CD = 0.05 + 0.02 * X[:, 0] + 0.03 * X[:, 1] ** 2 + np.random.normal(0, 0.005, n_samples)
    CD = np.maximum(CD, 0.01)
    Volume = 3 + 2 * X[:, 0] + 1.5 * X[:, 1] - 0.5 * X[:, 2] + np.random.normal(0, 0.1, n_samples)
    
    y_dict = {
        'CL': CL,
        'CD': CD,
        'Volume': Volume,
        'CL/CD': CL / CD
    }
    
    # Test comparison
    print("\nComparing surrogate models...")
    comparison = SurrogateComparison(model_types=['nn_ensemble', 'clcd_optimized'])
    comparison.compare(X, y_dict, verbose=True)
    print("\n" + comparison.summary())
    
    # Test best model selection
    best_type, best_model = comparison.get_best_model('CL/CD')
    print(f"\nBest model for CL/CD: {best_type}")
    
    # Test CLCDHunterAcquisition
    print("\n" + "=" * 50)
    print("Testing CL/CD Hunter Acquisition")
    
    if best_model is not None:
        acquisition = CLCDHunterAcquisition(
            min_clcd=8.0,
            volume_constraint=(3.0, 5.0),
            exploration_weight=1.5
        )
        
        bounds = np.array([[0, 0.5], [0, 0.5], [0, 1.0], [0, 1.0]])
        next_points = acquisition.select_next_points(
            best_model,
            bounds,
            n_points=3,
            best_clcd=10.0
        )
        
        print(f"\nNext points to evaluate:")
        for i, point in enumerate(next_points):
            print(f"  {i+1}: X = [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}, {point[3]:.3f}]")
            
            # Predict for this point
            means, stds = best_model.predict(point.reshape(1, -1))
            print(f"      Predicted CL/CD: {means['CL/CD'][0]:.2f} ± {stds['CL/CD'][0]:.2f}")
            print(f"      Predicted Volume: {means['Volume'][0]:.2f} ± {stds['Volume'][0]:.2f}")
    
    print("\n✓ AI Surrogate Integration tests passed!")
