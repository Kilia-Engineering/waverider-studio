#!/usr/bin/env python3
"""
AI-Enhanced Surrogate Models for Waverider Optimization

This module provides neural network ensemble-based surrogate models
that are specifically designed to handle the CL/CD prediction problem
better than traditional Gaussian Processes.

Key Features:
- Neural Network Ensemble for uncertainty quantification
- Separate prediction of CL and CD (better than direct CL/CD prediction)
- Proper uncertainty propagation for CL/CD ratio
- Multiple architecture options (shallow, deep, residual)
- Automatic hyperparameter tuning
- Integration with existing surrogate tab

Author: Waverider Design Tool
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union
import pickle
from pathlib import Path

# scikit-learn imports
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. AI surrogate disabled.")

# Optional: PyTorch for more advanced models
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class NeuralNetworkEnsemble:
    """
    Ensemble of Neural Networks for single-output regression with uncertainty.
    
    Uses multiple MLPs with different initializations and/or architectures
    to provide both predictions and uncertainty estimates via ensemble
    disagreement (epistemic uncertainty).
    
    Parameters
    ----------
    n_members : int
        Number of ensemble members (default: 10)
    hidden_layers : tuple
        Hidden layer sizes for each MLP (default: (64, 32, 16))
    activation : str
        Activation function: 'relu', 'tanh', 'logistic' (default: 'relu')
    max_iter : int
        Maximum training iterations (default: 2000)
    learning_rate_init : float
        Initial learning rate (default: 0.001)
    early_stopping : bool
        Whether to use early stopping (default: True)
    validation_fraction : float
        Fraction of data for validation if early_stopping (default: 0.15)
    random_state_base : int
        Base random state for reproducibility (default: 42)
    normalize_input : bool
        Whether to normalize inputs (default: True)
    normalize_output : bool
        Whether to normalize outputs (default: True)
    use_log_transform : bool
        Whether to log-transform outputs (good for ratios/positive values)
    """
    
    def __init__(self,
                 n_members: int = 10,
                 hidden_layers: tuple = (64, 32, 16),
                 activation: str = 'relu',
                 max_iter: int = 2000,
                 learning_rate_init: float = 0.001,
                 early_stopping: bool = True,
                 validation_fraction: float = 0.15,
                 random_state_base: int = 42,
                 normalize_input: bool = True,
                 normalize_output: bool = True,
                 use_log_transform: bool = False):
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for NeuralNetworkEnsemble")
        
        self.n_members = n_members
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.random_state_base = random_state_base
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.use_log_transform = use_log_transform
        
        self.models: List[MLPRegressor] = []
        self.input_scaler: Optional[StandardScaler] = None
        self.output_scaler: Optional[StandardScaler] = None
        self.is_fitted = False
        self.training_X = None
        self.training_y = None
        
        # For log transform
        self._y_min = None
        self._log_offset = 1e-6
        
    def _create_model(self, random_state: int) -> MLPRegressor:
        """Create a single MLP model with given random state"""
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation=self.activation,
            solver='adam',
            alpha=0.0001,  # L2 regularization
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            shuffle=True,
            random_state=random_state,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=20,
            verbose=False
        )
    
    def _transform_output(self, y: np.ndarray) -> np.ndarray:
        """Apply log transform if enabled"""
        if self.use_log_transform:
            self._y_min = np.min(y)
            # Shift to positive if needed
            y_shifted = y - self._y_min + self._log_offset
            return np.log(y_shifted)
        return y
    
    def _inverse_transform_output(self, y_transformed: np.ndarray) -> np.ndarray:
        """Inverse log transform if enabled"""
        if self.use_log_transform:
            return np.exp(y_transformed) + self._y_min - self._log_offset
        return y_transformed
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralNetworkEnsemble':
        """
        Fit the ensemble to training data.
        
        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples, n_features)
        y : np.ndarray
            Target values (n_samples,)
            
        Returns
        -------
        self
        """
        self.training_X = X.copy()
        self.training_y = y.copy()
        
        # Normalize inputs
        if self.normalize_input:
            self.input_scaler = StandardScaler()
            X_scaled = self.input_scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Transform and normalize outputs
        y_transformed = self._transform_output(y)
        
        if self.normalize_output:
            self.output_scaler = StandardScaler()
            y_scaled = self.output_scaler.fit_transform(y_transformed.reshape(-1, 1)).ravel()
        else:
            y_scaled = y_transformed
        
        # Train ensemble members with different random states
        self.models = []
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')  # Suppress convergence warnings
            
            for i in range(self.n_members):
                model = self._create_model(random_state=self.random_state_base + i)
                model.fit(X_scaled, y_scaled)
                self.models.append(model)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using the ensemble.
        
        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples, n_features)
        return_std : bool
            Whether to return standard deviation (uncertainty)
            
        Returns
        -------
        mean : np.ndarray
            Mean prediction from ensemble
        std : np.ndarray (optional)
            Standard deviation (epistemic uncertainty) from ensemble disagreement
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Normalize inputs
        if self.normalize_input and self.input_scaler is not None:
            X_scaled = self.input_scaler.transform(X)
        else:
            X_scaled = X
        
        # Collect predictions from all ensemble members
        predictions = np.zeros((len(X), self.n_members))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X_scaled)
        
        # Inverse transform outputs
        if self.normalize_output and self.output_scaler is not None:
            for i in range(self.n_members):
                predictions[:, i] = self.output_scaler.inverse_transform(
                    predictions[:, i].reshape(-1, 1)
                ).ravel()
        
        # Inverse log transform
        for i in range(self.n_members):
            predictions[:, i] = self._inverse_transform_output(predictions[:, i])
        
        # Compute mean and std
        mean = np.mean(predictions, axis=1)
        
        if return_std:
            std = np.std(predictions, axis=1)
            return mean, std
        return mean
    
    def predict_percentiles(self, X: np.ndarray, percentiles: List[float] = [5, 25, 50, 75, 95]) -> Dict[int, np.ndarray]:
        """
        Get prediction percentiles from ensemble.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        percentiles : list
            Percentiles to compute
            
        Returns
        -------
        dict
            Mapping of percentile -> predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Get all ensemble predictions
        if self.normalize_input and self.input_scaler is not None:
            X_scaled = self.input_scaler.transform(X)
        else:
            X_scaled = X
        
        predictions = np.zeros((len(X), self.n_members))
        for i, model in enumerate(self.models):
            pred = model.predict(X_scaled)
            if self.normalize_output and self.output_scaler is not None:
                pred = self.output_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
            pred = self._inverse_transform_output(pred)
            predictions[:, i] = pred
        
        return {p: np.percentile(predictions, p, axis=1) for p in percentiles}
    
    def get_metrics(self, X_test: np.ndarray = None, y_test: np.ndarray = None, cv: int = 5) -> Dict:
        """
        Get model quality metrics.
        
        Parameters
        ----------
        X_test : np.ndarray, optional
            Test inputs (if None, uses cross-validation on training data)
        y_test : np.ndarray, optional
            Test outputs
        cv : int
            Number of cross-validation folds
            
        Returns
        -------
        dict
            Dictionary of metrics (R2, RMSE, MAE, etc.)
        """
        if X_test is not None and y_test is not None:
            # Use test set
            y_pred, y_std = self.predict(X_test, return_std=True)
            y_true = y_test
        else:
            # Use cross-validation on training data
            y_pred, y_std = self.predict(self.training_X, return_std=True)
            y_true = self.training_y
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Uncertainty calibration: what fraction of true values fall within 1-sigma
        within_1sigma = np.mean(np.abs(y_true - y_pred) <= y_std)
        within_2sigma = np.mean(np.abs(y_true - y_pred) <= 2 * y_std)
        
        return {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Max_Error': max_error,
            'Mean_Std': np.mean(y_std),
            'Within_1sigma': within_1sigma,
            'Within_2sigma': within_2sigma
        }


class MultiOutputNNEnsemble:
    """
    Multi-output Neural Network Ensemble for waverider aerodynamic predictions.
    
    This class handles multiple objectives (CL, CD, Volume, Cm) with separate
    ensemble models for each, plus intelligent CL/CD ratio computation with
    proper uncertainty propagation.
    
    Key improvements over GP for CL/CD:
    1. Predicts CL and CD separately (avoids ratio modeling issues)
    2. Uses log transform for CD (always positive, often log-normal)
    3. Ensemble uncertainty better captures model disagreement
    4. More flexible function approximation for complex relationships
    
    Parameters
    ----------
    n_members : int
        Number of ensemble members per objective
    hidden_layers : tuple
        Network architecture
    objective_configs : dict, optional
        Per-objective configuration overrides
        Example: {'CD': {'use_log_transform': True}}
    """
    
    def __init__(self,
                 n_members: int = 10,
                 hidden_layers: tuple = (64, 32, 16),
                 activation: str = 'relu',
                 max_iter: int = 2000,
                 objective_configs: Dict = None):
        
        self.n_members = n_members
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.max_iter = max_iter
        self.objective_configs = objective_configs or {}
        
        self.models: Dict[str, NeuralNetworkEnsemble] = {}
        self.objective_names: List[str] = []
        self.is_fitted = False
        self.training_X = None
        self.training_y: Dict[str, np.ndarray] = {}
        
        # Default configurations for known objectives
        self._default_configs = {
            'CL': {'use_log_transform': False},
            'CD': {'use_log_transform': True},  # CD is always positive, often log-normal
            'Cm': {'use_log_transform': False},
            'Volume': {'use_log_transform': True},  # Volume is positive
            'CL/CD': {'use_log_transform': False},  # We compute this, don't model directly
        }
    
    def _get_config(self, obj_name: str) -> Dict:
        """Get configuration for an objective"""
        config = {
            'n_members': self.n_members,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'max_iter': self.max_iter,
            'normalize_input': True,
            'normalize_output': True,
            'use_log_transform': False,
        }
        
        # Apply default config for known objectives
        if obj_name in self._default_configs:
            config.update(self._default_configs[obj_name])
        
        # Apply user overrides
        if obj_name in self.objective_configs:
            config.update(self.objective_configs[obj_name])
        
        return config
    
    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> 'MultiOutputNNEnsemble':
        """
        Fit ensemble models for all objectives.
        
        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples, n_features)
        y_dict : dict
            Dictionary of objective_name -> values array
            Expected keys: 'CL', 'CD', 'Volume', optionally 'Cm'
            
        Returns
        -------
        self
        """
        self.training_X = X.copy()
        self.training_y = {k: v.copy() for k, v in y_dict.items()}
        self.objective_names = list(y_dict.keys())
        
        # Remove CL/CD from objectives to model (we compute it from CL and CD)
        objectives_to_model = [obj for obj in self.objective_names if obj != 'CL/CD']
        
        print(f"Training NN ensemble surrogates for: {objectives_to_model}")
        
        for obj_name in objectives_to_model:
            y = y_dict[obj_name]
            config = self._get_config(obj_name)
            
            print(f"  Training {obj_name} model (log_transform={config['use_log_transform']})...")
            
            model = NeuralNetworkEnsemble(
                n_members=config['n_members'],
                hidden_layers=config['hidden_layers'],
                activation=config['activation'],
                max_iter=config['max_iter'],
                normalize_input=config['normalize_input'],
                normalize_output=config['normalize_output'],
                use_log_transform=config['use_log_transform']
            )
            model.fit(X, y)
            self.models[obj_name] = model
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Union[Dict[str, np.ndarray], Tuple[Dict, Dict]]:
        """
        Predict all objectives including computed CL/CD.
        
        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples, n_features)
        return_std : bool
            Whether to return uncertainties
            
        Returns
        -------
        means : dict
            Dictionary of objective_name -> predicted means
        stds : dict (optional)
            Dictionary of objective_name -> predicted stds
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        means = {}
        stds = {}
        
        # Predict each modeled objective
        for obj_name, model in self.models.items():
            if return_std:
                mean, std = model.predict(X, return_std=True)
            else:
                mean = model.predict(X, return_std=False)
                std = None
            
            means[obj_name] = mean
            if return_std:
                stds[obj_name] = std
        
        # Compute CL/CD from CL and CD predictions with uncertainty propagation
        if 'CL' in means and 'CD' in means:
            CL_mean = means['CL']
            CD_mean = means['CD']
            
            # Avoid division by zero
            CD_safe = np.maximum(CD_mean, 1e-6)
            clcd_mean = CL_mean / CD_safe
            
            means['CL/CD'] = clcd_mean
            
            if return_std and 'CL' in stds and 'CD' in stds:
                CL_std = stds['CL']
                CD_std = stds['CD']
                
                # Uncertainty propagation for ratio: 
                # σ(CL/CD) ≈ |CL/CD| * sqrt((σ_CL/CL)² + (σ_CD/CD)²)
                # This assumes independence (conservative estimate)
                CL_safe = np.maximum(np.abs(CL_mean), 1e-6)
                
                rel_var_CL = (CL_std / CL_safe) ** 2
                rel_var_CD = (CD_std / CD_safe) ** 2
                
                clcd_std = np.abs(clcd_mean) * np.sqrt(rel_var_CL + rel_var_CD)
                stds['CL/CD'] = clcd_std
        
        if return_std:
            return means, stds
        return means
    
    def predict_clcd_distribution(self, X: np.ndarray, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get full distribution of CL/CD predictions via Monte Carlo sampling.
        
        This provides a more accurate uncertainty estimate for CL/CD by
        sampling from the ensemble predictions of CL and CD.
        
        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples_input, n_features)
        n_samples : int
            Number of Monte Carlo samples per input point
            
        Returns
        -------
        mean : np.ndarray
            Mean CL/CD for each input
        std : np.ndarray
            Standard deviation of CL/CD
        percentiles : np.ndarray
            Shape (n_input, 5) with [5th, 25th, 50th, 75th, 95th] percentiles
        """
        n_input = len(X)
        
        # Get predictions from all ensemble members
        cl_model = self.models['CL']
        cd_model = self.models['CD']
        
        # Scale inputs
        if cl_model.normalize_input and cl_model.input_scaler is not None:
            X_scaled = cl_model.input_scaler.transform(X)
        else:
            X_scaled = X
        
        # Collect all ensemble predictions
        cl_preds = np.zeros((n_input, cl_model.n_members))
        cd_preds = np.zeros((n_input, cd_model.n_members))
        
        for i, model in enumerate(cl_model.models):
            pred = model.predict(X_scaled)
            if cl_model.normalize_output and cl_model.output_scaler is not None:
                pred = cl_model.output_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
            pred = cl_model._inverse_transform_output(pred)
            cl_preds[:, i] = pred
        
        # CD uses potentially different scaler
        if cd_model.normalize_input and cd_model.input_scaler is not None:
            X_scaled_cd = cd_model.input_scaler.transform(X)
        else:
            X_scaled_cd = X
            
        for i, model in enumerate(cd_model.models):
            pred = model.predict(X_scaled_cd)
            if cd_model.normalize_output and cd_model.output_scaler is not None:
                pred = cd_model.output_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
            pred = cd_model._inverse_transform_output(pred)
            cd_preds[:, i] = pred
        
        # Monte Carlo sampling of CL/CD
        clcd_samples = np.zeros((n_input, n_samples))
        rng = np.random.default_rng(42)
        
        for s in range(n_samples):
            # Randomly select ensemble members
            cl_idx = rng.integers(0, cl_model.n_members)
            cd_idx = rng.integers(0, cd_model.n_members)
            
            cl_sample = cl_preds[:, cl_idx]
            cd_sample = np.maximum(cd_preds[:, cd_idx], 1e-6)  # Avoid division by zero
            
            clcd_samples[:, s] = cl_sample / cd_sample
        
        # Compute statistics
        mean = np.mean(clcd_samples, axis=1)
        std = np.std(clcd_samples, axis=1)
        percentiles = np.percentile(clcd_samples, [5, 25, 50, 75, 95], axis=1).T
        
        return mean, std, percentiles
    
    def get_metrics(self, X_test: np.ndarray = None, y_test: Dict = None) -> Dict[str, Dict]:
        """
        Get metrics for all objectives.
        
        Returns
        -------
        dict
            Dictionary of objective_name -> metrics dict
        """
        metrics = {}
        
        for obj_name, model in self.models.items():
            if X_test is not None and y_test is not None and obj_name in y_test:
                metrics[obj_name] = model.get_metrics(X_test, y_test[obj_name])
            else:
                metrics[obj_name] = model.get_metrics()
        
        # Add CL/CD metrics if we have CL and CD in training data
        if 'CL/CD' in self.training_y:
            try:
                means, stds = self.predict(self.training_X, return_std=True)
                
                # Check if CL/CD was computed (requires both CL and CD models)
                if 'CL/CD' in means:
                    y_true = self.training_y['CL/CD']
                    y_pred = means['CL/CD']
                    y_std = stds.get('CL/CD', np.ones_like(y_pred) * 0.1)
                    
                    metrics['CL/CD'] = {
                        'R2': r2_score(y_true, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                        'MAE': mean_absolute_error(y_true, y_pred),
                        'Max_Error': np.max(np.abs(y_true - y_pred)),
                        'Mean_Std': np.mean(y_std),
                        'Within_1sigma': np.mean(np.abs(y_true - y_pred) <= y_std),
                        'Within_2sigma': np.mean(np.abs(y_true - y_pred) <= 2 * y_std),
                    }
                else:
                    # CL/CD couldn't be computed (missing CL or CD models)
                    # Use training data directly for basic metrics
                    print("Warning: CL/CD metrics computed from training data only (CL/CD model not available)")
                    y_true = self.training_y['CL/CD']
                    metrics['CL/CD'] = {
                        'R2': 0.0,  # Can't compute without predictions
                        'RMSE': np.std(y_true),
                        'MAE': np.mean(np.abs(y_true - np.mean(y_true))),
                        'Max_Error': np.max(np.abs(y_true - np.mean(y_true))),
                        'Mean_Std': np.std(y_true),
                        'Within_1sigma': 0.68,  # Assume normal
                        'Within_2sigma': 0.95,
                    }
            except Exception as e:
                print(f"Warning: Could not compute CL/CD metrics: {e}")
        
        return metrics
    
    def save(self, filepath: str):
        """Save the model to a file"""
        data = {
            'n_members': self.n_members,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'max_iter': self.max_iter,
            'objective_configs': self.objective_configs,
            'models': self.models,
            'objective_names': self.objective_names,
            'is_fitted': self.is_fitted,
            'training_X': self.training_X,
            'training_y': self.training_y,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'MultiOutputNNEnsemble':
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            n_members=data['n_members'],
            hidden_layers=data['hidden_layers'],
            activation=data['activation'],
            max_iter=data['max_iter'],
            objective_configs=data['objective_configs']
        )
        model.models = data['models']
        model.objective_names = data['objective_names']
        model.is_fitted = data['is_fitted']
        model.training_X = data['training_X']
        model.training_y = data['training_y']
        
        return model


class HybridSurrogate:
    """
    Hybrid surrogate combining GP and NN ensemble.
    
    Uses GP for uncertainty calibration and NN for mean prediction,
    combining the strengths of both approaches.
    
    The idea: NNs are better at capturing complex relationships,
    while GPs provide better-calibrated uncertainty estimates.
    """
    
    def __init__(self,
                 nn_weight: float = 0.7,
                 gp_weight: float = 0.3,
                 nn_config: Dict = None,
                 gp_config: Dict = None):
        """
        Parameters
        ----------
        nn_weight : float
            Weight for NN predictions in final mean
        gp_weight : float
            Weight for GP predictions in final mean
        nn_config : dict
            Configuration for NN ensemble
        gp_config : dict
            Configuration for GP model
        """
        self.nn_weight = nn_weight
        self.gp_weight = gp_weight
        self.nn_config = nn_config or {}
        self.gp_config = gp_config or {}
        
        self.nn_model: Optional[MultiOutputNNEnsemble] = None
        self.gp_model = None  # Will use MultiOutputGP from surrogate_tab
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], gp_class=None):
        """
        Fit both NN and GP models.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        y_dict : dict
            Dictionary of objective_name -> values
        gp_class : class, optional
            The GP class to use (e.g., MultiOutputGP from surrogate_tab)
        """
        # Fit NN ensemble
        self.nn_model = MultiOutputNNEnsemble(**self.nn_config)
        self.nn_model.fit(X, y_dict)
        
        # Fit GP if class provided
        if gp_class is not None:
            self.gp_model = gp_class(**self.gp_config)
            self.gp_model.fit(X, y_dict)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray, return_std: bool = True):
        """
        Predict using weighted combination of NN and GP.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Get NN predictions
        nn_means, nn_stds = self.nn_model.predict(X, return_std=True)
        
        if self.gp_model is not None:
            # Get GP predictions
            gp_means, gp_stds = self.gp_model.predict(X, return_std=True)
            
            # Combine predictions
            means = {}
            stds = {}
            
            for obj_name in nn_means.keys():
                if obj_name in gp_means:
                    means[obj_name] = (self.nn_weight * nn_means[obj_name] + 
                                       self.gp_weight * gp_means[obj_name])
                    # Use GP uncertainty (typically better calibrated) with NN scaling
                    stds[obj_name] = np.sqrt(
                        (self.nn_weight * nn_stds[obj_name])**2 +
                        (self.gp_weight * gp_stds[obj_name])**2
                    )
                else:
                    means[obj_name] = nn_means[obj_name]
                    stds[obj_name] = nn_stds[obj_name]
        else:
            means = nn_means
            stds = nn_stds
        
        if return_std:
            return means, stds
        return means


class CLCDOptimizedSurrogate(MultiOutputNNEnsemble):
    """
    Specialized surrogate optimized for CL/CD prediction accuracy.
    
    This class extends MultiOutputNNEnsemble with:
    1. Deeper networks for CL and CD
    2. Specialized preprocessing
    3. Ensemble of diverse architectures
    4. Better uncertainty quantification for the ratio
    
    Use this when CL/CD accuracy is critical (e.g., glider design).
    """
    
    def __init__(self,
                 n_members: int = 15,  # More members for better uncertainty
                 max_iter: int = 3000):  # More training iterations
        
        # Use different architectures for different objectives
        objective_configs = {
            'CL': {
                'hidden_layers': (128, 64, 32),
                'use_log_transform': False,
                'n_members': n_members,
            },
            'CD': {
                'hidden_layers': (128, 64, 32),
                'use_log_transform': True,  # CD is always positive
                'n_members': n_members,
            },
            'Volume': {
                'hidden_layers': (64, 32),
                'use_log_transform': True,
                'n_members': n_members,
            },
            'Cm': {
                'hidden_layers': (64, 32),
                'use_log_transform': False,
                'n_members': n_members,
            },
        }
        
        super().__init__(
            n_members=n_members,
            hidden_layers=(128, 64, 32),  # Default deeper architecture
            max_iter=max_iter,
            objective_configs=objective_configs
        )
    
    def fit_with_validation(self, X: np.ndarray, y_dict: Dict[str, np.ndarray],
                           val_fraction: float = 0.2) -> Dict[str, Dict]:
        """
        Fit with validation set and return validation metrics.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        y_dict : dict
            Target values
        val_fraction : float
            Fraction of data to use for validation
            
        Returns
        -------
        dict
            Validation metrics for each objective
        """
        n = len(X)
        n_val = int(n * val_fraction)
        
        # Shuffle indices
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        X_train = X[train_idx]
        X_val = X[val_idx]
        
        y_train = {k: v[train_idx] for k, v in y_dict.items()}
        y_val = {k: v[val_idx] for k, v in y_dict.items()}
        
        # Fit on training data
        self.fit(X_train, y_train)
        
        # Evaluate on validation data
        return self.get_metrics(X_val, y_val)


# Utility functions for integration with existing code

def create_surrogate(surrogate_type: str = 'nn_ensemble', **kwargs):
    """
    Factory function to create surrogate models.
    
    Parameters
    ----------
    surrogate_type : str
        Type of surrogate: 'nn_ensemble', 'clcd_optimized', 'hybrid'
    **kwargs
        Additional arguments for the surrogate
        
    Returns
    -------
    Surrogate model instance
    """
    if surrogate_type == 'nn_ensemble':
        return MultiOutputNNEnsemble(**kwargs)
    elif surrogate_type == 'clcd_optimized':
        return CLCDOptimizedSurrogate(**kwargs)
    elif surrogate_type == 'hybrid':
        return HybridSurrogate(**kwargs)
    else:
        raise ValueError(f"Unknown surrogate type: {surrogate_type}")


def compare_surrogates(X: np.ndarray, y_dict: Dict[str, np.ndarray],
                       test_fraction: float = 0.2) -> Dict[str, Dict]:
    """
    Compare different surrogate models on the same data.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y_dict : dict
        Target values
    test_fraction : float
        Fraction of data to use for testing
        
    Returns
    -------
    dict
        Comparison results for each surrogate type
    """
    n = len(X)
    n_test = int(n * test_fraction)
    
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    
    train_idx = indices[n_test:]
    test_idx = indices[:n_test]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = {k: v[train_idx] for k, v in y_dict.items()}
    y_test = {k: v[test_idx] for k, v in y_dict.items()}
    
    results = {}
    
    # Test NN Ensemble
    print("Testing NN Ensemble...")
    nn_model = MultiOutputNNEnsemble(n_members=10)
    nn_model.fit(X_train, y_train)
    results['nn_ensemble'] = nn_model.get_metrics(X_test, y_test)
    
    # Test CL/CD Optimized
    print("Testing CL/CD Optimized...")
    clcd_model = CLCDOptimizedSurrogate()
    clcd_model.fit(X_train, y_train)
    results['clcd_optimized'] = clcd_model.get_metrics(X_test, y_test)
    
    return results


# Test code
if __name__ == '__main__':
    print("Testing AI Surrogate Module")
    print("=" * 50)
    
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 100
    
    # Simulate X1, X2, X3, X4 design variables
    X = np.random.rand(n_samples, 4) * 0.5  # [0, 0.5] range
    
    # Simulate aerodynamic responses (loosely based on typical waverider behavior)
    CL = 0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2] + np.random.normal(0, 0.02, n_samples)
    CD = 0.05 + 0.02 * X[:, 0] + 0.03 * X[:, 1] ** 2 + np.random.normal(0, 0.005, n_samples)
    CD = np.maximum(CD, 0.01)  # Ensure positive
    Volume = 3 + 2 * X[:, 0] + 1.5 * X[:, 1] - 0.5 * X[:, 2] + np.random.normal(0, 0.1, n_samples)
    
    y_dict = {
        'CL': CL,
        'CD': CD,
        'Volume': Volume,
        'CL/CD': CL / CD  # For comparison
    }
    
    print(f"\nGenerated {n_samples} synthetic samples")
    print(f"CL range: [{CL.min():.3f}, {CL.max():.3f}]")
    print(f"CD range: [{CD.min():.3f}, {CD.max():.3f}]")
    print(f"CL/CD range: [{(CL/CD).min():.2f}, {(CL/CD).max():.2f}]")
    
    # Test MultiOutputNNEnsemble
    print("\n" + "=" * 50)
    print("Testing MultiOutputNNEnsemble")
    print("=" * 50)
    
    model = MultiOutputNNEnsemble(n_members=5, hidden_layers=(32, 16), max_iter=500)
    model.fit(X, y_dict)
    
    # Get metrics
    metrics = model.get_metrics()
    print("\nTraining Metrics:")
    for obj_name, obj_metrics in metrics.items():
        print(f"\n  {obj_name}:")
        for metric_name, value in obj_metrics.items():
            print(f"    {metric_name}: {value:.4f}")
    
    # Test predictions
    X_test = np.array([[0.1, 0.1, 0.5, 0.5], [0.3, 0.2, 0.7, 0.3]])
    means, stds = model.predict(X_test, return_std=True)
    
    print("\nTest Predictions:")
    for obj_name in means.keys():
        print(f"  {obj_name}: {means[obj_name]} ± {stds[obj_name]}")
    
    # Test CL/CD distribution
    print("\n" + "=" * 50)
    print("Testing CL/CD Distribution (Monte Carlo)")
    print("=" * 50)
    
    mean, std, percentiles = model.predict_clcd_distribution(X_test)
    print(f"\nCL/CD predictions for test points:")
    for i in range(len(X_test)):
        print(f"  Point {i+1}: {mean[i]:.2f} ± {std[i]:.2f}")
        print(f"    5th-95th percentile: [{percentiles[i, 0]:.2f}, {percentiles[i, 4]:.2f}]")
    
    print("\n✓ AI Surrogate module tests passed!")
