#!/usr/bin/env python3
"""
Multi-Mach Hunter Tab for Waverider GUI
========================================

Finds waverider designs that perform well across a RANGE of Mach numbers,
not just a single point. Uses the off-design surrogate to predict performance
at multiple flight conditions and optimizes for:

- Maximum average CL/CD (best overall performer)
- Maximum minimum CL/CD (most robust - never bad)
- Minimum variation (most consistent across Mach)

This enables designing vehicles for missions that span multiple flight regimes.
"""

import sys
import os
import json
import pickle
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QGridLayout, QFrame,
                             QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
                             QProgressBar, QTextEdit, QTabWidget, QFileDialog,
                             QMessageBox, QSplitter, QScrollArea, QRadioButton,
                             QButtonGroup)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Scikit-learn for neural network
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# NEURAL NETWORK ENSEMBLE (same as other tabs)
# =============================================================================

class NNEnsemble:
    """Ensemble of neural networks for prediction with uncertainty."""
    
    def __init__(self, n_models=5, hidden_layers=(128, 64, 32), 
                 max_iter=500, random_state=42):
        self.n_models = n_models
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.random_state = random_state
        self.models = []
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        
    def predict(self, X, return_std=False):
        """Predict with optional uncertainty."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        X_scaled = self.scaler_X.transform(X)
        
        predictions = np.array([
            model.predict(X_scaled) for model in self.models
        ])
        
        predictions = np.array([
            self.scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
            for pred in predictions
        ])
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        if return_std:
            return mean_pred, std_pred
        return mean_pred
    
    @classmethod
    def load(cls, filepath):
        """Load ensemble from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        ensemble = cls(
            n_models=data['n_models'],
            hidden_layers=data['hidden_layers']
        )
        ensemble.models = data['models']
        ensemble.scaler_X = data['scaler_X']
        ensemble.scaler_y = data['scaler_y']
        ensemble.is_fitted = data['is_fitted']
        return ensemble


# =============================================================================
# MAIN TAB WIDGET
# =============================================================================

class MultiMachHunterTab(QWidget):
    """
    Tab for finding optimal multi-Mach waverider designs.
    """
    
    design_selected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        
        # Surrogate models (will be loaded)
        self.ensemble_CL_CD = None
        self.ensemble_Volume = None
        self.models_loaded = False
        self.has_volume_model = False
        
        # Results storage
        self.best_designs = []
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Results
        right_panel = self.create_results_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 600])
        main_layout.addWidget(splitter)
    
    def create_control_panel(self):
        """Create the control panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Header
        header = QLabel(
            "<h3>🌐 Multi-Mach Hunter</h3>"
            "<p style='color: #888888;'>Find designs that perform well across a range of Mach numbers</p>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)
        
        # Model Loading
        model_group = QGroupBox("🧠 Surrogate Model")
        model_layout = QVBoxLayout(model_group)
        
        self.model_status = QLabel("⚠️ No model loaded")
        self.model_status.setStyleSheet("QLabel { color: #F59E0B; background-color: #78350F; padding: 5px; border-radius: 3px; }")
        model_layout.addWidget(self.model_status)
        
        load_btn = QPushButton("📂 Load Surrogate Model...")
        load_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_btn)
        
        layout.addWidget(model_group)
        
        # Mach Range Selection
        mach_group = QGroupBox("🎯 Target Mach Range")
        mach_layout = QGridLayout(mach_group)
        
        mach_layout.addWidget(QLabel(
            "<i>Designs will be optimized for performance across this entire range</i>"
        ), 0, 0, 1, 4)
        
        mach_layout.addWidget(QLabel("Mach Range:"), 1, 0)
        
        self.mach_min_spin = QDoubleSpinBox()
        self.mach_min_spin.setRange(3.0, 8.0)
        self.mach_min_spin.setValue(4.0)
        self.mach_min_spin.setSingleStep(1.0)
        self.mach_min_spin.setPrefix("from M")
        mach_layout.addWidget(self.mach_min_spin, 1, 1)
        
        self.mach_max_spin = QDoubleSpinBox()
        self.mach_max_spin.setRange(3.0, 8.0)
        self.mach_max_spin.setValue(7.0)
        self.mach_max_spin.setSingleStep(1.0)
        self.mach_max_spin.setPrefix("to M")
        mach_layout.addWidget(self.mach_max_spin, 1, 2)
        
        mach_layout.addWidget(QLabel("AoA for evaluation:"), 2, 0)
        self.eval_aoa_spin = QDoubleSpinBox()
        self.eval_aoa_spin.setRange(0.0, 5.0)
        self.eval_aoa_spin.setValue(2.0)
        self.eval_aoa_spin.setSingleStep(0.5)
        self.eval_aoa_spin.setSuffix("°")
        mach_layout.addWidget(self.eval_aoa_spin, 2, 1, 1, 2)
        
        layout.addWidget(mach_group)
        
        # Optimization Objective
        obj_group = QGroupBox("📊 Optimization Objective")
        obj_layout = QVBoxLayout(obj_group)
        
        self.obj_button_group = QButtonGroup()
        
        self.obj_robust = QRadioButton("🛡️ Maximize MINIMUM CL/CD (Robust)")
        self.obj_robust.setToolTip("Best worst-case performance - design never drops below this value")
        self.obj_robust.setChecked(True)
        self.obj_button_group.addButton(self.obj_robust, 0)
        obj_layout.addWidget(self.obj_robust)
        
        self.obj_mean = QRadioButton("📈 Maximize MEAN CL/CD (Best Average)")
        self.obj_mean.setToolTip("Best average performance across all Mach numbers")
        self.obj_button_group.addButton(self.obj_mean, 1)
        obj_layout.addWidget(self.obj_mean)
        
        self.obj_consistent = QRadioButton("📏 Minimize VARIATION (Most Consistent)")
        self.obj_consistent.setToolTip("Most stable performance - smallest difference between best and worst")
        self.obj_button_group.addButton(self.obj_consistent, 2)
        obj_layout.addWidget(self.obj_consistent)
        
        self.obj_balanced = QRadioButton("⚖️ Balanced (High Mean + Low Variation)")
        self.obj_balanced.setToolTip("Combines good average with consistency")
        self.obj_button_group.addButton(self.obj_balanced, 3)
        obj_layout.addWidget(self.obj_balanced)
        
        layout.addWidget(obj_group)
        
        # Geometry Constraints
        geom_group = QGroupBox("📐 Geometry Constraints")
        geom_layout = QGridLayout(geom_group)
        
        row = 0
        geom_layout.addWidget(QLabel("Width (m):"), row, 0)
        self.width_min = QDoubleSpinBox()
        self.width_min.setRange(0.5, 5.0)
        self.width_min.setValue(1.0)
        self.width_min.setPrefix("min: ")
        geom_layout.addWidget(self.width_min, row, 1)
        
        self.width_max = QDoubleSpinBox()
        self.width_max.setRange(0.5, 5.0)
        self.width_max.setValue(3.0)
        self.width_max.setPrefix("max: ")
        geom_layout.addWidget(self.width_max, row, 2)
        row += 1
        
        geom_layout.addWidget(QLabel("Height (m):"), row, 0)
        self.height_min = QDoubleSpinBox()
        self.height_min.setRange(0.1, 5.0)
        self.height_min.setValue(0.5)
        self.height_min.setPrefix("min: ")
        geom_layout.addWidget(self.height_min, row, 1)
        
        self.height_max = QDoubleSpinBox()
        self.height_max.setRange(0.1, 5.0)
        self.height_max.setValue(2.5)
        self.height_max.setPrefix("max: ")
        geom_layout.addWidget(self.height_max, row, 2)
        row += 1
        
        geom_layout.addWidget(QLabel("Min Volume (m³):"), row, 0)
        self.min_volume = QDoubleSpinBox()
        self.min_volume.setRange(0.0, 50.0)
        self.min_volume.setValue(0.5)
        self.min_volume.setSingleStep(0.1)
        geom_layout.addWidget(self.min_volume, row, 1)
        
        self.volume_check = QCheckBox("Enforce")
        self.volume_check.setChecked(True)
        geom_layout.addWidget(self.volume_check, row, 2)
        
        layout.addWidget(geom_group)
        
        # Design Constraints
        design_group = QGroupBox("✈️ Design Constraints")
        design_layout = QGridLayout(design_group)
        
        design_layout.addWidget(QLabel("Design Mach:"), 0, 0)
        self.design_mach_min = QDoubleSpinBox()
        self.design_mach_min.setRange(3.0, 6.0)
        self.design_mach_min.setValue(4.0)
        self.design_mach_min.setPrefix("min: ")
        design_layout.addWidget(self.design_mach_min, 0, 1)
        
        self.design_mach_max = QDoubleSpinBox()
        self.design_mach_max.setRange(3.0, 6.0)
        self.design_mach_max.setValue(6.0)
        self.design_mach_max.setPrefix("max: ")
        design_layout.addWidget(self.design_mach_max, 0, 2)
        
        design_layout.addWidget(QLabel("Design β (deg):"), 1, 0)
        self.beta_min = QDoubleSpinBox()
        self.beta_min.setRange(10.0, 30.0)
        self.beta_min.setValue(15.0)
        self.beta_min.setPrefix("min: ")
        design_layout.addWidget(self.beta_min, 1, 1)
        
        self.beta_max = QDoubleSpinBox()
        self.beta_max.setRange(10.0, 30.0)
        self.beta_max.setValue(22.0)
        self.beta_max.setPrefix("max: ")
        design_layout.addWidget(self.beta_max, 1, 2)
        
        layout.addWidget(design_group)
        
        # Run Button
        self.run_btn = QPushButton("🚀 Run Multi-Mach Hunter")
        self.run_btn.clicked.connect(self.run_hunter)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #F59E0B;
                color: #0A0A0A;
                padding: 15px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #D97706; }
            QPushButton:disabled { background-color: #333333; color: #666666; }
        """)
        layout.addWidget(self.run_btn)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        layout.addStretch()
        
        scroll.setWidget(panel)
        return scroll
    
    def create_results_panel(self):
        """Create the results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Results text tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 10))
        self.results_text.setPlaceholderText(
            "Multi-Mach Hunter Results\n"
            "=========================\n\n"
            "Configure your target Mach range and constraints,\n"
            "then click 'Run Multi-Mach Hunter' to find optimal designs.\n\n"
            "The hunter will evaluate each design at multiple Mach numbers\n"
            "and rank them based on your chosen optimization objective."
        )
        results_layout.addWidget(self.results_text)
        
        # Apply button
        self.apply_btn = QPushButton("✅ Apply Best Design to Main GUI")
        self.apply_btn.clicked.connect(self.apply_best_design)
        self.apply_btn.setEnabled(False)
        results_layout.addWidget(self.apply_btn)
        
        self.results_tabs.addTab(results_tab, "📋 Results")
        
        # Plot tab
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        
        self.plot_fig = Figure(figsize=(8, 6))
        self.plot_canvas = FigureCanvas(self.plot_fig)
        plot_layout.addWidget(self.plot_canvas)
        
        self.results_tabs.addTab(plot_tab, "📊 Comparison Plot")
        
        layout.addWidget(self.results_tabs)
        
        return panel
    
    def load_model(self):
        """Load the surrogate model."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Surrogate Model Folder", "",
            QFileDialog.ShowDirsOnly
        )
        
        if not folder:
            return
        
        try:
            # Load CL/CD model (required)
            clcd_path = os.path.join(folder, 'ensemble_CL_CD.pkl')
            if not os.path.exists(clcd_path):
                raise FileNotFoundError("ensemble_CL_CD.pkl not found")
            
            self.ensemble_CL_CD = NNEnsemble.load(clcd_path)
            
            # Load Volume model (optional)
            vol_path = os.path.join(folder, 'ensemble_Volume.pkl')
            if os.path.exists(vol_path):
                self.ensemble_Volume = NNEnsemble.load(vol_path)
                self.has_volume_model = True
            else:
                self.has_volume_model = False
            
            self.models_loaded = True
            
            self.model_status.setText("✅ Model loaded!")
            self.model_status.setStyleSheet(
                "QLabel { color: #4ADE80; background-color: #14532D; "
                "padding: 5px; border-radius: 3px; }"
            )
            
            self.run_btn.setEnabled(True)
            
            msg = "CL/CD surrogate loaded!"
            if self.has_volume_model:
                msg += "\nVolume surrogate loaded!"
            QMessageBox.information(self, "Success", msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
    
    def predict_multi_mach(self, params, mach_range, aoa):
        """Predict CL/CD across multiple Mach numbers for a design."""
        results = {}
        
        for mach in mach_range:
            X = np.array([[
                params['X1'], params['X2'], params['X3'], params['X4'],
                params['design_Mach'], params['design_beta'],
                mach, aoa,
                params['width'], params['height']
            ]])
            
            pred, std = self.ensemble_CL_CD.predict(X, return_std=True)
            results[f'M{int(mach)}'] = {'value': pred[0], 'std': std[0]}
        
        # Compute aggregates
        values = [r['value'] for r in results.values()]
        stds = [r['std'] for r in results.values()]
        
        results['mean'] = np.mean(values)
        results['min'] = np.min(values)
        results['max'] = np.max(values)
        results['std'] = np.std(values)
        results['range'] = np.max(values) - np.min(values)
        results['avg_uncertainty'] = np.mean(stds)
        
        return results
    
    def predict_volume(self, params):
        """Predict volume for a design."""
        if not self.has_volume_model:
            # Approximate
            return params['width'] * params['height'] * 3 * params['width'] * 0.15
        
        X = np.array([[
            params['X1'], params['X2'], params['X3'], params['X4'],
            params['design_Mach'], params['design_beta'],
            params['width'], params['height']
        ]])
        
        pred, _ = self.ensemble_Volume.predict(X, return_std=True)
        return pred[0]
    
    def get_min_beta(self, design_mach):
        """Get minimum valid beta (shock angle) for a given design Mach.
        
        The shock angle must be greater than the Mach angle for an attached shock.
        Mach angle μ = arcsin(1/M)
        We add a small margin for numerical stability.
        """
        mach_angle = np.degrees(np.arcsin(1.0 / design_mach))
        # Add 1 degree margin for safety
        return mach_angle + 1.0
    
    def run_hunter(self):
        """Run the multi-Mach optimization."""
        if not self.models_loaded:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
        
        try:
            # Get parameters
            mach_min = self.mach_min_spin.value()
            mach_max = self.mach_max_spin.value()
            eval_aoa = self.eval_aoa_spin.value()
            
            if mach_min >= mach_max:
                QMessageBox.warning(self, "Warning", "Mach min must be less than max!")
                return
            
            # Create Mach range (integer steps)
            mach_range = list(range(int(mach_min), int(mach_max) + 1))
            
            # Get constraints
            width_min = self.width_min.value()
            width_max = self.width_max.value()
            height_min = self.height_min.value()
            height_max = self.height_max.value()
            min_volume = self.min_volume.value()
            enforce_volume = self.volume_check.isChecked()
            
            design_mach_min = self.design_mach_min.value()
            design_mach_max = self.design_mach_max.value()
            beta_min = self.beta_min.value()
            beta_max = self.beta_max.value()
            
            # Get objective
            obj_id = self.obj_button_group.checkedId()
            obj_names = {0: 'robust', 1: 'mean', 2: 'consistent', 3: 'balanced'}
            objective = obj_names[obj_id]
            
            # Build results header
            results = "=" * 75 + "\n"
            results += "           MULTI-MACH HUNTER - OPTIMAL DESIGN SEARCH\n"
            results += "=" * 75 + "\n\n"
            
            results += f"TARGET MACH RANGE: M{int(mach_min)} to M{int(mach_max)}\n"
            results += f"Evaluation AoA: {eval_aoa}°\n\n"
            
            results += f"OPTIMIZATION OBJECTIVE: {objective.upper()}\n"
            if objective == 'robust':
                results += "  → Maximize minimum CL/CD (best worst-case)\n"
            elif objective == 'mean':
                results += "  → Maximize mean CL/CD (best average)\n"
            elif objective == 'consistent':
                results += "  → Minimize variation (most stable)\n"
            else:
                results += "  → Balance mean and consistency\n"
            
            results += f"\nGEOMETRY CONSTRAINTS:\n"
            results += f"  Width: {width_min:.1f} - {width_max:.1f} m\n"
            results += f"  Height: {height_min:.1f} - {height_max:.1f} m\n"
            if enforce_volume:
                results += f"  Min Volume: {min_volume:.1f} m³\n"
                results += f"  ⚠️ NOTE: Volume predictions may be ~50% higher than actual.\n"
            
            results += f"\nDESIGN CONSTRAINTS:\n"
            results += f"  Design Mach: {design_mach_min:.0f} - {design_mach_max:.0f}\n"
            results += f"  Design β: {beta_min:.0f}° - {beta_max:.0f}°\n"
            
            # Show physics constraints
            results += f"\nPHYSICS CONSTRAINTS (Mach angle limits):\n"
            for dm in [3, 4, 5, 6]:
                if design_mach_min <= dm <= design_mach_max:
                    min_b = self.get_min_beta(dm)
                    results += f"  Design M{dm}: β > {min_b:.1f}°\n"
            
            results += "\nSearching...\n\n"
            self.results_text.setText(results)
            
            # Show progress
            self.progress.setVisible(True)
            self.progress.setValue(0)
            
            # Sample design space
            n_samples = 5000
            design_machs = [m for m in [3.0, 4.0, 5.0, 6.0] 
                          if design_mach_min <= m <= design_mach_max]
            
            np.random.seed(42)
            candidates = []
            n_rejected_physics = 0
            n_rejected_volume = 0
            n_rejected_design_space = 0
            
            for i in range(n_samples):
                if i % 500 == 0:
                    self.progress.setValue(int(100 * i / n_samples))
                    QApplication.processEvents()
                
                # Sample design Mach first
                design_mach = np.random.choice(design_machs)
                
                # Get minimum valid beta for this design Mach
                min_valid_beta = self.get_min_beta(design_mach)
                
                # Adjust beta range to ensure valid designs
                effective_beta_min = max(beta_min, min_valid_beta)
                if effective_beta_min >= beta_max:
                    # This design Mach can't produce valid designs in our beta range
                    n_rejected_physics += 1
                    continue
                
                params = {
                    'X1': np.random.uniform(0.05, 0.45),
                    'X2': np.random.uniform(0.1, 0.9),
                    'X3': np.random.uniform(0.1, 0.9),
                    'X4': np.random.uniform(0.1, 0.9),
                    'design_Mach': design_mach,
                    'design_beta': np.random.uniform(effective_beta_min, beta_max),
                    'width': np.random.uniform(width_min, width_max),
                    'height': np.random.uniform(height_min, height_max)
                }
                
                # Check design space constraint: X2/(1-X1)^4 < (7/64) × (W/H)^4
                # Apply 10% safety margin to avoid boundary cases that fail in GUI
                X1, X2 = params['X1'], params['X2']
                W, H = params['width'], params['height']
                constraint_value = X2 / ((1 - X1) ** 4) if X1 < 1 else float('inf')
                max_constraint = (7.0 / 64.0) * (W / H) ** 4
                
                if constraint_value >= 0.85 * max_constraint:
                    # Design space constraint violated (or too close to boundary)
                    n_rejected_design_space += 1
                    continue
                
                # Volume constraint
                if enforce_volume:
                    vol = self.predict_volume(params)
                    if vol < min_volume:
                        n_rejected_volume += 1
                        continue
                    params['volume'] = vol
                
                # Evaluate at all Mach numbers
                multi_results = self.predict_multi_mach(params, mach_range, eval_aoa)
                
                # Compute score based on objective
                if objective == 'robust':
                    score = multi_results['min']
                elif objective == 'mean':
                    score = multi_results['mean']
                elif objective == 'consistent':
                    # Higher score = better, so negate std
                    # Also require decent mean
                    if multi_results['mean'] < 3.0:
                        score = -999
                    else:
                        score = -multi_results['std'] + multi_results['mean'] * 0.1
                else:  # balanced
                    score = multi_results['mean'] - 0.5 * multi_results['std']
                
                candidates.append({
                    'params': params,
                    'multi_results': multi_results,
                    'score': score
                })
            
            self.progress.setValue(100)
            
            # Sort by score
            candidates.sort(key=lambda x: x['score'], reverse=True)
            self.best_designs = candidates[:20]
            
            # Display results
            results += "-" * 75 + "\n"
            results += f"SEARCH COMPLETE:\n"
            results += f"  Valid designs found: {len(candidates)}\n"
            if n_rejected_physics > 0:
                results += f"  Rejected (physics):  {n_rejected_physics} (β < Mach angle)\n"
            if n_rejected_design_space > 0:
                results += f"  Rejected (design space): {n_rejected_design_space} (X2/(1-X1)⁴ constraint)\n"
            if n_rejected_volume > 0:
                results += f"  Rejected (volume):   {n_rejected_volume}\n"
            results += "-" * 75 + "\n\n"
            
            if len(candidates) == 0:
                results += "❌ NO VALID DESIGNS FOUND!\n\n"
                results += "All designs were rejected. Try:\n"
                results += "  - Increasing β max (current limit may be below Mach angle)\n"
                results += "  - Using higher Design Mach (M4+ has lower β requirements)\n"
                results += "  - Relaxing volume constraints\n"
                self.results_text.setText(results)
                self.progress.setVisible(False)
                return
            
            results += "TOP 10 MULTI-MACH DESIGNS:\n"
            results += "-" * 75 + "\n"
            
            # Header with Mach columns
            header = f"{'#':<3} "
            for m in mach_range:
                header += f"{'M'+str(m):>6} "
            header += f"| {'Min':>6} {'Mean':>6} {'Std':>5} | {'dM':>3} {'β':>5}\n"
            results += header
            results += "-" * 75 + "\n"
            
            for i, cand in enumerate(self.best_designs[:10]):
                mr = cand['multi_results']
                p = cand['params']
                
                line = f"{i+1:<3} "
                for m in mach_range:
                    val = mr[f'M{m}']['value']
                    line += f"{val:>6.2f} "
                line += f"| {mr['min']:>6.2f} {mr['mean']:>6.2f} {mr['std']:>5.2f} "
                line += f"| {p['design_Mach']:>3.0f} {p['design_beta']:>5.1f}\n"
                results += line
            
            # Best design details
            results += "\n" + "=" * 75 + "\n"
            results += "🏆 BEST MULTI-MACH DESIGN\n"
            results += "=" * 75 + "\n\n"
            
            best = self.best_designs[0]
            bp = best['params']
            br = best['multi_results']
            
            results += "PERFORMANCE ACROSS MACH RANGE:\n"
            for m in mach_range:
                val = br[f'M{m}']['value']
                std = br[f'M{m}']['std']
                results += f"  Mach {m}: CL/CD = {val:.3f} ± {std:.3f}\n"
            
            results += f"\n  MEAN:  {br['mean']:.3f}\n"
            results += f"  MIN:   {br['min']:.3f} (worst case)\n"
            results += f"  MAX:   {br['max']:.3f} (best case)\n"
            results += f"  RANGE: {br['range']:.3f}\n"
            results += f"  STD:   {br['std']:.3f}\n"
            
            results += f"\nDESIGN PARAMETERS:\n"
            results += f"  Design Mach: {bp['design_Mach']:.0f}\n"
            results += f"  Design β:    {bp['design_beta']:.2f}°\n"
            results += f"  X1: {bp['X1']:.4f}\n"
            results += f"  X2: {bp['X2']:.4f}\n"
            results += f"  X3: {bp['X3']:.4f}\n"
            results += f"  X4: {bp['X4']:.4f}\n"
            results += f"  Width:  {bp['width']:.3f} m\n"
            results += f"  Height: {bp['height']:.3f} m\n"
            if 'volume' in bp:
                results += f"  Volume: {bp['volume']:.3f} m³\n"
            
            results += "\n" + "=" * 75 + "\n"
            
            self.results_text.setText(results)
            self.apply_btn.setEnabled(True)
            self.progress.setVisible(False)
            
            # Create comparison plot
            self.create_comparison_plot(mach_range)
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Hunter failed:\n{e}\n\n{traceback.format_exc()}")
            self.progress.setVisible(False)
    
    def create_comparison_plot(self, mach_range):
        """Create a plot comparing top designs across Mach range."""
        self.plot_fig.clear()
        
        ax = self.plot_fig.add_subplot(111)
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, min(5, len(self.best_designs))))
        
        for i, (cand, color) in enumerate(zip(self.best_designs[:5], colors)):
            mr = cand['multi_results']
            values = [mr[f'M{m}']['value'] for m in mach_range]
            stds = [mr[f'M{m}']['std'] for m in mach_range]
            
            label = f"#{i+1}: dM{cand['params']['design_Mach']:.0f}, β={cand['params']['design_beta']:.1f}°"
            ax.errorbar(mach_range, values, yerr=stds, 
                       marker='o', capsize=3, label=label, color=color, linewidth=2)
        
        ax.set_xlabel('Flight Mach Number', fontsize=12)
        ax.set_ylabel('CL/CD', fontsize=12)
        ax.set_title('Top 5 Designs: Performance Across Mach Range', fontsize=14)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(mach_range)
        
        self.plot_fig.tight_layout()
        self.plot_canvas.draw()
        
        # Switch to plot tab
        self.results_tabs.setCurrentIndex(1)
    
    def apply_best_design(self):
        """Apply best design to main GUI."""
        if not self.best_designs:
            return
        
        params = self.best_designs[0]['params']
        
        if self.parent_gui:
            try:
                if hasattr(self.parent_gui, 'mach_spin'):
                    self.parent_gui.mach_spin.setValue(params['design_Mach'])
                if hasattr(self.parent_gui, 'beta_spin'):
                    self.parent_gui.beta_spin.setValue(params['design_beta'])
                if hasattr(self.parent_gui, 'width_spin'):
                    self.parent_gui.width_spin.setValue(params['width'])
                if hasattr(self.parent_gui, 'height_spin'):
                    self.parent_gui.height_spin.setValue(params['height'])
                
                if hasattr(self.parent_gui, 'x1_slider'):
                    self.parent_gui.x1_slider.setValue(int(params['X1'] * 1000))
                if hasattr(self.parent_gui, 'x2_slider'):
                    self.parent_gui.x2_slider.setValue(int(params['X2'] * 1000))
                if hasattr(self.parent_gui, 'x3_slider'):
                    self.parent_gui.x3_slider.setValue(int(params['X3'] * 1000))
                if hasattr(self.parent_gui, 'x4_slider'):
                    self.parent_gui.x4_slider.setValue(int(params['X4'] * 1000))
                
                QMessageBox.information(
                    self, "Applied",
                    f"Multi-Mach optimal design applied!\n\n"
                    f"Design Mach: {params['design_Mach']:.0f}\n"
                    f"Design β: {params['design_beta']:.1f}°\n\n"
                    "Click 'Generate' in main window to create the waverider."
                )
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Could not apply all parameters:\n{e}")
        else:
            QMessageBox.information(
                self, "Best Design",
                f"Best multi-Mach design:\n\n"
                f"Design Mach: {params['design_Mach']:.0f}\n"
                f"Design β: {params['design_beta']:.1f}°\n"
                f"X1-X4: [{params['X1']:.3f}, {params['X2']:.3f}, {params['X3']:.3f}, {params['X4']:.3f}]"
            )


# Import for processEvents
from PyQt5.QtWidgets import QApplication

# For standalone testing
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MultiMachHunterTab()
    window.setWindowTitle("Multi-Mach Hunter")
    window.resize(1100, 800)
    window.show()
    sys.exit(app.exec_())
