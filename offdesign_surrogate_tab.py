#!/usr/bin/env python3
"""
Off-Design Neural Network Surrogate Tab for Waverider GUI
==========================================================

Provides neural network ensemble surrogate predictions for off-design
flight conditions. Trained on 11,835 samples across multiple design
and flight conditions.

Features:
- Quick aerodynamic predictions without running PySAGAS
- Uncertainty quantification via ensemble disagreement
- Flight envelope exploration
- CL/CD Hunter: Find optimal design for target flight conditions
"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QGridLayout, QFrame,
                             QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
                             QProgressBar, QTextEdit, QTabWidget, QFileDialog,
                             QMessageBox, QSplitter, QScrollArea, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor

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
# NEURAL NETWORK ENSEMBLE (same as training script)
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
# UNCERTAINTY GUIDANCE WIDGET
# =============================================================================

class UncertaintyGuide(QFrame):
    """Widget showing uncertainty interpretation guide."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        
        title = QLabel("📊 Uncertainty Interpretation Guide")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title)
        
        guide_text = """
<style>
    .low { color: #28a745; font-weight: bold; }
    .med { color: #ffc107; font-weight: bold; }
    .high { color: #dc3545; font-weight: bold; }
</style>
<p><span class="low">● Low uncertainty (σ &lt; 0.02)</span>: Trust the prediction</p>
<p><span class="med">● Medium uncertainty (0.02 ≤ σ &lt; 0.05)</span>: Prediction is reasonable</p>
<p><span class="high">● High uncertainty (σ ≥ 0.05)</span>: Run PySAGAS to verify</p>
<p style="margin-top: 10px; font-size: 9px; color: #666;">
This allows "smart" validation - only run expensive CFD when the surrogate is unsure.
</p>
"""
        guide_label = QLabel(guide_text)
        guide_label.setTextFormat(Qt.RichText)
        guide_label.setWordWrap(True)
        layout.addWidget(guide_label)


# =============================================================================
# MAIN TAB WIDGET
# =============================================================================

class OffDesignSurrogateTab(QWidget):
    """
    Tab for off-design neural network surrogate predictions.
    """
    
    # Signal to update main GUI with optimal design
    design_selected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        
        # Surrogate models
        self.ensemble_CL = None
        self.ensemble_CD = None
        self.ensemble_CL_CD = None
        self.ensemble_Volume = None
        self.model_config = None
        self.models_loaded = False
        self.has_volume_model = False
        
        # Feature order (must match training)
        self.feature_cols = ['X1', 'X2', 'X3', 'X4', 'design_Mach', 'design_beta',
                            'flight_Mach', 'flight_AoA', 'width', 'height']
        self.volume_feature_cols = ['X1', 'X2', 'X3', 'X4', 'design_Mach', 'design_beta',
                                    'width', 'height']
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        
        # Create splitter for left/right panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Results and plots
        right_panel = self.create_results_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 600])
        main_layout.addWidget(splitter)
    
    def create_control_panel(self):
        """Create the left control panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Model Loading Section
        model_group = QGroupBox("🧠 Neural Network Surrogate Model")
        model_layout = QVBoxLayout(model_group)
        
        # Status label
        self.model_status = QLabel("⚠️ No model loaded")
        self.model_status.setStyleSheet("QLabel { color: #856404; background-color: #fff3cd; padding: 5px; border-radius: 3px; }")
        model_layout.addWidget(self.model_status)
        
        # Load button
        load_btn = QPushButton("📂 Load Surrogate Model...")
        load_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_btn)
        
        # Model info
        self.model_info = QLabel("")
        self.model_info.setWordWrap(True)
        self.model_info.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        model_layout.addWidget(self.model_info)
        
        layout.addWidget(model_group)
        
        # Uncertainty Guide
        self.uncertainty_guide = UncertaintyGuide()
        layout.addWidget(self.uncertainty_guide)
        
        # ===== PREDICTION MODE SECTION =====
        pred_group = QGroupBox("🔮 Prediction Mode - Evaluate Specific Design")
        pred_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        pred_layout = QGridLayout(pred_group)
        
        row = 0
        pred_layout.addWidget(QLabel("<i>Enter all parameters to predict aerodynamics</i>"), row, 0, 1, 2)
        row += 1
        
        # Design parameters for prediction
        pred_layout.addWidget(QLabel("Design Mach:"), row, 0)
        self.design_mach_spin = QDoubleSpinBox()
        self.design_mach_spin.setRange(3.0, 8.0)
        self.design_mach_spin.setValue(5.0)
        self.design_mach_spin.setSingleStep(0.5)
        pred_layout.addWidget(self.design_mach_spin, row, 1)
        row += 1
        
        pred_layout.addWidget(QLabel("Design β (deg):"), row, 0)
        self.design_beta_spin = QDoubleSpinBox()
        self.design_beta_spin.setRange(10.0, 30.0)
        self.design_beta_spin.setValue(20.0)
        self.design_beta_spin.setSingleStep(1.0)
        pred_layout.addWidget(self.design_beta_spin, row, 1)
        row += 1
        
        pred_layout.addWidget(QLabel("Width (m):"), row, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.5, 5.0)
        self.width_spin.setValue(2.0)
        self.width_spin.setSingleStep(0.5)
        pred_layout.addWidget(self.width_spin, row, 1)
        row += 1
        
        pred_layout.addWidget(QLabel("Height (m):"), row, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.1, 5.0)
        self.height_spin.setValue(1.0)
        self.height_spin.setSingleStep(0.1)
        pred_layout.addWidget(self.height_spin, row, 1)
        row += 1
        
        # X parameters
        pred_layout.addWidget(QLabel("X1:"), row, 0)
        self.x1_spin = QDoubleSpinBox()
        self.x1_spin.setRange(0.0, 0.5)
        self.x1_spin.setValue(0.25)
        self.x1_spin.setSingleStep(0.05)
        self.x1_spin.setDecimals(4)
        pred_layout.addWidget(self.x1_spin, row, 1)
        row += 1
        
        pred_layout.addWidget(QLabel("X2:"), row, 0)
        self.x2_spin = QDoubleSpinBox()
        self.x2_spin.setRange(0.0, 1.0)
        self.x2_spin.setValue(0.5)
        self.x2_spin.setSingleStep(0.05)
        self.x2_spin.setDecimals(4)
        pred_layout.addWidget(self.x2_spin, row, 1)
        row += 1
        
        pred_layout.addWidget(QLabel("X3:"), row, 0)
        self.x3_spin = QDoubleSpinBox()
        self.x3_spin.setRange(0.05, 0.95)
        self.x3_spin.setValue(0.5)
        self.x3_spin.setSingleStep(0.05)
        self.x3_spin.setDecimals(4)
        pred_layout.addWidget(self.x3_spin, row, 1)
        row += 1
        
        pred_layout.addWidget(QLabel("X4:"), row, 0)
        self.x4_spin = QDoubleSpinBox()
        self.x4_spin.setRange(0.05, 0.95)
        self.x4_spin.setValue(0.5)
        self.x4_spin.setSingleStep(0.05)
        self.x4_spin.setDecimals(4)
        pred_layout.addWidget(self.x4_spin, row, 1)
        row += 1
        
        pred_layout.addWidget(QLabel("Flight Mach:"), row, 0)
        self.flight_mach_spin = QDoubleSpinBox()
        self.flight_mach_spin.setRange(2.0, 12.0)
        self.flight_mach_spin.setValue(5.0)
        self.flight_mach_spin.setSingleStep(0.5)
        pred_layout.addWidget(self.flight_mach_spin, row, 1)
        row += 1
        
        pred_layout.addWidget(QLabel("Flight AoA (deg):"), row, 0)
        self.flight_aoa_spin = QDoubleSpinBox()
        self.flight_aoa_spin.setRange(-5.0, 15.0)
        self.flight_aoa_spin.setValue(2.0)
        self.flight_aoa_spin.setSingleStep(0.5)
        pred_layout.addWidget(self.flight_aoa_spin, row, 1)
        row += 1
        
        # Buttons row
        btn_layout = QHBoxLayout()
        
        sync_btn = QPushButton("🔄 Sync from GUI")
        sync_btn.clicked.connect(self.sync_from_gui)
        btn_layout.addWidget(sync_btn)
        
        self.predict_btn = QPushButton("🔮 Predict")
        self.predict_btn.clicked.connect(self.predict_single)
        self.predict_btn.setEnabled(False)
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff; color: white;
                padding: 8px; font-weight: bold; border-radius: 5px;
            }
            QPushButton:hover { background-color: #0056b3; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        btn_layout.addWidget(self.predict_btn)
        
        self.envelope_btn = QPushButton("📈 Envelope")
        self.envelope_btn.clicked.connect(self.scan_envelope)
        self.envelope_btn.setEnabled(False)
        btn_layout.addWidget(self.envelope_btn)
        
        pred_layout.addLayout(btn_layout, row, 0, 1, 2)
        
        layout.addWidget(pred_group)
        
        # ===== HUNTER MODE SECTION =====
        hunter_group = QGroupBox("🎯 CL/CD Hunter - Find Optimal Design")
        hunter_group.setStyleSheet("QGroupBox { font-weight: bold; color: #28a745; }")
        hunter_layout = QGridLayout(hunter_group)
        
        row = 0
        info_label = QLabel(
            "<i>Specify target flight conditions and constraints.<br>"
            "Hunter will automatically search for optimal X1-X4, design_Mach, design_β.</i>"
        )
        info_label.setWordWrap(True)
        hunter_layout.addWidget(info_label, row, 0, 1, 4)
        row += 1
        
        # Target flight conditions
        hunter_layout.addWidget(QLabel("<b>Target Flight:</b>"), row, 0, 1, 4)
        row += 1
        
        hunter_layout.addWidget(QLabel("Flight Mach:"), row, 0)
        self.hunter_mach_spin = QDoubleSpinBox()
        self.hunter_mach_spin.setRange(2.0, 12.0)
        self.hunter_mach_spin.setValue(6.0)
        self.hunter_mach_spin.setSingleStep(0.5)
        hunter_layout.addWidget(self.hunter_mach_spin, row, 1)
        
        hunter_layout.addWidget(QLabel("Flight AoA:"), row, 2)
        self.hunter_aoa_spin = QDoubleSpinBox()
        self.hunter_aoa_spin.setRange(-5.0, 15.0)
        self.hunter_aoa_spin.setValue(2.0)
        self.hunter_aoa_spin.setSingleStep(0.5)
        hunter_layout.addWidget(self.hunter_aoa_spin, row, 3)
        row += 1
        
        # Geometry constraints
        hunter_layout.addWidget(QLabel("<b>Geometry Constraints:</b>"), row, 0, 1, 4)
        row += 1
        
        hunter_layout.addWidget(QLabel("Width (m):"), row, 0)
        self.hunter_width_min = QDoubleSpinBox()
        self.hunter_width_min.setRange(0.5, 5.0)
        self.hunter_width_min.setValue(1.0)
        self.hunter_width_min.setPrefix("min: ")
        hunter_layout.addWidget(self.hunter_width_min, row, 1)
        
        self.hunter_width_max = QDoubleSpinBox()
        self.hunter_width_max.setRange(0.5, 5.0)
        self.hunter_width_max.setValue(3.0)
        self.hunter_width_max.setPrefix("max: ")
        hunter_layout.addWidget(self.hunter_width_max, row, 2, 1, 2)
        row += 1
        
        hunter_layout.addWidget(QLabel("Height (m):"), row, 0)
        self.hunter_height_min = QDoubleSpinBox()
        self.hunter_height_min.setRange(0.1, 5.0)
        self.hunter_height_min.setValue(0.5)
        self.hunter_height_min.setPrefix("min: ")
        hunter_layout.addWidget(self.hunter_height_min, row, 1)
        
        self.hunter_height_max = QDoubleSpinBox()
        self.hunter_height_max.setRange(0.1, 5.0)
        self.hunter_height_max.setValue(2.5)
        self.hunter_height_max.setPrefix("max: ")
        hunter_layout.addWidget(self.hunter_height_max, row, 2, 1, 2)
        row += 1
        
        hunter_layout.addWidget(QLabel("Min Volume (m³):"), row, 0)
        self.hunter_min_volume = QDoubleSpinBox()
        self.hunter_min_volume.setRange(0.0, 50.0)
        self.hunter_min_volume.setValue(0.5)
        self.hunter_min_volume.setSingleStep(0.1)
        self.hunter_min_volume.setDecimals(2)
        hunter_layout.addWidget(self.hunter_min_volume, row, 1)
        
        self.hunter_volume_check = QCheckBox("Enforce")
        self.hunter_volume_check.setChecked(True)
        hunter_layout.addWidget(self.hunter_volume_check, row, 2)
        row += 1
        
        # Design constraints (to stay within training domain)
        hunter_layout.addWidget(QLabel("<b>Design Constraints:</b>"), row, 0, 1, 4)
        row += 1
        
        hunter_layout.addWidget(QLabel("Design Mach:"), row, 0)
        self.hunter_design_mach_min = QDoubleSpinBox()
        self.hunter_design_mach_min.setRange(3.0, 6.0)
        self.hunter_design_mach_min.setValue(3.0)
        self.hunter_design_mach_min.setSingleStep(1.0)
        self.hunter_design_mach_min.setPrefix("min: ")
        hunter_layout.addWidget(self.hunter_design_mach_min, row, 1)
        
        self.hunter_design_mach_max = QDoubleSpinBox()
        self.hunter_design_mach_max.setRange(3.0, 6.0)
        self.hunter_design_mach_max.setValue(6.0)
        self.hunter_design_mach_max.setSingleStep(1.0)
        self.hunter_design_mach_max.setPrefix("max: ")
        hunter_layout.addWidget(self.hunter_design_mach_max, row, 2, 1, 2)
        row += 1
        
        hunter_layout.addWidget(QLabel("Design β (deg):"), row, 0)
        self.hunter_beta_min = QDoubleSpinBox()
        self.hunter_beta_min.setRange(10.0, 30.0)
        self.hunter_beta_min.setValue(15.0)
        self.hunter_beta_min.setSingleStep(1.0)
        self.hunter_beta_min.setPrefix("min: ")
        hunter_layout.addWidget(self.hunter_beta_min, row, 1)
        
        self.hunter_beta_max = QDoubleSpinBox()
        self.hunter_beta_max.setRange(10.0, 30.0)
        self.hunter_beta_max.setValue(26.0)
        self.hunter_beta_max.setSingleStep(1.0)
        self.hunter_beta_max.setPrefix("max: ")
        hunter_layout.addWidget(self.hunter_beta_max, row, 2, 1, 2)
        row += 1
        
        # Training domain note
        domain_note = QLabel(
            "<i style='color: #666; font-size: 9px;'>"
            "💡 Tip: For flight M6, use design M4-6 to stay within training domain (lower σ)"
            "</i>"
        )
        domain_note.setWordWrap(True)
        hunter_layout.addWidget(domain_note, row, 0, 1, 4)
        row += 1
        
        # Volume model status
        self.volume_model_status = QLabel("⚠️ Volume model not loaded")
        self.volume_model_status.setStyleSheet("QLabel { color: #856404; font-size: 10px; }")
        hunter_layout.addWidget(self.volume_model_status, row, 0, 1, 4)
        row += 1
        
        # Hunter button
        self.hunter_btn = QPushButton("🎯 Run CL/CD Hunter")
        self.hunter_btn.clicked.connect(self.run_clcd_hunter)
        self.hunter_btn.setEnabled(False)
        self.hunter_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745; color: white;
                padding: 12px; font-size: 13px; font-weight: bold; border-radius: 5px;
            }
            QPushButton:hover { background-color: #1e7e34; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        hunter_layout.addWidget(self.hunter_btn, row, 0, 1, 4)
        
        layout.addWidget(hunter_group)
        
        # Spacer
        layout.addStretch()
        
        scroll.setWidget(panel)
        return scroll
    
    def create_results_panel(self):
        """Create the right results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Single Prediction Tab
        pred_tab = QWidget()
        pred_layout = QVBoxLayout(pred_tab)
        
        # Prediction results display
        self.pred_results = QTextEdit()
        self.pred_results.setReadOnly(True)
        self.pred_results.setFont(QFont("Courier", 10))
        self.pred_results.setPlaceholderText(
            "Prediction results will appear here...\n\n"
            "Load a surrogate model and click 'Predict Aerodynamics' to get started."
        )
        pred_layout.addWidget(self.pred_results)
        
        self.results_tabs.addTab(pred_tab, "🔮 Prediction")
        
        # Envelope Tab
        env_tab = QWidget()
        env_layout = QVBoxLayout(env_tab)
        
        self.envelope_fig = Figure(figsize=(8, 6))
        self.envelope_canvas = FigureCanvas(self.envelope_fig)
        env_layout.addWidget(self.envelope_canvas)
        
        self.results_tabs.addTab(env_tab, "📈 Envelope")
        
        # Hunter Tab
        hunter_tab = QWidget()
        hunter_layout = QVBoxLayout(hunter_tab)
        
        self.hunter_results = QTextEdit()
        self.hunter_results.setReadOnly(True)
        self.hunter_results.setFont(QFont("Courier", 10))
        self.hunter_results.setPlaceholderText(
            "CL/CD Hunter results will appear here...\n\n"
            "This feature searches for the optimal design parameters\n"
            "that maximize CL/CD for your target flight conditions."
        )
        hunter_layout.addWidget(self.hunter_results)
        
        # Apply best design button
        self.apply_hunter_btn = QPushButton("✅ Apply Best Design to Main GUI")
        self.apply_hunter_btn.clicked.connect(self.apply_hunter_result)
        self.apply_hunter_btn.setEnabled(False)
        hunter_layout.addWidget(self.apply_hunter_btn)
        
        self.results_tabs.addTab(hunter_tab, "🎯 Hunter")
        
        layout.addWidget(self.results_tabs)
        
        return panel
    
    def load_model(self):
        """Load the trained surrogate model."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Surrogate Model Folder", "",
            QFileDialog.ShowDirsOnly
        )
        
        if not folder:
            return
        
        try:
            # Check for required files
            required_files = ['ensemble_CL.pkl', 'ensemble_CD.pkl', 
                            'ensemble_CL_CD.pkl', 'config.json']
            
            for fname in required_files:
                if not os.path.exists(os.path.join(folder, fname)):
                    raise FileNotFoundError(f"Missing: {fname}")
            
            # Load models
            self.ensemble_CL = NNEnsemble.load(os.path.join(folder, 'ensemble_CL.pkl'))
            self.ensemble_CD = NNEnsemble.load(os.path.join(folder, 'ensemble_CD.pkl'))
            self.ensemble_CL_CD = NNEnsemble.load(os.path.join(folder, 'ensemble_CL_CD.pkl'))
            
            # Load Volume model if available
            volume_path = os.path.join(folder, 'ensemble_Volume.pkl')
            if os.path.exists(volume_path):
                self.ensemble_Volume = NNEnsemble.load(volume_path)
                self.has_volume_model = True
            else:
                self.ensemble_Volume = None
                self.has_volume_model = False
            
            # Load config
            with open(os.path.join(folder, 'config.json'), 'r') as f:
                self.model_config = json.load(f)
            
            self.models_loaded = True
            
            # Update UI
            self.model_status.setText("✅ Model loaded successfully!")
            self.model_status.setStyleSheet(
                "QLabel { color: #155724; background-color: #d4edda; "
                "padding: 5px; border-radius: 3px; }"
            )
            
            # Show model info
            metrics = self.model_config.get('metrics', {})
            info_text = f"Samples: {self.model_config.get('n_samples', 'N/A')}\n"
            info_text += f"Ensemble size: {self.model_config.get('ensemble_size', 'N/A')}\n"
            if 'CL_CD' in metrics:
                info_text += f"CL/CD R²: {metrics['CL_CD'].get('R2', 0):.4f}\n"
            if self.has_volume_model and 'Volume' in metrics:
                info_text += f"Volume R²: {metrics['Volume'].get('R2', 0):.4f}\n"
            info_text += f"Trained: {self.model_config.get('training_date', 'N/A')[:10]}"
            self.model_info.setText(info_text)
            
            # Update volume model status
            if self.has_volume_model:
                self.volume_model_status.setText("✅ Volume model loaded")
                self.volume_model_status.setStyleSheet("QLabel { color: #155724; font-size: 10px; }")
            else:
                self.volume_model_status.setText("⚠️ Volume model not available - constraint will be approximate")
                self.volume_model_status.setStyleSheet("QLabel { color: #856404; font-size: 10px; }")
            
            # Enable buttons
            self.predict_btn.setEnabled(True)
            self.envelope_btn.setEnabled(True)
            self.hunter_btn.setEnabled(True)
            
            msg = f"Surrogate model loaded!\n\n"
            msg += f"Samples: {self.model_config.get('n_samples', 'N/A')}\n"
            msg += f"CL/CD R²: {metrics.get('CL_CD', {}).get('R2', 0):.4f}\n"
            if self.has_volume_model:
                msg += f"Volume R²: {metrics.get('Volume', {}).get('R2', 0):.4f}\n"
                msg += "\n✅ Volume constraint will be enforced"
            else:
                msg += "\n⚠️ No Volume model - constraint will use approximation"
            
            QMessageBox.information(self, "Success", msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
            self.models_loaded = False
    
    def sync_from_gui(self):
        """Sync design parameters from main GUI."""
        if not self.parent_gui:
            QMessageBox.warning(self, "Warning", "No parent GUI connected")
            return
        
        try:
            # Get values from main GUI
            if hasattr(self.parent_gui, 'mach_spin'):
                self.design_mach_spin.setValue(self.parent_gui.mach_spin.value())
            if hasattr(self.parent_gui, 'beta_spin'):
                self.design_beta_spin.setValue(self.parent_gui.beta_spin.value())
            if hasattr(self.parent_gui, 'width_spin'):
                self.width_spin.setValue(self.parent_gui.width_spin.value())
            if hasattr(self.parent_gui, 'height_spin'):
                self.height_spin.setValue(self.parent_gui.height_spin.value())
            
            # X parameters (from sliders, scaled by 1000)
            if hasattr(self.parent_gui, 'x1_slider'):
                self.x1_spin.setValue(self.parent_gui.x1_slider.value() / 1000.0)
            if hasattr(self.parent_gui, 'x2_slider'):
                self.x2_spin.setValue(self.parent_gui.x2_slider.value() / 1000.0)
            if hasattr(self.parent_gui, 'x3_slider'):
                self.x3_spin.setValue(self.parent_gui.x3_slider.value() / 1000.0)
            if hasattr(self.parent_gui, 'x4_slider'):
                self.x4_spin.setValue(self.parent_gui.x4_slider.value() / 1000.0)
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not sync all parameters:\n{e}")
    
    def get_current_features(self):
        """Get current feature vector from UI inputs."""
        return np.array([[
            self.x1_spin.value(),
            self.x2_spin.value(),
            self.x3_spin.value(),
            self.x4_spin.value(),
            self.design_mach_spin.value(),
            self.design_beta_spin.value(),
            self.flight_mach_spin.value(),
            self.flight_aoa_spin.value(),
            self.width_spin.value(),
            self.height_spin.value()
        ]])
    
    def get_uncertainty_level(self, std):
        """Get uncertainty level and color."""
        if std < 0.02:
            return "LOW", "#28a745", "Trust the prediction"
        elif std < 0.05:
            return "MEDIUM", "#ffc107", "Prediction is reasonable"
        else:
            return "HIGH", "#dc3545", "Consider running PySAGAS"
    
    def predict_single(self):
        """Make a single prediction for current parameters."""
        if not self.models_loaded:
            QMessageBox.warning(self, "Warning", "Please load a surrogate model first")
            return
        
        try:
            X = self.get_current_features()
            
            # Predict
            CL, CL_std = self.ensemble_CL.predict(X, return_std=True)
            CD, CD_std = self.ensemble_CD.predict(X, return_std=True)
            CL_CD, CL_CD_std = self.ensemble_CL_CD.predict(X, return_std=True)
            
            CL, CL_std = CL[0], CL_std[0]
            CD, CD_std = CD[0], CD_std[0]
            CL_CD, CL_CD_std = CL_CD[0], CL_CD_std[0]
            
            # Get uncertainty levels
            cl_level, cl_color, cl_advice = self.get_uncertainty_level(CL_std)
            cd_level, cd_color, cd_advice = self.get_uncertainty_level(CD_std)
            clcd_level, clcd_color, clcd_advice = self.get_uncertainty_level(CL_CD_std)
            
            # Format results
            results = "=" * 60 + "\n"
            results += "      NEURAL NETWORK SURROGATE PREDICTION\n"
            results += "=" * 60 + "\n\n"
            
            results += "DESIGN PARAMETERS:\n"
            results += f"  Design Mach:  {self.design_mach_spin.value():.2f}\n"
            results += f"  Design β:     {self.design_beta_spin.value():.2f}°\n"
            results += f"  Width:        {self.width_spin.value():.3f} m\n"
            results += f"  Height:       {self.height_spin.value():.3f} m\n"
            results += f"  X1={self.x1_spin.value():.4f}  X2={self.x2_spin.value():.4f}  "
            results += f"X3={self.x3_spin.value():.4f}  X4={self.x4_spin.value():.4f}\n\n"
            
            results += "FLIGHT CONDITIONS:\n"
            results += f"  Flight Mach:  {self.flight_mach_spin.value():.2f}\n"
            results += f"  Flight AoA:   {self.flight_aoa_spin.value():.2f}°\n\n"
            
            results += "-" * 60 + "\n"
            results += "PREDICTED AERODYNAMICS:\n"
            results += "-" * 60 + "\n\n"
            
            results += f"  CL:     {CL:.4f} ± {CL_std:.4f}  [{cl_level}]\n"
            results += f"  CD:     {CD:.4f} ± {CD_std:.4f}  [{cd_level}]\n"
            results += f"  CL/CD:  {CL_CD:.3f} ± {CL_CD_std:.3f}  [{clcd_level}]\n\n"
            
            results += "-" * 60 + "\n"
            results += "CONFIDENCE ASSESSMENT:\n"
            results += "-" * 60 + "\n\n"
            
            # Overall assessment
            max_std = max(CL_std / max(abs(CL), 0.01), 
                         CD_std / max(abs(CD), 0.01),
                         CL_CD_std / max(abs(CL_CD), 0.01))
            
            if max_std < 0.05:
                results += "  ✅ HIGH CONFIDENCE: Predictions are reliable.\n"
                results += "     No PySAGAS validation needed.\n"
            elif max_std < 0.1:
                results += "  ⚠️ MEDIUM CONFIDENCE: Predictions are reasonable.\n"
                results += "     Consider PySAGAS validation for critical designs.\n"
            else:
                results += "  ❌ LOW CONFIDENCE: High uncertainty detected.\n"
                results += "     Recommend running PySAGAS for accurate results.\n"
            
            results += "\n" + "=" * 60 + "\n"
            
            self.pred_results.setText(results)
            self.results_tabs.setCurrentIndex(0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed:\n{e}")
    
    def scan_envelope(self):
        """Scan the flight envelope for current design."""
        if not self.models_loaded:
            QMessageBox.warning(self, "Warning", "Please load a surrogate model first")
            return
        
        try:
            # Get base parameters
            base_X = self.get_current_features()[0]
            
            # Define envelope
            mach_range = np.linspace(3, 8, 11)
            aoa_range = np.linspace(0, 4, 9)
            
            # Create meshgrid
            Mach_grid, AoA_grid = np.meshgrid(mach_range, aoa_range)
            CL_CD_grid = np.zeros_like(Mach_grid)
            std_grid = np.zeros_like(Mach_grid)
            
            # Predict for each point
            for i in range(len(aoa_range)):
                for j in range(len(mach_range)):
                    X = base_X.copy()
                    X[6] = mach_range[j]  # flight_Mach
                    X[7] = aoa_range[i]   # flight_AoA
                    
                    pred, std = self.ensemble_CL_CD.predict(X.reshape(1, -1), return_std=True)
                    CL_CD_grid[i, j] = pred[0]
                    std_grid[i, j] = std[0]
            
            # Plot
            self.envelope_fig.clear()
            
            ax1 = self.envelope_fig.add_subplot(121)
            cf = ax1.contourf(Mach_grid, AoA_grid, CL_CD_grid, levels=20, cmap='RdYlGn')
            ax1.set_xlabel('Flight Mach')
            ax1.set_ylabel('Flight AoA (deg)')
            ax1.set_title('CL/CD Across Flight Envelope')
            self.envelope_fig.colorbar(cf, ax=ax1, label='CL/CD')
            
            # Mark current point
            ax1.plot(self.flight_mach_spin.value(), self.flight_aoa_spin.value(), 
                    'ko', markersize=10, label='Current')
            
            # Mark best point
            best_idx = np.unravel_index(np.argmax(CL_CD_grid), CL_CD_grid.shape)
            ax1.plot(mach_range[best_idx[1]], aoa_range[best_idx[0]], 
                    'r*', markersize=15, label=f'Best: {CL_CD_grid[best_idx]:.2f}')
            ax1.legend()
            
            # Uncertainty plot
            ax2 = self.envelope_fig.add_subplot(122)
            cf2 = ax2.contourf(Mach_grid, AoA_grid, std_grid, levels=20, cmap='YlOrRd')
            ax2.set_xlabel('Flight Mach')
            ax2.set_ylabel('Flight AoA (deg)')
            ax2.set_title('Prediction Uncertainty (σ)')
            self.envelope_fig.colorbar(cf2, ax=ax2, label='Uncertainty')
            
            # Add guidance lines
            ax2.contour(Mach_grid, AoA_grid, std_grid, levels=[0.02, 0.05], 
                       colors=['green', 'red'], linestyles=['--', '--'])
            
            self.envelope_fig.tight_layout()
            self.envelope_canvas.draw()
            
            self.results_tabs.setCurrentIndex(1)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Envelope scan failed:\n{e}")
    
    def get_min_beta(self, design_mach):
        """Get minimum valid beta (shock angle) for a given design Mach.
        
        The shock angle must be greater than the Mach angle for an attached shock.
        Mach angle μ = arcsin(1/M)
        We add a small margin for numerical stability.
        """
        mach_angle = np.degrees(np.arcsin(1.0 / design_mach))
        # Add 1 degree margin for safety
        return mach_angle + 1.0
    
    def run_clcd_hunter(self):
        """Find optimal design for target flight conditions with constraints."""
        if not self.models_loaded:
            QMessageBox.warning(self, "Warning", "Please load a surrogate model first")
            return
        
        try:
            # Get target flight conditions from Hunter UI
            target_mach = self.hunter_mach_spin.value()
            target_aoa = self.hunter_aoa_spin.value()
            
            # Get geometry constraints from Hunter UI
            width_min = self.hunter_width_min.value()
            width_max = self.hunter_width_max.value()
            height_min = self.hunter_height_min.value()
            height_max = self.hunter_height_max.value()
            min_volume = self.hunter_min_volume.value()
            enforce_volume = self.hunter_volume_check.isChecked()
            
            # Get design constraints from Hunter UI
            design_mach_min = self.hunter_design_mach_min.value()
            design_mach_max = self.hunter_design_mach_max.value()
            beta_min = self.hunter_beta_min.value()
            beta_max = self.hunter_beta_max.value()
            
            # Validate constraints
            if width_min > width_max:
                QMessageBox.warning(self, "Warning", "Width min > max!")
                return
            if height_min > height_max:
                QMessageBox.warning(self, "Warning", "Height min > max!")
                return
            if design_mach_min > design_mach_max:
                QMessageBox.warning(self, "Warning", "Design Mach min > max!")
                return
            if beta_min > beta_max:
                QMessageBox.warning(self, "Warning", "Beta min > max!")
                return
            
            results = "=" * 70 + "\n"
            results += "            CL/CD HUNTER - OPTIMAL DESIGN SEARCH\n"
            results += "=" * 70 + "\n\n"
            results += "TARGET FLIGHT CONDITIONS:\n"
            results += f"  Flight Mach: {target_mach:.2f}\n"
            results += f"  Flight AoA:  {target_aoa:.2f}°\n\n"
            results += "GEOMETRY CONSTRAINTS:\n"
            results += f"  Width:  {width_min:.2f} - {width_max:.2f} m\n"
            results += f"  Height: {height_min:.2f} - {height_max:.2f} m\n"
            if enforce_volume:
                results += f"  Min Volume: {min_volume:.2f} m³"
                if self.has_volume_model:
                    results += " (enforced via surrogate)\n"
                    results += "  ⚠️ NOTE: Volume predictions may be ~50% higher than actual.\n"
                    results += "     Verify volumes in 3D View after applying design.\n"
                else:
                    results += " (approximate: W×H×L scaling)\n"
            else:
                results += "  Volume constraint: disabled\n"
            results += "\n"
            results += "DESIGN CONSTRAINTS:\n"
            results += f"  Design Mach: {design_mach_min:.0f} - {design_mach_max:.0f}\n"
            results += f"  Design β:    {beta_min:.1f}° - {beta_max:.1f}°\n"
            results += "\n"
            results += "PHYSICS CONSTRAINTS (β > Mach angle):\n"
            for dm in [3, 4, 5, 6]:
                if design_mach_min <= dm <= design_mach_max:
                    min_b = self.get_min_beta(dm)
                    status = "✓" if beta_max > min_b else "✗"
                    results += f"  {status} Design M{dm}: β > {min_b:.1f}°\n"
            results += "\n"
            results += "SEARCHING DESIGN SPACE...\n"
            results += "  (X1, X2, X3, X4 optimized within constraints)\n\n"
            
            # Build list of allowed design Mach values
            all_design_machs = [3.0, 4.0, 5.0, 6.0]
            design_machs = [m for m in all_design_machs if design_mach_min <= m <= design_mach_max]
            
            if not design_machs:
                QMessageBox.warning(self, "Warning", "No valid design Mach values in range!")
                return
            
            n_samples = 5000  # More samples for better coverage
            
            # Random sampling of design space
            np.random.seed(42)
            all_candidates = []
            filtered_candidates = []
            n_rejected_physics = 0
            n_rejected_volume = 0
            n_rejected_design_space = 0
            
            for _ in range(n_samples):
                # Sample design Mach first
                design_mach = np.random.choice(design_machs)
                
                # Get minimum valid beta for this design Mach (physics constraint)
                min_valid_beta = self.get_min_beta(design_mach)
                
                # Adjust beta range to ensure valid designs
                effective_beta_min = max(beta_min, min_valid_beta)
                if effective_beta_min >= beta_max:
                    # This design Mach can't produce valid designs in our beta range
                    n_rejected_physics += 1
                    continue
                
                # Sample design parameters (these are what Hunter finds)
                params = {
                    'X1': np.random.uniform(0.05, 0.45),
                    'X2': np.random.uniform(0.1, 0.9),
                    'X3': np.random.uniform(0.1, 0.9),
                    'X4': np.random.uniform(0.1, 0.9),
                    'design_Mach': design_mach,
                    'design_beta': np.random.uniform(effective_beta_min, beta_max),
                    # Sample within user-specified constraints
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
                
                # Predict CL/CD
                X_aero = np.array([[
                    params['X1'], params['X2'], params['X3'], params['X4'],
                    params['design_Mach'], params['design_beta'],
                    target_mach, target_aoa,
                    params['width'], params['height']
                ]])
                
                clcd_pred, clcd_std = self.ensemble_CL_CD.predict(X_aero, return_std=True)
                
                # Predict Volume (if model available)
                if self.has_volume_model and enforce_volume:
                    X_vol = np.array([[
                        params['X1'], params['X2'], params['X3'], params['X4'],
                        params['design_Mach'], params['design_beta'],
                        params['width'], params['height']
                    ]])
                    vol_pred, vol_std = self.ensemble_Volume.predict(X_vol, return_std=True)
                    params['volume'] = vol_pred[0]
                    params['volume_std'] = vol_std[0]
                elif enforce_volume:
                    # Approximate volume as W * H * L (assume L ≈ 3 * W for waverider)
                    approx_vol = params['width'] * params['height'] * (3.0 * params['width']) * 0.15
                    params['volume'] = approx_vol
                    params['volume_std'] = approx_vol * 0.3  # 30% uncertainty for approximation
                else:
                    params['volume'] = None
                    params['volume_std'] = None
                
                candidate = {
                    'clcd': clcd_pred[0],
                    'clcd_std': clcd_std[0],
                    'params': params
                }
                all_candidates.append(candidate)
                
                # Apply volume constraint
                if enforce_volume:
                    if params['volume'] >= min_volume:
                        filtered_candidates.append(candidate)
                    else:
                        n_rejected_volume += 1
                else:
                    filtered_candidates.append(candidate)
            
            # Sort by CL/CD
            filtered_candidates.sort(key=lambda x: x['clcd'], reverse=True)
            
            # Statistics
            n_passed = len(filtered_candidates)
            
            results += "-" * 70 + "\n"
            results += f"SEARCH RESULTS:\n"
            results += f"  Valid designs: {n_passed}\n"
            if n_rejected_physics > 0:
                results += f"  Rejected (β < Mach angle): {n_rejected_physics}\n"
            if n_rejected_design_space > 0:
                results += f"  Rejected (design space X2/(1-X1)⁴ constraint): {n_rejected_design_space}\n"
            if n_rejected_volume > 0:
                results += f"  Rejected (volume): {n_rejected_volume}\n"
            results += "-" * 70 + "\n\n"
            
            if n_passed == 0:
                results += "❌ NO DESIGNS FOUND!\n\n"
                results += "All designs were rejected. Try:\n"
                results += "  - Increasing β max (see physics constraints below)\n"
                results += "  - Using higher Design Mach (M4+ has lower β requirements)\n"
                results += "  - Increasing width/height ranges\n"
                results += "  - Decreasing minimum volume\n"
                results += "  - Reducing X2 max or increasing X1 min (design space constraint)\n\n"
                results += "Physics constraints (β must be greater than Mach angle):\n"
                for dm in [3, 4, 5, 6]:
                    min_b = self.get_min_beta(dm)
                    results += f"  Design M{dm}: β > {min_b:.1f}°\n"
                self.hunter_results.setText(results)
                self.results_tabs.setCurrentIndex(2)
                return
            
            # Store best for apply button
            self.best_hunter_result = filtered_candidates[0]['params']
            
            results += "TOP 10 DESIGNS:\n"
            results += "-" * 70 + "\n"
            
            header = f"{'Rank':<5} {'CL/CD':>8} {'±σ':>6} | {'dM':>3} {'β':>5} | "
            header += f"{'W':>5} {'H':>5}"
            if enforce_volume:
                header += f" {'Vol':>6}"
            header += f" | {'X1':>5} {'X2':>5} {'X3':>5} {'X4':>5}\n"
            results += header
            results += "-" * 70 + "\n"
            
            for i, cand in enumerate(filtered_candidates[:10]):
                p = cand['params']
                level, _, _ = self.get_uncertainty_level(cand['clcd_std'])
                
                line = f"{i+1:<5} {cand['clcd']:>8.3f} {cand['clcd_std']:>6.3f} | "
                line += f"{p['design_Mach']:>3.0f} {p['design_beta']:>5.1f} | "
                line += f"{p['width']:>5.2f} {p['height']:>5.2f}"
                if enforce_volume:
                    line += f" {p['volume']:>6.2f}"
                line += f" | {p['X1']:>5.3f} {p['X2']:>5.3f} {p['X3']:>5.3f} {p['X4']:>5.3f}\n"
                results += line
            
            results += "\n" + "=" * 70 + "\n"
            results += "🏆 BEST DESIGN\n"
            results += "=" * 70 + "\n\n"
            
            best = filtered_candidates[0]
            bp = best['params']
            
            results += f"  CL/CD:       {best['clcd']:.4f} ± {best['clcd_std']:.4f}\n"
            level, _, advice = self.get_uncertainty_level(best['clcd_std'])
            results += f"  Confidence:  {level} - {advice}\n\n"
            
            results += "  DESIGN PARAMETERS (found by Hunter):\n"
            results += f"    Design Mach: {bp['design_Mach']:.1f}\n"
            results += f"    Design β:    {bp['design_beta']:.2f}°\n"
            results += f"    X1: {bp['X1']:.4f}\n"
            results += f"    X2: {bp['X2']:.4f}\n"
            results += f"    X3: {bp['X3']:.4f}\n"
            results += f"    X4: {bp['X4']:.4f}\n\n"
            
            results += "  GEOMETRY (within your constraints):\n"
            results += f"    Width:  {bp['width']:.3f} m\n"
            results += f"    Height: {bp['height']:.3f} m\n"
            if bp['volume'] is not None:
                results += f"    Volume: {bp['volume']:.3f} m³"
                if self.has_volume_model:
                    results += f" ± {bp['volume_std']:.3f}\n"
                else:
                    results += " (approx)\n"
            
            results += "\n" + "=" * 70 + "\n"
            results += "Click 'Apply Best Design to Main GUI' to use this design.\n"
            results += "=" * 70 + "\n"
            
            self.hunter_results.setText(results)
            self.apply_hunter_btn.setEnabled(True)
            self.results_tabs.setCurrentIndex(2)
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Hunter search failed:\n{e}\n\n{traceback.format_exc()}")
    
    def apply_hunter_result(self):
        """Apply the best hunter result to main GUI."""
        if not hasattr(self, 'best_hunter_result') or not self.best_hunter_result:
            return
        
        params = self.best_hunter_result
        
        if self.parent_gui:
            try:
                # Update main GUI parameters
                if hasattr(self.parent_gui, 'mach_spin'):
                    self.parent_gui.mach_spin.setValue(params['design_Mach'])
                if hasattr(self.parent_gui, 'beta_spin'):
                    self.parent_gui.beta_spin.setValue(params['design_beta'])
                if hasattr(self.parent_gui, 'width_spin'):
                    self.parent_gui.width_spin.setValue(params['width'])
                if hasattr(self.parent_gui, 'height_spin'):
                    self.parent_gui.height_spin.setValue(params['height'])
                
                # X sliders (scaled by 1000)
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
                    f"Design applied to GUI!\n\n"
                    f"Design Mach: {params['design_Mach']:.1f}\n"
                    f"Design β: {params['design_beta']:.2f}°\n"
                    f"X1={params['X1']:.4f}, X2={params['X2']:.4f}\n"
                    f"X3={params['X3']:.4f}, X4={params['X4']:.4f}\n\n"
                    "Click 'Generate' in the main window to create the waverider."
                )
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Could not apply all parameters:\n{e}")
        else:
            # Just show the parameters
            QMessageBox.information(
                self, "Best Design",
                f"Best design parameters:\n\n"
                f"Design Mach: {params['design_Mach']:.1f}\n"
                f"Design β: {params['design_beta']:.2f}°\n"
                f"Width: {params['width']:.2f} m\n"
                f"Height: {params['height']:.3f} m\n"
                f"X1={params['X1']:.4f}, X2={params['X2']:.4f}\n"
                f"X3={params['X3']:.4f}, X4={params['X4']:.4f}"
            )


# For standalone testing
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = OffDesignSurrogateTab()
    window.setWindowTitle("Off-Design Surrogate Prediction")
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())
