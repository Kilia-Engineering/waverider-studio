#!/usr/bin/env python3
"""
Cone-Derived Waverider Tab for Waverider GUI
=============================================
Based on Adam Weaver's SHADOW methodology from Utah State University.

Features:
- Polynomial leading edge parameterization (2nd and 3rd order)
- Taylor-Maccoll flow field solver
- Streamline tracing for compression surface
- Export to STL, STEP formats
- Gmsh meshing integration
- PySAGAS aerodynamic analysis
- Design space exploration (Adam's optimization method)
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QGridLayout, QSlider,
                             QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
                             QProgressBar, QTextEdit, QTabWidget, QFileDialog,
                             QMessageBox, QSplitter, QFrame, QScrollArea,
                             QRadioButton, QButtonGroup, QDialog, QLineEdit,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Import the cone waverider generator
from cone_waverider import (
    ConeWaverider,
    create_second_order_waverider,
    create_third_order_waverider,
    optimal_shock_angle
)

# Check for optional dependencies
try:
    import cadquery as cq
    from cadquery import exporters
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False

try:
    from pysagas.cfd import OPM
    from pysagas.flow import FlowState
    PYSAGAS_AVAILABLE = True
except ImportError:
    PYSAGAS_AVAILABLE = False

try:
    from stl import mesh as stl_mesh
    NUMPY_STL_AVAILABLE = True
except ImportError:
    NUMPY_STL_AVAILABLE = False


class ConeWaveriderCanvas(FigureCanvas):
    """Canvas for 3D cone waverider visualization"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax.set_xlabel('X (Span)')
        self.ax.set_ylabel('Z (Streamwise)')
        self.ax.set_zlabel('Y (Vertical)')
        self.ax.set_title('Cone-Derived Waverider')
        
    def plot_waverider(self, waverider, show_upper=True, show_lower=True, 
                       show_le=True, show_wireframe=False, show_cg=True):
        """Plot the cone-derived waverider geometry"""
        self.ax.clear()
        
        if waverider is None:
            self.ax.set_title('No waverider generated')
            self.draw()
            return
        
        # Plot upper surface streamlines
        if show_upper:
            for i in range(len(waverider.upper_surface)):
                streamline = waverider.upper_surface[i]
                if show_wireframe:
                    self.ax.plot(streamline[:, 0], streamline[:, 2], streamline[:, 1], 
                                'b-', alpha=0.3, linewidth=0.5)
                else:
                    self.ax.plot(streamline[:, 0], streamline[:, 2], streamline[:, 1], 
                                'b-', alpha=0.5, linewidth=0.5)
        
        # Plot lower surface streamlines
        if show_lower:
            for i in range(len(waverider.lower_surface)):
                streamline = waverider.lower_surface[i]
                if show_wireframe:
                    self.ax.plot(streamline[:, 0], streamline[:, 2], streamline[:, 1], 
                                'r-', alpha=0.3, linewidth=0.5)
                else:
                    self.ax.plot(streamline[:, 0], streamline[:, 2], streamline[:, 1], 
                                'r-', alpha=0.5, linewidth=0.5)
        
        # Plot leading edge
        if show_le:
            le = waverider.leading_edge
            self.ax.plot(le[:, 0], le[:, 2], le[:, 1], 'k-', linewidth=2, label='Leading Edge')
        
        # Plot spanwise lines at a few stations
        n_stream = waverider.n_streamwise
        for j in [0, n_stream//4, n_stream//2, 3*n_stream//4, n_stream-1]:
            if show_upper:
                upper_line = waverider.upper_surface[:, j, :]
                self.ax.plot(upper_line[:, 0], upper_line[:, 2], upper_line[:, 1], 'b-', alpha=0.3)
            if show_lower:
                lower_j = min(j, waverider.lower_surface.shape[1]-1)
                lower_line = waverider.lower_surface[:, lower_j, :]
                self.ax.plot(lower_line[:, 0], lower_line[:, 2], lower_line[:, 1], 'r-', alpha=0.3)
        
        # Plot CG location
        if show_cg and waverider.cg is not None:
            cg = waverider.cg
            self.ax.scatter([cg[0]], [cg[2]], [cg[1]], c='g', s=100, marker='*', label='CG')
        
        self.ax.set_xlabel('X (Span)')
        self.ax.set_ylabel('Z (Streamwise)')
        self.ax.set_zlabel('Y (Vertical)')
        self.ax.set_title(f'Cone-Derived Waverider (Mach {waverider.mach:.1f}, β={waverider.shock_angle:.1f}°)')
        self.ax.legend()
        
        # Set equal aspect ratio
        try:
            max_range = max(
                waverider.upper_surface[:, :, 0].max() - waverider.upper_surface[:, :, 0].min(),
                waverider.upper_surface[:, :, 2].max() - waverider.upper_surface[:, :, 2].min(),
                abs(waverider.upper_surface[:, :, 1].max() - waverider.lower_surface[:, :, 1].min())
            )
            
            mid_x = (waverider.upper_surface[:, :, 0].max() + waverider.upper_surface[:, :, 0].min()) / 2
            mid_z = (waverider.upper_surface[:, :, 2].max() + waverider.upper_surface[:, :, 2].min()) / 2
            mid_y = (waverider.upper_surface[:, :, 1].max() + waverider.lower_surface[:, :, 1].min()) / 2
            
            self.ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            self.ax.set_ylim(mid_z - max_range/2, mid_z + max_range/2)
            self.ax.set_zlim(mid_y - max_range/2, mid_y + max_range/2)
        except:
            pass
        
        self.draw()


class DesignSpaceCanvas(FigureCanvas):
    """Canvas for design space exploration visualization"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def plot_design_space(self, results_df, x_param='A2', y_param='A0', 
                          color_param='L/D', show_stability=True):
        """Plot design space exploration results"""
        self.ax.clear()
        
        if results_df is None or len(results_df) == 0:
            self.ax.set_title('No results to display')
            self.draw()
            return
        
        # Scatter plot colored by objective
        scatter = self.ax.scatter(
            results_df[x_param], 
            results_df[y_param],
            c=results_df[color_param],
            cmap='viridis',
            s=50,
            alpha=0.7
        )
        
        self.fig.colorbar(scatter, ax=self.ax, label=color_param)
        
        # Overlay stability regions if available
        if show_stability and 'Stable' in results_df.columns:
            stable = results_df[results_df['Stable'] == True]
            unstable = results_df[results_df['Stable'] == False]
            
            if len(stable) > 0:
                self.ax.scatter(stable[x_param], stable[y_param], 
                               facecolors='none', edgecolors='green', 
                               s=100, linewidths=2, label='Stable')
            if len(unstable) > 0:
                self.ax.scatter(unstable[x_param], unstable[y_param], 
                               facecolors='none', edgecolors='red', 
                               s=100, linewidths=1, alpha=0.5, label='Unstable')
        
        # Mark best point
        if color_param in results_df.columns:
            best_idx = results_df[color_param].idxmax()
            best = results_df.loc[best_idx]
            self.ax.scatter([best[x_param]], [best[y_param]], 
                           c='gold', s=200, marker='*', edgecolors='black',
                           linewidths=2, label=f'Best {color_param}={best[color_param]:.3f}', zorder=10)
        
        self.ax.set_xlabel(x_param)
        self.ax.set_ylabel(y_param)
        self.ax.set_title(f'Design Space: {color_param} vs ({x_param}, {y_param})')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.draw()


class AnalysisWorker(QThread):
    """Worker thread for PySAGAS analysis"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, stl_file, mach, aoa, pressure, temperature, A_ref):
        super().__init__()
        self.stl_file = stl_file
        self.mach = mach
        self.aoa = aoa
        self.pressure = pressure
        self.temperature = temperature
        self.A_ref = A_ref
        
    def run(self):
        try:
            from pysagas.cfd import OPM
            from pysagas.flow import FlowState
            from pysagas.geometry.parsers import MeshIO
            
            self.progress.emit("Loading mesh...")
            
            # Load mesh
            cells = MeshIO.read_mesh(self.stl_file)
            
            self.progress.emit("Setting up flow conditions...")
            
            # Create flow state
            flow = FlowState(
                mach=self.mach,
                pressure=self.pressure,
                temperature=self.temperature,
                aoa=np.radians(self.aoa)
            )
            
            self.progress.emit("Running OPM analysis...")
            
            # Run analysis
            solver = OPM(cells=cells, freestream=flow, verbosity=0)
            solver.solve()
            
            self.progress.emit("Extracting coefficients...")
            
            # Get coefficients
            CL, CD, Cm = solver.flow_result.coefficients()
            LD = CL / CD if CD != 0 else float('inf')
            
            results = {
                'CL': float(CL),
                'CD': float(CD),
                'Cm': float(Cm),
                'L/D': float(LD)
            }
            
            self.progress.emit("Analysis complete!")
            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class DesignSpaceWorker(QThread):
    """Worker thread for design space exploration"""
    progress = pyqtSignal(int, int, str)  # current, total, message
    point_complete = pyqtSignal(dict)  # Single point result
    finished = pyqtSignal(list)  # All results
    error = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self._is_cancelled = False
        
    def cancel(self):
        self._is_cancelled = True
        
    def run(self):
        try:
            results = []
            
            mach = self.params['mach']
            shock_angle = self.params['shock_angle']
            poly_order = self.params['poly_order']
            
            # Generate parameter grid
            if poly_order == 2:
                A2_range = np.linspace(self.params['A2_min'], self.params['A2_max'], 
                                       self.params['n_A2'])
                A0_range = np.linspace(self.params['A0_min'], self.params['A0_max'], 
                                       self.params['n_A0'])
                
                total = len(A2_range) * len(A0_range)
                current = 0
                
                for A2 in A2_range:
                    for A0 in A0_range:
                        if self._is_cancelled:
                            self.finished.emit(results)
                            return
                        
                        current += 1
                        self.progress.emit(current, total, f"A2={A2:.2f}, A0={A0:.3f}")
                        
                        try:
                            wr = create_second_order_waverider(
                                mach=mach,
                                shock_angle=shock_angle,
                                A2=A2,
                                A0=A0,
                                n_leading_edge=self.params.get('n_le', 15),
                                n_streamwise=self.params.get('n_stream', 15)
                            )
                            
                            result = {
                                'A2': A2,
                                'A0': A0,
                                'cone_angle': wr.cone_angle_deg,
                                'planform_area': wr.planform_area,
                                'volume': wr.volume,
                                'mac': wr.mac,
                                'valid': True
                            }
                            
                            # Run aero analysis if requested
                            if self.params.get('run_aero', False):
                                aero = self._run_quick_analysis(wr)
                                result.update(aero)
                            
                        except Exception as e:
                            result = {
                                'A2': A2,
                                'A0': A0,
                                'valid': False,
                                'error': str(e)
                            }
                        
                        results.append(result)
                        self.point_complete.emit(result)
            
            else:  # 3rd order
                A3_range = np.linspace(self.params['A3_min'], self.params['A3_max'], 
                                       self.params['n_A3'])
                A2_range = np.linspace(self.params['A2_min'], self.params['A2_max'], 
                                       self.params['n_A2'])
                A0 = self.params['A0_fixed']
                
                total = len(A3_range) * len(A2_range)
                current = 0
                
                for A3 in A3_range:
                    for A2 in A2_range:
                        if self._is_cancelled:
                            self.finished.emit(results)
                            return
                        
                        current += 1
                        self.progress.emit(current, total, f"A3={A3:.1f}, A2={A2:.2f}")
                        
                        try:
                            wr = create_third_order_waverider(
                                mach=mach,
                                shock_angle=shock_angle,
                                A3=A3,
                                A2=A2,
                                A0=A0,
                                n_leading_edge=self.params.get('n_le', 15),
                                n_streamwise=self.params.get('n_stream', 15)
                            )
                            
                            result = {
                                'A3': A3,
                                'A2': A2,
                                'A0': A0,
                                'cone_angle': wr.cone_angle_deg,
                                'planform_area': wr.planform_area,
                                'volume': wr.volume,
                                'mac': wr.mac,
                                'valid': True
                            }
                            
                            if self.params.get('run_aero', False):
                                aero = self._run_quick_analysis(wr)
                                result.update(aero)
                            
                        except Exception as e:
                            result = {
                                'A3': A3,
                                'A2': A2,
                                'A0': A0,
                                'valid': False,
                                'error': str(e)
                            }
                        
                        results.append(result)
                        self.point_complete.emit(result)
            
            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")
    
    def _run_quick_analysis(self, waverider):
        """Run quick aero analysis on a waverider"""
        if not PYSAGAS_AVAILABLE:
            return {'L/D': 0, 'CL': 0, 'CD': 0}
        
        try:
            # Export to temp STL
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
                temp_stl = f.name
            
            waverider.export_stl(temp_stl)
            
            # Quick analysis
            from pysagas.cfd import OPM
            from pysagas.flow import FlowState
            from pysagas.geometry.parsers import MeshIO
            
            cells = MeshIO.read_mesh(temp_stl)
            flow = FlowState(
                mach=waverider.mach,
                pressure=self.params.get('pressure', 101325),
                temperature=self.params.get('temperature', 288.15),
                aoa=0
            )
            
            solver = OPM(cells=cells, freestream=flow, verbosity=0)
            solver.solve()
            
            CL, CD, Cm = solver.flow_result.coefficients()
            LD = CL / CD if CD != 0 else 0
            
            os.unlink(temp_stl)
            
            return {'L/D': float(LD), 'CL': float(CL), 'CD': float(CD), 'Cm': float(Cm)}
            
        except Exception as e:
            return {'L/D': 0, 'CL': 0, 'CD': 0, 'aero_error': str(e)}


class ConeWaveriderTab(QWidget):
    """Main tab for cone-derived waverider design"""
    
    waverider_generated = pyqtSignal(object)  # Emit waverider object
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.waverider = None
        self.last_stl_file = None
        self.last_step_file = None
        self.design_space_results = None
        self.analysis_worker = None
        self.design_worker = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QHBoxLayout(self)
        
        # Left panel - Parameters and controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Visualization and results
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_left_panel(self):
        """Create the left parameter panel"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # === Flow Conditions ===
        flow_group = QGroupBox("Flow Conditions")
        flow_layout = QGridLayout()
        
        flow_layout.addWidget(QLabel("Mach Number:"), 0, 0)
        self.mach_spin = QDoubleSpinBox()
        self.mach_spin.setRange(2.0, 30.0)
        self.mach_spin.setValue(6.0)
        self.mach_spin.setSingleStep(0.5)
        self.mach_spin.setDecimals(1)
        flow_layout.addWidget(self.mach_spin, 0, 1)
        
        flow_layout.addWidget(QLabel("Shock Angle β (°):"), 1, 0)
        self.shock_spin = QDoubleSpinBox()
        self.shock_spin.setRange(5.0, 60.0)
        self.shock_spin.setValue(12.0)
        self.shock_spin.setSingleStep(0.5)
        self.shock_spin.setDecimals(1)
        flow_layout.addWidget(self.shock_spin, 1, 1)
        
        # Auto-compute optimal shock angle button
        auto_shock_btn = QPushButton("Auto (Optimal)")
        auto_shock_btn.clicked.connect(self.auto_shock_angle)
        auto_shock_btn.setToolTip("Set shock angle to estimated optimal for L/D")
        flow_layout.addWidget(auto_shock_btn, 1, 2)
        
        flow_group.setLayout(flow_layout)
        layout.addWidget(flow_group)
        
        # === Polynomial Parameters ===
        poly_group = QGroupBox("Leading Edge Polynomial")
        poly_layout = QGridLayout()
        
        # Polynomial order selection
        poly_layout.addWidget(QLabel("Order:"), 0, 0)
        self.order_combo = QComboBox()
        self.order_combo.addItems(["2nd Order (y = A₂x² + A₀)", 
                                    "3rd Order (y = A₃x³ + A₂x² + A₀)"])
        self.order_combo.currentIndexChanged.connect(self.on_order_changed)
        poly_layout.addWidget(self.order_combo, 0, 1, 1, 2)
        
        # A3 coefficient (for 3rd order)
        poly_layout.addWidget(QLabel("A₃ (cubic):"), 1, 0)
        self.a3_spin = QDoubleSpinBox()
        self.a3_spin.setRange(-100.0, 100.0)
        self.a3_spin.setValue(0.0)
        self.a3_spin.setSingleStep(5.0)
        self.a3_spin.setDecimals(1)
        self.a3_spin.setEnabled(False)
        poly_layout.addWidget(self.a3_spin, 1, 1, 1, 2)
        
        # A2 coefficient
        poly_layout.addWidget(QLabel("A₂ (quadratic):"), 2, 0)
        self.a2_spin = QDoubleSpinBox()
        self.a2_spin.setRange(-50.0, 50.0)
        self.a2_spin.setValue(-2.0)
        self.a2_spin.setSingleStep(0.5)
        self.a2_spin.setDecimals(2)
        poly_layout.addWidget(self.a2_spin, 2, 1, 1, 2)
        
        # A0 coefficient (y-intercept)
        poly_layout.addWidget(QLabel("A₀ (y-intercept):"), 3, 0)
        self.a0_spin = QDoubleSpinBox()
        self.a0_spin.setRange(-1.0, 0.0)
        self.a0_spin.setValue(-0.15)
        self.a0_spin.setSingleStep(0.01)
        self.a0_spin.setDecimals(3)
        poly_layout.addWidget(self.a0_spin, 3, 1, 1, 2)
        
        poly_group.setLayout(poly_layout)
        layout.addWidget(poly_group)
        
        # === Mesh Resolution ===
        mesh_group = QGroupBox("Mesh Resolution")
        mesh_layout = QGridLayout()
        
        mesh_layout.addWidget(QLabel("Leading Edge Points:"), 0, 0)
        self.n_le_spin = QSpinBox()
        self.n_le_spin.setRange(11, 101)
        self.n_le_spin.setValue(21)
        self.n_le_spin.setSingleStep(2)
        mesh_layout.addWidget(self.n_le_spin, 0, 1)
        
        mesh_layout.addWidget(QLabel("Streamwise Points:"), 1, 0)
        self.n_stream_spin = QSpinBox()
        self.n_stream_spin.setRange(10, 100)
        self.n_stream_spin.setValue(20)
        self.n_stream_spin.setSingleStep(5)
        mesh_layout.addWidget(self.n_stream_spin, 1, 1)
        
        mesh_layout.addWidget(QLabel("Scale Factor:"), 2, 0)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.001, 1000.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(1.0)
        self.scale_spin.setDecimals(3)
        self.scale_spin.setToolTip("Scale factor for export (1.0 = unit length)")
        mesh_layout.addWidget(self.scale_spin, 2, 1)
        
        mesh_group.setLayout(mesh_layout)
        layout.addWidget(mesh_group)
        
        # === Generate Button ===
        generate_btn = QPushButton("🚀 Generate Waverider")
        generate_btn.setStyleSheet("QPushButton { font-size: 14px; font-weight: bold; padding: 10px; }")
        generate_btn.clicked.connect(self.generate_waverider)
        layout.addWidget(generate_btn)
        
        # === Export Group ===
        export_group = QGroupBox("Export")
        export_layout = QGridLayout()

        stl_btn = QPushButton("Export STL")
        stl_btn.clicked.connect(self.export_stl)
        export_layout.addWidget(stl_btn, 0, 0)

        step_btn = QPushButton("Export STEP")
        step_btn.clicked.connect(self.export_step)
        step_btn.setEnabled(CADQUERY_AVAILABLE)
        if not CADQUERY_AVAILABLE:
            step_btn.setToolTip("CadQuery not installed")
        export_layout.addWidget(step_btn, 0, 1)

        tri_btn = QPushButton("Export TRI")
        tri_btn.clicked.connect(self.export_tri)
        export_layout.addWidget(tri_btn, 1, 0)

        self.half_vehicle_check = QCheckBox("Half vehicle")
        self.half_vehicle_check.setToolTip(
            "Export only the right half (positive Z) without mirroring.\n"
            "Useful for CFD meshing with symmetry boundary conditions.")
        export_layout.addWidget(self.half_vehicle_check, 1, 1)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # === LE Blunting Group ===
        blunt_group = QGroupBox("LE Fillet (STEP only)")
        blunt_layout = QGridLayout()

        self.blunting_check = QCheckBox("Enable LE fillet")
        self.blunting_check.setToolTip(
            "Apply a fillet to the leading edge during STEP export.\n"
            "Uses OpenCASCADE BRepFilletAPI with optional variable radius.")
        self.blunting_check.stateChanged.connect(self._on_blunting_toggled)
        blunt_layout.addWidget(self.blunting_check, 0, 0, 1, 2)

        blunt_layout.addWidget(QLabel("Radius (m):"), 1, 0)
        self.blunting_radius_spin = QDoubleSpinBox()
        self.blunting_radius_spin.setRange(0.0001, 1.0)
        self.blunting_radius_spin.setValue(0.005)
        self.blunting_radius_spin.setSingleStep(0.001)
        self.blunting_radius_spin.setDecimals(4)
        self.blunting_radius_spin.setEnabled(False)
        blunt_layout.addWidget(self.blunting_radius_spin, 1, 1)

        blunt_layout.addWidget(QLabel("Spanwise:"), 2, 0)
        self.blunting_sweep_combo = QComboBox()
        self.blunting_sweep_combo.addItems([
            "Uniform radius",
            "Sweep-scaled"])
        self.blunting_sweep_combo.setToolTip(
            "Uniform: Same fillet radius across the entire span\n"
            "Sweep-scaled: Radius tapers toward wingtip based on\n"
            "  local sweep angle (R_tip = R * cos(sweep))")
        self.blunting_sweep_combo.setEnabled(False)
        blunt_layout.addWidget(self.blunting_sweep_combo, 2, 1)

        blunt_group.setLayout(blunt_layout)
        layout.addWidget(blunt_group)
        
        # === Gmsh Meshing ===
        gmsh_group = QGroupBox("Gmsh Meshing")
        gmsh_layout = QGridLayout()
        
        gmsh_layout.addWidget(QLabel("Min Element Size:"), 0, 0)
        self.gmsh_min_spin = QDoubleSpinBox()
        self.gmsh_min_spin.setRange(0.0001, 10.0)
        self.gmsh_min_spin.setValue(0.01)
        self.gmsh_min_spin.setDecimals(4)
        gmsh_layout.addWidget(self.gmsh_min_spin, 0, 1)
        
        gmsh_layout.addWidget(QLabel("Max Element Size:"), 1, 0)
        self.gmsh_max_spin = QDoubleSpinBox()
        self.gmsh_max_spin.setRange(0.001, 10.0)
        self.gmsh_max_spin.setValue(0.05)
        self.gmsh_max_spin.setDecimals(4)
        gmsh_layout.addWidget(self.gmsh_max_spin, 1, 1)
        
        gmsh_btn = QPushButton("Generate Gmsh Mesh")
        gmsh_btn.clicked.connect(self.generate_gmsh_mesh)
        gmsh_btn.setEnabled(GMSH_AVAILABLE)
        if not GMSH_AVAILABLE:
            gmsh_btn.setToolTip("Gmsh not installed")
        gmsh_layout.addWidget(gmsh_btn, 2, 0, 1, 2)
        
        gmsh_group.setLayout(gmsh_layout)
        layout.addWidget(gmsh_group)
        
        # === PySAGAS Analysis ===
        analysis_group = QGroupBox("PySAGAS Analysis")
        analysis_layout = QGridLayout()
        
        analysis_layout.addWidget(QLabel("Angle of Attack (°):"), 0, 0)
        self.aoa_spin = QDoubleSpinBox()
        self.aoa_spin.setRange(-20.0, 20.0)
        self.aoa_spin.setValue(0.0)
        self.aoa_spin.setSingleStep(0.5)
        analysis_layout.addWidget(self.aoa_spin, 0, 1)
        
        analysis_layout.addWidget(QLabel("Pressure (Pa):"), 1, 0)
        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(100, 1e7)
        self.pressure_spin.setValue(101325)
        self.pressure_spin.setDecimals(0)
        analysis_layout.addWidget(self.pressure_spin, 1, 1)
        
        analysis_layout.addWidget(QLabel("Temperature (K):"), 2, 0)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(100, 500)
        self.temp_spin.setValue(288.15)
        self.temp_spin.setDecimals(2)
        analysis_layout.addWidget(self.temp_spin, 2, 1)
        
        analyze_btn = QPushButton("🔬 Run Analysis")
        analyze_btn.clicked.connect(self.run_analysis)
        analyze_btn.setEnabled(PYSAGAS_AVAILABLE)
        if not PYSAGAS_AVAILABLE:
            analyze_btn.setToolTip("PySAGAS not installed")
        analysis_layout.addWidget(analyze_btn, 3, 0, 1, 2)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Add stretch at the bottom
        layout.addStretch()
        
        scroll.setWidget(panel)
        return scroll
    
    def create_right_panel(self):
        """Create the right visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        
        # === 3D View Tab ===
        tab_3d = QWidget()
        layout_3d = QVBoxLayout(tab_3d)
        
        # Display options
        options_layout = QHBoxLayout()
        self.show_upper_check = QCheckBox("Upper Surface")
        self.show_upper_check.setChecked(True)
        self.show_lower_check = QCheckBox("Lower Surface")
        self.show_lower_check.setChecked(True)
        self.show_le_check = QCheckBox("Leading Edge")
        self.show_le_check.setChecked(True)
        self.show_cg_check = QCheckBox("CG")
        self.show_cg_check.setChecked(True)
        
        options_layout.addWidget(self.show_upper_check)
        options_layout.addWidget(self.show_lower_check)
        options_layout.addWidget(self.show_le_check)
        options_layout.addWidget(self.show_cg_check)
        options_layout.addStretch()
        
        update_btn = QPushButton("Update View")
        update_btn.clicked.connect(self.update_3d_view)
        options_layout.addWidget(update_btn)
        
        layout_3d.addLayout(options_layout)
        
        self.canvas_3d = ConeWaveriderCanvas()
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, tab_3d)
        layout_3d.addWidget(self.toolbar_3d)
        layout_3d.addWidget(self.canvas_3d)
        
        self.tab_widget.addTab(tab_3d, "3D View")
        
        # === Design Space Tab ===
        tab_design = self.create_design_space_tab()
        self.tab_widget.addTab(tab_design, "🔍 Design Space")
        
        # === Results Tab ===
        tab_results = QWidget()
        layout_results = QVBoxLayout(tab_results)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 10))
        layout_results.addWidget(self.results_text)
        
        self.tab_widget.addTab(tab_results, "📊 Results")
        
        layout.addWidget(self.tab_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        return panel
    
    def create_design_space_tab(self):
        """Create the design space exploration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Splitter for controls and visualization
        splitter = QSplitter(Qt.Vertical)
        
        # Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Parameter ranges
        ranges_group = QGroupBox("Parameter Ranges")
        ranges_layout = QGridLayout()
        
        # Second order parameters
        ranges_layout.addWidget(QLabel("A₂ Range:"), 0, 0)
        self.ds_a2_min = QDoubleSpinBox()
        self.ds_a2_min.setRange(-50, 50)
        self.ds_a2_min.setValue(-10)
        ranges_layout.addWidget(self.ds_a2_min, 0, 1)
        
        ranges_layout.addWidget(QLabel("to"), 0, 2)
        self.ds_a2_max = QDoubleSpinBox()
        self.ds_a2_max.setRange(-50, 50)
        self.ds_a2_max.setValue(10)
        ranges_layout.addWidget(self.ds_a2_max, 0, 3)
        
        ranges_layout.addWidget(QLabel("Steps:"), 0, 4)
        self.ds_a2_steps = QSpinBox()
        self.ds_a2_steps.setRange(3, 50)
        self.ds_a2_steps.setValue(10)
        ranges_layout.addWidget(self.ds_a2_steps, 0, 5)
        
        ranges_layout.addWidget(QLabel("A₀ Range:"), 1, 0)
        self.ds_a0_min = QDoubleSpinBox()
        self.ds_a0_min.setRange(-1, 0)
        self.ds_a0_min.setValue(-0.3)
        self.ds_a0_min.setDecimals(3)
        ranges_layout.addWidget(self.ds_a0_min, 1, 1)
        
        ranges_layout.addWidget(QLabel("to"), 1, 2)
        self.ds_a0_max = QDoubleSpinBox()
        self.ds_a0_max.setRange(-1, 0)
        self.ds_a0_max.setValue(-0.05)
        self.ds_a0_max.setDecimals(3)
        ranges_layout.addWidget(self.ds_a0_max, 1, 3)
        
        ranges_layout.addWidget(QLabel("Steps:"), 1, 4)
        self.ds_a0_steps = QSpinBox()
        self.ds_a0_steps.setRange(3, 50)
        self.ds_a0_steps.setValue(10)
        ranges_layout.addWidget(self.ds_a0_steps, 1, 5)
        
        # Third order parameters
        ranges_layout.addWidget(QLabel("A₃ Range:"), 2, 0)
        self.ds_a3_min = QDoubleSpinBox()
        self.ds_a3_min.setRange(-100, 100)
        self.ds_a3_min.setValue(-50)
        ranges_layout.addWidget(self.ds_a3_min, 2, 1)
        
        ranges_layout.addWidget(QLabel("to"), 2, 2)
        self.ds_a3_max = QDoubleSpinBox()
        self.ds_a3_max.setRange(-100, 100)
        self.ds_a3_max.setValue(50)
        ranges_layout.addWidget(self.ds_a3_max, 2, 3)
        
        ranges_layout.addWidget(QLabel("Steps:"), 2, 4)
        self.ds_a3_steps = QSpinBox()
        self.ds_a3_steps.setRange(3, 50)
        self.ds_a3_steps.setValue(10)
        ranges_layout.addWidget(self.ds_a3_steps, 2, 5)
        
        ranges_group.setLayout(ranges_layout)
        controls_layout.addWidget(ranges_group)
        
        # Options
        options_layout = QHBoxLayout()
        
        self.run_aero_check = QCheckBox("Run Aero Analysis (slower)")
        self.run_aero_check.setChecked(False)
        self.run_aero_check.setEnabled(PYSAGAS_AVAILABLE)
        options_layout.addWidget(self.run_aero_check)
        
        options_layout.addStretch()
        
        # Run button
        self.run_ds_btn = QPushButton("▶ Run Design Space Exploration")
        self.run_ds_btn.clicked.connect(self.run_design_space)
        options_layout.addWidget(self.run_ds_btn)
        
        self.cancel_ds_btn = QPushButton("⏹ Cancel")
        self.cancel_ds_btn.clicked.connect(self.cancel_design_space)
        self.cancel_ds_btn.setEnabled(False)
        options_layout.addWidget(self.cancel_ds_btn)
        
        controls_layout.addLayout(options_layout)
        
        # Progress
        self.ds_progress = QProgressBar()
        self.ds_progress.setVisible(False)
        controls_layout.addWidget(self.ds_progress)
        
        self.ds_status = QLabel("Ready to explore design space")
        controls_layout.addWidget(self.ds_status)
        
        splitter.addWidget(controls_widget)
        
        # Visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        
        self.ds_canvas = DesignSpaceCanvas()
        self.ds_toolbar = NavigationToolbar(self.ds_canvas, viz_widget)
        viz_layout.addWidget(self.ds_toolbar)
        viz_layout.addWidget(self.ds_canvas)
        
        # Export results button
        export_layout = QHBoxLayout()
        export_results_btn = QPushButton("Export Results to CSV")
        export_results_btn.clicked.connect(self.export_design_space_results)
        export_layout.addWidget(export_results_btn)
        export_layout.addStretch()
        viz_layout.addLayout(export_layout)
        
        splitter.addWidget(viz_widget)
        
        splitter.setSizes([200, 400])
        layout.addWidget(splitter)
        
        return tab
    
    def on_order_changed(self, index):
        """Handle polynomial order change"""
        self.a3_spin.setEnabled(index == 1)
        
    def auto_shock_angle(self):
        """Automatically set optimal shock angle"""
        mach = self.mach_spin.value()
        optimal = optimal_shock_angle(mach)
        self.shock_spin.setValue(optimal)
        self.status_label.setText(f"Set shock angle to {optimal:.1f}° (estimated optimal for Mach {mach})")
    
    def generate_waverider(self):
        """Generate the cone-derived waverider"""
        try:
            mach = self.mach_spin.value()
            shock_angle = self.shock_spin.value()
            order = self.order_combo.currentIndex()
            
            self.status_label.setText("Generating waverider...")
            
            if order == 0:  # 2nd order
                self.waverider = create_second_order_waverider(
                    mach=mach,
                    shock_angle=shock_angle,
                    A2=self.a2_spin.value(),
                    A0=self.a0_spin.value(),
                    n_leading_edge=self.n_le_spin.value(),
                    n_streamwise=self.n_stream_spin.value()
                )
            else:  # 3rd order
                self.waverider = create_third_order_waverider(
                    mach=mach,
                    shock_angle=shock_angle,
                    A3=self.a3_spin.value(),
                    A2=self.a2_spin.value(),
                    A0=self.a0_spin.value(),
                    n_leading_edge=self.n_le_spin.value(),
                    n_streamwise=self.n_stream_spin.value()
                )
            
            # Update visualization
            self.update_3d_view()
            
            # Update results
            self.update_results_display()
            
            self.status_label.setText("Waverider generated successfully!")
            self.waverider_generated.emit(self.waverider)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate waverider:\n{str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
    
    def update_3d_view(self):
        """Update the 3D visualization"""
        self.canvas_3d.plot_waverider(
            self.waverider,
            show_upper=self.show_upper_check.isChecked(),
            show_lower=self.show_lower_check.isChecked(),
            show_le=self.show_le_check.isChecked(),
            show_cg=self.show_cg_check.isChecked()
        )
    
    def update_results_display(self):
        """Update the results text display"""
        if self.waverider is None:
            return
        
        refs = self.waverider.get_reference_values(scale=self.scale_spin.value())
        
        text = f"""
{'='*60}
CONE-DERIVED WAVERIDER SUMMARY
{'='*60}

FLOW CONDITIONS
---------------
Mach Number:        {self.waverider.mach:.2f}
Shock Angle:        {self.waverider.shock_angle:.2f}°
Cone Angle:         {self.waverider.cone_angle_deg:.2f}°
Deflection Angle:   {np.degrees(self.waverider.deflection_angle):.2f}°
Post-shock Mach:    {self.waverider.post_shock_mach:.2f}

GEOMETRY (Non-dimensional)
--------------------------
Length:             {self.waverider.length:.4f}
Planform Area:      {self.waverider.planform_area:.4f}
Volume:             {self.waverider.volume:.6f}
MAC:                {self.waverider.mac:.4f}
CG Location:        [{self.waverider.cg[0]:.4f}, {self.waverider.cg[1]:.4f}, {self.waverider.cg[2]:.4f}]

GEOMETRY (Scaled by {self.scale_spin.value()})
--------------------------
Length:             {refs['length']:.2f}
Planform Area:      {refs['area']:.2f}

POLYNOMIAL PARAMETERS
---------------------
Order:              {self.waverider.poly_order}
Coefficients:       {self.waverider.poly_coeffs}

MESH
----
Leading Edge Pts:   {self.waverider.n_leading_edge}
Streamwise Pts:     {self.waverider.n_streamwise}

{'='*60}
"""
        self.results_text.setText(text)
    
    def _on_blunting_toggled(self, state):
        enabled = bool(state)
        self.blunting_radius_spin.setEnabled(enabled)
        self.blunting_sweep_combo.setEnabled(enabled)

    def export_stl(self):
        """Export waverider to STL format"""
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate a waverider first!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save STL File", "cone_waverider.stl", "STL Files (*.stl)"
        )
        
        if filename:
            try:
                self.waverider.export_stl(filename)
                self.last_stl_file = filename
                self.status_label.setText(f"Exported to {filename}")
                QMessageBox.information(self, "Success", f"STL exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export STL:\n{str(e)}")
    
    def export_tri(self):
        """Export waverider to NASA Cart3D TRI format"""
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate a waverider first!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save TRI File", "cone_waverider.tri", "TRI Files (*.tri)"
        )
        
        if filename:
            try:
                self.waverider.export_tri(filename)
                self.status_label.setText(f"Exported to {filename}")
                QMessageBox.information(self, "Success", f"TRI exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export TRI:\n{str(e)}")
    
    def export_step(self):
        """Export waverider to STEP format using CadQuery NURBS surfaces"""
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate a waverider first!")
            return

        if not CADQUERY_AVAILABLE:
            QMessageBox.warning(self, "Warning", "CadQuery is not installed!")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save STEP File", "cone_waverider.step", "STEP Files (*.step *.stp)"
        )

        if filename:
            try:
                self.status_label.setText("Exporting to STEP (this may take a moment)...")

                # STEP uses mm; geometry is in meters → scale * 1000
                scale = self.scale_spin.value() * 1000.0
                blunting_radius = 0.0
                sweep_scaled = False
                if hasattr(self, 'blunting_check') and self.blunting_check.isChecked():
                    blunting_radius = self.blunting_radius_spin.value()
                    sweep_scaled = (self.blunting_sweep_combo.currentIndex() == 1)
                half_only = hasattr(self, 'half_vehicle_check') and self.half_vehicle_check.isChecked()

                print(f"[ConeWaverider Export] blunting_radius={blunting_radius}, "
                      f"sweep_scaled={sweep_scaled}, scale={scale}, half_only={half_only}")
                self._export_step_nurbs(filename, scale,
                                        blunting_radius=blunting_radius,
                                        sweep_scaled=sweep_scaled,
                                        half_only=half_only)

                self.last_step_file = filename
                self.status_label.setText(f"Exported to {filename}")
                QMessageBox.information(self, "Success", f"STEP exported to:\n{filename}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"Failed to export STEP:\n{str(e)}")

    def _export_step_nurbs(self, filename, scale, blunting_radius=0.0,
                           sweep_scaled=False, half_only=False):
        """
        Export STEP with smooth NURBS surfaces.

        The cone waverider uses coords [x=span, y=vertical, z=streamwise].
        We swizzle to [X=streamwise, Y=vertical, Z=span] for the shared
        solid builder, then build right half → fillet → mirror → export.
        """
        import cadquery as cq

        wr = self.waverider
        # Swizzle from [x_span, y_vert, z_stream] to [X_stream, Y_vert, Z_span]
        upper_raw = wr.upper_surface[:, :, [2, 1, 0]]  # (n_le, n_stream, 3)
        lower_raw = wr.lower_surface[:, :, [2, 1, 0]]
        le_raw = wr.leading_edge[:, [2, 1, 0]]          # (n_le, 3)

        n_le = upper_raw.shape[0]
        center_idx = n_le // 2

        # Right half = positive Z (positive original x_span)
        upper_half = upper_raw[center_idx:, :, :]
        lower_half = lower_raw[center_idx:, :, :]
        le_curve = le_raw[center_idx:]
        n_half = upper_half.shape[0]

        # Extract boundary curves
        centerline_upper = upper_half[0, :, :]
        centerline_lower = lower_half[0, :, :]
        te_upper = upper_half[:, -1, :]
        te_lower = lower_half[:, -1, :]
        upper_streams = [upper_half[i, :, :] for i in range(n_half)]
        lower_streams = [lower_half[i, :, :] for i in range(n_half)]

        # Build 4-face NURBS solid
        from waverider_generator.cad_export import build_waverider_solid
        right_side = build_waverider_solid(
            upper_streams, lower_streams, le_curve,
            centerline_upper, centerline_lower,
            te_upper, te_lower)

        # Scale to mm for STEP
        right_side = right_side.scale(scale)

        # Apply LE fillet if enabled
        if blunting_radius > 0:
            print(f"[ConeWaverider STEP] LE fillet: radius={blunting_radius * scale:.4f}mm, "
                  f"sweep_scaled={sweep_scaled}")
            from waverider_generator.cad_export import _apply_le_fillet
            le_pts = le_curve * scale
            right_side = _apply_le_fillet(
                right_side, blunting_radius * scale, le_pts,
                nose_cap=False, sweep_scaled=sweep_scaled)

        if half_only:
            result = cq.Workplane("XY").newObject([right_side])
        else:
            left_side = right_side.mirror(mirrorPlane='XY')
            result = cq.Workplane("XY").newObject([right_side]).union(left_side)

        cq.exporters.export(result, filename)
    
    def generate_gmsh_mesh(self):
        """Generate mesh using Gmsh"""
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate a waverider first!")
            return
        
        if not GMSH_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Gmsh is not installed!")
            return
        
        # First export to STL
        temp_stl = tempfile.mktemp(suffix='.stl')
        self.waverider.export_stl(temp_stl)
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Gmsh Mesh", "cone_waverider_mesh.stl", "STL Files (*.stl);;MSH Files (*.msh)"
        )
        
        if filename:
            try:
                self.status_label.setText("Generating Gmsh mesh...")
                
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal", 0)
                
                gmsh.merge(temp_stl)
                
                # Set mesh size
                gmsh.option.setNumber("Mesh.MeshSizeMin", self.gmsh_min_spin.value())
                gmsh.option.setNumber("Mesh.MeshSizeMax", self.gmsh_max_spin.value())
                
                # Generate 2D mesh
                gmsh.model.mesh.generate(2)
                
                # Save
                gmsh.write(filename)
                gmsh.finalize()
                
                os.unlink(temp_stl)
                
                self.last_stl_file = filename
                self.status_label.setText(f"Gmsh mesh saved to {filename}")
                QMessageBox.information(self, "Success", f"Mesh saved to:\n{filename}")
                
            except Exception as e:
                gmsh.finalize()
                QMessageBox.critical(self, "Error", f"Gmsh meshing failed:\n{str(e)}")
    
    def run_analysis(self):
        """Run PySAGAS aerodynamic analysis"""
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate a waverider first!")
            return
        
        if not PYSAGAS_AVAILABLE:
            QMessageBox.warning(self, "Warning", "PySAGAS is not installed!")
            return
        
        # Export to temp STL
        temp_stl = tempfile.mktemp(suffix='.stl')
        scale = self.scale_spin.value()
        
        # Scale and export
        vertices, triangles = self.waverider.get_mesh()
        vertices_scaled = vertices * scale
        
        # Write scaled STL
        with open(temp_stl, 'w') as f:
            f.write("solid waverider\n")
            for tri in triangles:
                v0, v1, v2 = vertices_scaled[tri[0]], vertices_scaled[tri[1]], vertices_scaled[tri[2]]
                e1 = v1 - v0
                e2 = v2 - v0
                normal = np.cross(e1, e2)
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normal = normal / norm
                else:
                    normal = np.array([0, 0, 1])
                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write("endsolid waverider\n")
        
        self.last_stl_file = temp_stl
        
        # Start analysis worker
        refs = self.waverider.get_reference_values(scale=scale)
        
        self.analysis_worker = AnalysisWorker(
            stl_file=temp_stl,
            mach=self.mach_spin.value(),
            aoa=self.aoa_spin.value(),
            pressure=self.pressure_spin.value(),
            temperature=self.temp_spin.value(),
            A_ref=refs['area']
        )
        
        self.analysis_worker.progress.connect(self.on_analysis_progress)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.error.connect(self.on_analysis_error)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Running PySAGAS analysis...")
        
        self.analysis_worker.start()
    
    def on_analysis_progress(self, msg):
        """Handle analysis progress update"""
        self.status_label.setText(msg)
    
    def on_analysis_finished(self, results):
        """Handle analysis completion"""
        self.progress_bar.setVisible(False)
        
        text = self.results_text.toPlainText()
        text += f"""

AERODYNAMIC ANALYSIS RESULTS
============================
Mach:       {self.mach_spin.value():.2f}
AoA:        {self.aoa_spin.value():.2f}°

CL:         {results['CL']:.6f}
CD:         {results['CD']:.6f}
Cm:         {results['Cm']:.6f}
L/D:        {results['L/D']:.3f}

{'='*60}
"""
        self.results_text.setText(text)
        self.status_label.setText(f"Analysis complete! L/D = {results['L/D']:.3f}")
    
    def on_analysis_error(self, error_msg):
        """Handle analysis error"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis failed!")
        QMessageBox.critical(self, "Analysis Error", error_msg)
    
    def run_design_space(self):
        """Run design space exploration"""
        order = self.order_combo.currentIndex()
        
        params = {
            'mach': self.mach_spin.value(),
            'shock_angle': self.shock_spin.value(),
            'poly_order': order + 2,  # 2 or 3
            'n_le': max(11, self.n_le_spin.value() // 2),  # Reduce for speed
            'n_stream': max(10, self.n_stream_spin.value() // 2),
            'run_aero': self.run_aero_check.isChecked(),
            'pressure': self.pressure_spin.value(),
            'temperature': self.temp_spin.value(),
            
            # 2nd order ranges
            'A2_min': self.ds_a2_min.value(),
            'A2_max': self.ds_a2_max.value(),
            'n_A2': self.ds_a2_steps.value(),
            'A0_min': self.ds_a0_min.value(),
            'A0_max': self.ds_a0_max.value(),
            'n_A0': self.ds_a0_steps.value(),
            
            # 3rd order ranges
            'A3_min': self.ds_a3_min.value(),
            'A3_max': self.ds_a3_max.value(),
            'n_A3': self.ds_a3_steps.value(),
            'A0_fixed': self.a0_spin.value(),
        }
        
        total = params['n_A2'] * params['n_A0'] if order == 0 else params['n_A3'] * params['n_A2']
        
        self.design_worker = DesignSpaceWorker(params)
        self.design_worker.progress.connect(self.on_ds_progress)
        self.design_worker.point_complete.connect(self.on_ds_point_complete)
        self.design_worker.finished.connect(self.on_ds_finished)
        self.design_worker.error.connect(self.on_ds_error)
        
        self.ds_progress.setVisible(True)
        self.ds_progress.setRange(0, total)
        self.ds_progress.setValue(0)
        self.run_ds_btn.setEnabled(False)
        self.cancel_ds_btn.setEnabled(True)
        self.ds_status.setText("Starting design space exploration...")
        
        self.design_space_results = []
        self.design_worker.start()
    
    def cancel_design_space(self):
        """Cancel design space exploration"""
        if self.design_worker:
            self.design_worker.cancel()
            self.ds_status.setText("Cancelling...")
    
    def on_ds_progress(self, current, total, msg):
        """Handle design space progress"""
        self.ds_progress.setValue(current)
        self.ds_status.setText(f"[{current}/{total}] {msg}")
    
    def on_ds_point_complete(self, result):
        """Handle single point completion"""
        self.design_space_results.append(result)
        
        # Update plot periodically
        if len(self.design_space_results) % 10 == 0:
            self.update_design_space_plot()
    
    def on_ds_finished(self, results):
        """Handle design space exploration completion"""
        self.ds_progress.setVisible(False)
        self.run_ds_btn.setEnabled(True)
        self.cancel_ds_btn.setEnabled(False)
        
        self.design_space_results = results
        
        valid_results = [r for r in results if r.get('valid', False)]
        self.ds_status.setText(f"Completed! {len(valid_results)}/{len(results)} valid designs")
        
        self.update_design_space_plot()
    
    def on_ds_error(self, error_msg):
        """Handle design space error"""
        self.ds_progress.setVisible(False)
        self.run_ds_btn.setEnabled(True)
        self.cancel_ds_btn.setEnabled(False)
        self.ds_status.setText("Error occurred!")
        QMessageBox.critical(self, "Error", error_msg)
    
    def update_design_space_plot(self):
        """Update the design space visualization"""
        if not self.design_space_results:
            return
        
        import pandas as pd
        
        valid_results = [r for r in self.design_space_results if r.get('valid', False)]
        if not valid_results:
            return
        
        df = pd.DataFrame(valid_results)
        
        # Determine which parameters to plot
        order = self.order_combo.currentIndex()
        
        if order == 0:  # 2nd order
            x_param = 'A2'
            y_param = 'A0'
        else:  # 3rd order
            x_param = 'A3'
            y_param = 'A2'
        
        # Color by L/D if available, otherwise volume
        if 'L/D' in df.columns and df['L/D'].notna().any():
            color_param = 'L/D'
        elif 'volume' in df.columns:
            color_param = 'volume'
        else:
            color_param = 'planform_area'
        
        self.ds_canvas.plot_design_space(df, x_param, y_param, color_param)
    
    def export_design_space_results(self):
        """Export design space results to CSV"""
        if not self.design_space_results:
            QMessageBox.warning(self, "Warning", "No results to export!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "design_space_results.csv", "CSV Files (*.csv)"
        )
        
        if filename:
            import pandas as pd
            df = pd.DataFrame(self.design_space_results)
            df.to_csv(filename, index=False)
            QMessageBox.information(self, "Success", f"Results exported to:\n{filename}")


# For testing standalone
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create main window with just this tab
    window = QMainWindow()
    window.setWindowTitle("Cone-Derived Waverider Designer")
    window.setGeometry(100, 100, 1400, 900)
    
    tab = ConeWaveriderTab()
    window.setCentralWidget(tab)
    
    window.show()
    sys.exit(app.exec_())
