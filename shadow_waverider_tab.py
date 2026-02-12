#!/usr/bin/env python3
"""
Cone-Derived Waverider Tab for Waverider GUI
=============================================
Based on Adam Weaver's SHADOW methodology from Utah State University thesis.
"""

import sys
import os
import numpy as np
from datetime import datetime
import tempfile

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QGridLayout,
                             QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
                             QProgressBar, QTextEdit, QTabWidget, QFileDialog,
                             QMessageBox, QSplitter, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from shadow_waverider import (
    ShadowWaverider, create_second_order_waverider,
    create_third_order_waverider, optimal_shock_angle
)

# Optional dependencies
try:
    import cadquery as cq
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
    from pysagas.geometry.parsers import MeshIO
    PYSAGAS_AVAILABLE = True
except ImportError:
    PYSAGAS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


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
            self.progress.emit("Loading mesh...")
            cells = MeshIO.load_from_file(self.stl_file)
            
            self.progress.emit("Setting up flow...")
            flow = FlowState(mach=self.mach, pressure=self.pressure,
                           temperature=self.temperature, aoa=np.radians(self.aoa))
            
            self.progress.emit("Running OPM...")
            solver = OPM(cells=cells, freestream=flow, verbosity=0)
            solver.solve()
            
            CL, CD, Cm = solver.flow_result.coefficients()
            LD = CL / CD if CD != 0 else float('inf')
            
            self.finished.emit({'CL': float(CL), 'CD': float(CD), 
                              'Cm': float(Cm), 'L/D': float(LD)})
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class DesignSpaceWorker(QThread):
    """Worker thread for design space exploration"""
    progress = pyqtSignal(int, int, str)
    point_complete = pyqtSignal(dict)
    finished = pyqtSignal(list)
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
            include_aero = self.params.get('include_aero', False)
            
            if poly_order == 2:
                A2_range = np.linspace(self.params['A2_min'], self.params['A2_max'], self.params['n_A2'])
                A0_range = np.linspace(self.params['A0_min'], self.params['A0_max'], self.params['n_A0'])
                total = len(A2_range) * len(A0_range)
                current = 0
                
                for A2 in A2_range:
                    for A0 in A0_range:
                        if self._is_cancelled:
                            self.finished.emit(results)
                            return
                        current += 1
                        self.progress.emit(current, total, f"A2={A2:.2f}, A0={A0:.3f}")
                        result = self._eval_2nd(mach, shock_angle, A2, A0, include_aero)
                        results.append(result)
                        self.point_complete.emit(result)
            else:
                A3_range = np.linspace(self.params['A3_min'], self.params['A3_max'], self.params['n_A3'])
                A2_range = np.linspace(self.params['A2_min'], self.params['A2_max'], self.params['n_A2'])
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
                        result = self._eval_3rd(mach, shock_angle, A3, A2, A0, include_aero)
                        results.append(result)
                        self.point_complete.emit(result)
            
            self.finished.emit(results)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")
    
    def _run_pysagas(self, wr, mach):
        """Run PySAGAS analysis on a waverider"""
        if not PYSAGAS_AVAILABLE:
            return {}
        
        try:
            # Create temporary STL
            verts, tris = wr.get_mesh()
            
            # Write STL to temp file
            temp_stl = tempfile.mktemp(suffix='.stl')
            with open(temp_stl, 'w') as f:
                f.write("solid waverider\n")
                for tri in tris:
                    v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    n = np.cross(edge1, edge2)
                    norm = np.linalg.norm(n)
                    if norm > 1e-10:
                        n = n / norm
                    else:
                        n = np.array([0, 0, 1])
                    f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                    f.write("    outer loop\n")
                    f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                    f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                    f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                    f.write("    endloop\n")
                    f.write("  endfacet\n")
                f.write("endsolid waverider\n")
            
            # Run PySAGAS
            cells = MeshIO.read_mesh(temp_stl)
            pressure = self.params.get('pressure', 101325)
            temperature = self.params.get('temperature', 288.15)
            aoa = self.params.get('aoa', 0)
            
            flow = FlowState(mach=mach, pressure=pressure, 
                           temperature=temperature, aoa=np.radians(aoa))
            solver = OPM(cells=cells, freestream=flow, verbosity=0)
            solver.solve()
            
            CL, CD, Cm = solver.flow_result.coefficients()
            LD = CL / CD if abs(CD) > 1e-10 else 0
            
            # Cleanup
            try:
                os.unlink(temp_stl)
            except:
                pass
            
            return {'CL': float(CL), 'CD': float(CD), 'Cm': float(Cm), 'L/D': float(LD)}
        except Exception as e:
            return {'aero_error': str(e)}
    
    def _eval_2nd(self, mach, shock_angle, A2, A0, include_aero=False):
        try:
            wr = create_second_order_waverider(mach=mach, shock_angle=shock_angle,
                A2=A2, A0=A0, n_leading_edge=self.params.get('n_le', 15),
                n_streamwise=self.params.get('n_stream', 15))
            result = {'A2': A2, 'A0': A0, 'cone_angle': wr.cone_angle_deg,
                   'planform_area': wr.planform_area, 'volume': wr.volume, 'valid': True}
            
            if include_aero:
                aero = self._run_pysagas(wr, mach)
                result.update(aero)
            
            return result
        except Exception as e:
            return {'A2': A2, 'A0': A0, 'valid': False, 'error': str(e)}
    
    def _eval_3rd(self, mach, shock_angle, A3, A2, A0, include_aero=False):
        try:
            wr = create_third_order_waverider(mach=mach, shock_angle=shock_angle,
                A3=A3, A2=A2, A0=A0, n_leading_edge=self.params.get('n_le', 15),
                n_streamwise=self.params.get('n_stream', 15))
            result = {'A3': A3, 'A2': A2, 'A0': A0, 'cone_angle': wr.cone_angle_deg,
                   'planform_area': wr.planform_area, 'volume': wr.volume, 'valid': True}
            
            if include_aero:
                aero = self._run_pysagas(wr, mach)
                result.update(aero)
            
            return result
        except Exception as e:
            return {'A3': A3, 'A2': A2, 'A0': A0, 'valid': False, 'error': str(e)}


class ShadowWaveriderCanvas(FigureCanvas):
    """Canvas for 3D visualization"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8), facecolor='#0A0A0A')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1A1A1A')
        super().__init__(self.fig)
        self.setParent(parent)
        
    def plot_waverider(self, wr, show_upper=True, show_lower=True, show_le=True, show_cg=True):
        self.ax.clear()
        if wr is None:
            self.ax.set_title('No waverider generated')
            self.draw()
            return
        
        # Surface data shape: (n_le, n_stream, 3)
        # After coordinate transform: X=streamwise, Y=vertical, Z=span
        # For plotting: we'll use Z(span) as plot X, X(streamwise) as plot Y, Y(vertical) as plot Z
        upper = wr.upper_surface
        lower = wr.lower_surface
        
        if show_upper:
            X_plot = upper[:, :, 2]   # Z = Span -> plot X
            Y_plot = upper[:, :, 0]   # X = Streamwise -> plot Y
            Z_plot = upper[:, :, 1]   # Y = Vertical -> plot Z
            self.ax.plot_surface(X_plot, Y_plot, Z_plot, color='steelblue', alpha=0.4, 
                                linewidth=0, antialiased=True, shade=True)
        
        if show_lower:
            X_plot = lower[:, :, 2]
            Y_plot = lower[:, :, 0]
            Z_plot = lower[:, :, 1]
            self.ax.plot_surface(X_plot, Y_plot, Z_plot, color='indianred', alpha=0.4,
                                linewidth=0, antialiased=True, shade=True)
        
        # Build legend with proxy artists
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = []
        if show_upper:
            legend_elements.append(Patch(facecolor='steelblue', alpha=0.4, label='Upper Surface'))
        if show_lower:
            legend_elements.append(Patch(facecolor='indianred', alpha=0.4, label='Lower Surface'))
        
        if show_le and hasattr(wr, 'leading_edge'):
            le = wr.leading_edge
            # Plot: Z(span), X(streamwise), Y(vertical)
            self.ax.plot(le[:, 2], le[:, 0], le[:, 1], 'k-', linewidth=2.5)
            legend_elements.append(Line2D([0], [0], color='black', linewidth=2.5, label='Leading Edge'))
        
        if show_cg and hasattr(wr, 'cg') and wr.cg is not None:
            cg = wr.cg
            # Plot: Z(span), X(streamwise), Y(vertical)
            self.ax.scatter([cg[2]], [cg[0]], [cg[1]], c='lime', s=150, marker='*', 
                           edgecolors='black', linewidths=1, zorder=10)
            legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='lime',
                                         markeredgecolor='black', markersize=12, label='CG'))
        
        self.ax.set_xlabel('Z (Span)', color='#FFFFFF')
        self.ax.set_ylabel('X (Streamwise)', color='#FFFFFF')
        self.ax.set_zlabel('Y (Vertical)', color='#FFFFFF')
        self.ax.set_title(f'SHADOW Waverider (M={wr.mach:.1f}, β={wr.shock_angle:.1f}°)', color='#FFFFFF')
        self.ax.tick_params(colors='#888888')
        if legend_elements:
            self.ax.legend(handles=legend_elements, loc='upper left')
        self._set_axes_equal()
        self.fig.tight_layout()
        self.draw()
    
    def _set_axes_equal(self):
        """Set equal aspect ratio for 3D plot"""
        try:
            limits = np.array([self.ax.get_xlim3d(), self.ax.get_ylim3d(), self.ax.get_zlim3d()])
            center = np.mean(limits, axis=1)
            radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
            self.ax.set_xlim3d([center[0] - radius, center[0] + radius])
            self.ax.set_ylim3d([center[1] - radius, center[1] + radius])
            self.ax.set_zlim3d([center[2] - radius, center[2] + radius])
        except:
            pass


class DesignSpaceCanvas(FigureCanvas):
    """Canvas for design space visualization"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8), facecolor='#0A0A0A')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1A1A1A')
        self.colorbar = None  # Track colorbar to remove it on updates
        super().__init__(self.fig)
        self.setParent(parent)
        
    def plot_design_space(self, df, x_param, y_param, color_param):
        # Remove old colorbar if it exists
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except:
                pass
            self.colorbar = None
        
        self.ax.clear()
        
        if df is None or len(df) == 0:
            self.ax.set_title('No results')
            self.draw()
            return
        
        valid_df = df[df['valid'] == True] if 'valid' in df.columns else df
        invalid_df = df[df['valid'] == False] if 'valid' in df.columns else None
        
        if len(valid_df) == 0:
            self.ax.set_title('No valid results')
            self.draw()
            return
        
        # Plot valid points
        if color_param in valid_df.columns:
            sc = self.ax.scatter(valid_df[x_param], valid_df[y_param], c=valid_df[color_param],
                               cmap='viridis', s=80, alpha=0.8, edgecolors='white', linewidths=0.5)
            self.colorbar = self.fig.colorbar(sc, ax=self.ax, label=color_param)
            
            # Mark best point
            best_idx = valid_df[color_param].idxmax()
            best = valid_df.loc[best_idx]
            self.ax.scatter([best[x_param]], [best[y_param]], c='gold', s=300, marker='*',
                          edgecolors='black', linewidths=2, zorder=10, label=f'Best: {best[color_param]:.4f}')
        else:
            self.ax.scatter(valid_df[x_param], valid_df[y_param], s=80, alpha=0.8)
        
        # Plot invalid points
        if invalid_df is not None and len(invalid_df) > 0:
            self.ax.scatter(invalid_df[x_param], invalid_df[y_param], c='red', marker='x', 
                          s=50, alpha=0.5, label='Invalid')
        
        self.ax.set_xlabel(x_param, fontsize=12, color='#FFFFFF')
        self.ax.set_ylabel(y_param, fontsize=12, color='#FFFFFF')
        self.ax.set_title(f'{color_param} vs ({x_param}, {y_param})', fontsize=14, color='#FFFFFF')
        self.ax.tick_params(colors='#888888')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        self.fig.tight_layout()
        self.draw()


class ShadowWaveriderTab(QWidget):
    """Main tab for cone-derived waverider design"""
    
    waverider_generated = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.waverider = None
        self.last_stl_file = None
        self.design_space_results = None
        self.analysis_worker = None
        self.design_worker = None
        self.init_ui()
        
    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Left panel (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self._create_flow_group())
        left_layout.addWidget(self._create_poly_group())
        left_layout.addWidget(self._create_mesh_group())
        left_layout.addWidget(self._create_blunting_group())
        left_layout.addWidget(self._create_generate_group())
        left_layout.addWidget(self._create_export_group())
        left_layout.addWidget(self._create_analysis_group())
        left_layout.addStretch()

        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(400)

        # Right panel (tabs)
        right_panel = QTabWidget()

        # 3D View tab
        view_widget = QWidget()
        view_layout = QVBoxLayout(view_widget)

        opts = QHBoxLayout()
        self.show_upper = QCheckBox("Upper"); self.show_upper.setChecked(True)
        self.show_lower = QCheckBox("Lower"); self.show_lower.setChecked(True)
        self.show_le = QCheckBox("LE"); self.show_le.setChecked(True)
        self.show_cg = QCheckBox("CG"); self.show_cg.setChecked(True)
        opts.addWidget(self.show_upper); opts.addWidget(self.show_lower)
        opts.addWidget(self.show_le); opts.addWidget(self.show_cg)
        opts.addStretch()
        update_btn = QPushButton("Update")
        update_btn.clicked.connect(self.update_view)
        opts.addWidget(update_btn)
        view_layout.addLayout(opts)

        self.canvas_3d = ShadowWaveriderCanvas()
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, view_widget)
        view_layout.addWidget(self.toolbar_3d)
        view_layout.addWidget(self.canvas_3d)
        right_panel.addTab(view_widget, "3D View")

        # Design Space tab
        ds_widget = self._create_design_space_widget()
        right_panel.addTab(ds_widget, "Design Space")

        # Results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 10))
        results_layout.addWidget(self.results_text)
        right_panel.addTab(results_widget, "Results")

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)
    
    def _create_flow_group(self):
        group = QGroupBox("Flow Conditions")
        layout = QGridLayout()
        
        layout.addWidget(QLabel("Mach:"), 0, 0)
        self.mach_spin = QDoubleSpinBox()
        self.mach_spin.setRange(2.0, 30.0); self.mach_spin.setValue(6.0)
        self.mach_spin.setSingleStep(0.5); self.mach_spin.setDecimals(1)
        self.mach_spin.valueChanged.connect(self.update_shock_recommendations)
        layout.addWidget(self.mach_spin, 0, 1)
        
        layout.addWidget(QLabel("Shock β (°):"), 1, 0)
        self.shock_spin = QDoubleSpinBox()
        self.shock_spin.setRange(5.0, 60.0); self.shock_spin.setValue(12.0)
        self.shock_spin.setSingleStep(0.5); self.shock_spin.setDecimals(1)
        layout.addWidget(self.shock_spin, 1, 1)
        
        auto_btn = QPushButton("Auto")
        auto_btn.clicked.connect(self.auto_shock)
        layout.addWidget(auto_btn, 1, 2)
        
        # Shock angle recommendations label
        self.shock_rec_label = QLabel("")
        self.shock_rec_label.setStyleSheet("color: #888888; font-size: 9px;")
        self.shock_rec_label.setWordWrap(True)
        layout.addWidget(self.shock_rec_label, 2, 0, 1, 3)
        
        layout.addWidget(QLabel("Cone θc (°):"), 3, 0)
        self.cone_label = QLabel("--")
        self.cone_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.cone_label, 3, 1)
        
        group.setLayout(layout)
        
        # Initialize recommendations
        self.update_shock_recommendations()
        
        return group
    
    def update_shock_recommendations(self):
        """Update shock angle recommendations based on Mach number"""
        mach = self.mach_spin.value()
        
        # Mach angle (minimum possible shock angle)
        mach_angle = np.degrees(np.arcsin(1.0 / mach))
        
        # Optimal shock angle for L/D (empirical estimate)
        optimal = optimal_shock_angle(mach)
        
        # Recommended range (±1-2 degrees around optimal)
        rec_min = max(mach_angle + 0.5, optimal - 2.0)
        rec_max = optimal + 2.0
        
        self.shock_rec_label.setText(
            f"β range: {mach_angle:.1f}° (Mach angle) to ~45° | "
            f"Recommended: {optimal:.1f}° (range: {rec_min:.1f}°-{rec_max:.1f}°)"
        )
    
    def _create_poly_group(self):
        group = QGroupBox("Polynomial (y = A₃x³ + A₂x² + A₀)")
        layout = QGridLayout()
        
        layout.addWidget(QLabel("Order:"), 0, 0)
        self.order_combo = QComboBox()
        self.order_combo.addItems(["2nd Order", "3rd Order"])
        self.order_combo.currentIndexChanged.connect(self.on_order_change)
        layout.addWidget(self.order_combo, 0, 1)
        
        layout.addWidget(QLabel("A₃:"), 1, 0)
        self.a3_spin = QDoubleSpinBox()
        self.a3_spin.setRange(-100, 100); self.a3_spin.setValue(0); self.a3_spin.setEnabled(False)
        layout.addWidget(self.a3_spin, 1, 1)
        
        layout.addWidget(QLabel("A₂:"), 2, 0)
        self.a2_spin = QDoubleSpinBox()
        self.a2_spin.setRange(-50, 50); self.a2_spin.setValue(-2.0); self.a2_spin.setDecimals(2)
        layout.addWidget(self.a2_spin, 2, 1)
        
        layout.addWidget(QLabel("A₀:"), 3, 0)
        self.a0_spin = QDoubleSpinBox()
        self.a0_spin.setRange(-1, 0); self.a0_spin.setValue(-0.15); self.a0_spin.setDecimals(3)
        layout.addWidget(self.a0_spin, 3, 1)
        
        group.setLayout(layout)
        return group
    
    def _create_mesh_group(self):
        group = QGroupBox("Mesh")
        layout = QGridLayout()
        
        layout.addWidget(QLabel("LE Points:"), 0, 0)
        self.n_le_spin = QSpinBox()
        self.n_le_spin.setRange(11, 101); self.n_le_spin.setValue(21)
        layout.addWidget(self.n_le_spin, 0, 1)
        
        layout.addWidget(QLabel("Streamwise:"), 1, 0)
        self.n_stream_spin = QSpinBox()
        self.n_stream_spin.setRange(10, 100); self.n_stream_spin.setValue(20)
        layout.addWidget(self.n_stream_spin, 1, 1)
        
        layout.addWidget(QLabel("Length (m):"), 2, 0)
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.1, 100.0); self.length_spin.setValue(1.0); self.length_spin.setDecimals(2)
        self.length_spin.setToolTip("Waverider length in meters (streamwise extent)")
        layout.addWidget(self.length_spin, 2, 1)
        
        layout.addWidget(QLabel("Scale (1.0 = meters):"), 3, 0)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.001, 10000); self.scale_spin.setValue(1.0); self.scale_spin.setDecimals(3)
        self.scale_spin.setToolTip("Scale factor for export (1.0 = SI meters)")
        layout.addWidget(self.scale_spin, 3, 1)
        
        group.setLayout(layout)
        return group
    
    def _create_blunting_group(self):
        group = QGroupBox("Leading Edge Blunting")
        layout = QGridLayout()

        self.blunting_check = QCheckBox("Enable LE blunting")
        self.blunting_check.setToolTip("Apply circular arc blunting to the sharp LE during STEP export")
        self.blunting_check.stateChanged.connect(self._on_blunting_toggled)
        layout.addWidget(self.blunting_check, 0, 0, 1, 2)

        layout.addWidget(QLabel("Radius (m):"), 1, 0)
        self.blunting_radius_spin = QDoubleSpinBox()
        self.blunting_radius_spin.setRange(0.0001, 1.0)
        self.blunting_radius_spin.setValue(0.005)
        self.blunting_radius_spin.setSingleStep(0.001)
        self.blunting_radius_spin.setDecimals(4)
        self.blunting_radius_spin.setEnabled(False)
        layout.addWidget(self.blunting_radius_spin, 1, 1)

        self.blunting_preview_btn = QPushButton("Show LE Preview")
        self.blunting_preview_btn.setToolTip("Visualize blunted vs original LE on the 3D view.\nBlunting is applied automatically during STEP export.")
        self.blunting_preview_btn.clicked.connect(self._preview_blunting)
        self.blunting_preview_btn.setEnabled(False)
        self.blunting_preview_btn.setStyleSheet(
            "QPushButton { background-color: #1A1A1A; color: #F59E0B; border: 1px solid #78350F; padding: 5px; }"
            "QPushButton:hover { background-color: #78350F; color: #FFFFFF; }"
            "QPushButton:disabled { color: #555555; border-color: #333333; }"
        )
        layout.addWidget(self.blunting_preview_btn, 2, 0, 1, 2)

        group.setLayout(layout)
        return group

    def _on_blunting_toggled(self, state):
        enabled = bool(state)
        self.blunting_radius_spin.setEnabled(enabled)
        self.blunting_preview_btn.setEnabled(enabled and self.waverider is not None)

    def _preview_blunting(self):
        """Show blunted LE preview on the 3D view."""
        if self.waverider is None:
            QMessageBox.warning(self, "No waverider", "Generate a waverider first.")
            return

        radius = self.blunting_radius_spin.value()
        if radius <= 0:
            return

        try:
            wr = self.waverider
            # ConeWaverider has upper_surface/lower_surface as (n_span, n_stream, 3) arrays
            # Leading edge is at streamwise index 0
            original_le = wr.leading_edge  # (n_le, 3)

            # Compute blunted LE points using local tangent information
            n_le = wr.upper_surface.shape[0]
            blunted_points = []

            for i in range(n_le):
                le_pt = wr.upper_surface[i, 0, :]
                # Upper tangent (downstream from LE)
                if wr.upper_surface.shape[1] >= 2:
                    t_u = wr.upper_surface[i, 1, :] - wr.upper_surface[i, 0, :]
                    n = np.linalg.norm(t_u)
                    t_u = t_u / n if n > 1e-12 else np.array([1, 0, 0], dtype=float)
                else:
                    t_u = np.array([1, 0, 0], dtype=float)

                if wr.lower_surface.shape[1] >= 2:
                    t_l = wr.lower_surface[i, 1, :] - wr.lower_surface[i, 0, :]
                    n = np.linalg.norm(t_l)
                    t_l = t_l / n if n > 1e-12 else np.array([1, 0, 0], dtype=float)
                else:
                    t_l = np.array([1, 0, 0], dtype=float)

                bisector = t_u + t_l
                b_norm = np.linalg.norm(bisector)
                if b_norm > 1e-12:
                    bisector = bisector / b_norm
                else:
                    bisector = np.array([1, 0, 0], dtype=float)

                cos_half = np.clip(np.dot(t_u, t_l), -1, 1)
                half_angle = np.arccos(cos_half) / 2.0

                if half_angle < 1e-6:
                    blunted_points.append(le_pt)
                    continue

                d_center = radius / np.sin(half_angle)
                center = le_pt + d_center * bisector

                tp_upper = le_pt + np.dot(center - le_pt, t_u) * t_u
                tp_lower = le_pt + np.dot(center - le_pt, t_l) * t_l

                v_up = tp_upper - center
                v_lo = tp_lower - center
                v_up_hat = v_up / (np.linalg.norm(v_up) + 1e-12)
                v_lo_hat = v_lo / (np.linalg.norm(v_lo) + 1e-12)
                v_mid = v_up_hat + v_lo_hat
                v_mid_norm = np.linalg.norm(v_mid)
                if v_mid_norm > 1e-12:
                    v_mid = v_mid / v_mid_norm
                arc_mid = center + radius * v_mid
                blunted_points.append(arc_mid)

            blunted_le = np.array(blunted_points)

            # Draw on 3D canvas
            ax = self.canvas_3d.ax
            for line in list(ax.lines):
                if hasattr(line, '_blunting_preview'):
                    line.remove()

            line_orig, = ax.plot(
                original_le[:, 0], original_le[:, 1], original_le[:, 2],
                'r--', linewidth=1.5, label='Original LE')
            line_orig._blunting_preview = True

            line_blunt, = ax.plot(
                blunted_le[:, 0], blunted_le[:, 1], blunted_le[:, 2],
                color='#4ADE80', linewidth=2.5, label='Blunted LE')
            line_blunt._blunting_preview = True

            ax.legend(loc='upper right', fontsize=8)
            self.canvas_3d.draw()

            self.info_label.setText(
                f"LE blunting preview: r = {radius:.4f} m | "
                f"Original (red) vs Blunted (green)")

        except Exception as e:
            QMessageBox.critical(self, "Preview error",
                                 f"Failed to preview blunting:\n\n{str(e)}")

    def _create_generate_group(self):
        group = QGroupBox("Generate")
        layout = QVBoxLayout()
        
        btn = QPushButton("🚀 Generate Waverider")
        btn.setStyleSheet("background-color: #F59E0B; color: #0A0A0A; font-weight: bold; padding: 10px;")
        btn.clicked.connect(self.generate)
        layout.addWidget(btn)
        
        self.info_label = QLabel("Ready")
        self.info_label.setStyleSheet("color: #888888; font-size: 10px;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        group.setLayout(layout)
        return group
    
    def _create_export_group(self):
        group = QGroupBox("Export")
        layout = QGridLayout()
        
        stl_btn = QPushButton("STL"); stl_btn.clicked.connect(self.export_stl)
        tri_btn = QPushButton("TRI"); tri_btn.clicked.connect(self.export_tri)
        step_btn = QPushButton("STEP"); step_btn.clicked.connect(self.export_step)
        step_btn.setEnabled(CADQUERY_AVAILABLE)
        gmsh_btn = QPushButton("Gmsh"); gmsh_btn.clicked.connect(self.gmsh_mesh)
        gmsh_btn.setEnabled(GMSH_AVAILABLE)
        
        layout.addWidget(stl_btn, 0, 0); layout.addWidget(tri_btn, 0, 1)
        layout.addWidget(step_btn, 1, 0); layout.addWidget(gmsh_btn, 1, 1)
        
        layout.addWidget(QLabel("Gmsh min [m]:"), 2, 0)
        self.gmsh_min = QDoubleSpinBox()
        self.gmsh_min.setRange(0.00001, 10.0); self.gmsh_min.setValue(0.005); self.gmsh_min.setDecimals(5)
        layout.addWidget(self.gmsh_min, 2, 1)

        layout.addWidget(QLabel("Gmsh max [m]:"), 3, 0)
        self.gmsh_max = QDoubleSpinBox()
        self.gmsh_max.setRange(0.0001, 100.0); self.gmsh_max.setValue(0.05); self.gmsh_max.setDecimals(5)
        layout.addWidget(self.gmsh_max, 3, 1)
        
        group.setLayout(layout)
        return group
    
    def _create_analysis_group(self):
        group = QGroupBox("PySAGAS Analysis")
        layout = QGridLayout()

        if not PYSAGAS_AVAILABLE:
            layout.addWidget(QLabel("PySAGAS not available"), 0, 0, 1, 2)

        layout.addWidget(QLabel("AoA:"), 1, 0)
        self.aoa_spin = QDoubleSpinBox()
        self.aoa_spin.setRange(-20, 20); self.aoa_spin.setValue(0)
        layout.addWidget(self.aoa_spin, 1, 1)

        layout.addWidget(QLabel("P (Pa):"), 2, 0)
        self.p_spin = QDoubleSpinBox()
        self.p_spin.setRange(100, 1e7); self.p_spin.setValue(101325); self.p_spin.setDecimals(0)
        layout.addWidget(self.p_spin, 2, 1)

        layout.addWidget(QLabel("T (K):"), 3, 0)
        self.t_spin = QDoubleSpinBox()
        self.t_spin.setRange(100, 500); self.t_spin.setValue(288.15)
        layout.addWidget(self.t_spin, 3, 1)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setEnabled(PYSAGAS_AVAILABLE)
        layout.addWidget(self.analyze_btn, 4, 0, 1, 2)

        self.analysis_progress = QProgressBar()
        self.analysis_progress.setVisible(False)
        layout.addWidget(self.analysis_progress, 5, 0, 1, 2)

        self.analysis_status = QLabel("")
        self.analysis_status.setStyleSheet("color: #888888; font-style: italic;")
        layout.addWidget(self.analysis_status, 6, 0, 1, 2)

        group.setLayout(layout)
        return group
    
    def _create_design_space_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        group = QGroupBox("Design Space Exploration")
        gl = QGridLayout()
        
        gl.addWidget(QLabel("A₂:"), 0, 0)
        self.ds_a2_min = QDoubleSpinBox(); self.ds_a2_min.setRange(-50, 50); self.ds_a2_min.setValue(-10)
        self.ds_a2_max = QDoubleSpinBox(); self.ds_a2_max.setRange(-50, 50); self.ds_a2_max.setValue(0)
        self.ds_a2_n = QSpinBox(); self.ds_a2_n.setRange(3, 50); self.ds_a2_n.setValue(10)
        gl.addWidget(self.ds_a2_min, 0, 1); gl.addWidget(QLabel("to"), 0, 2)
        gl.addWidget(self.ds_a2_max, 0, 3); gl.addWidget(self.ds_a2_n, 0, 4)
        
        gl.addWidget(QLabel("A₀:"), 1, 0)
        self.ds_a0_min = QDoubleSpinBox(); self.ds_a0_min.setRange(-1, 0); self.ds_a0_min.setValue(-0.3); self.ds_a0_min.setDecimals(3)
        self.ds_a0_max = QDoubleSpinBox(); self.ds_a0_max.setRange(-1, 0); self.ds_a0_max.setValue(-0.05); self.ds_a0_max.setDecimals(3)
        self.ds_a0_n = QSpinBox(); self.ds_a0_n.setRange(3, 50); self.ds_a0_n.setValue(10)
        gl.addWidget(self.ds_a0_min, 1, 1); gl.addWidget(QLabel("to"), 1, 2)
        gl.addWidget(self.ds_a0_max, 1, 3); gl.addWidget(self.ds_a0_n, 1, 4)
        
        gl.addWidget(QLabel("A₃:"), 2, 0)
        self.ds_a3_min = QDoubleSpinBox(); self.ds_a3_min.setRange(-100, 100); self.ds_a3_min.setValue(-50)
        self.ds_a3_max = QDoubleSpinBox(); self.ds_a3_max.setRange(-100, 100); self.ds_a3_max.setValue(50)
        self.ds_a3_n = QSpinBox(); self.ds_a3_n.setRange(3, 50); self.ds_a3_n.setValue(10)
        gl.addWidget(self.ds_a3_min, 2, 1); gl.addWidget(QLabel("to"), 2, 2)
        gl.addWidget(self.ds_a3_max, 2, 3); gl.addWidget(self.ds_a3_n, 2, 4)
        
        # Aero analysis checkbox
        self.ds_include_aero = QCheckBox("Include Aero (PySAGAS)")
        self.ds_include_aero.setEnabled(PYSAGAS_AVAILABLE)
        if not PYSAGAS_AVAILABLE:
            self.ds_include_aero.setToolTip("PySAGAS not available")
        else:
            self.ds_include_aero.setToolTip("Run PySAGAS for each design (slower but gives L/D)")
        gl.addWidget(self.ds_include_aero, 3, 0, 1, 3)
        
        # Color-by selector
        gl.addWidget(QLabel("Color by:"), 3, 3)
        self.ds_color_combo = QComboBox()
        self.ds_color_combo.addItems(["volume", "planform_area", "L/D", "CL", "CD"])
        self.ds_color_combo.currentTextChanged.connect(self.update_ds_plot)
        gl.addWidget(self.ds_color_combo, 3, 4)
        
        btn_layout = QHBoxLayout()
        self.run_ds_btn = QPushButton("▶ Run")
        self.run_ds_btn.clicked.connect(self.run_design_space)
        self.cancel_ds_btn = QPushButton("⏹ Cancel")
        self.cancel_ds_btn.clicked.connect(self.cancel_ds)
        self.cancel_ds_btn.setEnabled(False)
        btn_layout.addWidget(self.run_ds_btn); btn_layout.addWidget(self.cancel_ds_btn)
        gl.addLayout(btn_layout, 4, 0, 1, 5)
        
        group.setLayout(gl)
        layout.addWidget(group)
        
        self.ds_progress = QProgressBar(); self.ds_progress.setVisible(False)
        layout.addWidget(self.ds_progress)
        self.ds_status = QLabel("Ready")
        layout.addWidget(self.ds_status)
        
        # Best design info panel
        self.best_design_group = QGroupBox("⭐ Best Design Found")
        best_layout = QGridLayout()
        self.best_design_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 2px solid #F59E0B; border-radius: 5px; margin-top: 10px; padding-top: 10px; }"
            "QGroupBox::title { color: #F59E0B; }"
        )
        
        best_layout.addWidget(QLabel("A₃:"), 0, 0)
        self.best_a3_label = QLabel("--")
        self.best_a3_label.setStyleSheet("font-weight: bold; color: #FFFFFF;")
        best_layout.addWidget(self.best_a3_label, 0, 1)
        
        best_layout.addWidget(QLabel("A₂:"), 0, 2)
        self.best_a2_label = QLabel("--")
        self.best_a2_label.setStyleSheet("font-weight: bold; color: #FFFFFF;")
        best_layout.addWidget(self.best_a2_label, 0, 3)
        
        best_layout.addWidget(QLabel("A₀:"), 0, 4)
        self.best_a0_label = QLabel("--")
        self.best_a0_label.setStyleSheet("font-weight: bold; color: #FFFFFF;")
        best_layout.addWidget(self.best_a0_label, 0, 5)
        
        best_layout.addWidget(QLabel("Volume:"), 1, 0)
        self.best_volume_label = QLabel("--")
        self.best_volume_label.setStyleSheet("font-weight: bold; color: #4ADE80;")
        best_layout.addWidget(self.best_volume_label, 1, 1)
        
        best_layout.addWidget(QLabel("Area:"), 1, 2)
        self.best_area_label = QLabel("--")
        self.best_area_label.setStyleSheet("font-weight: bold; color: #4ADE80;")
        best_layout.addWidget(self.best_area_label, 1, 3)
        
        best_layout.addWidget(QLabel("θc:"), 1, 4)
        self.best_cone_label = QLabel("--")
        self.best_cone_label.setStyleSheet("font-weight: bold; color: #4ADE80;")
        best_layout.addWidget(self.best_cone_label, 1, 5)
        
        best_layout.addWidget(QLabel("L/D:"), 2, 0)
        self.best_ld_label = QLabel("--")
        self.best_ld_label.setStyleSheet("font-weight: bold; color: #EF4444; font-size: 14px;")
        best_layout.addWidget(self.best_ld_label, 2, 1)
        
        # Apply best design button
        apply_best_btn = QPushButton("📋 Apply to Main Panel")
        apply_best_btn.clicked.connect(self.apply_best_design)
        apply_best_btn.setStyleSheet("background-color: #F59E0B; color: #0A0A0A; font-weight: bold; padding: 5px;")
        best_layout.addWidget(apply_best_btn, 2, 2, 1, 4)
        
        self.best_design_group.setLayout(best_layout)
        self.best_design_group.setVisible(False)  # Hidden until we have results
        layout.addWidget(self.best_design_group)
        
        self.ds_canvas = DesignSpaceCanvas()
        self.ds_toolbar = NavigationToolbar(self.ds_canvas, widget)
        layout.addWidget(self.ds_toolbar)
        layout.addWidget(self.ds_canvas)
        
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self.export_ds_csv)
        layout.addWidget(export_btn)
        
        return widget
    
    def apply_best_design(self):
        """Apply the best design parameters to the main panel"""
        if not hasattr(self, 'best_design_params') or self.best_design_params is None:
            return
        
        params = self.best_design_params
        if 'A3' in params:
            self.a3_spin.setValue(params['A3'])
        if 'A2' in params:
            self.a2_spin.setValue(params['A2'])
        if 'A0' in params:
            self.a0_spin.setValue(params['A0'])
        
        self.info_label.setText("✓ Applied best design parameters")
    
    # === Slot methods ===
    def on_order_change(self, idx):
        self.a3_spin.setEnabled(idx == 1)
    
    def auto_shock(self):
        opt = optimal_shock_angle(self.mach_spin.value())
        self.shock_spin.setValue(opt)
        self.info_label.setText(f"Set β={opt:.1f}°")
    
    def generate(self):
        try:
            mach = self.mach_spin.value()
            shock = self.shock_spin.value()
            length = self.length_spin.value()
            
            if self.order_combo.currentIndex() == 0:
                self.waverider = create_second_order_waverider(
                    mach=mach, shock_angle=shock, A2=self.a2_spin.value(),
                    A0=self.a0_spin.value(), n_leading_edge=self.n_le_spin.value(),
                    n_streamwise=self.n_stream_spin.value(), length=length)
            else:
                self.waverider = create_third_order_waverider(
                    mach=mach, shock_angle=shock, A3=self.a3_spin.value(),
                    A2=self.a2_spin.value(), A0=self.a0_spin.value(),
                    n_leading_edge=self.n_le_spin.value(), n_streamwise=self.n_stream_spin.value(),
                    length=length)
            
            self.cone_label.setText(f"{self.waverider.cone_angle_deg:.2f}")
            self.update_view()
            self.update_results()
            self.info_label.setText(f"✓ θc={self.waverider.cone_angle_deg:.1f}°, Area={self.waverider.planform_area:.4f}, L={length:.2f}m")
            if self.blunting_check.isChecked():
                self.blunting_preview_btn.setEnabled(True)
            self.waverider_generated.emit(self.waverider)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.info_label.setText(f"✗ {str(e)}")
    
    def update_view(self):
        self.canvas_3d.plot_waverider(self.waverider, self.show_upper.isChecked(),
            self.show_lower.isChecked(), self.show_le.isChecked(), self.show_cg.isChecked())
    
    def update_results(self):
        if self.waverider is None: return
        wr = self.waverider
        self.results_text.setText(f"""
{'='*50}
CONE-DERIVED WAVERIDER
{'='*50}
Mach:           {wr.mach:.2f}
Shock β:        {wr.shock_angle:.2f}°
Cone θc:        {wr.cone_angle_deg:.2f}°
Post-shock M:   {wr.post_shock_mach:.2f}

Polynomial:     Order {wr.poly_order}
Coefficients:   {wr.poly_coeffs}

Length:         {wr.length:.4f}
Planform Area:  {wr.planform_area:.4f}
Volume:         {wr.volume:.6f}
MAC:            {wr.mac:.4f}
CG:             [{wr.cg[0]:.4f}, {wr.cg[1]:.4f}, {wr.cg[2]:.4f}]
{'='*50}
""")
    
    # === Export methods ===
    def export_stl(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate waverider first!")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save STL", "shadow_waverider.stl", "STL (*.stl)")
        if fn:
            self.waverider.export_stl(fn)
            self.last_stl_file = fn
            QMessageBox.information(self, "Success", f"Saved: {fn}")
    
    def export_tri(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate waverider first!")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save TRI", "shadow_waverider.tri", "TRI (*.tri)")
        if fn:
            self.waverider.export_tri(fn)
            QMessageBox.information(self, "Success", f"Saved: {fn}")
    
    def export_step(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate waverider first!")
            return
        if not CADQUERY_AVAILABLE:
            QMessageBox.warning(self, "Warning", "CadQuery not installed")
            return
        
        # Ask user which method to use
        from PyQt5.QtWidgets import QInputDialog
        methods = ["NURBS Surfaces (smooth)", "Quad Faces (faceted)"]
        method, ok = QInputDialog.getItem(self, "STEP Export Method", 
            "Select export method:", methods, 0, False)
        if not ok:
            return
            
        fn, _ = QFileDialog.getSaveFileName(self, "Save STEP", "shadow_waverider.step", "STEP (*.step)")
        if fn:
            try:
                scale = self.scale_spin.value()
                blunting_radius = 0.0
                if self.blunting_check.isChecked():
                    blunting_radius = self.blunting_radius_spin.value()
                if method == methods[0]:
                    self._export_step_nurbs(fn, scale, blunting_radius=blunting_radius)
                else:
                    self._export_step_faces(fn, scale)
                QMessageBox.information(self, "Success", f"Saved: {fn}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"STEP export failed:\n{str(e)}")
    
    def _export_step_nurbs(self, filename, scale, blunting_radius=0.0):
        """
        Export STEP with smooth NURBS surfaces using interpPlate.
        Now that shadow waverider uses same coords as cad_export.py:
        X = streamwise, Y = vertical, Z = span

        We build one half (positive Z / right side), then mirror.
        """
        import cadquery as cq
        
        wr = self.waverider
        
        n_le = wr.upper_surface.shape[0]
        n_stream = wr.upper_surface.shape[1]
        center_idx = n_le // 2
        
        # Get right half (positive Z side) - indices from center to end
        # After transform: Z is span, so right half has positive Z values
        upper_half = wr.upper_surface[center_idx:, :, :] * scale
        lower_half = wr.lower_surface[center_idx:, :, :] * scale

        n_half = upper_half.shape[0]

        # Apply point-level LE blunting BEFORE building CAD surfaces
        blunted_span_indices = set()
        if blunting_radius > 0:
            blunting_r = blunting_radius * scale
            max_offset = 5.0 * blunting_r
            n_blunted = 0
            for i in range(n_half):
                le_pt = upper_half[i, 0, :].copy()
                # Upper surface tangent (downstream from LE)
                t_u = upper_half[i, 1, :] - upper_half[i, 0, :]
                t_u_norm = np.linalg.norm(t_u)
                t_u = t_u / t_u_norm if t_u_norm > 1e-12 else np.array([1, 0, 0], dtype=float)
                # Lower surface tangent
                t_l = lower_half[i, 1, :] - lower_half[i, 0, :]
                t_l_norm = np.linalg.norm(t_l)
                t_l = t_l / t_l_norm if t_l_norm > 1e-12 else np.array([1, 0, 0], dtype=float)

                bisector = t_u + t_l
                b_norm = np.linalg.norm(bisector)
                if b_norm < 1e-12:
                    continue
                bisector = bisector / b_norm

                cos_half = np.clip(np.dot(t_u, t_l), -1, 1)
                half_angle = np.arccos(cos_half) / 2.0
                if half_angle < 0.05:  # ~3 degrees minimum opening angle
                    continue

                d_center = blunting_r / np.sin(half_angle)
                center = le_pt + d_center * bisector

                tp_upper = le_pt + np.dot(center - le_pt, t_u) * t_u
                tp_lower = le_pt + np.dot(center - le_pt, t_l) * t_l

                # Skip if offset too large (nearly-flat LE sections)
                offset = np.linalg.norm(tp_upper - le_pt)
                if offset > max_offset:
                    continue

                upper_half[i, 0, :] = tp_upper
                lower_half[i, 0, :] = tp_lower
                blunted_span_indices.add(i)
                n_blunted += 1
            print(f"[Blunting] Applied LE blunting to {n_blunted}/{n_half} span stations (r={blunting_r:.4f})")

        # Extract key curves
        # Leading edge: j=0 (first streamwise station)
        le_upper = upper_half[:, 0, :]   # (n_half, 3)
        le_lower = lower_half[:, 0, :]

        # Trailing edge: j=-1 (last streamwise station)
        te_upper = upper_half[:, -1, :]
        te_lower = lower_half[:, -1, :]

        # Centerline: i=0 (Z≈0)
        centerline_upper = upper_half[0, :, :]  # (n_stream, 3)
        centerline_lower = lower_half[0, :, :]

        # Interior points for upper surface (exclude boundaries)
        # Skip first 2 streamwise points near LE for blunted span stations
        us_points = []
        for i in range(1, n_half - 1):
            start_j = 1
            if i in blunted_span_indices:
                start_j = min(3, n_stream - 1)
            for j in range(start_j, n_stream - 1):
                us_points.append(tuple(upper_half[i, j, :]))
        
        # Interior points for lower surface (same skip logic near LE)
        ls_points = []
        for i in range(1, n_half - 1):
            start_j = 1
            if i in blunted_span_indices:
                start_j = min(3, n_stream - 1)
            for j in range(start_j, n_stream - 1):
                ls_points.append(tuple(lower_half[i, j, :]))
        
        # Build boundary for upper surface
        # Need closed loop: centerline -> TE edge -> wingtip -> LE edge back to start
        workplane = cq.Workplane("XY")
        
        # Symmetry line (centerline from LE to TE)
        sym_start = tuple(centerline_upper[0])    # LE at centerline
        sym_end = tuple(centerline_upper[-1])     # TE at centerline
        
        edge_wire_upper = workplane.moveTo(sym_start[0], sym_start[1])
        edge_wire_upper = edge_wire_upper.lineTo(sym_end[0], sym_end[1])
        
        # Add LE spline and TE spline
        edge_wire_upper = edge_wire_upper.add(cq.Workplane("XY").spline([tuple(x) for x in le_upper]))
        edge_wire_upper = edge_wire_upper.add(cq.Workplane("XY").spline([tuple(x) for x in te_upper]))
        
        # Create upper surface
        upper_surface = cq.Workplane("XY").interpPlate(edge_wire_upper, us_points, 0)
        
        # Build boundary for lower surface
        sym_start_lower = tuple(centerline_lower[0])
        sym_end_lower = tuple(centerline_lower[-1])
        
        edge_wire_lower = workplane.moveTo(sym_start_lower[0], sym_start_lower[1])
        edge_wire_lower = edge_wire_lower.lineTo(sym_end_lower[0], sym_end_lower[1])
        edge_wire_lower = edge_wire_lower.add(cq.Workplane("XY").spline([tuple(x) for x in le_lower]))
        edge_wire_lower = edge_wire_lower.add(cq.Workplane("XY").spline([tuple(x) for x in te_lower]))
        
        lower_surface = cq.Workplane("XY").interpPlate(edge_wire_lower, ls_points, 0)
        
        # Back face (trailing edge closure)
        e1 = cq.Edge.makeSpline([cq.Vector(*tuple(x)) for x in te_lower])
        e2 = cq.Edge.makeSpline([cq.Vector(*tuple(x)) for x in te_upper])
        v1 = cq.Vector(*sym_end)
        v2 = cq.Vector(*sym_end_lower)
        e3 = cq.Edge.makeLine(v1, v2)
        back = cq.Face.makeFromWires(cq.Wire.assembleEdges([e1, e2, e3]))
        
        # Symmetry face (at Z=0)
        v_le = cq.Vector(*sym_start)
        v_te_upper = cq.Vector(*sym_end)
        v_te_lower = cq.Vector(*sym_end_lower)
        e4 = cq.Edge.makeLine(v_le, v_te_upper)
        e5 = cq.Edge.makeLine(v_le, v_te_lower)
        sym_face = cq.Face.makeFromWires(cq.Wire.assembleEdges([e3, e4, e5]))
        
        # Create solid
        shell = cq.Shell.makeShell([
            upper_surface.val(),
            lower_surface.val(),
            back,
            sym_face
        ])
        right_side = cq.Solid.makeSolid(shell)

        # Mirror across XY plane (Z=0) to get left side
        left_side = right_side.mirror(mirrorPlane='XY')

        # Union both halves
        result = cq.Workplane("XY").newObject([right_side]).union(left_side)

        cq.exporters.export(result, filename)
    
    def _export_step_faces(self, filename, scale):
        """
        Export STEP by creating individual quad faces and combining them.
        This creates a surface model (not solid) but it will open correctly in CAD.
        """
        import cadquery as cq
        
        wr = self.waverider
        upper = wr.upper_surface * scale  # (n_le, n_stream, 3)
        lower = wr.lower_surface * scale
        
        n_le = upper.shape[0]
        n_stream = upper.shape[1]
        
        faces = []
        
        # Create faces for upper surface (quads split into triangles or as quads)
        for i in range(n_le - 1):
            for j in range(n_stream - 1):
                try:
                    p00 = cq.Vector(*upper[i, j, :])
                    p01 = cq.Vector(*upper[i, j+1, :])
                    p10 = cq.Vector(*upper[i+1, j, :])
                    p11 = cq.Vector(*upper[i+1, j+1, :])
                    
                    # Create quad face
                    wire = cq.Wire.makePolygon([p00, p01, p11, p10], close=True)
                    face = cq.Face.makeFromWires(wire)
                    faces.append(face)
                except:
                    pass
        
        # Create faces for lower surface
        for i in range(n_le - 1):
            for j in range(n_stream - 1):
                try:
                    p00 = cq.Vector(*lower[i, j, :])
                    p01 = cq.Vector(*lower[i, j+1, :])
                    p10 = cq.Vector(*lower[i+1, j, :])
                    p11 = cq.Vector(*lower[i+1, j+1, :])
                    
                    # Create quad face (reverse winding for outward normal)
                    wire = cq.Wire.makePolygon([p00, p10, p11, p01], close=True)
                    face = cq.Face.makeFromWires(wire)
                    faces.append(face)
                except:
                    pass
        
        # Create base (trailing edge) faces
        for i in range(n_le - 1):
            try:
                # Upper TE points
                u0 = cq.Vector(*upper[i, -1, :])
                u1 = cq.Vector(*upper[i+1, -1, :])
                # Lower TE points
                l0 = cq.Vector(*lower[i, -1, :])
                l1 = cq.Vector(*lower[i+1, -1, :])
                
                wire = cq.Wire.makePolygon([u0, u1, l1, l0], close=True)
                face = cq.Face.makeFromWires(wire)
                faces.append(face)
            except:
                pass
        
        if not faces:
            raise RuntimeError("No faces could be created")
        
        # Try to make a shell and solid
        try:
            shell = cq.Shell.makeShell(faces)
            solid = cq.Solid.makeSolid(shell)
            cq.exporters.export(solid, filename)
            return
        except Exception as e:
            print(f"Solid creation failed: {e}, exporting as shell")
        
        # Try shell only
        try:
            shell = cq.Shell.makeShell(faces)
            cq.exporters.export(shell, filename)
            return
        except Exception as e:
            print(f"Shell creation failed: {e}, exporting as compound")
        
        # Fallback: export as compound of faces
        compound = cq.Compound.makeCompound(faces)
        cq.exporters.export(compound, filename)
    
    def gmsh_mesh(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate waverider first!")
            return
        if not GMSH_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Gmsh not installed")
            return

        # Export STL in meters (SI units)
        temp_stl = tempfile.mktemp(suffix='.stl')
        self.waverider.export_stl(temp_stl)

        fn, _ = QFileDialog.getSaveFileName(self, "Save Mesh", "shadow_waverider_gmsh.stl", "STL (*.stl);;MSH (*.msh)")
        if fn:
            try:
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.merge(temp_stl)
                gmsh.option.setNumber("Mesh.MeshSizeMin", self.gmsh_min.value())
                gmsh.option.setNumber("Mesh.MeshSizeMax", self.gmsh_max.value())
                gmsh.model.mesh.generate(2)

                # Get mesh statistics
                num_nodes = len(gmsh.model.mesh.getNodes()[0])
                num_triangles = len(gmsh.model.mesh.getElementsByType(2)[0])

                gmsh.write(fn)
                gmsh.finalize()
                os.unlink(temp_stl)

                file_size_kb = os.path.getsize(fn) / 1024
                self.last_stl_file = fn

                QMessageBox.information(
                    self, "Mesh Generated",
                    f"Mesh generated successfully!\n\n"
                    f"Triangles: {num_triangles}\n"
                    f"Nodes: {num_nodes}\n"
                    f"File size: {file_size_kb:.1f} KB\n"
                    f"Saved to: {fn}\n\n"
                    f"You can now run PySAGAS analysis."
                )
            except Exception as e:
                gmsh.finalize()
                QMessageBox.critical(self, "Error", str(e))
    
    # === Analysis ===
    def run_analysis(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate waverider first!")
            return
        if not PYSAGAS_AVAILABLE: return

        temp_stl = tempfile.mktemp(suffix='.stl')
        scale = self.scale_spin.value()
        verts, tris = self.waverider.get_mesh()
        verts = verts * scale

        with open(temp_stl, 'w') as f:
            f.write("solid waverider\n")
            for tri in tris:
                v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
                n = np.cross(v1-v0, v2-v0); n = n/np.linalg.norm(n) if np.linalg.norm(n) > 1e-10 else [0,0,1]
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n    endloop\n  endfacet\n")
            f.write("endsolid waverider\n")

        # Show progress bar and disable button
        self.analyze_btn.setEnabled(False)
        self.analysis_progress.setVisible(True)
        self.analysis_progress.setRange(0, 0)  # Indeterminate
        self.analysis_status.setText("Starting analysis...")
        self.analysis_status.setStyleSheet("color: #F59E0B;")

        refs = self.waverider.get_reference_values(scale=scale)
        self.analysis_worker = AnalysisWorker(temp_stl, self.mach_spin.value(), self.aoa_spin.value(),
            self.p_spin.value(), self.t_spin.value(), refs['area'])
        self.analysis_worker.progress.connect(self._on_analysis_progress)
        self.analysis_worker.finished.connect(self.on_analysis_done)
        self.analysis_worker.error.connect(self._on_analysis_error)
        self.analysis_worker.start()

    def _on_analysis_progress(self, message):
        self.analysis_status.setText(message)
        self.info_label.setText(message)

    def _on_analysis_error(self, error_msg):
        self.analysis_progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.analysis_status.setText("Analysis failed")
        self.analysis_status.setStyleSheet("color: #EF4444;")
        QMessageBox.critical(self, "Error", error_msg)
    
    def on_analysis_done(self, r):
        # Hide progress bar, re-enable button
        self.analysis_progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.analysis_status.setText("Analysis complete")
        self.analysis_status.setStyleSheet("color: #4ADE80;")

        result_text = "\n" + "="*60 + "\n"
        result_text += "AERODYNAMIC ANALYSIS RESULTS\n"
        result_text += "="*60 + "\n\n"
        result_text += f"Conditions:\n"
        result_text += f"  Mach number:     {self.mach_spin.value():.2f}\n"
        result_text += f"  Angle of attack: {self.aoa_spin.value():.2f}\n"
        result_text += f"  Pressure:        {self.p_spin.value():.0f} Pa\n"
        result_text += f"  Temperature:     {self.t_spin.value():.2f} K\n\n"
        result_text += f"Coefficients:\n"
        result_text += f"  CL (Lift):   {r['CL']:.6f}\n"
        result_text += f"  CD (Drag):   {r['CD']:.6f}\n"
        result_text += f"  Cm (Moment): {r['Cm']:.6f}\n"
        result_text += f"  L/D Ratio:   {r['L/D']:.3f}\n\n"
        result_text += "="*60 + "\n"

        txt = self.results_text.toPlainText()
        self.results_text.setText(txt + result_text)
        self.info_label.setText(f"L/D = {r['L/D']:.3f}")

        QMessageBox.information(
            self, "Analysis Complete",
            f"Analysis finished successfully!\n\n"
            f"CL = {r['CL']:.6f}\n"
            f"CD = {r['CD']:.6f}\n"
            f"Cm = {r['Cm']:.6f}\n"
            f"L/D = {r['L/D']:.3f}"
        )
    
    # === Design Space ===
    def run_design_space(self):
        order = self.order_combo.currentIndex()
        include_aero = self.ds_include_aero.isChecked() and PYSAGAS_AVAILABLE
        
        params = {
            'mach': self.mach_spin.value(), 'shock_angle': self.shock_spin.value(),
            'poly_order': order + 2, 'n_le': 15, 'n_stream': 15,
            'A2_min': self.ds_a2_min.value(), 'A2_max': self.ds_a2_max.value(), 'n_A2': self.ds_a2_n.value(),
            'A0_min': self.ds_a0_min.value(), 'A0_max': self.ds_a0_max.value(), 'n_A0': self.ds_a0_n.value(),
            'A3_min': self.ds_a3_min.value(), 'A3_max': self.ds_a3_max.value(), 'n_A3': self.ds_a3_n.value(),
            'A0_fixed': self.a0_spin.value(),
            'include_aero': include_aero,
            'pressure': self.p_spin.value(),
            'temperature': self.t_spin.value(),
            'aoa': self.aoa_spin.value()
        }
        total = params['n_A2'] * params['n_A0'] if order == 0 else params['n_A3'] * params['n_A2']
        
        # Warn user if aero is enabled
        if include_aero:
            estimated_time = total * 5  # ~5 seconds per design
            mins = estimated_time // 60
            secs = estimated_time % 60
            reply = QMessageBox.question(self, "Aero Analysis", 
                f"Running PySAGAS for {total} designs.\n"
                f"Estimated time: ~{mins}m {secs}s\n\n"
                f"Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.No:
                return
        
        self.design_worker = DesignSpaceWorker(params)
        self.design_worker.progress.connect(self.on_ds_progress)
        self.design_worker.point_complete.connect(self.on_ds_point)
        self.design_worker.finished.connect(self.on_ds_done)
        self.design_worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        
        self.ds_progress.setVisible(True); self.ds_progress.setRange(0, total)
        self.run_ds_btn.setEnabled(False); self.cancel_ds_btn.setEnabled(True)
        self.design_space_results = []
        self.best_design_group.setVisible(False)
        self.design_worker.start()
    
    def cancel_ds(self):
        if self.design_worker: self.design_worker.cancel()
    
    def on_ds_progress(self, cur, tot, msg):
        self.ds_progress.setValue(cur)
        self.ds_status.setText(f"[{cur}/{tot}] {msg}")
    
    def on_ds_point(self, r):
        self.design_space_results.append(r)
        if len(self.design_space_results) % 10 == 0: self.update_ds_plot()
    
    def on_ds_done(self, results):
        self.ds_progress.setVisible(False)
        self.run_ds_btn.setEnabled(True); self.cancel_ds_btn.setEnabled(False)
        self.design_space_results = results
        valid = sum(1 for r in results if r.get('valid', False))
        self.ds_status.setText(f"✓ {valid}/{len(results)} valid")
        self.update_ds_plot()
        self.update_best_design_panel()
    
    def update_best_design_panel(self):
        """Update the best design info panel"""
        if not self.design_space_results or not PANDAS_AVAILABLE:
            self.best_design_group.setVisible(False)
            return
        
        df = pd.DataFrame(self.design_space_results)
        valid_df = df[df['valid'] == True] if 'valid' in df.columns else df
        
        if len(valid_df) == 0:
            self.best_design_group.setVisible(False)
            return
        
        # Find best by selected color parameter
        color_param = self.ds_color_combo.currentText()
        if color_param not in valid_df.columns:
            color_param = 'volume' if 'volume' in valid_df.columns else 'planform_area'
        
        best_idx = valid_df[color_param].idxmax()
        best = valid_df.loc[best_idx]
        
        # Store for apply button
        self.best_design_params = {
            'A3': best.get('A3', 0),
            'A2': best.get('A2', 0),
            'A0': best.get('A0', 0)
        }
        
        # Update labels
        self.best_a3_label.setText(f"{best.get('A3', 0):.3f}" if 'A3' in best else "N/A")
        self.best_a2_label.setText(f"{best.get('A2', 0):.3f}")
        self.best_a0_label.setText(f"{best.get('A0', 0):.4f}")
        self.best_volume_label.setText(f"{best.get('volume', 0):.4f}")
        self.best_area_label.setText(f"{best.get('planform_area', 0):.4f}")
        self.best_cone_label.setText(f"{best.get('cone_angle', 0):.2f}°")
        
        # Show L/D if available
        if 'L/D' in best:
            self.best_ld_label.setText(f"{best.get('L/D', 0):.3f}")
        else:
            self.best_ld_label.setText("--")
        
        self.best_design_group.setVisible(True)
    
    def update_ds_plot(self):
        if not self.design_space_results or not PANDAS_AVAILABLE: return
        df = pd.DataFrame(self.design_space_results)
        order = self.order_combo.currentIndex()
        x, y = ('A2', 'A0') if order == 0 else ('A3', 'A2')
        
        # Use selected color parameter
        color = self.ds_color_combo.currentText()
        valid_df = df[df['valid'] == True] if 'valid' in df.columns else df
        if color not in valid_df.columns:
            # Fallback if selected metric not available
            color = 'volume' if 'volume' in valid_df.columns else 'planform_area'
        
        self.ds_canvas.plot_design_space(df, x, y, color)
    
    def export_ds_csv(self):
        if not self.design_space_results:
            QMessageBox.warning(self, "Warning", "No results"); return
        if not PANDAS_AVAILABLE:
            QMessageBox.warning(self, "Warning", "Pandas not available"); return
        fn, _ = QFileDialog.getSaveFileName(self, "Save CSV", "design_space.csv", "CSV (*.csv)")
        if fn:
            pd.DataFrame(self.design_space_results).to_csv(fn, index=False)
            QMessageBox.information(self, "Success", f"Saved: {fn}")


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication, QMainWindow
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("SHADOW Waverider Tab (Test)")
    win.setGeometry(100, 100, 1400, 900)
    win.setCentralWidget(ShadowWaveriderTab())
    win.show()
    sys.exit(app.exec_())
