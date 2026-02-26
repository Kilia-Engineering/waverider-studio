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

            # Cone angle depends only on Mach/shock_angle, not polynomial coefficients.
            # Compute once and reuse for all designs to guarantee constancy.
            try:
                ref_wr = create_second_order_waverider(
                    mach=mach, shock_angle=shock_angle,
                    A2=-5.0, A0=-0.1, n_leading_edge=5, n_streamwise=5)
                constant_cone_angle = ref_wr.cone_angle_deg
            except Exception:
                constant_cone_angle = None

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
                        if constant_cone_angle is not None:
                            result['cone_angle'] = constant_cone_angle
                        results.append(result)
                        self.point_complete.emit(result)
            else:
                A3_range = np.linspace(self.params['A3_min'], self.params['A3_max'], self.params['n_A3'])
                A2_range = np.linspace(self.params['A2_min'], self.params['A2_max'], self.params['n_A2'])
                A0_range = np.linspace(self.params['A0_min'], self.params['A0_max'], self.params['n_A0'])
                total = len(A3_range) * len(A2_range) * len(A0_range)
                current = 0

                for A3 in A3_range:
                    for A2 in A2_range:
                        for A0 in A0_range:
                            if self._is_cancelled:
                                self.finished.emit(results)
                                return
                            current += 1
                            self.progress.emit(current, total, f"A3={A3:.1f}, A2={A2:.2f}, A0={A0:.3f}")
                            result = self._eval_3rd(mach, shock_angle, A3, A2, A0, include_aero)
                            if constant_cone_angle is not None:
                                result['cone_angle'] = constant_cone_angle
                            results.append(result)
                            self.point_complete.emit(result)
            
            self.finished.emit(results)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")
    
    def _run_pysagas(self, wr, mach):
        """Run PySAGAS analysis on a SHADOW waverider.

        Uses waverider's computed planform_area and mac as reference values
        for proper coefficient non-dimensionalization.
        """
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

            # Use waverider's computed reference values for proper non-dimensionalization
            A_ref = getattr(wr, 'planform_area', 1.0) or 1.0
            c_ref = getattr(wr, 'mac', 1.0) or 1.0
            CL, CD, Cm = solver.flow_result.coefficients(A_ref=A_ref, c_ref=c_ref)
            LD = CL / CD if abs(CD) > 1e-10 else 0

            # Cleanup
            try:
                os.unlink(temp_stl)
            except:
                pass

            return {'CL': float(CL), 'CD': float(CD), 'Cm': float(Cm), 'L/D': float(LD)}
        except Exception as e:
            return {'aero_error': str(e)}
    
    def _run_stability(self, wr, mach):
        """Run stability derivative analysis on a SHADOW waverider."""
        try:
            from stability_analysis import analyze_waverider_stability
            pressure = self.params.get('pressure', 101325)
            temperature = self.params.get('temperature', 288.15)
            aoa = self.params.get('aoa', 0)
            vtk_dir = self.params.get('vtk_dir', None)

            result = analyze_waverider_stability(
                wr, mach, pressure=pressure, temperature=temperature,
                alpha_deg=aoa, save_vtk_prefix=vtk_dir)

            # Return subset of keys for design space (exclude perturbed data)
            return {
                'CL': result['CL'], 'CD': result['CD'], 'Cm': result['Cm'],
                'L/D': result['L/D'],
                'Cl_beta': result['Cl_beta'],
                'Cn_beta': result['Cn_beta'],
                'Cm_alpha': result['Cm_alpha'],
                'pitch_stable': result['pitch_stable'],
                'yaw_stable': result['yaw_stable'],
                'roll_stable': result['roll_stable'],
                'fully_stable': result['fully_stable'],
            }
        except Exception as e:
            return {'stability_error': str(e)}

    def _eval_2nd(self, mach, shock_angle, A2, A0, include_aero=False):
        try:
            wr = create_second_order_waverider(mach=mach, shock_angle=shock_angle,
                A2=A2, A0=A0, n_leading_edge=self.params.get('n_le', 15),
                n_streamwise=self.params.get('n_stream', 15))
            result = {'A2': A2, 'A0': A0, 'cone_angle': wr.cone_angle_deg,
                   'planform_area': wr.planform_area, 'volume': wr.volume,
                   'vol_efficiency': (wr.volume ** (2.0/3.0)) / wr.planform_area if wr.planform_area > 1e-6 else 0.0,
                   'mac': wr.mac, 'valid': True}

            include_stability = self.params.get('include_stability', False)
            if include_stability:
                stab = self._run_stability(wr, mach)
                result.update(stab)
            elif include_aero:
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
                   'planform_area': wr.planform_area, 'volume': wr.volume,
                   'vol_efficiency': (wr.volume ** (2.0/3.0)) / wr.planform_area if wr.planform_area > 1e-6 else 0.0,
                   'mac': wr.mac, 'valid': True}

            include_stability = self.params.get('include_stability', False)
            if include_stability:
                stab = self._run_stability(wr, mach)
                result.update(stab)
            elif include_aero:
                aero = self._run_pysagas(wr, mach)
                result.update(aero)

            return result
        except Exception as e:
            return {'A3': A3, 'A2': A2, 'A0': A0, 'valid': False, 'error': str(e)}


class GradientOptWorker(QThread):
    """Worker thread for gradient-based optimization using ShadowOptimizer."""
    progress = pyqtSignal(int, dict)
    finished_signal = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, mach, shock_angle, poly_order, x0, bounds,
                 objective='CL/CD', method='SLSQP', maxiter=50,
                 stability_constrained=False, save_vtk=True,
                 pressure=101325.0, temperature=288.15, alpha_deg=0.0,
                 mesh_min=0.005, mesh_max=0.05, save_geometry_vtk=True):
        super().__init__()
        self.mach = mach
        self.shock_angle = shock_angle
        self.poly_order = poly_order
        self.x0 = np.array(x0, dtype=float)
        self.bounds = bounds
        self.objective = objective
        self.method = method
        self.maxiter = maxiter
        self.stability_constrained = stability_constrained
        self.save_vtk = save_vtk
        self.pressure = pressure
        self.temperature = temperature
        self.alpha_deg = alpha_deg
        self.mesh_min = mesh_min
        self.mesh_max = mesh_max
        self.save_geometry_vtk = save_geometry_vtk

    def run(self):
        try:
            from pysagas_optimizer import ShadowOptimizer

            optimizer = ShadowOptimizer(
                mach=self.mach,
                shock_angle=self.shock_angle,
                poly_order=self.poly_order,
                pressure=self.pressure,
                temperature=self.temperature,
                alpha_deg=self.alpha_deg,
                objective=self.objective,
                method=self.method,
                stability_constrained=self.stability_constrained,
                save_vtk=self.save_vtk,
                output_dir='optimization_results',
                verbose=False,
                mesh_min=self.mesh_min,
                mesh_max=self.mesh_max,
                save_geometry_vtk=self.save_geometry_vtk
            )

            # Wire progress callback to emit Qt signal
            optimizer.progress_callback = lambda iteration, entry: \
                self.progress.emit(iteration, entry)

            result = optimizer.optimize(
                x0=self.x0,
                bounds=self.bounds,
                maxiter=self.maxiter
            )

            # Remove non-serializable items before emitting
            result.pop('waverider', None)
            result.pop('scipy_result', None)
            # Convert sensitivity DataFrames to dicts for Qt signal
            sens = result.get('sensitivity')
            if sens is not None:
                try:
                    result['sensitivity'] = {
                        'f_sens': sens['f_sens'].to_dict(),
                        'm_sens': sens['m_sens'].to_dict(),
                        'parameters': sens['parameters'],
                    }
                except Exception:
                    result.pop('sensitivity', None)
            self.finished_signal.emit(result)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class ShadowWaveriderCanvas(FigureCanvas):
    """Canvas for 3D visualization"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8), facecolor='#0A0A0A')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1A1A1A')
        super().__init__(self.fig)
        self.setParent(parent)
        
    def plot_waverider(self, wr, show_upper=True, show_lower=True, show_le=True, show_cg=True, show_info=True):
        self.ax.clear()
        # Remove previous info text
        if hasattr(self, '_info_text') and self._info_text is not None:
            try:
                self._info_text.remove()
            except Exception:
                pass
            self._info_text = None
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
        if show_info:
            self._draw_info_panel(wr)
        self.fig.tight_layout()
        self.draw()

    def _draw_info_panel(self, wr):
        """Draw waverider info overlay on the 3D view."""
        if wr is None:
            return

        vol_eff = (wr.volume ** (2.0/3.0)) / wr.planform_area if wr.planform_area > 0 else 0.0

        info = (
            "WAVERIDER INFO\n"
            f"  Method          Shadow (Cone-Derived)\n"
            f"  Mach            {wr.mach:.1f}\n"
            f"  Shock \u03b2         {wr.shock_angle:.1f}\u00b0\n"
            f"  Cone \u03b8c         {wr.cone_angle_deg:.2f}\u00b0\n"
            f"  Post-shock M    {wr.post_shock_mach:.2f}\n"
            f"  Length           {wr.length:.4f} m\n"
            f"  Planform Area    {wr.planform_area:.4f} m\u00b2\n"
            f"  Volume           {wr.volume:.6f} m\u00b3\n"
            f"  Vol Efficiency   {vol_eff:.6f}\n"
            f"  CG              [{wr.cg[0]:.4f}, {wr.cg[1]:.4f}, {wr.cg[2]:.4f}]"
        )

        self._info_text = self.fig.text(
            0.02, 0.98, info,
            transform=self.fig.transFigure,
            fontsize=8, fontfamily='monospace',
            verticalalignment='top',
            color='white',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='#1A1A1A',
                edgecolor='#D97706',
                alpha=0.85
            )
        )

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
    point_clicked = pyqtSignal(dict)  # Emits clicked design's result dict

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8), facecolor='#0A0A0A')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1A1A1A')
        self.colorbar = None  # Track colorbar to remove it on updates
        self._scatter = None  # Main scatter artist for pick events
        self._valid_df = None  # DataFrame for looking up clicked points
        self._click_highlight = None  # Highlight ring for clicked point
        self._x_param = None
        self._y_param = None
        self._z_param = None   # Z-axis parameter for 3D mode
        self._is_3d = False    # Track current plot dimensionality
        super().__init__(self.fig)
        self.setParent(parent)
        self.mpl_connect('pick_event', self._on_pick)
        
    def plot_design_space(self, df, x_param, y_param, color_param, z_param=None):
        need_3d = z_param is not None

        # Remove old colorbar if it exists
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except:
                pass
            self.colorbar = None

        # Recreate axes (2D↔3D axes can't be converted in-place)
        self.fig.clear()
        if need_3d:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self._is_3d = True
        else:
            self.ax = self.fig.add_subplot(111)
            self._is_3d = False
        self.ax.set_facecolor('#1A1A1A')

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

        # Store references for click-to-inspect
        self._valid_df = valid_df.reset_index(drop=True)
        self._x_param = x_param
        self._y_param = y_param
        self._z_param = z_param
        self._scatter = None
        self._click_highlight = None

        # Special handling for "stability" color mode
        if color_param == 'stability' and 'fully_stable' in valid_df.columns:
            self._plot_stability_overlay(valid_df, x_param, y_param, z_param)
        elif color_param in valid_df.columns:
            if need_3d:
                self._scatter = self.ax.scatter(
                    valid_df[x_param], valid_df[y_param], valid_df[z_param],
                    c=valid_df[color_param], cmap='viridis', s=80, alpha=0.8,
                    edgecolors='white', linewidths=0.5, picker=True)
            else:
                self._scatter = self.ax.scatter(
                    valid_df[x_param], valid_df[y_param],
                    c=valid_df[color_param], cmap='viridis', s=80, alpha=0.8,
                    edgecolors='white', linewidths=0.5, picker=True)
            self.colorbar = self.fig.colorbar(
                self._scatter, ax=self.ax, label=color_param,
                shrink=0.7 if need_3d else 1.0)

            # Mark best point
            best_idx = valid_df[color_param].idxmax()
            best = valid_df.loc[best_idx]
            if need_3d:
                self.ax.scatter([best[x_param]], [best[y_param]], [best[z_param]],
                    c='gold', s=300, marker='*', edgecolors='black', linewidths=2,
                    zorder=10, label=f'Best: {best[color_param]:.4f}')
            else:
                self.ax.scatter([best[x_param]], [best[y_param]],
                    c='gold', s=300, marker='*', edgecolors='black', linewidths=2,
                    zorder=10, label=f'Best: {best[color_param]:.4f}')
        else:
            if need_3d:
                self._scatter = self.ax.scatter(
                    valid_df[x_param], valid_df[y_param], valid_df[z_param],
                    s=80, alpha=0.8, picker=True)
            else:
                self._scatter = self.ax.scatter(
                    valid_df[x_param], valid_df[y_param], s=80, alpha=0.8, picker=True)

        # Plot invalid points
        if invalid_df is not None and len(invalid_df) > 0:
            if need_3d:
                self.ax.scatter(invalid_df[x_param], invalid_df[y_param], invalid_df[z_param],
                    c='red', marker='x', s=50, alpha=0.5, label='Invalid')
            else:
                self.ax.scatter(invalid_df[x_param], invalid_df[y_param],
                    c='red', marker='x', s=50, alpha=0.5, label='Invalid')

        # Axis labels and styling
        self.ax.set_xlabel(x_param, fontsize=12, color='#FFFFFF')
        self.ax.set_ylabel(y_param, fontsize=12, color='#FFFFFF')
        if need_3d:
            self.ax.set_zlabel(z_param, fontsize=12, color='#FFFFFF')
            self.ax.set_title(f'{color_param} vs ({x_param}, {y_param}, {z_param})',
                              fontsize=14, color='#FFFFFF')
            # Dark theme for 3D panes
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False
            self.ax.xaxis.pane.set_edgecolor('#333333')
            self.ax.yaxis.pane.set_edgecolor('#333333')
            self.ax.zaxis.pane.set_edgecolor('#333333')
            self.ax.tick_params(axis='x', colors='#888888')
            self.ax.tick_params(axis='y', colors='#888888')
            self.ax.tick_params(axis='z', colors='#888888')
            self.ax.grid(True, alpha=0.2)
        else:
            self.ax.set_title(f'{color_param} vs ({x_param}, {y_param})',
                              fontsize=14, color='#FFFFFF')
            self.ax.tick_params(colors='#888888')
            self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        self.fig.tight_layout()
        self.draw()

    def _plot_stability_overlay(self, df, x_param, y_param, z_param=None):
        """Plot design space colored by stability criteria (thesis Figs 5.5-5.18)."""
        use_3d = z_param is not None

        # Count how many stability criteria each design meets
        stab_count = (
            df['pitch_stable'].astype(int) +
            df['yaw_stable'].astype(int) +
            df['roll_stable'].astype(int)
        )

        # Unstable (0 criteria)
        unstable = df[stab_count == 0]
        if len(unstable) > 0:
            args = (unstable[x_param], unstable[y_param])
            if use_3d: args = args + (unstable[z_param],)
            self.ax.scatter(*args,
                          c='#EF4444', s=60, alpha=0.6, marker='o',
                          edgecolors='#991B1B', linewidths=0.5, label='Unstable (0/3)')

        # Partially stable (1-2 criteria)
        partial = df[(stab_count >= 1) & (stab_count <= 2)]
        if len(partial) > 0:
            args = (partial[x_param], partial[y_param])
            if use_3d: args = args + (partial[z_param],)
            self.ax.scatter(*args,
                          c='#F59E0B', s=80, alpha=0.7, marker='s',
                          edgecolors='#92400E', linewidths=0.5, label='Partial (1-2/3)')

        # Fully stable (all 3 criteria)
        stable = df[stab_count == 3]
        if len(stable) > 0:
            args = (stable[x_param], stable[y_param])
            if use_3d: args = args + (stable[z_param],)
            self.ax.scatter(*args,
                          c='#10B981', s=100, alpha=0.9, marker='D',
                          edgecolors='#065F46', linewidths=0.5, label='Stable (3/3)')

            # Mark best CL/CD among fully stable
            if 'L/D' in stable.columns and len(stable) > 0:
                best_idx = stable['L/D'].idxmax()
                best = stable.loc[best_idx]
                star_args = ([best[x_param]], [best[y_param]])
                if use_3d: star_args = star_args + ([best[z_param]],)
                self.ax.scatter(*star_args, c='gold', s=300, marker='*',
                              edgecolors='black', linewidths=2, zorder=10,
                              label=f'Best stable CL/CD: {best["L/D"]:.3f}')

    def _on_pick(self, event):
        """Handle click on scatter plot point."""
        if self._scatter is None or event.artist != self._scatter:
            return
        if self._valid_df is None or len(event.ind) == 0:
            return
        ind = event.ind[0]
        if ind >= len(self._valid_df):
            return
        row = self._valid_df.iloc[ind]

        # Remove previous highlight
        if self._click_highlight is not None:
            try:
                self._click_highlight.remove()
            except Exception:
                pass
        # Draw cyan ring around clicked point (2D or 3D)
        if self._is_3d and self._z_param is not None:
            self._click_highlight = self.ax.scatter(
                [row[self._x_param]], [row[self._y_param]], [row[self._z_param]],
                s=200, facecolors='none', edgecolors='cyan', linewidths=2.5, zorder=9
            )
        else:
            self._click_highlight = self.ax.scatter(
                [row[self._x_param]], [row[self._y_param]],
                s=200, facecolors='none', edgecolors='cyan', linewidths=2.5, zorder=9
            )
        self.draw()

        # Emit the clicked point's data
        self.point_clicked.emit(row.to_dict())


class ShadowWaveriderTab(QWidget):
    """Main tab for cone-derived waverider design"""

    # Mapping: objective name → (dict_key, display_label, higher_is_better)
    _OBJ_MAP = {
        'CL/CD':          ('L/D',            'CL/CD',          True),
        'L/D':            ('L/D',            'CL/CD',          True),
        '-CD':            ('CD',             'CD',             False),
        'CL':             ('CL',             'CL',             True),
        'Vol Efficiency': ('vol_efficiency', 'Vol Efficiency', True),
    }

    waverider_generated = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.waverider = None
        self.last_stl_file = None
        self.design_space_results = None
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
        left_layout.addWidget(self._create_min_thickness_group())
        left_layout.addWidget(self._create_generate_group())
        left_layout.addWidget(self._create_export_group())
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
        self.show_info = QCheckBox("Info"); self.show_info.setChecked(True)
        opts.addWidget(self.show_upper); opts.addWidget(self.show_lower)
        opts.addWidget(self.show_le); opts.addWidget(self.show_cg)
        opts.addWidget(self.show_info)
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

        # Gradient Optimization tab
        opt_widget = self._create_gradient_opt_widget()
        right_panel.addTab(opt_widget, "Gradient Opt")

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
        self.a3_spin.setToolTip(
            "Cubic coefficient (3rd order only)\n"
            "Controls S-shaped inflection of the LE planform.\n"
            "Positive: wingtip curves upward\n"
            "Negative: wingtip curves downward more")
        layout.addWidget(self.a3_spin, 1, 1)

        layout.addWidget(QLabel("A₂:"), 2, 0)
        self.a2_spin = QDoubleSpinBox()
        self.a2_spin.setRange(-50, 50); self.a2_spin.setValue(-2.0); self.a2_spin.setDecimals(2)
        self.a2_spin.setToolTip(
            "Quadratic coefficient \u2014 controls LE sweep curvature.\n"
            "More negative: sharper sweep, narrower body, thicker vehicle\n"
            "Less negative: wider body, less sweep, risk of surface intersection\n"
            "Typical range: -1 to -10")
        layout.addWidget(self.a2_spin, 2, 1)

        layout.addWidget(QLabel("A₀:"), 3, 0)
        self.a0_spin = QDoubleSpinBox()
        self.a0_spin.setRange(-1, 0); self.a0_spin.setValue(-0.15); self.a0_spin.setDecimals(3)
        self.a0_spin.setToolTip(
            "Y-intercept \u2014 vertical position of the nose tip.\n"
            "More negative: nose sits deeper on shock cone, more volume/thickness\n"
            "Less negative: shallower nose, thinner vehicle, risk of surface crossing\n"
            "Typical range: -0.05 to -0.3")
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
        self.scale_spin.setToolTip("Additional scale factor for export (1.0 = SI meters).\nSTEP mm conversion is applied automatically.")
        layout.addWidget(self.scale_spin, 3, 1)
        
        group.setLayout(layout)
        return group
    
    def _create_blunting_group(self):
        group = QGroupBox("Leading Edge Blunting")
        layout = QGridLayout()

        self.blunting_check = QCheckBox("Enable LE fillet")
        self.blunting_check.setToolTip(
            "Apply a fillet to the leading edge during STEP export.\n"
            "Uses OpenCASCADE BRepFilletAPI with optional variable radius.")
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

        layout.addWidget(QLabel("Spanwise:"), 2, 0)
        self.blunting_sweep_combo = QComboBox()
        self.blunting_sweep_combo.addItems([
            "Uniform radius",
            "Sweep-scaled"])
        self.blunting_sweep_combo.setToolTip(
            "Uniform: Same fillet radius across the entire span\n"
            "Sweep-scaled: Radius tapers toward wingtip based on\n"
            "  local sweep angle (R_tip = R * cos(sweep))")
        self.blunting_sweep_combo.setEnabled(False)
        layout.addWidget(self.blunting_sweep_combo, 2, 1)

        self.blunting_preview_btn = QPushButton("Show LE Preview")
        self.blunting_preview_btn.setToolTip("Visualize blunted vs original LE on the 3D view.\nBlunting is applied automatically during STEP export.")
        self.blunting_preview_btn.clicked.connect(self._preview_blunting)
        self.blunting_preview_btn.setEnabled(False)
        self.blunting_preview_btn.setStyleSheet(
            "QPushButton { background-color: #1A1A1A; color: #F59E0B; border: 1px solid #78350F; padding: 5px; }"
            "QPushButton:hover { background-color: #78350F; color: #FFFFFF; }"
            "QPushButton:disabled { color: #555555; border-color: #333333; }"
        )
        layout.addWidget(self.blunting_preview_btn, 3, 0, 1, 2)

        group.setLayout(layout)
        return group

    def _create_min_thickness_group(self):
        group = QGroupBox("Minimum Nose Thickness")
        layout = QGridLayout()

        self.min_thickness_check = QCheckBox("Enforce minimum thickness")
        self.min_thickness_check.setToolTip(
            "Ensure the nose region has a minimum thickness so that\n"
            "the exported CAD solid is not infinitely thin at the tip.\n"
            "Recommended when using LE blunting.")
        self.min_thickness_check.stateChanged.connect(self._on_min_thickness_toggled)
        layout.addWidget(self.min_thickness_check, 0, 0, 1, 2)

        layout.addWidget(QLabel("Thickness (% L):"), 1, 0)
        self.min_thickness_spin = QDoubleSpinBox()
        self.min_thickness_spin.setRange(0.1, 10.0)
        self.min_thickness_spin.setValue(1.0)
        self.min_thickness_spin.setSingleStep(0.1)
        self.min_thickness_spin.setDecimals(1)
        self.min_thickness_spin.setSuffix(" %")
        self.min_thickness_spin.setToolTip(
            "Minimum thickness as a percentage of vehicle length.\n"
            "Default 1% — increase if nose filleting still fails.")
        self.min_thickness_spin.setEnabled(False)
        layout.addWidget(self.min_thickness_spin, 1, 1)

        group.setLayout(layout)
        return group

    def _on_min_thickness_toggled(self, state):
        self.min_thickness_spin.setEnabled(bool(state))

    def _on_blunting_toggled(self, state):
        enabled = bool(state)
        self.blunting_radius_spin.setEnabled(enabled)
        self.blunting_sweep_combo.setEnabled(enabled)
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
            # Taper radius near nose: full at wingtip, near-zero at center
            n_le = wr.upper_surface.shape[0]
            n_stream = wr.upper_surface.shape[1]
            center_idx = n_le // 2  # nose/center index
            blunted_points = []

            # Use a point well downstream for robust tangent estimation
            # (cone-derived upper is flat, lower curves gradually)
            j_tan = max(2, min(n_stream // 4, n_stream - 1))

            for i in range(n_le):
                le_pt = wr.upper_surface[i, 0, :]

                # Taper: full radius everywhere, quick taper only near nose
                dist_from_center = abs(i - center_idx)
                max_dist = max(center_idx, n_le - 1 - center_idx)
                frac = dist_from_center / max_dist if max_dist > 0 else 1.0
                # Full radius for 85%+ of the LE, taper in last 15% near nose
                taper_zone = 0.15
                if frac < taper_zone:
                    taper = frac / taper_zone  # 0→1 within taper zone
                else:
                    taper = 1.0
                local_radius = radius * taper

                if local_radius < 1e-6:
                    blunted_points.append(le_pt)
                    continue

                # Upper tangent (downstream from LE)
                t_u = wr.upper_surface[i, j_tan, :] - wr.upper_surface[i, 0, :]
                n = np.linalg.norm(t_u)
                t_u = t_u / n if n > 1e-12 else np.array([1, 0, 0], dtype=float)

                t_l = wr.lower_surface[i, j_tan, :] - wr.lower_surface[i, 0, :]
                n = np.linalg.norm(t_l)
                t_l = t_l / n if n > 1e-12 else np.array([1, 0, 0], dtype=float)

                bisector = t_u + t_l
                b_norm = np.linalg.norm(bisector)
                if b_norm > 1e-12:
                    bisector = bisector / b_norm
                else:
                    bisector = np.array([1, 0, 0], dtype=float)

                cos_half = np.clip(np.dot(t_u, t_l), -1, 1)
                half_angle = np.arccos(cos_half) / 2.0

                # Skip if surfaces are nearly tangent
                if half_angle < 0.05:
                    blunted_points.append(le_pt)
                    continue

                d_center = local_radius / np.sin(half_angle)
                d_center = min(d_center, local_radius * 5)
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
                arc_mid = center + local_radius * v_mid
                blunted_points.append(arc_mid)

            blunted_le = np.array(blunted_points)

            # Draw on 3D canvas using same axis mapping as plot_waverider:
            # Z(span) -> plot X, X(streamwise) -> plot Y, Y(vertical) -> plot Z
            ax = self.canvas_3d.ax
            for line in list(ax.lines):
                if hasattr(line, '_blunting_preview'):
                    line.remove()

            line_orig, = ax.plot(
                original_le[:, 2], original_le[:, 0], original_le[:, 1],
                'r--', linewidth=1.5, label='Original LE')
            line_orig._blunting_preview = True

            line_blunt, = ax.plot(
                blunted_le[:, 2], blunted_le[:, 0], blunted_le[:, 1],
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

        self.half_vehicle_check = QCheckBox("Half vehicle (right side only)")
        self.half_vehicle_check.setToolTip(
            "Export only the right half (positive Z) without mirroring.\n"
            "Useful for CFD meshing with symmetry boundary conditions.")

        layout.addWidget(stl_btn, 0, 0); layout.addWidget(tri_btn, 0, 1)
        layout.addWidget(step_btn, 1, 0, 1, 2)
        layout.addWidget(self.half_vehicle_check, 2, 0, 1, 2)

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
            self.ds_include_aero.setToolTip("Run PySAGAS for each design (slower but gives CL/CD)")
        gl.addWidget(self.ds_include_aero, 3, 0, 1, 2)

        # Stability analysis checkbox
        self.ds_include_stability = QCheckBox("Include Stability")
        self.ds_include_stability.setEnabled(PYSAGAS_AVAILABLE)
        if not PYSAGAS_AVAILABLE:
            self.ds_include_stability.setToolTip("PySAGAS not available")
        else:
            self.ds_include_stability.setToolTip(
                "Compute stability derivatives via perturbation (5x slower than aero-only)")
        self.ds_include_stability.toggled.connect(
            lambda checked: self.ds_include_aero.setChecked(True) if checked else None)
        gl.addWidget(self.ds_include_stability, 3, 2)

        # Color-by selector
        gl.addWidget(QLabel("Color by:"), 3, 3)
        self.ds_color_combo = QComboBox()
        self.ds_color_combo.addItems([
            "volume", "planform_area", "vol_efficiency", "CL/CD", "CL", "CD",
            "Cm_alpha", "Cl_beta", "Cn_beta", "stability"])
        self.ds_color_combo.currentTextChanged.connect(self.update_ds_plot)
        gl.addWidget(self.ds_color_combo, 3, 4)

        # Aero flow conditions (used when Include Aero is checked)
        aero_row = QHBoxLayout()
        aero_row.addWidget(QLabel("AoA:"))
        self.aoa_spin = QDoubleSpinBox()
        self.aoa_spin.setRange(-20, 20); self.aoa_spin.setValue(0)
        aero_row.addWidget(self.aoa_spin)
        aero_row.addWidget(QLabel("P (Pa):"))
        self.p_spin = QDoubleSpinBox()
        self.p_spin.setRange(100, 1e7); self.p_spin.setValue(101325); self.p_spin.setDecimals(0)
        aero_row.addWidget(self.p_spin)
        aero_row.addWidget(QLabel("T (K):"))
        self.t_spin = QDoubleSpinBox()
        self.t_spin.setRange(100, 500); self.t_spin.setValue(288.15)
        aero_row.addWidget(self.t_spin)
        gl.addLayout(aero_row, 4, 0, 1, 5)

        btn_layout = QHBoxLayout()
        self.run_ds_btn = QPushButton("▶ Run")
        self.run_ds_btn.clicked.connect(self.run_design_space)
        self.cancel_ds_btn = QPushButton("⏹ Cancel")
        self.cancel_ds_btn.clicked.connect(self.cancel_ds)
        self.cancel_ds_btn.setEnabled(False)
        btn_layout.addWidget(self.run_ds_btn); btn_layout.addWidget(self.cancel_ds_btn)
        gl.addLayout(btn_layout, 5, 0, 1, 5)
        
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
        
        best_layout.addWidget(QLabel("CL/CD:"), 2, 0)
        self.best_ld_label = QLabel("--")
        self.best_ld_label.setStyleSheet("font-weight: bold; color: #EF4444; font-size: 14px;")
        best_layout.addWidget(self.best_ld_label, 2, 1)
        
        # Stability info (hidden until stability analysis is run)
        self.best_stability_header = QLabel("Stable:")
        self.best_stability_label = QLabel("--")
        self.best_stability_label.setStyleSheet("font-weight: bold; color: #10B981;")
        best_layout.addWidget(self.best_stability_header, 2, 2)
        best_layout.addWidget(self.best_stability_label, 2, 3, 1, 2)
        self.best_stability_header.setVisible(False)
        self.best_stability_label.setVisible(False)

        self.best_stable_ld_header = QLabel("Best Stable CL/CD:")
        self.best_stable_ld_label = QLabel("--")
        self.best_stable_ld_label.setStyleSheet("font-weight: bold; color: #10B981;")
        best_layout.addWidget(self.best_stable_ld_header, 3, 0, 1, 2)
        best_layout.addWidget(self.best_stable_ld_label, 3, 2, 1, 3)
        self.best_stable_ld_header.setVisible(False)
        self.best_stable_ld_label.setVisible(False)

        # Apply best design button
        apply_best_btn = QPushButton("Apply to Main Panel")
        apply_best_btn.clicked.connect(self.apply_best_design)
        apply_best_btn.setStyleSheet("background-color: #F59E0B; color: #0A0A0A; font-weight: bold; padding: 5px;")
        best_layout.addWidget(apply_best_btn, 4, 0, 1, 3)

        # Show best design button (useful after clicking a different point)
        show_best_btn = QPushButton("Show Best Design")
        show_best_btn.clicked.connect(self.update_best_design_panel)
        show_best_btn.setStyleSheet("padding: 5px; color: #F59E0B; border: 1px solid #F59E0B;")
        best_layout.addWidget(show_best_btn, 4, 3, 1, 3)
        
        self.best_design_group.setLayout(best_layout)
        self.best_design_group.setVisible(False)  # Hidden until we have results
        layout.addWidget(self.best_design_group)
        
        self.ds_canvas = DesignSpaceCanvas()
        self.ds_canvas.point_clicked.connect(self._on_ds_point_clicked)
        self.ds_toolbar = NavigationToolbar(self.ds_canvas, widget)
        layout.addWidget(self.ds_toolbar)
        layout.addWidget(self.ds_canvas)
        
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self.export_ds_csv)
        layout.addWidget(export_btn)
        
        return widget

    def _create_gradient_opt_widget(self):
        """Create the gradient-based optimization panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        group = QGroupBox("Gradient-Based Optimization (SLSQP/COBYLA)")
        gl = QGridLayout()

        # Objective
        gl.addWidget(QLabel("Objective:"), 0, 0)
        self.opt_objective = QComboBox()
        self.opt_objective.addItems(["CL/CD", "-CD", "CL", "Vol Efficiency"])
        gl.addWidget(self.opt_objective, 0, 1)

        # Method
        gl.addWidget(QLabel("Method:"), 0, 2)
        self.opt_method = QComboBox()
        self.opt_method.addItems(["SLSQP", "COBYLA", "Nelder-Mead"])
        gl.addWidget(self.opt_method, 0, 3)

        # Max iterations
        gl.addWidget(QLabel("Max Iter:"), 1, 0)
        self.opt_maxiter = QSpinBox()
        self.opt_maxiter.setRange(5, 500)
        self.opt_maxiter.setValue(50)
        gl.addWidget(self.opt_maxiter, 1, 1)

        # Stability constraints
        self.opt_stability = QCheckBox("Stability Constraints")
        self.opt_stability.setToolTip("Enforce Cm_alpha<0, Cn_beta>0, Cl_beta<0")
        self.opt_stability.setEnabled(PYSAGAS_AVAILABLE)
        gl.addWidget(self.opt_stability, 1, 2, 1, 2)

        # Save VTK
        self.opt_save_vtk = QCheckBox("Save Pressure VTK")
        self.opt_save_vtk.setToolTip("Save pressure VTK files at each iteration")
        self.opt_save_vtk.setChecked(True)
        gl.addWidget(self.opt_save_vtk, 2, 0)

        # Save geometry VTK for animation
        self.opt_save_geom_vtk = QCheckBox("Save Geometry VTK (animation)")
        self.opt_save_geom_vtk.setToolTip(
            "Save waverider mesh as VTK at each iteration for ParaView animation")
        self.opt_save_geom_vtk.setChecked(True)
        gl.addWidget(self.opt_save_geom_vtk, 2, 2, 1, 2)

        group.setLayout(gl)
        layout.addWidget(group)

        # Gmsh Mesh Settings group
        mesh_group = QGroupBox("Gmsh Mesh Settings")
        mg = QGridLayout()

        mg.addWidget(QLabel("Min Element Size [m]:"), 0, 0)
        self.opt_mesh_min = QDoubleSpinBox()
        self.opt_mesh_min.setRange(0.00001, 10.0)
        self.opt_mesh_min.setValue(0.005)
        self.opt_mesh_min.setSingleStep(0.001)
        self.opt_mesh_min.setDecimals(5)
        self.opt_mesh_min.setToolTip("Minimum triangle edge length in meters")
        mg.addWidget(self.opt_mesh_min, 0, 1)

        mg.addWidget(QLabel("Max Element Size [m]:"), 1, 0)
        self.opt_mesh_max = QDoubleSpinBox()
        self.opt_mesh_max.setRange(0.0001, 100.0)
        self.opt_mesh_max.setValue(0.05)
        self.opt_mesh_max.setSingleStep(0.005)
        self.opt_mesh_max.setDecimals(5)
        self.opt_mesh_max.setToolTip("Maximum triangle edge length in meters")
        mg.addWidget(self.opt_mesh_max, 1, 1)

        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))
        coarse_btn = QPushButton("Coarse")
        coarse_btn.clicked.connect(lambda: (
            self.opt_mesh_min.setValue(0.01), self.opt_mesh_max.setValue(0.1)))
        preset_layout.addWidget(coarse_btn)
        medium_btn = QPushButton("Medium")
        medium_btn.clicked.connect(lambda: (
            self.opt_mesh_min.setValue(0.005), self.opt_mesh_max.setValue(0.05)))
        preset_layout.addWidget(medium_btn)
        fine_btn = QPushButton("Fine")
        fine_btn.clicked.connect(lambda: (
            self.opt_mesh_min.setValue(0.002), self.opt_mesh_max.setValue(0.02)))
        preset_layout.addWidget(fine_btn)
        preset_layout.addStretch()
        mg.addLayout(preset_layout, 2, 0, 1, 2)

        mesh_group.setLayout(mg)
        layout.addWidget(mesh_group)

        # Bounds group
        bounds_group = QGroupBox("Design Variable Bounds")
        gl2 = QGridLayout()

        # A2 bounds (tighter defaults to avoid degenerate geometries)
        gl2.addWidget(QLabel("A2 bounds:"), 0, 0)
        self.opt_a2_lo = QDoubleSpinBox()
        self.opt_a2_lo.setRange(-50, 0); self.opt_a2_lo.setValue(-15)
        gl2.addWidget(self.opt_a2_lo, 0, 1)
        self.opt_a2_hi = QDoubleSpinBox()
        self.opt_a2_hi.setRange(-50, 0); self.opt_a2_hi.setValue(-0.5)
        gl2.addWidget(self.opt_a2_hi, 0, 2)

        # A0 bounds (tighter defaults to avoid CL sign flips)
        gl2.addWidget(QLabel("A0 bounds:"), 1, 0)
        self.opt_a0_lo = QDoubleSpinBox()
        self.opt_a0_lo.setRange(-1, 0); self.opt_a0_lo.setValue(-0.4); self.opt_a0_lo.setDecimals(3)
        gl2.addWidget(self.opt_a0_lo, 1, 1)
        self.opt_a0_hi = QDoubleSpinBox()
        self.opt_a0_hi.setRange(-1, 0); self.opt_a0_hi.setValue(-0.02); self.opt_a0_hi.setDecimals(3)
        gl2.addWidget(self.opt_a0_hi, 1, 2)

        bounds_group.setLayout(gl2)
        layout.addWidget(bounds_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.opt_run_btn = QPushButton("Run Optimization")
        self.opt_run_btn.clicked.connect(self.run_gradient_opt)
        self.opt_run_btn.setEnabled(PYSAGAS_AVAILABLE)
        self.opt_run_btn.setStyleSheet(
            "background-color: #10B981; color: white; font-weight: bold; padding: 8px;")
        btn_layout.addWidget(self.opt_run_btn)

        self.opt_cancel_btn = QPushButton("Cancel")
        self.opt_cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.opt_cancel_btn)

        self.opt_anim_btn = QPushButton("Generate Animation GIF")
        self.opt_anim_btn.setToolTip(
            "Generate a GIF showing waverider shape evolution from the last optimization")
        self.opt_anim_btn.clicked.connect(self._generate_animation)
        self.opt_anim_btn.setEnabled(False)
        btn_layout.addWidget(self.opt_anim_btn)

        layout.addLayout(btn_layout)

        # Progress
        self.opt_progress = QLabel("Ready")
        layout.addWidget(self.opt_progress)

        # Results log
        self.opt_log = QTextEdit()
        self.opt_log.setReadOnly(True)
        self.opt_log.setFont(QFont("Courier", 9))
        self.opt_log.setMaximumHeight(300)
        layout.addWidget(self.opt_log)

        # Convergence plot
        self.opt_canvas = FigureCanvas(Figure(figsize=(8, 4), facecolor='#0A0A0A'))
        self.opt_ax = self.opt_canvas.figure.add_subplot(111)
        self.opt_ax.set_facecolor('#1A1A1A')
        layout.addWidget(self.opt_canvas)

        layout.addStretch()
        return widget

    def run_gradient_opt(self):
        """Launch gradient-based optimization in a worker thread."""
        if not PYSAGAS_AVAILABLE:
            QMessageBox.warning(self, "Error", "PySAGAS is not available")
            return

        order = self.order_combo.currentIndex() + 2
        mach = self.mach_spin.value()
        shock = self.shock_spin.value()

        # Build initial point from current panel values
        if order == 2:
            x0 = [self.a2_spin.value(), self.a0_spin.value()]
            bounds = [(self.opt_a2_lo.value(), self.opt_a2_hi.value()),
                      (self.opt_a0_lo.value(), self.opt_a0_hi.value())]
        else:
            x0 = [self.a3_spin.value(), self.a2_spin.value(), self.a0_spin.value()]
            bounds = [(-50, 50),
                      (self.opt_a2_lo.value(), self.opt_a2_hi.value()),
                      (self.opt_a0_lo.value(), self.opt_a0_hi.value())]

        self.opt_run_btn.setEnabled(False)
        self.opt_log.clear()
        self._opt_objective_name = self.opt_objective.currentText()
        self.opt_log.append(f"Starting {self.opt_method.currentText()} optimization...")
        self.opt_log.append(f"  Mach={mach}, shock={shock}, order={order}")
        self.opt_log.append(f"  Objective: maximize {self._opt_objective_name}")
        self.opt_log.append(f"  x0 = {x0}")
        self.opt_log.append("")

        # Run in thread
        self._opt_worker = GradientOptWorker(
            mach=mach, shock_angle=shock, poly_order=order,
            x0=x0, bounds=bounds,
            objective=self.opt_objective.currentText(),
            method=self.opt_method.currentText(),
            maxiter=self.opt_maxiter.value(),
            stability_constrained=self.opt_stability.isChecked(),
            save_vtk=self.opt_save_vtk.isChecked(),
            pressure=self.p_spin.value(),
            temperature=self.t_spin.value(),
            alpha_deg=self.aoa_spin.value(),
            mesh_min=self.opt_mesh_min.value(),
            mesh_max=self.opt_mesh_max.value(),
            save_geometry_vtk=self.opt_save_geom_vtk.isChecked())
        self._opt_worker.progress.connect(self._on_opt_progress)
        self._opt_worker.finished_signal.connect(self._on_opt_done)
        self._opt_worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self._opt_worker.start()

    def _on_opt_progress(self, iteration, entry):
        """Handle gradient optimization progress updates."""
        obj_name = getattr(self, '_opt_objective_name', 'CL/CD')
        dict_key, label, _ = self._OBJ_MAP.get(obj_name, ('L/D', 'CL/CD', True))
        val = entry.get(dict_key, 0)

        self.opt_progress.setText(
            f"Eval {iteration}: {label}={val:.4f}, "
            f"obj={entry.get('objective', 0):.6f}")
        self.opt_log.append(
            f"Eval {iteration}: {label}={val:.4f} "
            f"CD={entry.get('CD', 0):.6f}")

        # Update convergence plot
        if hasattr(self, '_opt_history'):
            self._opt_history.append(entry)
        else:
            self._opt_history = [entry]

        self.opt_ax.clear()
        iters = [e['iteration'] for e in self._opt_history]
        vals = [e.get(dict_key, 0) for e in self._opt_history]
        self.opt_ax.plot(iters, vals, 'o-', color='#10B981', linewidth=2)
        self.opt_ax.set_xlabel('Iteration', color='#FFFFFF')
        self.opt_ax.set_ylabel(label, color='#FFFFFF')
        self.opt_ax.set_title('Convergence', color='#FFFFFF')
        self.opt_ax.tick_params(colors='#888888')
        self.opt_ax.grid(True, alpha=0.3)
        self.opt_canvas.draw()

    def _on_opt_done(self, result):
        """Handle gradient optimization completion."""
        self.opt_run_btn.setEnabled(True)
        self._opt_history = []

        obj_name = getattr(self, '_opt_objective_name', 'CL/CD')
        dict_key, label, _ = self._OBJ_MAP.get(obj_name, ('L/D', 'CL/CD', True))

        if result.get('success', False):
            x_opt = result['x_optimal']
            final = result.get('final_evaluation', {})
            self.opt_log.append(f"\nOptimization CONVERGED")
            self.opt_log.append(f"  Optimal: {x_opt}")
            self.opt_log.append(f"  {label} = {final.get(dict_key, 'N/A')}")
            self.opt_log.append(f"  CL = {final.get('CL', 'N/A')}")
            self.opt_log.append(f"  CD = {final.get('CD', 'N/A')}")
            if 'Cm_alpha' in final:
                self.opt_log.append(f"  Cm_alpha = {final.get('Cm_alpha', 'N/A')}")
                self.opt_log.append(f"  Cn_beta  = {final.get('Cn_beta', 'N/A')}")
                self.opt_log.append(f"  Cl_beta  = {final.get('Cl_beta', 'N/A')}")

            # Ask user if they want to apply optimal design
            obj_val = final.get(dict_key, 0)
            try:
                obj_str = f"{float(obj_val):.4f}"
            except (ValueError, TypeError):
                obj_str = str(obj_val)
            reply = QMessageBox.question(self, "Optimization Complete",
                f"{label} = {obj_str}\n\n"
                f"Apply optimal design to main panel?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                if len(x_opt) == 2:
                    self.a2_spin.setValue(x_opt[0])
                    self.a0_spin.setValue(x_opt[1])
                else:
                    self.a3_spin.setValue(x_opt[0])
                    self.a2_spin.setValue(x_opt[1])
                    self.a0_spin.setValue(x_opt[2])
                self.generate()
        else:
            self.opt_log.append(f"\nOptimization FAILED: {result.get('message', 'unknown')}")

            # Show best design found before failure
            best = result.get('best_found')
            if best is not None:
                self.opt_log.append(f"\nBest design found before failure:")
                best_x = best.get('x', None)
                if best_x is not None:
                    x_str = ", ".join(f"{v:.4f}" for v in best_x)
                    self.opt_log.append(f"  x = [{x_str}]")
                try:
                    self.opt_log.append(f"  {label} = {float(best.get(dict_key, 0)):.4f}")
                except (ValueError, TypeError):
                    self.opt_log.append(f"  {label} = N/A")
                self.opt_log.append(f"  CL  = {best.get('CL', 'N/A')}")
                self.opt_log.append(f"  CD  = {best.get('CD', 'N/A')}")

            self.opt_log.append(f"\nSuggestions:")
            self.opt_log.append(f"  - Try 'Nelder-Mead' method (gradient-free, more robust)")
            self.opt_log.append(f"  - Narrow the A2/A0 bounds")
            self.opt_log.append(f"  - Try a different initial point")
            if best is not None and best.get('x') is not None:
                self.opt_log.append(f"  - Use the best-found design as new starting point")

            # Offer to apply best-found design anyway
            if best is not None and best.get('x') is not None:
                best_val = best.get(dict_key, 0)
                try:
                    best_str = f"{float(best_val):.4f}"
                except (ValueError, TypeError):
                    best_str = str(best_val)
                reply = QMessageBox.question(self, "Optimization Failed",
                    f"Optimization did not converge, but found:\n"
                    f"{label} = {best_str}\n\n"
                    f"Apply best-found design to main panel?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    best_x = best['x']
                    if len(best_x) == 2:
                        self.a2_spin.setValue(best_x[0])
                        self.a0_spin.setValue(best_x[1])
                    else:
                        self.a3_spin.setValue(best_x[0])
                        self.a2_spin.setValue(best_x[1])
                        self.a0_spin.setValue(best_x[2])
                    self.generate()

        # Log GIF and sensitivity results
        gif_path = result.get('gif_path')
        if gif_path and os.path.exists(gif_path):
            self.opt_log.append(f"\nAnimation saved: {gif_path}")

        sens = result.get('sensitivity')
        if sens is not None:
            self.opt_log.append(f"\nShape sensitivities (dF/dp):")
            try:
                f_sens = sens['f_sens']
                params = sens.get('parameters', [])
                for param in params:
                    dFx = f_sens.get('dFx/dp', {}).get(param, 0)
                    dFy = f_sens.get('dFy/dp', {}).get(param, 0)
                    dFz = f_sens.get('dFz/dp', {}).get(param, 0)
                    self.opt_log.append(
                        f"  {param}: dFx={dFx:.4e}, dFy={dFy:.4e}, dFz={dFz:.4e}")
            except Exception:
                self.opt_log.append("  (see console for details)")
            self.opt_log.append(
                f"  Sensitivity VTK: optimization_results/optimized_sensitivities.vtu")

        # Enable animation button for regeneration
        self.opt_anim_btn.setEnabled(True)

        self.opt_progress.setText("Done")

    def _generate_animation(self):
        """Generate or regenerate animation GIF from last optimization."""
        history_path = os.path.join('optimization_results', 'convergence_history.json')
        if not os.path.exists(history_path):
            QMessageBox.warning(self, "No History",
                                "No optimization history found. Run optimization first.")
            return

        try:
            from animation_utils import generate_gif_from_history_file
            gif_path = generate_gif_from_history_file(
                history_json_path=history_path,
                mach=self.mach_spin.value(),
                shock_angle=self.shock_spin.value(),
                poly_order=self.order_combo.currentIndex() + 2,
            )
            if gif_path:
                self.opt_log.append(f"\nAnimation saved: {gif_path}")
                QMessageBox.information(self, "Animation Complete",
                                        f"GIF saved to:\n{gif_path}")
            else:
                QMessageBox.warning(self, "Animation Failed",
                                    "Could not generate animation. Check console for details.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Animation generation failed:\n{e}")

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
        
        self.info_label.setText("\u2713 Applied best design parameters")

    def _on_ds_point_clicked(self, result):
        """Update the best design panel to show a clicked point's data."""
        self.best_design_group.setTitle("\U0001f4cd Selected Design")
        self.best_a3_label.setText(f"{result.get('A3', 0):.3f}" if 'A3' in result else "N/A")
        self.best_a2_label.setText(f"{result.get('A2', 0):.3f}")
        self.best_a0_label.setText(f"{result.get('A0', 0):.4f}")
        self.best_volume_label.setText(f"{result.get('volume', 0):.4f}")
        self.best_area_label.setText(f"{result.get('planform_area', 0):.4f}")
        self.best_cone_label.setText(f"{result.get('cone_angle', 0):.2f}\u00b0")
        if 'L/D' in result and result.get('L/D') is not None:
            try:
                self.best_ld_label.setText(f"{float(result['L/D']):.3f}")
            except (ValueError, TypeError):
                self.best_ld_label.setText("--")
        else:
            self.best_ld_label.setText("--")
        # Update apply button to use clicked design's params
        self.best_design_params = {
            'A3': result.get('A3', 0),
            'A2': result.get('A2', 0),
            'A0': result.get('A0', 0)
        }
        self.best_design_group.setVisible(True)

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

            # Check surface health and warn user if surfaces are too thin
            health = self.waverider.check_surface_health()
            if not health['healthy']:
                msg = "\n".join(health['suggestions'])
                QMessageBox.warning(self, "Surface Intersection Warning",
                                    f"The generated geometry has surface "
                                    f"intersection issues:\n\n{msg}")

            self.waverider_generated.emit(self.waverider)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.info_label.setText(f"✗ {str(e)}")
    
    def update_view(self):
        self.canvas_3d.plot_waverider(self.waverider, self.show_upper.isChecked(),
            self.show_lower.isChecked(), self.show_le.isChecked(), self.show_cg.isChecked(),
            self.show_info.isChecked())
    
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
                # STEP files use millimeters (OCCT convention);
                # geometry is in meters → multiply by 1000
                scale = self.scale_spin.value() * 1000.0
                blunting_radius = 0.0
                sweep_scaled = False
                if self.blunting_check.isChecked():
                    blunting_radius = self.blunting_radius_spin.value()
                    sweep_scaled = (self.blunting_sweep_combo.currentIndex() == 1)
                min_thickness = 0.0
                if self.min_thickness_check.isChecked():
                    wr = self.waverider
                    x_vals = wr.upper_surface[:, :, 0]
                    veh_length = float(x_vals.max() - x_vals.min())
                    pct = self.min_thickness_spin.value()
                    min_thickness = veh_length * pct / 100.0
                half_only = getattr(self, 'half_vehicle_check', None) and self.half_vehicle_check.isChecked()
                print(f"[Shadow Export] method='{method}', blunting_radius={blunting_radius}, "
                      f"sweep_scaled={sweep_scaled}, min_thickness={min_thickness}, "
                      f"scale={scale}, half_only={half_only}")
                if method == methods[0]:
                    self._export_step_nurbs(fn, scale,
                                            blunting_radius=blunting_radius,
                                            sweep_scaled=sweep_scaled,
                                            min_thickness=min_thickness,
                                            half_only=half_only)
                else:
                    self._export_step_faces(fn, scale, min_thickness=min_thickness)
                QMessageBox.information(self, "Success", f"Saved: {fn}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"STEP export failed:\n{str(e)}")
    
    def _export_step_nurbs(self, filename, scale, blunting_radius=0.0,
                           sweep_scaled=False, min_thickness=0.0,
                           half_only=False):
        """
        Export STEP with smooth NURBS surfaces using interpPlate.
        Coords: X = streamwise, Y = vertical, Z = span.

        Builds one half (positive Z / right side), optionally applies LE
        fillet, then mirrors to create the full vehicle.
        """
        import cadquery as cq

        wr = self.waverider
        upper_surf = wr.upper_surface
        lower_surf = wr.lower_surface

        # Apply minimum thickness enforcement before building CAD
        if min_thickness > 0:
            from waverider_generator.cad_export import enforce_min_thickness_arrays
            upper_surf, lower_surf = enforce_min_thickness_arrays(
                upper_surf, lower_surf, min_thickness)

        n_le = upper_surf.shape[0]
        center_idx = n_le // 2

        # Get right half (positive Z side) — stay in SI meters for all CAD ops
        upper_half = upper_surf[center_idx:, :, :]
        lower_half = lower_surf[center_idx:, :, :]
        n_half = upper_half.shape[0]

        # Extract curves from arrays
        le_curve = upper_half[:, 0, :]
        centerline_upper = upper_half[0, :, :]
        centerline_lower = lower_half[0, :, :]
        te_upper = upper_half[:, -1, :]
        te_lower = lower_half[:, -1, :]
        upper_streams = [upper_half[i, :, :] for i in range(n_half)]
        lower_streams = [lower_half[i, :, :] for i in range(n_half)]

        # Build 4-face NURBS solid (in SI meters)
        from waverider_generator.cad_export import build_waverider_solid
        right_side = build_waverider_solid(
            upper_streams, lower_streams, le_curve,
            centerline_upper, centerline_lower,
            te_upper, te_lower)

        # Scale from SI meters to mm for STEP export
        right_side = right_side.scale(scale)

        # Apply post-solid LE fillet if blunting is enabled
        if blunting_radius > 0:
            print(f"[Shadow STEP] LE fillet: radius={blunting_radius * scale:.4f}mm, "
                  f"sweep_scaled={sweep_scaled}")
            from waverider_generator.cad_export import _apply_le_fillet
            le_pts = le_curve * scale
            right_side = _apply_le_fillet(
                right_side, blunting_radius * scale, le_pts,
                nose_cap=False, sweep_scaled=sweep_scaled)

        if half_only:
            result = cq.Workplane("XY").newObject([right_side])
        else:
            # Mirror across XY plane (Z=0) to get left side
            left_side = right_side.mirror(mirrorPlane='XY')
            result = cq.Workplane("XY").newObject([right_side]).union(left_side)

        cq.exporters.export(result, filename)
    
    def _export_step_faces(self, filename, scale, min_thickness=0.0):
        """
        Export STEP by creating individual quad faces and combining them.
        This creates a surface model (not solid) but it will open correctly in CAD.
        """
        import cadquery as cq

        wr = self.waverider
        upper_surf = wr.upper_surface
        lower_surf = wr.lower_surface

        if min_thickness > 0:
            from waverider_generator.cad_export import enforce_min_thickness_arrays
            upper_surf, lower_surf = enforce_min_thickness_arrays(
                upper_surf, lower_surf, min_thickness)

        upper = upper_surf * scale  # (n_le, n_stream, 3)
        lower = lower_surf * scale
        
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
    
    # === Design Space ===
    def run_design_space(self):
        order = self.order_combo.currentIndex()
        include_aero = self.ds_include_aero.isChecked() and PYSAGAS_AVAILABLE
        include_stability = self.ds_include_stability.isChecked() and PYSAGAS_AVAILABLE

        params = {
            'mach': self.mach_spin.value(), 'shock_angle': self.shock_spin.value(),
            'poly_order': order + 2, 'n_le': 15, 'n_stream': 15,
            'A2_min': self.ds_a2_min.value(), 'A2_max': self.ds_a2_max.value(), 'n_A2': self.ds_a2_n.value(),
            'A0_min': self.ds_a0_min.value(), 'A0_max': self.ds_a0_max.value(), 'n_A0': self.ds_a0_n.value(),
            'A3_min': self.ds_a3_min.value(), 'A3_max': self.ds_a3_max.value(), 'n_A3': self.ds_a3_n.value(),
            'A0_fixed': self.a0_spin.value(),
            'include_aero': include_aero,
            'include_stability': include_stability,
            'pressure': self.p_spin.value(),
            'temperature': self.t_spin.value(),
            'aoa': self.aoa_spin.value()
        }
        total = params['n_A2'] * params['n_A0'] if order == 0 else params['n_A3'] * params['n_A2'] * params['n_A0']

        # Warn user about time estimates
        if include_stability:
            estimated_time = total * 25  # ~25 seconds per design (5 PySAGAS runs)
            mins = estimated_time // 60
            secs = estimated_time % 60
            reply = QMessageBox.question(self, "Stability Analysis",
                f"Running stability analysis for {total} designs.\n"
                f"(5 PySAGAS runs per design for perturbation)\n"
                f"Estimated time: ~{mins}m {secs}s\n\n"
                f"Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.No:
                return
        elif include_aero:
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
        """Update the best design info panel (resets to best design)"""
        self.best_design_group.setTitle("\u2b50 Best Design Found")
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

        # Show stability info if available
        if 'fully_stable' in best:
            stab_text = []
            if best.get('pitch_stable', False):
                stab_text.append("Pitch")
            if best.get('yaw_stable', False):
                stab_text.append("Yaw")
            if best.get('roll_stable', False):
                stab_text.append("Roll")
            stab_str = ", ".join(stab_text) if stab_text else "None"
            self.best_stability_label.setText(stab_str)
            self.best_stability_label.setStyleSheet(
                f"font-weight: bold; color: {'#10B981' if best.get('fully_stable') else '#F59E0B'};")
            self.best_stability_label.setVisible(True)
            self.best_stability_header.setVisible(True)

            # Also find best L/D among fully stable designs
            valid_stable = valid_df[valid_df.get('fully_stable', False) == True]
            if len(valid_stable) > 0 and 'L/D' in valid_stable.columns:
                best_stable_idx = valid_stable['L/D'].idxmax()
                best_stable = valid_stable.loc[best_stable_idx]
                self.best_stable_ld_label.setText(
                    f"{best_stable['L/D']:.3f} (A2={best_stable.get('A2', 0):.2f})")
                self.best_stable_ld_label.setVisible(True)
                self.best_stable_ld_header.setVisible(True)
            else:
                self.best_stable_ld_label.setText("No fully stable designs")
                self.best_stable_ld_label.setVisible(True)
                self.best_stable_ld_header.setVisible(True)
        else:
            self.best_stability_label.setVisible(False)
            self.best_stability_header.setVisible(False)
            self.best_stable_ld_label.setVisible(False)
            self.best_stable_ld_header.setVisible(False)

        self.best_design_group.setVisible(True)
    
    def update_ds_plot(self):
        if not self.design_space_results or not PANDAS_AVAILABLE: return
        df = pd.DataFrame(self.design_space_results)
        order = self.order_combo.currentIndex()
        if order == 0:
            x, y, z = 'A2', 'A0', None       # 2nd order: 2D scatter
        else:
            x, y, z = 'A3', 'A2', 'A0'       # 3rd order: 3D scatter

        # Use selected color parameter
        # Map display names to DataFrame column names
        _DISPLAY_TO_COL = {'CL/CD': 'L/D'}
        color = self.ds_color_combo.currentText()
        color_col = _DISPLAY_TO_COL.get(color, color)  # Translate for DataFrame indexing
        valid_df = df[df['valid'] == True] if 'valid' in df.columns else df
        if color_col not in valid_df.columns:
            # Fallback if selected metric not available
            color_col = 'volume' if 'volume' in valid_df.columns else 'planform_area'

        self.ds_canvas.plot_design_space(df, x, y, color_col, z_param=z)
    
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
