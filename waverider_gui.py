#!/usr/bin/env python3
"""
Interactive Waverider Design GUI
Allows real-time parameter adjustment and 3D visualization
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QGroupBox, QGridLayout, QSlider, QDoubleSpinBox,
                             QMessageBox, QTabWidget, QCheckBox, QSpinBox,
                             QProgressBar, QTextEdit, QFileDialog, QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Add project path
# Ensure local GUI directory is on sys.path (robust across launch locations)
GUI_ROOT = os.path.dirname(os.path.abspath(__file__))
if GUI_ROOT not in sys.path:
    sys.path.insert(0, GUI_ROOT)

from waverider_generator.generator import waverider as wr
from waverider_generator.cad_export import to_CAD


# Import plot windows for proper Qt-compatible plotting
try:
    from plot_windows import AerodeckPlotWindow
    AERODECK_PLOT_AVAILABLE = True
except ImportError:
    AERODECK_PLOT_AVAILABLE = False


def calculate_waverider_volume(waverider_obj):
    """
    Calculate internal volume of waverider using trapezoidal rule integration.
    
    Uses cross-sectional area integration along the streamwise direction.
    This is the most accurate method for arbitrary waverider shapes.
    
    Parameters
    ----------
    waverider_obj : WaveriderGenerator
        Waverider geometry object with upper_surface_streams and lower_surface_streams
        
    Returns
    -------
    volume : float
        Internal volume in m³
        
    Notes
    -----
    The waverider geometry only stores ONE HALF (y >= 0) due to symmetry.
    The shoelace formula gives the area of half the cross-section, so we
    multiply by 2 to get the full volume.
    """
    # Get streamlines
    upper_streams = waverider_obj.upper_surface_streams
    lower_streams = waverider_obj.lower_surface_streams
    
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
        
        # Shoelace formula for polygon area (this is HALF the cross-section due to symmetry)
        area = 0.5 * abs(np.dot(z_points, np.roll(y_points, 1)) - 
                         np.dot(y_points, np.roll(z_points, 1)))
        
        areas.append(area)
    
    if len(areas) < 2:
        return 0.0
    
    # Integrate using trapezoidal rule (gives half-volume)
    try:
        half_volume = np.trapezoid(areas, x_positions)
    except AttributeError:
        # Fallback for older numpy versions
        half_volume = np.trapz(areas, x_positions)
    
    # Full volume (symmetric waverider - multiply by 2)
    return 2.0 * abs(half_volume)

# Import reference area calculator
try:
    from reference_area_calculator import (
        calculate_planform_area_from_waverider,
        calculate_wetted_area_from_waverider
    )
    AREA_CALC_AVAILABLE = True
except ImportError:
    AREA_CALC_AVAILABLE = False

# Import PySAGAS for aerodynamic analysis
try:
    from pysagas.cfd import OPM
    from pysagas.flow import FlowState
    from pysagas.geometry.parsers import MeshIO
    PYSAGAS_AVAILABLE = True
except ImportError:
    PYSAGAS_AVAILABLE = False


# Import optimization tab
from optimization_tab import OptimizationTab

# Import surrogate optimization tab
try:
    from surrogate_tab import SurrogateTab
    SURROGATE_AVAILABLE = True
except ImportError as e:
    print(f"Surrogate tab not available: {e}")
    SURROGATE_AVAILABLE = False
    
# Import off-design surrogate tab
try:
    from offdesign_surrogate_tab import OffDesignSurrogateTab
    OFFDESIGN_SURROGATE_AVAILABLE = True
except ImportError as e:
    print(f"Off-design surrogate tab not available: {e}")
    OFFDESIGN_SURROGATE_AVAILABLE = False
    
# Import multi-mach hunter tab
try:
    from multimach_hunter_tab import MultiMachHunterTab
    MULTIMACH_HUNTER_AVAILABLE = True
except ImportError as e:
    print(f"Multi-mach hunter tab not available: {e}")
    MULTIMACH_HUNTER_AVAILABLE = False
    
# Import cone-waverider tab
try:
    from shadow_waverider_tab import ShadowWaveriderTab
    CONE_WAVERIDER_AVAILABLE = True
except ImportError as e:
    print(f"Cone waverider tab not available: {e}")
    CONE_WAVERIDER_AVAILABLE = False
    
# Import Claude assistant tab
try:
    from claude_assistant_tab import ClaudeAssistantTab
    CLAUDE_ASSISTANT_AVAILABLE = True
except ImportError as e:
    print(f"Claude assistant tab not available: {e}")
    CLAUDE_ASSISTANT_AVAILABLE = False



class WaveriderCanvas(FigureCanvas):
    """Canvas for 3D waverider visualization"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Initialize plot
        self.ax.set_xlabel('X (Streamwise) [m]')
        self.ax.set_ylabel('Y (Vertical) [m]')
        self.ax.set_zlabel('Z (Spanwise) [m]')
        self.ax.set_title('Waverider 3D Visualization')
        
    def plot_waverider(self, waverider_obj, show_upper=True, show_lower=True, 
                      show_le=True, show_wireframe=False):
        """Plot the waverider geometry"""
        self.ax.clear()
        
        # Upper surface - RIGHT HALF (positive Z)
        if show_upper:
            if show_wireframe:
                self.ax.plot_wireframe(waverider_obj.upper_surface_x, 
                                     waverider_obj.upper_surface_y, 
                                     waverider_obj.upper_surface_z,
                                     color='blue', alpha=0.3, linewidth=0.5)
            else:
                self.ax.plot_surface(waverider_obj.upper_surface_x, 
                                   waverider_obj.upper_surface_y, 
                                   waverider_obj.upper_surface_z,
                                   color='cyan', alpha=0.8, edgecolor='none',
                                   shade=True, antialiased=True)
            
            # MIRROR - LEFT HALF (negative Z)
            if show_wireframe:
                self.ax.plot_wireframe(waverider_obj.upper_surface_x, 
                                     waverider_obj.upper_surface_y, 
                                     -waverider_obj.upper_surface_z,
                                     color='blue', alpha=0.3, linewidth=0.5)
            else:
                self.ax.plot_surface(waverider_obj.upper_surface_x, 
                                   waverider_obj.upper_surface_y, 
                                   -waverider_obj.upper_surface_z,
                                   color='cyan', alpha=0.8, edgecolor='none',
                                   shade=True, antialiased=True)
        
        # Lower surface - create surfaces from streamlines
        if show_lower:
            # Convert streamlines to surfaces
            n_streams = len(waverider_obj.lower_surface_streams)
            for i in range(n_streams - 1):
                stream1 = waverider_obj.lower_surface_streams[i]
                stream2 = waverider_obj.lower_surface_streams[i + 1]
                
                # Handle different lengths by interpolating
                n_points = min(len(stream1), len(stream2))
                
                if len(stream1) != len(stream2):
                    from scipy.interpolate import interp1d
                    t1 = np.linspace(0, 1, len(stream1))
                    t2 = np.linspace(0, 1, len(stream2))
                    t_common = np.linspace(0, 1, n_points)
                    
                    stream1_x = interp1d(t1, stream1[:, 0])(t_common)
                    stream1_y = interp1d(t1, stream1[:, 1])(t_common)
                    stream1_z = interp1d(t1, stream1[:, 2])(t_common)
                    stream2_x = interp1d(t2, stream2[:, 0])(t_common)
                    stream2_y = interp1d(t2, stream2[:, 1])(t_common)
                    stream2_z = interp1d(t2, stream2[:, 2])(t_common)
                else:
                    stream1_x = stream1[:, 0]
                    stream1_y = stream1[:, 1]
                    stream1_z = stream1[:, 2]
                    stream2_x = stream2[:, 0]
                    stream2_y = stream2[:, 1]
                    stream2_z = stream2[:, 2]
                
                # Create surface between two streamlines - RIGHT HALF
                x_surf = np.array([stream1_x, stream2_x])
                y_surf = np.array([stream1_y, stream2_y])
                z_surf = np.array([stream1_z, stream2_z])
                
                if show_wireframe:
                    self.ax.plot_wireframe(x_surf, y_surf, z_surf,
                                         color='orange', alpha=0.3, linewidth=0.5)
                else:
                    self.ax.plot_surface(x_surf, y_surf, z_surf,
                                       color='orange', alpha=0.8, edgecolor='none',
                                       shade=True, antialiased=True)
                
                # MIRROR - LEFT HALF
                if show_wireframe:
                    self.ax.plot_wireframe(x_surf, y_surf, -z_surf,
                                         color='orange', alpha=0.3, linewidth=0.5)
                else:
                    self.ax.plot_surface(x_surf, y_surf, -z_surf,
                                       color='orange', alpha=0.8, edgecolor='none',
                                       shade=True, antialiased=True)
        
        # Leading edge - RIGHT and LEFT halves
        if show_le:
            le = waverider_obj.leading_edge
            self.ax.plot(le[:, 0], le[:, 1], le[:, 2], 
                       'g-', linewidth=3, label='Leading Edge')
            # Mirror
            self.ax.plot(le[:, 0], le[:, 1], -le[:, 2], 
                       'g-', linewidth=3)
        
        # Set labels and equal aspect
        self.ax.set_xlabel('X (Streamwise) [m]')
        self.ax.set_ylabel('Y (Vertical) [m]')
        self.ax.set_zlabel('Z (Spanwise) [m]')
        self.ax.set_title('Waverider 3D Visualization')
        
        # Set equal aspect ratio
        max_range = np.array([
            waverider_obj.upper_surface_x.max() - waverider_obj.upper_surface_x.min(),
            waverider_obj.upper_surface_y.max() - waverider_obj.upper_surface_y.min(),
            waverider_obj.upper_surface_z.max() - waverider_obj.upper_surface_z.min()
        ]).max() / 2.0
        
        mid_x = (waverider_obj.upper_surface_x.max() + waverider_obj.upper_surface_x.min()) * 0.5
        mid_y = (waverider_obj.upper_surface_y.max() + waverider_obj.upper_surface_y.min()) * 0.5
        mid_z = (waverider_obj.upper_surface_z.max() + waverider_obj.upper_surface_z.min()) * 0.5
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        self.ax.legend()
        self.draw()


class BasePlaneCanvas(FigureCanvas):
    """Canvas for base plane visualization"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def plot_base_plane(self, waverider_obj):
        """Plot the base plane view"""
        self.ax.clear()
        
        # Get data
        inters = waverider_obj.local_intersections_us
        inters = np.vstack([np.array([0, waverider_obj.height]), inters, waverider_obj.us_P3])
        
        shockwave = np.column_stack([waverider_obj.z_local_shockwave, 
                                     waverider_obj.y_local_shockwave])
        shockwave = np.vstack([np.array([0, 0]), shockwave, waverider_obj.s_P4])
        
        lower_surface = waverider_obj.lower_surface_streams
        lower_surface = np.vstack([stream[-1, :] for stream in lower_surface])
        z_ls = lower_surface[:, 2]
        y_ls = lower_surface[:, 1] + waverider_obj.height
        
        # Plot symmetry line
        self.ax.plot([0, 0], [0, waverider_obj.height], 'b-', linewidth=2)
        
        # Plot osculating planes
        for i, (point1, point2) in enumerate(zip(inters, shockwave)):
            x_values = [point1[0], point2[0]]
            y_values = [point1[1], point2[1]]
            label = 'Osculating Planes' if i == 0 else None
            self.ax.plot(x_values, y_values, 'b-', alpha=0.3, label=label)
        
        # Plot curves
        self.ax.plot(shockwave[:, 0], shockwave[:, 1], 'go--', linewidth=2, label="Shockwave")
        self.ax.plot(inters[:, 0], inters[:, 1], 'r-o', linewidth=2, label="Upper Surface")
        self.ax.plot(z_ls, y_ls, '-ok', linewidth=2, label="Lower Surface")
        self.ax.plot(waverider_obj.us_P3[0], waverider_obj.us_P3[1], 'bo', 
                    markersize=10, label="Tip")
        
        self.ax.set_xlabel('z [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.set_title(f'Base Plane [X1={waverider_obj.X1:.2f}, X2={waverider_obj.X2:.2f}, '
                         f'X3={waverider_obj.X3:.2f}, X4={waverider_obj.X4:.2f}]')
        self.ax.set_aspect('equal')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.draw()


class LECanvas(FigureCanvas):
    """Canvas for leading edge visualization"""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def plot_leading_edge(self, waverider_obj):
        """Plot the leading edge shape (top view)"""
        self.ax.clear()
        
        le = waverider_obj.leading_edge
        base_point = [waverider_obj.length, 0]
        
        self.ax.plot(le[:, 2], le[:, 0], 'b-', linewidth=2, label='Leading Edge')
        self.ax.plot([le[0, 2], base_point[1]], [le[0, 0], base_point[0]], 
                    '--k', linewidth=1.5, label='Symmetry Plane')
        self.ax.plot([le[-1, 2], base_point[1]], [le[-1, 0], base_point[0]], 
                    '--r', linewidth=1.5, label='Base Plane')
        
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.ax.set_title(f"Leading Edge Shape (Top View)")
        self.ax.set_xlabel('z [m]')
        self.ax.set_ylabel('x [m]')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.draw()


class GeometrySchematicCanvas(FigureCanvas):
    """Canvas showing a simple schematic of height and width definitions."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_schematic(self, height, width):
        self.ax.clear()

        self.ax.axhline(0.0, color='black', linewidth=2, label="Base plane (y = 0)")
        self.ax.axhline(height, color='gray', linestyle='--', linewidth=1.5,
                        label="Lower surface level")

        self.ax.annotate("", xy=(0.0, height), xytext=(0.0, 0.0),
                         arrowprops=dict(arrowstyle="<->", linewidth=1.5))
        self.ax.text(0.0, 0.5 * height, "height", ha="left", va="center", fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

        self.ax.annotate("", xy=(width, 0.0), xytext=(0.0, 0.0),
                         arrowprops=dict(arrowstyle="<->", linewidth=1.5))
        self.ax.text(0.5 * width, 0.0, "width (half-span)", ha="center", va="bottom",
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

        self.ax.plot(0.0, 0.0, 'ko')
        self.ax.text(0.0, -0.03 * max(height, 1e-3), "symmetry plane (z = 0)",
                     ha="center", va="top", fontsize=9)

        x_max = max(width * 1.2, 1e-3)
        y_max = max(height * 1.2, 1e-3)
        self.ax.set_xlim(-0.1 * x_max, 1.1 * x_max)
        self.ax.set_ylim(-0.1 * y_max, 1.1 * y_max)

        self.ax.set_xlabel("z (spanwise)")
        self.ax.set_ylabel("y (vertical)")
        self.ax.set_title("Definition of height and width")

        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_aspect("equal", adjustable="box")
        self.draw()


class AnalysisWorker(QThread):
    """Worker thread for PySAGAS analysis (keeps GUI responsive)"""
    finished = pyqtSignal(dict)  # Emits results
    error = pyqtSignal(str)  # Emits error message
    progress = pyqtSignal(str)  # Emits progress updates
    
    def __init__(self, stl_file, freestream_dict, aoa, A_ref):
        super().__init__()
        self.stl_file = stl_file
        self.freestream_dict = freestream_dict
        self.aoa = aoa
        self.A_ref = A_ref
        
    def run(self):
        import io
        import contextlib
        import warnings
        
        # Suppress warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        try:
            self.progress.emit("Loading STL mesh...")
            print(f"\n{'='*60}")
            print("PySAGAS Analysis Starting")
            print(f"{'='*60}")
            print(f"STL file: {self.stl_file}")
            print(f"Angle of attack: {self.aoa}°")
            print(f"Reference area: {self.A_ref:.4f} m²")
            sys.stdout.flush()
            
            # Load STL file
            cells = MeshIO.load_from_file(self.stl_file)
            
            msg = f"Loaded {len(cells)} cells"
            self.progress.emit(msg)
            print(msg)
            sys.stdout.flush()
            
            # Reference area (our calculated value)
            A_ref = self.A_ref
            
            # Instantiate solver
            freestream = FlowState(
                mach=self.freestream_dict['mach'],
                pressure=self.freestream_dict['pressure'],
                temperature=self.freestream_dict['temperature']
            )
            
            msg = "Initializing OPM solver..."
            self.progress.emit(msg)
            print(msg)
            sys.stdout.flush()
            
            solver = OPM(cells, freestream)
            
            # Run solver
            msg = f"Running analysis at α={self.aoa}°..."
            self.progress.emit(msg)
            print(msg)
            self.progress.emit("(Running PySAGAS; console output suppressed to keep GUI responsive)")
            sys.stdout.flush()

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                result = solver.solve(aoa=self.aoa)

            # Show last lines of PySAGAS output
            tail = "\n".join(buf.getvalue().splitlines()[-40:])
            if tail.strip():
                self.progress.emit("PySAGAS log tail:\n" + tail)
            
            # Save results to VTK file for visualization
            msg = "Saving VTK file..."
            self.progress.emit(msg)
            print(msg)
            sys.stdout.flush()
            
            try:
                solver.save("waverider")
                print("✓ VTK file saved: waverider.vtu")
            except Exception as e:
                print(f"⚠️  Could not save VTK: {e}")
            
            # Get aero coefficients
            msg = "Extracting coefficients..."
            self.progress.emit(msg)
            print(msg)
            sys.stdout.flush()
            
            # Get aero coefficients (no A_ref parameter - PySAGAS calculates internally)
            CL, CD, Cm = solver.flow_result.coefficients()
            
            print(f"\n  Coefficients from PySAGAS:")
            print(f"    CL = {CL:.6f}")
            print(f"    CD = {CD:.6f}")
            print(f"    Cm = {Cm:.6f}")
            print(f"  Reference area used: {A_ref:.4f} m²")
            sys.stdout.flush()
            
            # Calculate L/D
            LD = CL / CD if CD != 0 else float('inf')
            
            results = {
                'CL': float(CL),
                'CD': float(CD),
                'Cm': float(Cm),
                'CL/CD': float(LD)
            }
            
            msg = "Analysis complete!"
            self.progress.emit(msg)
            print(f"\n{'='*60}")
            print("Results:")
            print(f"  CL   = {CL:.6f}")
            print(f"  CD   = {CD:.6f}")
            print(f"  Cm   = {Cm:.6f}")
            print(f"  CL/CD  = {LD:.3f}")
            print(f"{'='*60}\n")
            sys.stdout.flush()
            
            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self.error.emit(error_msg)
            print(f"\n✗ ERROR: {error_msg}\n")
            sys.stdout.flush()




class MeshCanvas(FigureCanvas):
    """Canvas for visualizing STL mesh with interactive controls."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Enable mouse interaction
        self.ax.mouse_init()

        # Show axes (reverted from previous change)
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.ax.set_title('STL Mesh Preview')

    def plot_stl_mesh(self, stl_file):
        """Load and plot an STL file"""
        try:
            from stl import mesh as stl_mesh
            
            # Load the STL file
            mesh_data = stl_mesh.Mesh.from_file(stl_file)
            
            self.ax.clear()
            
            # Restore labels after clear
            self.ax.set_xlabel('X [m]')
            self.ax.set_ylabel('Y [m]')
            self.ax.set_zlabel('Z [m]')
            
            # Extract vertices
            vectors = mesh_data.vectors
            
            # Plot with better appearance
            collection = self.create_mesh_collection(vectors)
            self.ax.add_collection3d(collection)
            
            # Set limits with some padding
            all_points = vectors.reshape(-1, 3)
            for dim, axis in enumerate([self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim]):
                pmin, pmax = all_points[:, dim].min(), all_points[:, dim].max()
                padding = (pmax - pmin) * 0.1
                axis(pmin - padding, pmax + padding)
            
            # Set equal aspect ratio for proper proportions
            try:
                self.ax.set_box_aspect([
                    np.ptp(all_points[:, 0]),
                    np.ptp(all_points[:, 1]),
                    np.ptp(all_points[:, 2])
                ])
            except:
                pass  # Older matplotlib versions don't have this
            
            self.ax.set_title(f'STL Mesh ({len(vectors)} triangles)', fontsize=12)
            
            # Set a good initial view angle
            self.ax.view_init(elev=20, azim=45)
            
            self.fig.tight_layout()
            self.draw()
            
            return len(vectors)
            
        except ImportError:
            raise ImportError("numpy-stl not installed. Install with: pip install numpy-stl")
        except Exception as e:
            raise Exception(f"Could not load STL file: {str(e)}")
    
    def create_mesh_collection(self, vectors):
        """Create a 3D polygon collection from triangle vectors"""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Create collection with good appearance
        collection = Poly3DCollection(
            vectors,
            facecolors='lightblue',
            edgecolors='navy',
            alpha=0.6,
            linewidths=0.3
        )
        
        return collection



class WaveriderGUI(QMainWindow):
    """Main GUI window for waverider design"""
    
    def __init__(self):
        super().__init__()
        self.waverider = None
        self.waverider_volume = 0.0  # Stored volume in m³
        self.analysis_worker = None
        self.last_stl_file = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Interactive Waverider Design Tool')
        self.setGeometry(100, 100, 1600, 900)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel - Parameters
        left_panel = self.create_parameter_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Visualization
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, 3)
        
        # Set default values
        self.set_default_parameters()
        
    def create_parameter_panel(self):
        """Create the parameter input panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Flow conditions group
        flow_group = QGroupBox("Flow Conditions")
        flow_layout = QGridLayout()
        
        flow_layout.addWidget(QLabel("Mach Number (M∞):"), 0, 0)
        self.m_inf_spin = QDoubleSpinBox()
        self.m_inf_spin.setRange(1.1, 20.0)
        self.m_inf_spin.setValue(5.0)
        self.m_inf_spin.setSingleStep(0.1)
        self.m_inf_spin.valueChanged.connect(self.update_beta_hint)
        flow_layout.addWidget(self.m_inf_spin, 0, 1)
        
        flow_layout.addWidget(QLabel("Shock Angle β (deg):"), 1, 0)
        beta_layout = QHBoxLayout()
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(5.0, 89.0)
        self.beta_spin.setValue(15.0)
        self.beta_spin.setSingleStep(0.5)
        beta_layout.addWidget(self.beta_spin)
        
        # Auto-calculate beta button
        self.auto_beta_btn = QPushButton("📐 Auto")
        self.auto_beta_btn.setToolTip("Auto-calculate recommended β for current Mach")
        self.auto_beta_btn.setMaximumWidth(60)
        self.auto_beta_btn.clicked.connect(self.auto_calculate_beta)
        beta_layout.addWidget(self.auto_beta_btn)
        flow_layout.addLayout(beta_layout, 1, 1)
        
        # Beta hint label
        self.beta_hint_label = QLabel("")
        self.beta_hint_label.setStyleSheet("color: #666; font-size: 10px;")
        self.beta_hint_label.setWordWrap(True)
        flow_layout.addWidget(self.beta_hint_label, 2, 0, 1, 2)
        
        flow_group.setLayout(flow_layout)
        layout.addWidget(flow_group)
        
        # Initialize beta hint
        self.update_beta_hint()
        
        # Geometry group
        geom_group = QGroupBox("Geometry")
        geom_layout = QGridLayout()
        
        geom_layout.addWidget(QLabel("Height (m):"), 0, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.1, 10.0)
        self.height_spin.setValue(1.34)
        self.height_spin.setSingleStep(0.1)
        geom_layout.addWidget(self.height_spin, 0, 1)
        
        geom_layout.addWidget(QLabel("Width (m):"), 1, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 20.0)
        self.width_spin.setValue(3.0)
        self.width_spin.setSingleStep(0.1)
        geom_layout.addWidget(self.width_spin, 1, 1)
        
        # Volume display (calculated automatically after geometry generation)
        geom_layout.addWidget(QLabel("Volume (m³):"), 2, 0)
        self.volume_label = QLabel("N/A")
        self.volume_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        self.volume_label.setToolTip("Internal volume calculated after geometry generation")
        geom_layout.addWidget(self.volume_label, 2, 1)
        
        # Match lower surface to shockwave option (max volume)
        self.match_shock_check = QCheckBox("Match lower surface to shockwave (Max Volume)")
        self.match_shock_check.setToolTip(
            "When enabled, the lower surface follows the shockwave curve\n"
            "instead of tracing streamlines through the conical flowfield.\n\n"
            "This maximizes internal volume for the given geometry,\n"
            "but may affect aerodynamic performance predictions."
        )
        self.match_shock_check.setChecked(False)
        geom_layout.addWidget(self.match_shock_check, 3, 0, 1, 2)
        
        geom_group.setLayout(geom_layout)
        layout.addWidget(geom_group)
        
        # Geometry constraint hint (from paper Equation 8)
        self.geom_constraint_label = QLabel("")
        self.geom_constraint_label.setStyleSheet("color: #666; font-size: 10px; padding: 2px;")
        self.geom_constraint_label.setWordWrap(True)
        layout.addWidget(self.geom_constraint_label)
        
        # Connect geometry changes to update hint
        self.height_spin.valueChanged.connect(self.update_constraint_hints)
        self.width_spin.valueChanged.connect(self.update_constraint_hints)
        
        # Design parameters group (X1, X2, X3, X4)
        dp_group = QGroupBox("Design Parameters")
        dp_layout = QGridLayout()
        
        # X1
        dp_layout.addWidget(QLabel("X1 (Flat Region):"), 0, 0)
        self.x1_spin = QDoubleSpinBox()
        self.x1_spin.setRange(0.0, 1.0)
        self.x1_spin.setValue(0.11)
        self.x1_spin.setSingleStep(0.01)
        self.x1_spin.setDecimals(3)
        dp_layout.addWidget(self.x1_spin, 0, 1)
        
        self.x1_slider = QSlider(Qt.Horizontal)
        self.x1_slider.setRange(0, 1000)
        self.x1_slider.setValue(110)
        self.x1_slider.valueChanged.connect(
            lambda v: self.x1_spin.setValue(v/1000.0))
        self.x1_spin.valueChanged.connect(
            lambda v: self.x1_slider.setValue(int(v*1000)))
        dp_layout.addWidget(self.x1_slider, 0, 2)
        
        # X2
        dp_layout.addWidget(QLabel("X2 (Shock Height):"), 1, 0)
        self.x2_spin = QDoubleSpinBox()
        self.x2_spin.setRange(0.0, 1.0)
        self.x2_spin.setValue(0.63)
        self.x2_spin.setSingleStep(0.01)
        self.x2_spin.setDecimals(3)
        dp_layout.addWidget(self.x2_spin, 1, 1)
        
        self.x2_slider = QSlider(Qt.Horizontal)
        self.x2_slider.setRange(0, 1000)
        self.x2_slider.setValue(630)
        self.x2_slider.valueChanged.connect(
            lambda v: self.x2_spin.setValue(v/1000.0))
        self.x2_spin.valueChanged.connect(
            lambda v: self.x2_slider.setValue(int(v*1000)))
        dp_layout.addWidget(self.x2_slider, 1, 2)
        
        # X3
        dp_layout.addWidget(QLabel("X3 (Upper Surface 1):"), 2, 0)
        self.x3_spin = QDoubleSpinBox()
        self.x3_spin.setRange(0.0, 1.0)
        self.x3_spin.setValue(0.0)
        self.x3_spin.setSingleStep(0.01)
        self.x3_spin.setDecimals(3)
        dp_layout.addWidget(self.x3_spin, 2, 1)
        
        self.x3_slider = QSlider(Qt.Horizontal)
        self.x3_slider.setRange(0, 1000)
        self.x3_slider.setValue(0)
        self.x3_slider.valueChanged.connect(
            lambda v: self.x3_spin.setValue(v/1000.0))
        self.x3_spin.valueChanged.connect(
            lambda v: self.x3_slider.setValue(int(v*1000)))
        dp_layout.addWidget(self.x3_slider, 2, 2)
        
        # X4
        dp_layout.addWidget(QLabel("X4 (Upper Surface 2):"), 3, 0)
        self.x4_spin = QDoubleSpinBox()
        self.x4_spin.setRange(0.0, 1.0)
        self.x4_spin.setValue(0.46)
        self.x4_spin.setSingleStep(0.01)
        self.x4_spin.setDecimals(3)
        dp_layout.addWidget(self.x4_spin, 3, 1)
        
        self.x4_slider = QSlider(Qt.Horizontal)
        self.x4_slider.setRange(0, 1000)
        self.x4_slider.setValue(460)
        self.x4_slider.valueChanged.connect(
            lambda v: self.x4_spin.setValue(v/1000.0))
        self.x4_spin.valueChanged.connect(
            lambda v: self.x4_slider.setValue(int(v*1000)))
        dp_layout.addWidget(self.x4_slider, 3, 2)
        
        dp_group.setLayout(dp_layout)
        layout.addWidget(dp_group)
        
        # Design space constraint hint (X1, X2 relationship from paper Equation 8)
        self.design_constraint_label = QLabel("")
        self.design_constraint_label.setStyleSheet("color: #666; font-size: 10px; padding: 2px;")
        self.design_constraint_label.setWordWrap(True)
        layout.addWidget(self.design_constraint_label)
        
        # Connect X1, X2 changes to update hint
        self.x1_spin.valueChanged.connect(self.update_constraint_hints)
        self.x2_spin.valueChanged.connect(self.update_constraint_hints)
        
        # Initial update of constraint hints
        # (will be called after GUI is fully initialized via QTimer)
        
        # Mesh parameters group
        mesh_group = QGroupBox("Mesh Parameters")
        mesh_layout = QGridLayout()
        
        mesh_layout.addWidget(QLabel("n_planes:"), 0, 0)
        self.n_planes_spin = QSpinBox()
        self.n_planes_spin.setRange(10, 200)
        self.n_planes_spin.setValue(40)
        mesh_layout.addWidget(self.n_planes_spin, 0, 1)
        
        mesh_layout.addWidget(QLabel("n_streamwise:"), 1, 0)
        self.n_streamwise_spin = QSpinBox()
        self.n_streamwise_spin.setRange(10, 200)
        self.n_streamwise_spin.setValue(30)
        mesh_layout.addWidget(self.n_streamwise_spin, 1, 1)
        
        mesh_layout.addWidget(QLabel("delta_streamwise:"), 2, 0)
        self.delta_streamwise_spin = QDoubleSpinBox()
        self.delta_streamwise_spin.setRange(0.01, 0.2)
        self.delta_streamwise_spin.setValue(0.1)
        self.delta_streamwise_spin.setSingleStep(0.01)
        mesh_layout.addWidget(self.delta_streamwise_spin, 2, 1)
        
        mesh_layout.addWidget(QLabel("n_upper_surface:"), 3, 0)
        self.n_us_spin = QSpinBox()
        self.n_us_spin.setRange(10, 200000)
        self.n_us_spin.setValue(1000)
        self.n_us_spin.setToolTip("Number of interpolation points for upper surface Bézier curve")
        mesh_layout.addWidget(self.n_us_spin, 3, 1)
        
        mesh_layout.addWidget(QLabel("n_shockwave:"), 4, 0)
        self.n_sw_spin = QSpinBox()
        self.n_sw_spin.setRange(10, 200000)
        self.n_sw_spin.setValue(1000)
        self.n_sw_spin.setToolTip("Number of interpolation points for shockwave Bézier curve")
        mesh_layout.addWidget(self.n_sw_spin, 4, 1)
        
        mesh_group.setLayout(mesh_layout)
        layout.addWidget(mesh_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        generate_btn = QPushButton("Generate Waverider")
        generate_btn.clicked.connect(self.generate_waverider)
        generate_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        button_layout.addWidget(generate_btn)
        
        export_btn = QPushButton("Export CAD")
        export_btn.clicked.connect(self.export_cad)
        export_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        button_layout.addWidget(export_btn)
        
        layout.addLayout(button_layout)
        
        # Info label
        self.info_label = QLabel("Ready to generate waverider")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }")
        layout.addWidget(self.info_label)
        
        layout.addStretch()
        
        return panel
    
    def create_visualization_panel(self):
        """Create the visualization panel with tabs"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # 3D view tab
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
        self.show_wireframe_check = QCheckBox("Wireframe")
        self.show_wireframe_check.setChecked(False)
        
        options_layout.addWidget(QLabel("Display:"))
        options_layout.addWidget(self.show_upper_check)
        options_layout.addWidget(self.show_lower_check)
        options_layout.addWidget(self.show_le_check)
        options_layout.addWidget(self.show_wireframe_check)
        options_layout.addStretch()
        
        update_view_btn = QPushButton("Update View")
        update_view_btn.clicked.connect(self.update_3d_view)
        options_layout.addWidget(update_view_btn)
        
        layout_3d.addLayout(options_layout)
        
        # 3D canvas
        self.canvas_3d = WaveriderCanvas()
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, tab_3d)
        layout_3d.addWidget(self.toolbar_3d)
        layout_3d.addWidget(self.canvas_3d)
        
        self.tab_widget.addTab(tab_3d, "3D View")
        
        # Base plane tab
        tab_base = QWidget()
        layout_base = QVBoxLayout(tab_base)
        self.canvas_base = BasePlaneCanvas()
        self.toolbar_base = NavigationToolbar(self.canvas_base, tab_base)
        layout_base.addWidget(self.toolbar_base)
        layout_base.addWidget(self.canvas_base)
        self.tab_widget.addTab(tab_base, "Base Plane")
        
        # Leading edge tab
        tab_le = QWidget()
        layout_le = QVBoxLayout(tab_le)
        self.canvas_le = LECanvas()
        self.toolbar_le = NavigationToolbar(self.canvas_le, tab_le)
        layout_le.addWidget(self.toolbar_le)
        layout_le.addWidget(self.canvas_le)
        self.tab_widget.addTab(tab_le, "Leading Edge")
        
        # Geometry schematic tab
        tab_geom = QWidget()
        layout_geom = QVBoxLayout(tab_geom)
        self.canvas_geom = GeometrySchematicCanvas()
        self.toolbar_geom = NavigationToolbar(self.canvas_geom, tab_geom)
        layout_geom.addWidget(self.toolbar_geom)
        layout_geom.addWidget(self.canvas_geom)
        self.tab_widget.addTab(tab_geom, "Geometry Schematic")

        # Aero Analysis tab
        tab_analysis = self.create_analysis_tab()
        self.tab_widget.addTab(tab_analysis, "🔬 Aero Analysis")
        
        # Optimization tab
        self.optimization_tab = OptimizationTab(parent=self)
        self.tab_widget.addTab(self.optimization_tab, "🧬 Optimization")
        
        # Surrogate Optimization tab
        if SURROGATE_AVAILABLE:
            self.surrogate_tab = SurrogateTab(parent=self)
            self.tab_widget.addTab(self.surrogate_tab, "🔮 Surrogate Opt")
        else:
            # Create placeholder tab if surrogate not available
            surrogate_placeholder = QWidget()
            placeholder_layout = QVBoxLayout(surrogate_placeholder)
            placeholder_label = QLabel(
                "⚠️ Surrogate Optimization not available.\n\n"
                "Required: scikit-learn\n"
                "Install with: pip install scikit-learn"
            )
            placeholder_label.setStyleSheet(
                "QLabel { background-color: #fff3cd; padding: 20px; "
                "border-radius: 5px; font-size: 12px; }"
            )
            placeholder_label.setAlignment(Qt.AlignCenter)
            placeholder_layout.addWidget(placeholder_label)
            placeholder_layout.addStretch()
            self.tab_widget.addTab(surrogate_placeholder, "🔮 Surrogate Opt")
            
            
        # Off-Design Surrogate tab
        if OFFDESIGN_SURROGATE_AVAILABLE:
            self.offdesign_tab = OffDesignSurrogateTab(parent=self)
            self.tab_widget.addTab(self.offdesign_tab, "🎯 Off-Design NN")
        else:
            # Create placeholder tab if not available
            offdesign_placeholder = QWidget()
            offdesign_layout = QVBoxLayout(offdesign_placeholder)
            offdesign_label = QLabel(
                "⚠️ Off-Design Neural Network Surrogate not available.\\n\\n"
                "Required: scikit-learn, trained model files\\n"
                "Files needed in surrogate_model/ folder:\\n"
                "  - ensemble_CL.pkl\\n"
                "  - ensemble_CD.pkl\\n"
                "  - ensemble_CL_CD.pkl\\n"
                "  - config.json"
            )
            offdesign_label.setStyleSheet(
                "QLabel { background-color: #fff3cd; padding: 20px; "
                "border-radius: 5px; font-size: 12px; }"
            )
            offdesign_label.setAlignment(Qt.AlignCenter)
            offdesign_layout.addWidget(offdesign_label)
            offdesign_layout.addStretch()
            self.tab_widget.addTab(offdesign_placeholder, "🎯 Off-Design NN")
            
        # Multi-Mach Hunter tab
        if MULTIMACH_HUNTER_AVAILABLE:
            self.multimach_tab = MultiMachHunterTab(parent=self)
            self.tab_widget.addTab(self.multimach_tab, "🌐 Multi-Mach")
            
        # Cone-waverider tab    
        if CONE_WAVERIDER_AVAILABLE:
            self.shadow_waverider_tab = ShadowWaveriderTab(parent=self)
            self.tab_widget.addTab(self.shadow_waverider_tab, "🔷 SHADOW Waverider")
            
        # Claude Assistant tab
        if CLAUDE_ASSISTANT_AVAILABLE:
            self.claude_tab = ClaudeAssistantTab(parent=self)
            self.tab_widget.addTab(self.claude_tab, "🤖 Claude Assistant")
        else:
            claude_placeholder = QWidget()
            claude_layout = QVBoxLayout(claude_placeholder)
            claude_label = QLabel(
                "⚠️ Claude Assistant not available.\\n\\n"
                "Required: pip install anthropic"
            )
            claude_label.setStyleSheet(
                "QLabel { background-color: #fff3cd; padding: 20px; "
                "border-radius: 5px; font-size: 12px; }"
            )
            claude_label.setAlignment(Qt.AlignCenter)
            claude_layout.addWidget(claude_label)
            claude_layout.addStretch()
            self.tab_widget.addTab(claude_placeholder, "🤖 Claude Assistant")

        layout.addWidget(self.tab_widget)
        
        return panel
    

    def create_analysis_tab(self):
        """Create the PySAGAS analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        if not PYSAGAS_AVAILABLE:
            warning_label = QLabel("⚠️ PySAGAS not available. Install with: pip install pysagas")
            warning_label.setStyleSheet("QLabel { background-color: #ffcccc; padding: 10px; }")
            layout.addWidget(warning_label)

        # Analysis parameters
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QGridLayout()

        # Mach number for analysis
        params_layout.addWidget(QLabel("Mach Number M∞:"), 0, 0)
        self.analysis_mach_spin = QDoubleSpinBox()
        self.analysis_mach_spin.setRange(0.1, 25.0)
        self.analysis_mach_spin.setValue(5.0)
        self.analysis_mach_spin.setSingleStep(0.1)
        self.analysis_mach_spin.setDecimals(2)
        self.analysis_mach_spin.setToolTip("Freestream Mach number for aerodynamic analysis")
        params_layout.addWidget(self.analysis_mach_spin, 0, 1)

        # Angle of attack
        params_layout.addWidget(QLabel("Angle of Attack α (deg):"), 1, 0)
        self.aoa_spin = QDoubleSpinBox()
        self.aoa_spin.setRange(-20.0, 20.0)
        self.aoa_spin.setValue(0.0)
        self.aoa_spin.setSingleStep(0.5)
        self.aoa_spin.setDecimals(2)
        self.aoa_spin.setToolTip("Angle of attack for aerodynamic analysis")
        params_layout.addWidget(self.aoa_spin, 1, 1)

        # Reference area
        params_layout.addWidget(QLabel("Reference Area A_ref (m²):"), 2, 0)
        self.aref_spin = QDoubleSpinBox()
        self.aref_spin.setRange(0.1, 100.0)
        self.aref_spin.setValue(21.6)  # More realistic default for baseline
        self.aref_spin.setSingleStep(0.1)
        self.aref_spin.setDecimals(4)
        self.aref_spin.setToolTip(
            "Reference area for coefficient normalization.\n"
            "Use 'Calculate Accurate A_ref' for precise value!\n"
            "Simple w×h is very inaccurate (~400% error)."
        )
        params_layout.addWidget(self.aref_spin, 2, 1)

        # Auto-update A_ref button
        auto_aref_btn = QPushButton("🎯 Calculate Accurate A_ref")
        auto_aref_btn.clicked.connect(self.auto_set_aref)
        auto_aref_btn.setToolTip(
            "Calculate accurate planform area from waverider geometry.\n"
            "Much more accurate than simple width × height!\n"
            "Requires waverider to be generated first."
        )
        params_layout.addWidget(auto_aref_btn, 2, 2)

        # Freestream pressure
        params_layout.addWidget(QLabel("Pressure P∞ (Pa):"), 3, 0)
        self.pressure_spin = QDoubleSpinBox()
        self.pressure_spin.setRange(100, 1e7)
        self.pressure_spin.setValue(101325)  # Sea level
        self.pressure_spin.setSingleStep(1000)
        self.pressure_spin.setDecimals(0)
        self.pressure_spin.setToolTip("Freestream static pressure")
        params_layout.addWidget(self.pressure_spin, 3, 1)

        # Freestream temperature
        params_layout.addWidget(QLabel("Temperature T∞ (K):"), 4, 0)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(100, 400)
        self.temperature_spin.setValue(288.15)  # Sea level ISA
        self.temperature_spin.setSingleStep(1)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setToolTip("Freestream static temperature")
        params_layout.addWidget(self.temperature_spin, 4, 1)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # STL Mesh Generation Group
        mesh_gen_group = QGroupBox("STL Mesh Generation (using Gmsh)")
        mesh_gen_layout = QGridLayout()
        
        # Mesh size controls
        mesh_gen_layout.addWidget(QLabel("Min Element Size [m]:"), 0, 0)
        self.mesh_min_spin = QDoubleSpinBox()
        self.mesh_min_spin.setRange(0.00001, 1000.0)
        self.mesh_min_spin.setValue(0.005)  # 5mm default
        self.mesh_min_spin.setSingleStep(0.005)
        self.mesh_min_spin.setDecimals(5)
        self.mesh_min_spin.setToolTip("Minimum triangle edge length (smaller = finer mesh)")
        mesh_gen_layout.addWidget(self.mesh_min_spin, 0, 1)
        
        mesh_gen_layout.addWidget(QLabel("Max Element Size [m]:"), 1, 0)
        self.mesh_max_spin = QDoubleSpinBox()
        self.mesh_max_spin.setRange(0.00001, 1000)
        self.mesh_max_spin.setValue(0.05)  # 50mm default
        self.mesh_max_spin.setSingleStep(0.005)
        self.mesh_max_spin.setDecimals(5)
        self.mesh_max_spin.setToolTip("Maximum triangle edge length (smaller = finer mesh)")
        mesh_gen_layout.addWidget(self.mesh_max_spin, 1, 1)
        
        # Quality presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))
        
        coarse_btn = QPushButton("Coarse (fast)")
        coarse_btn.clicked.connect(lambda: self.set_mesh_preset(0.01, 0.1))
        preset_layout.addWidget(coarse_btn)
        
        medium_btn = QPushButton("Medium")
        medium_btn.clicked.connect(lambda: self.set_mesh_preset(0.005, 0.05))
        preset_layout.addWidget(medium_btn)
        
        fine_btn = QPushButton("Fine (slow)")
        fine_btn.clicked.connect(lambda: self.set_mesh_preset(0.002, 0.02))
        preset_layout.addWidget(fine_btn)
        
        preset_layout.addStretch()
        mesh_gen_layout.addLayout(preset_layout, 2, 0, 1, 2)
        
        # Generate mesh button
        self.generate_mesh_btn = QPushButton("🔧 Generate STL Mesh with Gmsh")
        self.generate_mesh_btn.clicked.connect(self.generate_stl_mesh)
        self.generate_mesh_btn.setEnabled(False)
        self.generate_mesh_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; }"
        )
        mesh_gen_layout.addWidget(self.generate_mesh_btn, 3, 0, 1, 2)
        
        # Mesh info
        self.mesh_gen_info = QLabel("Generate waverider first, then create mesh")
        self.mesh_gen_info.setStyleSheet("color: gray; font-style: italic;")
        mesh_gen_layout.addWidget(self.mesh_gen_info, 4, 0, 1, 2)
        
        mesh_gen_group.setLayout(mesh_gen_layout)
        layout.addWidget(mesh_gen_group)
        
        # STL Mesh Visualization Group
        viz_group = QGroupBox("STL Mesh Preview")
        viz_layout = QVBoxLayout()
        
        # Add mesh canvas
        self.mesh_canvas = MeshCanvas(self)
        self.mesh_toolbar = NavigationToolbar(self.mesh_canvas, self)
        
        viz_layout.addWidget(self.mesh_toolbar)
        viz_layout.addWidget(self.mesh_canvas)
        
        # Mesh info label
        self.mesh_info_label = QLabel("No mesh loaded - Export CAD first to create STL")
        self.mesh_info_label.setStyleSheet("color: gray; font-style: italic;")
        viz_layout.addWidget(self.mesh_info_label)
        
        # Load/Refresh mesh button
        mesh_btn_layout = QHBoxLayout()
        self.load_mesh_btn = QPushButton("📊 Load/Refresh STL Mesh")
        self.load_mesh_btn.clicked.connect(self.load_and_display_mesh)
        self.load_mesh_btn.setEnabled(False)
        mesh_btn_layout.addWidget(self.load_mesh_btn)
        mesh_btn_layout.addStretch()
        viz_layout.addLayout(mesh_btn_layout)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Run analysis buttons
        run_layout = QHBoxLayout()
        self.run_analysis_btn = QPushButton("🚀 Run PySAGAS Analysis")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.run_analysis_btn.setStyleSheet(
            "QPushButton { background-color: #FF5722; color: white; "
            "font-weight: bold; padding: 12px; font-size: 14px; }"
        )
        self.run_analysis_btn.setEnabled(False)
        run_layout.addWidget(self.run_analysis_btn)
        
        # Stop button (initially hidden)
        self.stop_analysis_btn = QPushButton("⛔ Stop Analysis")
        self.stop_analysis_btn.clicked.connect(self.stop_analysis)
        self.stop_analysis_btn.setStyleSheet(
            "QPushButton { background-color: #ff6b6b; color: white; "
            "font-weight: bold; padding: 12px; font-size: 14px; }"
        )
        self.stop_analysis_btn.setVisible(False)
        run_layout.addWidget(self.stop_analysis_btn)
        
        layout.addLayout(run_layout)
        
        # AeroDeck Sweep Group
        sweep_group = QGroupBox("📊 AeroDeck Sweep (Multi-Point Analysis)")
        sweep_layout = QGridLayout()
        
        # Enable sweep checkbox
        self.enable_sweep_check = QCheckBox("Enable AoA/Mach Sweep")
        self.enable_sweep_check.setToolTip(
            "Run analysis at multiple angles of attack and Mach numbers.\n"
            "Results are saved to an AeroDeck CSV file."
        )
        self.enable_sweep_check.stateChanged.connect(self.on_sweep_enabled_changed)
        sweep_layout.addWidget(self.enable_sweep_check, 0, 0, 1, 4)
        
        # AoA range
        sweep_layout.addWidget(QLabel("AoA range (°):"), 1, 0)
        self.aoa_min_spin = QDoubleSpinBox()
        self.aoa_min_spin.setRange(-30.0, 30.0)
        self.aoa_min_spin.setValue(-5.0)
        self.aoa_min_spin.setDecimals(1)
        self.aoa_min_spin.setEnabled(False)
        self.aoa_min_spin.valueChanged.connect(self.update_sweep_info)
        sweep_layout.addWidget(self.aoa_min_spin, 1, 1)
        
        sweep_layout.addWidget(QLabel("to"), 1, 2)
        self.aoa_max_spin = QDoubleSpinBox()
        self.aoa_max_spin.setRange(-30.0, 30.0)
        self.aoa_max_spin.setValue(10.0)
        self.aoa_max_spin.setDecimals(1)
        self.aoa_max_spin.setEnabled(False)
        self.aoa_max_spin.valueChanged.connect(self.update_sweep_info)
        sweep_layout.addWidget(self.aoa_max_spin, 1, 3)
        
        sweep_layout.addWidget(QLabel("Step:"), 1, 4)
        self.aoa_step_spin = QDoubleSpinBox()
        self.aoa_step_spin.setRange(0.5, 10.0)
        self.aoa_step_spin.setValue(1.0)
        self.aoa_step_spin.setDecimals(1)
        self.aoa_step_spin.setEnabled(False)
        self.aoa_step_spin.valueChanged.connect(self.update_sweep_info)
        sweep_layout.addWidget(self.aoa_step_spin, 1, 5)
        
        # Mach range
        sweep_layout.addWidget(QLabel("Mach range:"), 2, 0)
        self.mach_min_spin = QDoubleSpinBox()
        self.mach_min_spin.setRange(1.5, 25.0)
        self.mach_min_spin.setValue(4.0)
        self.mach_min_spin.setDecimals(1)
        self.mach_min_spin.setEnabled(False)
        self.mach_min_spin.valueChanged.connect(self.update_sweep_info)
        sweep_layout.addWidget(self.mach_min_spin, 2, 1)
        
        sweep_layout.addWidget(QLabel("to"), 2, 2)
        self.mach_max_spin = QDoubleSpinBox()
        self.mach_max_spin.setRange(1.5, 25.0)
        self.mach_max_spin.setValue(8.0)
        self.mach_max_spin.setDecimals(1)
        self.mach_max_spin.setEnabled(False)
        self.mach_max_spin.valueChanged.connect(self.update_sweep_info)
        sweep_layout.addWidget(self.mach_max_spin, 2, 3)
        
        sweep_layout.addWidget(QLabel("Step:"), 2, 4)
        self.mach_step_spin = QDoubleSpinBox()
        self.mach_step_spin.setRange(0.5, 5.0)
        self.mach_step_spin.setValue(1.0)
        self.mach_step_spin.setDecimals(1)
        self.mach_step_spin.setEnabled(False)
        self.mach_step_spin.valueChanged.connect(self.update_sweep_info)
        sweep_layout.addWidget(self.mach_step_spin, 2, 5)
        
        # Sweep info label
        self.sweep_info_label = QLabel("Enable sweep to analyze multiple flight conditions")
        self.sweep_info_label.setStyleSheet("color: #666; font-style: italic;")
        sweep_layout.addWidget(self.sweep_info_label, 3, 0, 1, 6)
        
        # Run sweep button
        self.run_sweep_btn = QPushButton("🔄 Run AeroDeck Sweep")
        self.run_sweep_btn.clicked.connect(self.run_aerodeck_sweep)
        self.run_sweep_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 10px; }"
        )
        self.run_sweep_btn.setEnabled(False)
        sweep_layout.addWidget(self.run_sweep_btn, 4, 0, 1, 3)
        
        # Plot results button
        self.plot_sweep_btn = QPushButton("📈 Plot AeroDeck Results")
        self.plot_sweep_btn.clicked.connect(self.plot_aerodeck_results)
        self.plot_sweep_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; }"
        )
        self.plot_sweep_btn.setEnabled(False)
        sweep_layout.addWidget(self.plot_sweep_btn, 4, 3, 1, 3)
        
        sweep_group.setLayout(sweep_layout)
        layout.addWidget(sweep_group)

        # Progress bar
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setVisible(False)
        layout.addWidget(self.analysis_progress)

        # Results display
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(
            "QTextEdit { font-family: 'Courier New'; font-size: 11pt; }"
        )
        self.results_text.setText("No analysis results yet.\n\n"
                                  "Steps:\n"
                                  "1. Generate a waverider\n"
                                  "2. Export to CAD (creates STL)\n"
                                  "3. Set analysis parameters\n"
                                  "4. Click 'Run PySAGAS Analysis'")
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        return tab

    def set_default_parameters(self):
        """Set default parameters from example"""
        # Already set in create_parameter_panel
        # Initialize constraint hints
        self.update_constraint_hints()
    
    def generate_waverider(self):
        """Generate the waverider with current parameters"""
        try:
            self.info_label.setText("Generating waverider... Please wait.")
            QApplication.processEvents()
            
            # Get parameters
            M_inf = self.m_inf_spin.value()
            beta = self.beta_spin.value()
            height = self.height_spin.value()
            width = self.width_spin.value()
            dp = [
                self.x1_spin.value(),
                self.x2_spin.value(),
                self.x3_spin.value(),
                self.x4_spin.value()
            ]
            n_planes = self.n_planes_spin.value()
            n_streamwise = self.n_streamwise_spin.value()
            delta_streamwise = self.delta_streamwise_spin.value()
            n_upper_surface = self.n_us_spin.value()
            n_shockwave = self.n_sw_spin.value()
            
            dp = [
                self.x1_spin.value(),
                self.x2_spin.value(),
                self.x3_spin.value(),
                self.x4_spin.value()
                ]
            print(f"DEBUG: dp = {dp}")  # Add this line
            print(f"DEBUG: x2_spin.value() = {self.x2_spin.value()}")  # And this
            
            # Check design space constraint
            constraint = dp[1] / ((1 - dp[0])**4)
            max_constraint = (7/64) * (width/height)**4
            
            if constraint >= max_constraint:
                QMessageBox.warning(self, "Design Space Violation",
                    f"Design parameters violate the design space constraint!\n\n"
                    f"Constraint value: {constraint:.4f}\n"
                    f"Maximum allowed: {max_constraint:.4f}\n\n"
                    f"Try reducing X2 or increasing X1.")
                self.info_label.setText("Design space constraint violated!")
                return
            
            # Generate waverider
            match_shockwave = self.match_shock_check.isChecked()
            self.waverider = wr(
                M_inf=M_inf,
                beta=beta,
                height=height,
                width=width,
                dp=dp,
                n_upper_surface=n_upper_surface,
                n_shockwave=n_shockwave,
                n_planes=n_planes,
                n_streamwise=n_streamwise,
                delta_streamwise=delta_streamwise,
                match_shockwave=match_shockwave
            )
            
            # Calculate and display volume
            try:
                self.waverider_volume = calculate_waverider_volume(self.waverider)
                self.volume_label.setText(f"{self.waverider_volume:.4f}")
                self.volume_label.setStyleSheet("font-weight: bold; color: #0066cc;")
            except Exception as vol_err:
                self.waverider_volume = 0.0
                self.volume_label.setText("Error")
                self.volume_label.setStyleSheet("font-weight: bold; color: #cc0000;")
                print(f"Volume calculation error: {vol_err}")
            
            # Update all views
            self.update_all_views()
            
            # Calculate some properties
            length = self.waverider.length
            
            self.info_label.setText(
                f"✓ Waverider generated successfully!\n\n"
                f"Length: {length:.3f} m\n"
                f"Width: {width:.3f} m\n"
                f"Height: {height:.3f} m\n"
                f"Volume: {self.waverider_volume:.4f} m³\n"
                f"Constraint: {constraint:.4f} / {max_constraint:.4f}\n"
                f"Design Point: [{dp[0]:.3f}, {dp[1]:.3f}, {dp[2]:.3f}, {dp[3]:.3f}]"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate waverider:\n\n{str(e)}")
            self.info_label.setText(f"Error: {str(e)}")
    
    def update_all_views(self):
        """Update all visualization views"""
        if self.waverider is None:
            return
        
        self.update_3d_view()
        self.canvas_base.plot_base_plane(self.waverider)
        self.canvas_le.plot_leading_edge(self.waverider)
    
    def update_constraint_hints(self):
        """
        Update constraint hint labels based on current parameter values.
        
        From the paper (Equation 8), the design space constraint is:
            X2 / (1 - X1)^4 < (7/64) * (width/height)^4
        
        This can be rearranged to find:
        - Max height given width: height_max = width / ((64/7) * X2 / (1-X1)^4)^0.25
        - Max X2 given X1: X2_max = (7/64) * (width/height)^4 * (1-X1)^4
        """
        try:
            width = self.width_spin.value()
            height = self.height_spin.value()
            X1 = self.x1_spin.value()
            X2 = self.x2_spin.value()
            
            # Calculate the constraint value (RHS of inequality)
            # X2 / (1 - X1)^4 < (7/64) * (width/height)^4
            one_minus_x1 = max(1 - X1, 0.001)  # Avoid division by zero
            rhs = (7.0 / 64.0) * (width / height) ** 4
            rhs_safe = 0.9 * rhs  # 10% safety margin
            
            # Current constraint value (LHS)
            lhs = X2 / (one_minus_x1 ** 4)
            
            # Check if constraint is satisfied
            is_valid = lhs < rhs_safe
            
            # Calculate max X2 given current X1, width, height
            max_x2 = rhs_safe * (one_minus_x1 ** 4)
            max_x2 = min(max_x2, 1.0)  # Cap at 1.0
            
            # Calculate max height given current width, X1, X2
            if X2 > 0.001:
                # Rearrange: height_max = width / ((64/7) * X2 / (1-X1)^4)^0.25
                ratio = (64.0 / 7.0) * X2 / (one_minus_x1 ** 4)
                if ratio > 0:
                    max_height = width / (ratio ** 0.25) * 0.9  # With safety margin
                else:
                    max_height = 10.0
            else:
                max_height = 10.0  # No constraint when X2 is very small
            
            # Update geometry constraint label
            if height > max_height:
                self.geom_constraint_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 2px; font-weight: bold;")
                geom_text = f"⚠️ Height too large! Max height ≈ {max_height:.2f} m for current X1, X2"
            else:
                self.geom_constraint_label.setStyleSheet("color: #388e3c; font-size: 10px; padding: 2px;")
                geom_text = f"✓ Max height ≈ {max_height:.2f} m (current: {height:.2f} m)"
            self.geom_constraint_label.setText(geom_text)
            
            # Update design variable constraint label
            if X2 > max_x2:
                self.design_constraint_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 2px; font-weight: bold;")
                design_text = f"⚠️ X2 too large! Max X2 ≈ {max_x2:.3f} for current X1={X1:.3f}"
            else:
                self.design_constraint_label.setStyleSheet("color: #388e3c; font-size: 10px; padding: 2px;")
                design_text = f"✓ Max X2 ≈ {max_x2:.3f} (current: {X2:.3f})"
            self.design_constraint_label.setText(design_text)
            
        except Exception as e:
            # Don't crash on hint update errors
            self.geom_constraint_label.setText("")
            self.design_constraint_label.setText("")
    
    def update_beta_hint(self):
        """Update the beta hint label showing valid range for current Mach."""
        try:
            M = self.m_inf_spin.value()
            
            # Calculate Mach angle (minimum possible shock angle)
            beta_min = np.degrees(np.arcsin(1.0 / M))
            
            # Get recommended beta values from lookup table
            recommended = self.get_recommended_beta(M)
            
            # Update hint label
            hint_text = f"β range: {beta_min:.1f}° (Mach angle) to ~45°  |  "
            hint_text += f"Recommended: {recommended['mid']:.1f}° (range: {recommended['low']:.1f}°-{recommended['high']:.1f}°)"
            
            self.beta_hint_label.setText(hint_text)
            
            # Check if current beta is valid
            current_beta = self.beta_spin.value()
            if current_beta < beta_min:
                self.beta_hint_label.setStyleSheet("color: #d32f2f; font-size: 10px; font-weight: bold;")
            else:
                self.beta_hint_label.setStyleSheet("color: #666; font-size: 10px;")
                
        except Exception as e:
            self.beta_hint_label.setText("")
    
    def get_recommended_beta(self, M):
        """
        Get recommended shock angle (β) for a given Mach number.
        
        These values are derived from oblique shock theory and practical
        waverider design experience. The recommended range gives attached
        shocks with reasonable cone angles.
        
        Parameters
        ----------
        M : float
            Freestream Mach number
            
        Returns
        -------
        dict
            Contains 'low', 'mid', 'high' recommended beta values
        """
        # Lookup table based on the paper and oblique shock theory
        # Format: Mach -> (low, mid, high) beta values in degrees
        beta_table = {
            2.0: (35.0, 40.0, 45.0),
            2.5: (30.0, 34.0, 38.0),
            3.0: (25.5, 26.5, 28.0),
            3.5: (22.0, 23.5, 25.0),
            4.0: (20.0, 21.0, 22.0),
            4.5: (18.5, 19.5, 20.5),
            5.0: (17.0, 18.0, 19.0),
            5.5: (16.0, 17.0, 18.0),
            6.0: (15.0, 16.0, 17.0),
            7.0: (13.5, 14.5, 15.5),
            8.0: (12.5, 13.5, 14.5),
            10.0: (11.0, 12.0, 13.0),
            12.0: (10.0, 11.0, 12.0),
            15.0: (9.0, 10.0, 11.0),
        }
        
        # Find closest Mach numbers for interpolation
        mach_values = sorted(beta_table.keys())
        
        if M <= mach_values[0]:
            low, mid, high = beta_table[mach_values[0]]
        elif M >= mach_values[-1]:
            low, mid, high = beta_table[mach_values[-1]]
        else:
            # Linear interpolation
            for i in range(len(mach_values) - 1):
                if mach_values[i] <= M <= mach_values[i + 1]:
                    M1, M2 = mach_values[i], mach_values[i + 1]
                    t = (M - M1) / (M2 - M1)
                    
                    low1, mid1, high1 = beta_table[M1]
                    low2, mid2, high2 = beta_table[M2]
                    
                    low = low1 + t * (low2 - low1)
                    mid = mid1 + t * (mid2 - mid1)
                    high = high1 + t * (high2 - high1)
                    break
        
        # Ensure beta is above Mach angle
        beta_min = np.degrees(np.arcsin(1.0 / M)) + 0.5  # Small margin
        low = max(low, beta_min)
        mid = max(mid, beta_min)
        high = max(high, beta_min)
        
        return {'low': low, 'mid': mid, 'high': high}
    
    def auto_calculate_beta(self):
        """Auto-calculate and set recommended beta for current Mach."""
        M = self.m_inf_spin.value()
        recommended = self.get_recommended_beta(M)
        
        # Set to middle recommended value
        self.beta_spin.setValue(recommended['mid'])
        
        # Show info message
        QMessageBox.information(
            self, "Auto β Calculation",
            f"For Mach {M:.1f}, recommended β values:\n\n"
            f"  Low:  {recommended['low']:.2f}°\n"
            f"  Mid:  {recommended['mid']:.2f}° ← (selected)\n"
            f"  High: {recommended['high']:.2f}°\n\n"
            f"Lower β → sharper leading edge, lower drag\n"
            f"Higher β → blunter leading edge, more volume"
        )
    
    def update_3d_view(self):
        """Update the 3D view with current display options"""
        if self.waverider is None:
            return
        
        self.canvas_3d.plot_waverider(
            self.waverider,
            show_upper=self.show_upper_check.isChecked(),
            show_lower=self.show_lower_check.isChecked(),
            show_le=self.show_le_check.isChecked(),
            show_wireframe=self.show_wireframe_check.isChecked()
        )
    


    # ========== AERO ANALYSIS METHODS ==========

    def auto_set_aref(self):
        """Automatically calculate accurate A_ref from waverider geometry"""
        if self.waverider is None:
            QMessageBox.warning(
                self, "No waverider",
                "Generate a waverider first to calculate accurate reference area."
            )
            return
        
        width = self.width_spin.value()
        height = self.height_spin.value()
        simple_area = width * height
        
        # Check if calculator is available
        if not AREA_CALC_AVAILABLE:
            QMessageBox.warning(
                self, "Calculator not available",
                "Reference area calculator module not found.\n\n"
                "The module should be in the same directory as the GUI.\n"
                "Falling back to simple width × height approximation.\n\n"
                f"A_ref = {simple_area:.4f} m² (width × height)\n\n"
                "⚠️ This is approximate and may have ~400% error!"
            )
            self.aref_spin.setValue(simple_area)
            self.info_label.setText(
                f"A_ref set to {simple_area:.4f} m² (width × height)\n"
                f"⚠️ Calculator module not found - this is approximate!"
            )
            return
        
        # Try to use accurate calculation
        try:
            print(f"Calculating accurate planform area for waverider...")
            accurate_area, method = calculate_planform_area_from_waverider(self.waverider)
            print(f"Result: {accurate_area:.4f} m² using {method}")
            
            self.aref_spin.setValue(accurate_area)
            
            difference_pct = 100 * (accurate_area - simple_area) / simple_area
            
            info_msg = (
                f"A_ref set to {accurate_area:.4f} m²\n\n"
                f"Method: {method}\n"
                f"Simple approximation (w×h): {simple_area:.4f} m²\n"
                f"Accurate planform area: {accurate_area:.4f} m²\n"
                f"Difference: {difference_pct:+.1f}%"
            )
            
            if abs(difference_pct) > 10:
                info_msg += f"\n\n⚠️ Using simple w×h would cause {abs(difference_pct):.1f}% error in coefficients!"
            
            self.info_label.setText(info_msg)
            
            # Also show wetted areas for reference
            try:
                upper, lower, total = calculate_wetted_area_from_waverider(self.waverider)
                QMessageBox.information(
                    self, "Reference Areas Calculated",
                    f"✓ Accurate calculation successful!\n\n"
                    f"Planform Area (for A_ref):\n"
                    f"  {accurate_area:.4f} m²\n\n"
                    f"Wetted Areas (for comparison):\n"
                    f"  Upper surface: {upper:.4f} m²\n"
                    f"  Lower surface: {lower:.4f} m²\n"
                    f"  Total wetted:  {total:.4f} m²\n"
                    f"  (SolidWorks should show ~{total:.2f} m²)\n\n"
                    f"Simple w×h: {simple_area:.4f} m²\n"
                    f"Error if using simple: {difference_pct:+.1f}%\n\n"
                    f"The planform area ({accurate_area:.4f} m²) has been\n"
                    f"set as your A_ref for aerodynamic analysis."
                )
            except Exception as e:
                print(f"Wetted area calculation failed: {e}")
                QMessageBox.information(
                    self, "Reference Area Calculated",
                    f"✓ Accurate planform area calculated!\n\n"
                    f"A_ref = {accurate_area:.4f} m²\n\n"
                    f"This is {difference_pct:+.1f}% different from\n"
                    f"simple w×h = {simple_area:.4f} m²"
                )
                    
        except Exception as e:
            # Fall back to simple
            QMessageBox.critical(
                self, "Calculation failed",
                f"Accurate calculation failed with error:\n\n{str(e)}\n\n"
                f"Falling back to simple approximation:\n"
                f"A_ref = {simple_area:.4f} m² (width × height)"
            )
            self.aref_spin.setValue(simple_area)
            self.info_label.setText(
                f"A_ref set to {simple_area:.4f} m² (w×h)\n"
                f"⚠️ Accurate calculation failed: {str(e)}"
            )

    # -------- Logic / callbacks -------- #

    def generate_waverider(self):
        """Instantiate the waverider object and update all views."""
        try:
            self.info_label.setText("Generating waverider...")
            QApplication.processEvents()

            M_inf = self.m_inf_spin.value()
            beta = self.beta_spin.value()
            height = self.height_spin.value()
            width = self.width_spin.value()

            dp = [
                self.x1_spin.value(),
                self.x2_spin.value(),
                self.x3_spin.value(),
                self.x4_spin.value(),
            ]

            n_planes = self.n_planes_spin.value()
            n_streamwise = self.n_streamwise_spin.value()
            n_upper_surface = self.n_us_spin.value()
            n_shockwave = self.n_sw_spin.value()
            delta_streamwise = self.delta_streamwise_spin.value()

            # Design-space constraint
            X1, X2 = dp[0], dp[1]
            constraint = X2 / ((1 - X1) ** 4) if X1 < 1 else float('inf')
            max_constraint = (7.0 / 64.0) * (width / height) ** 4

            if not (0.0 <= X1 < 1.0 and 0.0 <= X2 <= 1.0):
                QMessageBox.warning(self, "Invalid X1/X2 range",
                                    "X1 must be on [0,1), X2 must be in [0,1].")
                self.info_label.setText("Invalid design parameters X1/X2.")
                return

            if not (constraint < max_constraint):
                max_x2_for_x1 = max(0.0, min(1.0, max_constraint * (1.0 - X1) ** 4))
                suggested_x2 = round(max_x2_for_x1, 3)

                QMessageBox.warning(
                    self, "Design-space violation",
                    f"Constraint value: {constraint:.4f}\n"
                    f"Maximum allowed: {max_constraint:.4f}\n\n"
                    f"Suggestion: keep X1={X1:.3f}, choose X2 ≤ {suggested_x2:.3f}"
                )
                self.info_label.setText("Design-space constraint violated.")
                return

            # Build the waverider
            match_shockwave = self.match_shock_check.isChecked()
            self.waverider = wr(
                M_inf=M_inf,
                beta=beta,
                height=height,
                width=width,
                dp=dp,
                n_upper_surface=n_upper_surface,
                n_shockwave=n_shockwave,
                n_planes=n_planes,
                n_streamwise=n_streamwise,
                delta_streamwise=delta_streamwise,
                match_shockwave=match_shockwave
            )
            
            # Calculate and display volume
            try:
                self.waverider_volume = calculate_waverider_volume(self.waverider)
                self.volume_label.setText(f"{self.waverider_volume:.4f}")
                self.volume_label.setStyleSheet("font-weight: bold; color: #0066cc;")
            except Exception as vol_err:
                self.waverider_volume = 0.0
                self.volume_label.setText("Error")
                self.volume_label.setStyleSheet("font-weight: bold; color: #cc0000;")
                print(f"Volume calculation error: {vol_err}")

            # Update plots
            self.update_all_views()

            # Enable mesh generation and analysis buttons
            self.generate_mesh_btn.setEnabled(True)
            self.mesh_gen_info.setText("✓ Ready to generate mesh")
            self.mesh_gen_info.setStyleSheet("color: green;")

            self.info_label.setText(
                "✓ Waverider generated successfully.\n\n"
                f"Length: {self.waverider.length:.3f} m\n"
                f"Width:  {width:.3f} m\n"
                f"Height: {height:.3f} m\n"
                f"Volume: {self.waverider_volume:.4f} m³\n"
                f"Constraint: {constraint:.4f} / {max_constraint:.4f}\n"
                f"Design point: [{dp[0]:.3f}, {dp[1]:.3f}, {dp[2]:.3f}, {dp[3]:.3f}]"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate waverider:\n\n{str(e)}")
            self.info_label.setText(f"Error: {str(e)}")

    def update_all_views(self):
        if self.waverider is None:
            return
        self.update_3d_view()
        self.canvas_base.plot_base_plane(self.waverider)
        self.canvas_le.plot_leading_edge(self.waverider)
        self.canvas_geom.plot_schematic(
            height=self.height_spin.value(),
            width=self.width_spin.value()
        )

    def update_3d_view(self):
        if self.waverider is None:
            return
        self.canvas_3d.plot_waverider(
            self.waverider,
            show_upper=self.show_upper_check.isChecked(),
            show_lower=self.show_lower_check.isChecked(),
            show_le=self.show_le_check.isChecked(),
            show_wireframe=self.show_wireframe_check.isChecked(),
        )

    def export_cad(self):
        """Export STEP via cad_export.to_CAD and save STL for analysis"""
        if self.waverider is None:
            QMessageBox.warning(self, "No waverider",
                                "Generate a waverider before exporting CAD.")
            return

        # Ask for geometry type
        items = ["Full vehicle (mirrored, both sides)", "Half only (right side)"]
        choice, ok = QInputDialog.getItem(
            self, "Export options",
            "Select geometry to export:",
            items, 0, False
        )
        if not ok:
            return

        sides = "both" if "Full vehicle" in choice else "right"

        # File dialog
        default_name = "waverider.step"
        filter_str = "STEP files (*.step *.stp);;All files (*)"

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export waverider",
            default_name,
            filter_str
        )
        if not filename:
            return

        try:
            self.info_label.setText("Exporting STEP file...")
            QApplication.processEvents()

            to_CAD(
                waverider=self.waverider,
                sides=sides,
                export=True,
                filename=filename,
                scale=1000.0
            )
            QMessageBox.information(
                self, "Export successful",
                f"STEP file exported to:\n{filename}\n\n"
                f"Units: MILLIMETERS (m×1000)\n\n"
                f"To create STL mesh for analysis:\n"
                f"1. Go to 'Aerodynamic Analysis' tab\n"
                f"2. Set mesh parameters\n"
                f"3. Click 'Generate STL Mesh'"
            )

            self.info_label.setText(f"✓ STEP file exported to: {filename}")

        except Exception as e:
            QMessageBox.critical(
                self, "Export error",
                f"Failed to export CAD file:\n\n{str(e)}"
            )
            self.info_label.setText(f"Export error: {str(e)}")

    def set_mesh_preset(self, min_size, max_size):
        """Set mesh size preset"""
        self.mesh_min_spin.setValue(min_size)
        self.mesh_max_spin.setValue(max_size)
    
    def generate_stl_mesh(self):
        """Generate high-quality STL mesh using Gmsh"""
        if self.waverider is None:
            QMessageBox.warning(
                self, "No waverider",
                "Generate a waverider first before creating mesh."
            )
            return
        
        try:
            # Check if gmsh is available
            try:
                import gmsh
            except ImportError:
                reply = QMessageBox.question(
                    self, "Gmsh not installed",
                    "Gmsh is required for high-quality mesh generation.\n\n"
                    "Install with: pip install gmsh\n\n"
                    "Continue with lower-quality CadQuery meshing instead?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
                else:
                    # Fallback to CadQuery
                    self.generate_stl_mesh_cadquery()
                    return
            
            # Get mesh parameters
            min_size = self.mesh_min_spin.value()
            max_size = self.mesh_max_spin.value()

            self.mesh_gen_info.setText("Generating mesh with Gmsh...")
            self.mesh_gen_info.setStyleSheet("color: orange;")
            QApplication.processEvents()

            print(f"\n{'='*60}")
            print(f"Gmsh Mesh Generation")
            print(f"{'='*60}")
            print(f"Min element size:  ({min_size:.5f} m)")
            print(f"Max element size:  ({max_size:.5f} m)")
            sys.stdout.flush()

            # First, export STEP file temporarily
            import tempfile
            import cadquery as cq

            temp_step = tempfile.NamedTemporaryFile(suffix='.step', delete=False).name

            print("Exporting STEP geometry...")
            sys.stdout.flush()

            to_CAD(
                waverider=self.waverider,
                sides="both",
                export=True,
                filename=temp_step,
                scale=1000.0  # 1.0 = meters (geometry units)
            )
            
            # Initialize Gmsh
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)

            # Verbosity + logger for diagnosing "hangs"
            gmsh.option.setNumber("General.Verbosity", 5)
            gmsh.option.setNumber("General.Terminal", 1)  # stream gmsh messages to stdout
            gmsh.logger.start()

            # OCC healing options - AGGRESSIVE settings to fix problematic geometry
            gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
            gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
            gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
            gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
            gmsh.option.setNumber("Geometry.Tolerance", 1e-6)  # Tolerance for geometry operations
            gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-6)
            
            # Additional mesh quality options to prevent getting stuck
            gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.1)
            gmsh.option.setNumber("Mesh.AnisoMax", 1e10)
            gmsh.option.setNumber("Mesh.AllowSwapAngle", 30)

            print("Loading geometry into Gmsh...")
            sys.stdout.flush()

            # Load STEP file
            gmsh.model.occ.importShapes(temp_step)
            gmsh.model.occ.synchronize()

            # Remove duplicate entities introduced by STEP import, then resync
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            # Set mesh parameters
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)
            gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay (usually best)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
            
            # Limit element count to prevent infinite refinement
            gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
            gmsh.option.setNumber("Mesh.MinimumCirclePoints", 7)  # Reduce for faster meshing
            
            print("Generating surface mesh...")
            print("(If this takes >60 seconds, try increasing mesh size or simplifying geometry)")
            sys.stdout.flush()
            
            # Generate 2D mesh (surface only)
            try:
                gmsh.model.mesh.generate(2)
            except Exception as mesh_err:
                print(f"Mesh generation failed with Algorithm 6, trying Algorithm 1...")
                sys.stdout.flush()
                # Try simpler MeshAdapt algorithm as fallback
                gmsh.option.setNumber("Mesh.Algorithm", 1)  # MeshAdapt
                gmsh.model.mesh.generate(2)

            # Dump gmsh log (helps identify tiny edges/faces or repeated refinement)
            try:
                log_lines = gmsh.logger.get()
                if log_lines:
                    print("--- Gmsh log (most recent) ---")
                    for line in log_lines[-400:]:  # limit output
                        print(line)
                    print("--- End Gmsh log ---")
                    sys.stdout.flush()
            except Exception:
                pass
            
            # Get mesh statistics
            num_nodes = len(gmsh.model.mesh.getNodes()[0])
            num_triangles = len(gmsh.model.mesh.getElementsByType(2)[0])
            
            print(f"✓ Mesh generated: {num_triangles} triangles, {num_nodes} nodes")
            sys.stdout.flush()
            
            # Save STL
            stl_filename, _ = QFileDialog.getSaveFileName(
                self, "Save STL Mesh",
                "waverider_mesh.stl",
                "STL files (*.stl);;All files (*)"
            )
            
            if not stl_filename:
                try:
                    gmsh.logger.stop()
                except Exception:
                    pass
                gmsh.finalize()
                os.unlink(temp_step)
                self.mesh_gen_info.setText("Mesh generation cancelled")
                self.mesh_gen_info.setStyleSheet("color: gray; font-style: italic;")
                return
            
            print(f"Saving STL to: {stl_filename}")
            sys.stdout.flush()
            
            gmsh.write(stl_filename)

            # Stop logger before finalizing
            try:
                gmsh.logger.stop()
            except Exception:
                pass

            gmsh.finalize()
            
            # Clean up temp file
            os.unlink(temp_step)
            
            # Store STL file path
            self.last_stl_file = stl_filename
            
            # Get file size
            file_size_kb = os.path.getsize(stl_filename) / 1024
            
            print(f"✓ STL saved: {file_size_kb:.1f} KB")
            print(f"{'='*60}\n")
            sys.stdout.flush()
            
            # Update UI
            self.mesh_gen_info.setText(
                f"✓ Mesh generated: {num_triangles} triangles, {file_size_kb:.1f} KB"
            )
            self.mesh_gen_info.setStyleSheet("color: green;")
            
            # Enable buttons
            self.load_mesh_btn.setEnabled(True)
            self.run_analysis_btn.setEnabled(True)
            
            # Enable sweep button if sweep checkbox is enabled
            if hasattr(self, 'enable_sweep_check') and hasattr(self, 'run_sweep_btn'):
                if self.enable_sweep_check.isChecked():
                    self.run_sweep_btn.setEnabled(True)
                    # Update the info label
                    if hasattr(self, 'update_sweep_info'):
                        self.update_sweep_info()
            
            # Show success message
            QMessageBox.information(
                self, "Mesh Generated",
                f"High-quality mesh generated successfully!\n\n"
                f"Triangles: {num_triangles}\n"
                f"Nodes: {num_nodes}\n"
                f"File size: {file_size_kb:.1f} KB\n"
                f"Saved to: {stl_filename}\n\n"
                f"You can now preview the mesh or run analysis."
            )
            
        except Exception as e:
            # Ensure gmsh is finalized if it was initialized
            try:
                try:
                    gmsh.logger.stop()
                except Exception:
                    pass
                gmsh.finalize()
            except Exception:
                pass
            try:
                if 'temp_step' in locals() and os.path.exists(temp_step):
                    os.unlink(temp_step)
            except Exception:
                pass

            self.mesh_gen_info.setText(f"✗ Mesh generation failed")
            self.mesh_gen_info.setStyleSheet("color: red;")
            QMessageBox.critical(
                self, "Mesh Generation Failed",
                f"Could not generate mesh:\n\n{str(e)}"
            )
            print(f"\n✗ Mesh generation error: {str(e)}\n")
            sys.stdout.flush()
    
    def generate_stl_mesh_cadquery(self):
        """Fallback: Generate STL using CadQuery (lower quality)"""
        try:
            import cadquery as cq
            
            self.mesh_gen_info.setText("Generating mesh with CadQuery...")
            QApplication.processEvents()
            
            waverider_cad = to_CAD(
                waverider=self.waverider,
                sides="both",
                export=False,
                filename="",
                scale=1000.0
            )
            
            # Ask for filename
            stl_filename, _ = QFileDialog.getSaveFileName(
                self, "Save STL Mesh",
                "waverider_mesh.stl",
                "STL files (*.stl);;All files (*)"
            )
            
            if not stl_filename:
                return
            
            # Export with best settings CadQuery can do
            cq.exporters.export(
                waverider_cad,
                stl_filename,
                tolerance=0.001,
                angularTolerance=0.05
            )
            
            self.last_stl_file = stl_filename
            
            # Get stats
            from stl import mesh as stl_mesh
            mesh_data = stl_mesh.Mesh.from_file(stl_filename)
            num_triangles = len(mesh_data.vectors)
            file_size_kb = os.path.getsize(stl_filename) / 1024
            
            self.mesh_gen_info.setText(
                f"✓ Mesh generated (CadQuery): {num_triangles} triangles"
            )
            self.mesh_gen_info.setStyleSheet("color: orange;")
            
            self.load_mesh_btn.setEnabled(True)
            self.run_analysis_btn.setEnabled(True)
            
            # Enable sweep button if sweep checkbox is enabled
            if hasattr(self, 'enable_sweep_check') and hasattr(self, 'run_sweep_btn'):
                if self.enable_sweep_check.isChecked():
                    self.run_sweep_btn.setEnabled(True)
                    if hasattr(self, 'update_sweep_info'):
                        self.update_sweep_info()
            
            QMessageBox.warning(
                self, "Lower Quality Mesh",
                f"Mesh generated with CadQuery (not Gmsh).\n\n"
                f"Quality will be lower than with Gmsh.\n"
                f"Install gmsh for better results: pip install gmsh\n\n"
                f"Triangles: {num_triangles}\n"
                f"File: {stl_filename}"
            )
            
        except Exception as e:
            self.mesh_gen_info.setText("✗ Mesh generation failed")
            self.mesh_gen_info.setStyleSheet("color: red;")
            QMessageBox.critical(
                self, "Mesh Generation Failed",
                f"Could not generate mesh:\n\n{str(e)}"
            )

    def load_and_display_mesh(self):
        """Load and display the STL mesh in the preview"""
        if self.last_stl_file is None or not os.path.exists(self.last_stl_file):
            QMessageBox.warning(
                self, "No STL file",
                "No STL file found.\n\n"
                "Please export CAD first (this automatically creates the STL file)."
            )
            return
        
        try:
            # Try to import numpy-stl
            try:
                from stl import mesh as stl_mesh
            except ImportError:
                reply = QMessageBox.question(
                    self, "Missing dependency",
                    "The numpy-stl package is required for STL visualization.\n\n"
                    "Would you like installation instructions?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    QMessageBox.information(
                        self, "Installation",
                        "Install numpy-stl with:\n\n"
                        "pip install numpy-stl\n\n"
                        "or\n\n"
                        "conda install numpy-stl"
                    )
                return
            
            # Update status
            self.mesh_info_label.setText("Loading mesh...")
            QApplication.processEvents()
            
            # Load and display
            num_triangles = self.mesh_canvas.plot_stl_mesh(self.last_stl_file)
            
            # Update info
            file_size = os.path.getsize(self.last_stl_file) / 1024  # KB
            self.mesh_info_label.setText(
                f"✓ Loaded: {num_triangles} triangles, {file_size:.1f} KB"
            )
            self.mesh_info_label.setStyleSheet("color: green;")
            
        except Exception as e:
            self.mesh_info_label.setText(f"✗ Error loading mesh: {str(e)}")
            self.mesh_info_label.setStyleSheet("color: red;")
            QMessageBox.critical(
                self, "Mesh loading failed",
                f"Could not load STL mesh:\n\n{str(e)}"
            )
    
    def stop_analysis(self):
        """Stop the running analysis"""
        if self.analysis_worker is not None and self.analysis_worker.isRunning():
            reply = QMessageBox.question(
                self, "Stop Analysis",
                "Are you sure you want to stop the analysis?\n\n"
                "The solver will be terminated and no results will be available.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Terminate the thread
                print("\n⚠️  User requested analysis stop")
                print("Terminating worker thread...")
                sys.stdout.flush()
                
                self.analysis_worker.terminate()
                self.analysis_worker.wait()  # Wait for thread to finish
                
                # Reset UI
                self.analysis_progress.setVisible(False)
                self.run_analysis_btn.setEnabled(True)
                self.stop_analysis_btn.setVisible(False)
                
                self.results_text.append("\n⚠️  Analysis stopped by user")
                
                print("✓ Worker thread terminated")
                sys.stdout.flush()

    def run_analysis(self):
        """Run PySAGAS aerodynamic analysis"""
        if not PYSAGAS_AVAILABLE:
            QMessageBox.warning(
                self, "PySAGAS unavailable",
                "PySAGAS is not installed.\n\nInstall with: pip install pysagas"
            )
            return

        if self.waverider is None:
            QMessageBox.warning(
                self, "No waverider",
                "Generate a waverider first."
            )
            return

        # Check if we have an STL file
        if self.last_stl_file is None or not os.path.exists(self.last_stl_file):
            QMessageBox.warning(
                self, "No STL file",
                "No STL mesh found.\n\n"
                "Please generate the STL mesh first:\n"
                "1. Set mesh parameters (min/max element size)\n"
                "2. Click 'Generate STL Mesh with Gmsh'\n"
                "3. Review the mesh quality\n"
                "4. Then run analysis"
            )
            return

        # Get analysis parameters
        aoa = self.aoa_spin.value()
        A_ref = self.aref_spin.value()
        freestream_dict = {
            'mach': self.analysis_mach_spin.value(),
            'pressure': self.pressure_spin.value(),
            'temperature': self.temperature_spin.value()
        }

        # Disable run button, show stop button and progress
        self.run_analysis_btn.setEnabled(False)
        self.stop_analysis_btn.setVisible(True)
        self.analysis_progress.setVisible(True)
        self.analysis_progress.setRange(0, 0)  # Indeterminate

        self.results_text.setText("Starting analysis...\n")
        QApplication.processEvents()

        # Create and start worker thread
        self.analysis_worker = AnalysisWorker(
            self.last_stl_file,
            freestream_dict,
            aoa,
            A_ref
        )
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.analysis_worker.progress.connect(self.on_analysis_progress)
        self.analysis_worker.start()
        self.analysis_worker.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_worker.error.connect(self.analysis_worker.deleteLater)


    def on_analysis_progress(self, message):
        """Update progress message"""
        self.results_text.append(message)
        

    def on_analysis_finished(self, results):
        """Handle analysis completion"""
        self.analysis_progress.setVisible(False)
        self.run_analysis_btn.setEnabled(True)
        self.stop_analysis_btn.setVisible(False)

        # Format results
        result_text = "\n" + "="*60 + "\n"
        result_text += "AERODYNAMIC ANALYSIS RESULTS\n"
        result_text += "="*60 + "\n\n"

        result_text += f"Conditions:\n"
        result_text += f"  Mach number:     {self.analysis_mach_spin.value():.2f}\n"
        result_text += f"  Angle of attack: {self.aoa_spin.value():.2f}°\n"
        result_text += f"  Pressure:        {self.pressure_spin.value():.0f} Pa\n"
        result_text += f"  Temperature:     {self.temperature_spin.value():.2f} K\n"
        result_text += f"  Reference area:  {self.aref_spin.value():.3f} m²\n\n"

        result_text += f"Coefficients:\n"
        result_text += f"  CL (Coefficient of Lift):       {results['CL']:.6f}\n"
        result_text += f"  CD (Drag):       {results['CD']:.6f}\n"
        result_text += f"  Cm (Moment):     {results['Cm']:.6f}\n"
        result_text += f"  CL/CD Ratio:       {results['CL/CD']:.3f}\n\n"

        result_text += "="*60 + "\n"
        result_text += "Analysis complete! ✓\n"

        self.results_text.setText(result_text)
        
        # Clean up worker thread
        if self.analysis_worker:
            self.analysis_worker.quit()
            self.analysis_worker.wait()
            self.analysis_worker = None

        # Show summary message (non-blocking would be better, but this is okay)
        QMessageBox.information(
            self, "Analysis Complete",
            f"Analysis finished successfully!\n\n"
            f"CL = {results['CL']:.6f}\n"
            f"CD = {results['CD']:.6f}\n"
            f"CL/CD = {results['CL/CD']:.3f}"
        )

    def on_analysis_error(self, error_msg):
        """Handle analysis error"""
        self.analysis_progress.setVisible(False)
        self.run_analysis_btn.setEnabled(True)
        self.stop_analysis_btn.setVisible(False)
        
        # Clean up worker thread
        if self.analysis_worker:
            self.analysis_worker.quit()
            self.analysis_worker.wait()
            self.analysis_worker = None

        self.results_text.append(f"\n❌ Error: {error_msg}\n")

        QMessageBox.critical(
            self, "Analysis Failed",
            f"Analysis failed with error:\n\n{error_msg}"
        )


    # ========== AERODECK SWEEP METHODS ==========
    
    def on_sweep_enabled_changed(self, state):
        """Enable/disable sweep controls"""
        enabled = state == Qt.Checked
        self.aoa_min_spin.setEnabled(enabled)
        self.aoa_max_spin.setEnabled(enabled)
        self.aoa_step_spin.setEnabled(enabled)
        self.mach_min_spin.setEnabled(enabled)
        self.mach_max_spin.setEnabled(enabled)
        self.mach_step_spin.setEnabled(enabled)
        
        # Check if STL file exists
        stl_exists = False
        if self.last_stl_file is not None:
            stl_exists = os.path.exists(self.last_stl_file)
        
        # Enable run button if sweep is enabled AND we have a valid STL file
        self.run_sweep_btn.setEnabled(enabled and stl_exists)
        
        if enabled:
            self.update_sweep_info()
            if stl_exists:
                self.sweep_info_label.setStyleSheet("color: #2196F3; font-weight: bold;")
            else:
                self.sweep_info_label.setText(
                    "⚠️ Generate STL mesh first, then run sweep"
                )
                self.sweep_info_label.setStyleSheet("color: orange; font-style: italic;")
        else:
            self.sweep_info_label.setText("Enable sweep to analyze multiple flight conditions")
            self.sweep_info_label.setStyleSheet("color: #666; font-style: italic;")
    
    def update_sweep_info(self):
        """Update sweep info label with point count"""
        try:
            aoa_range = np.arange(self.aoa_min_spin.value(), 
                                  self.aoa_max_spin.value() + 0.01, 
                                  self.aoa_step_spin.value())
            mach_range = np.arange(self.mach_min_spin.value(), 
                                   self.mach_max_spin.value() + 0.01, 
                                   self.mach_step_spin.value())
            n_points = len(aoa_range) * len(mach_range)
            est_time = n_points * 45 / 60  # ~45 sec per point, in minutes
            
            self.sweep_info_label.setText(
                f"📊 {len(aoa_range)} AoA × {len(mach_range)} Mach = {n_points} points "
                f"(~{est_time:.0f} min)"
            )
            self.sweep_info_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        except:
            pass
    
    def run_aerodeck_sweep(self):
        """Run AeroDeck sweep analysis using PySAGAS"""
        if not PYSAGAS_AVAILABLE:
            QMessageBox.warning(self, "PySAGAS unavailable", 
                              "PySAGAS is not installed.")
            return
        
        if self.last_stl_file is None or not os.path.exists(self.last_stl_file):
            QMessageBox.warning(self, "No STL", "Generate STL mesh first.")
            return
        
        # Get sweep ranges
        aoa_list = list(np.arange(self.aoa_min_spin.value(), 
                              self.aoa_max_spin.value() + 0.01, 
                              self.aoa_step_spin.value()))
        mach_list = list(np.arange(self.mach_min_spin.value(), 
                               self.mach_max_spin.value() + 0.01, 
                               self.mach_step_spin.value()))
        
        n_points = len(aoa_list) * len(mach_list)
        
        reply = QMessageBox.question(
            self, "Run AeroDeck Sweep",
            f"This will run {n_points} PySAGAS analyses:\n\n"
            f"  AoA: {aoa_list[0]:.1f}° to {aoa_list[-1]:.1f}° ({len(aoa_list)} points)\n"
            f"  Mach: {mach_list[0]:.1f} to {mach_list[-1]:.1f} ({len(mach_list)} points)\n\n"
            f"Estimated time: ~{n_points * 45 / 60:.0f} minutes\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Get other parameters
        pressure = self.pressure_spin.value()
        temperature = self.temperature_spin.value()
        A_ref = self.aref_spin.value()
        
        # Disable buttons
        self.run_sweep_btn.setEnabled(False)
        self.run_analysis_btn.setEnabled(False)
        self.analysis_progress.setVisible(True)
        self.analysis_progress.setRange(0, n_points)
        self.analysis_progress.setValue(0)
        
        # Results storage
        self.aerodeck_results = {
            'aoa': [], 'mach': [], 
            'CL': [], 'CD': [], 'Cm': [], 'CL_CD': [],
            'pressure': pressure, 'temperature': temperature, 'A_ref': A_ref
        }
        
        self.results_text.clear()
        self.results_text.append("=" * 50)
        self.results_text.append("AERODECK SWEEP ANALYSIS")
        self.results_text.append("=" * 50)
        self.results_text.append(f"\nSTL File: {self.last_stl_file}")
        self.results_text.append(f"Pressure: {pressure:.0f} Pa")
        self.results_text.append(f"Temperature: {temperature:.2f} K")
        self.results_text.append(f"A_ref: {A_ref:.4f} m²")
        self.results_text.append(f"\nRunning {n_points} analysis points...\n")
        QApplication.processEvents()
        
        # Run sweep using PySAGAS - matching their AeroDeck example exactly
        try:
            from pysagas.flow import FlowState
            from pysagas.geometry.parsers import MeshIO
            from pysagas.cfd import OPM, AeroDeck
            
            # Load mesh using MeshIO
            self.results_text.append("Loading mesh...")
            QApplication.processEvents()
            
            cells = MeshIO.load_from_file(self.last_stl_file)
            self.results_text.append(f"  Loaded {len(cells)} cells")
            self.results_text.append(f"  Using A_ref = {A_ref:.4f} m²\n")
            QApplication.processEvents()
            
            # Instantiate flow solver (NO freestream at init - like their example)
            flow_solver = OPM(cells)
            
            # Create AeroDeck for storing results
            aerodeck = AeroDeck(inputs=["aoa", "mach"])
            
            # Perform sweep (aoa outer loop, mach inner - like their example)
            point_count = 0
            for aoa in aoa_list:
                for mach in mach_list:
                    point_count += 1
                    
                    # Update progress
                    self.analysis_progress.setValue(point_count)
                    QApplication.processEvents()
                    
                    try:
                        # Define freestream with BOTH mach AND aoa
                        freestream = FlowState(
                            mach=float(mach), 
                            pressure=float(pressure), 
                            temperature=float(temperature),
                            aoa=float(aoa)
                        )
                        
                        # Run flow solver
                        aero = flow_solver.solve(freestream=freestream)
                        
                        # Insert results into AeroDeck
                        aerodeck.insert(result=aero, aoa=aoa, mach=mach)
                        
                        self.results_text.append(f"  α={aoa:+.1f}°, M={mach:.1f} → Done")
                        
                    except Exception as e:
                        self.results_text.append(
                            f"  α={aoa:+.1f}°, M={mach:.1f} → FAILED: {str(e)[:50]}"
                        )
            
            # Save AeroDeck to CSV using PySAGAS method
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            aerodeck_filename = f"aerodeck_{timestamp}"
            aerodeck.to_csv(aerodeck_filename)
            self.aerodeck_csv_file = f"{aerodeck_filename}.csv"
            self.results_text.append(f"\n📁 AeroDeck saved: {self.aerodeck_csv_file}")
            
            # Read results back from CSV to populate our results dict
            self.results_text.append("\nReading results from CSV...")
            import pandas as pd
            
            # AeroDeck saves as comma-separated CSV
            df = pd.read_csv(self.aerodeck_csv_file)
            
            # Debug: show columns found
            self.results_text.append(f"  Columns found: {list(df.columns)}")
            self.results_text.append(f"  Number of points: {len(df)}")
            
            # Clear and repopulate results
            self.aerodeck_results = {
                'aoa': df['aoa'].tolist(),
                'mach': df['mach'].tolist(),
                'CL': df['CL'].tolist(),
                'CD': df['CD'].tolist(),
                'Cm': df['Cm'].tolist(),
                'CL_CD': (df['CL'] / df['CD']).tolist(),
                'pressure': pressure, 
                'temperature': temperature, 
                'A_ref': A_ref
            }
            
            # Display results
            self.results_text.append("\nResults:")
            for i in range(len(df)):
                aoa = df['aoa'].iloc[i]
                mach = df['mach'].iloc[i]
                CL = df['CL'].iloc[i]
                CD = df['CD'].iloc[i]
                CL_CD = CL / CD if abs(CD) > 1e-10 else 0.0
                self.results_text.append(
                    f"  α={aoa:+.1f}°, M={mach:.1f} → CL={CL:.4f}, CD={CD:.4f}, CL/CD={CL_CD:.2f}"
                )
            
            self.results_text.append("\n" + "=" * 50)
            self.results_text.append("SWEEP COMPLETE")
            self.results_text.append("=" * 50)
            
            # Summary statistics
            valid_cl_cd = [x for x in self.aerodeck_results['CL_CD'] if not np.isnan(x) and x != 0]
            if valid_cl_cd:
                max_ld = max(valid_cl_cd)
                max_idx = self.aerodeck_results['CL_CD'].index(max_ld)
                best_aoa = self.aerodeck_results['aoa'][max_idx]
                best_mach = self.aerodeck_results['mach'][max_idx]
                self.results_text.append(f"\n🏆 Best CL/CD = {max_ld:.2f} at α={best_aoa:.1f}°, M={best_mach:.1f}")
            
            # Enable plot button
            self.plot_sweep_btn.setEnabled(True)
            
        except Exception as e:
            self.results_text.append(f"\n❌ Sweep failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.analysis_progress.setVisible(False)
            self.run_sweep_btn.setEnabled(True)
            self.run_analysis_btn.setEnabled(True)
    
    def save_aerodeck_csv(self):
        """Save AeroDeck results to CSV"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aerodeck_{timestamp}.csv"
            
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['AoA_deg', 'Mach', 'CL', 'CD', 'Cm', 'CL_CD'])
                
                for i in range(len(self.aerodeck_results['aoa'])):
                    writer.writerow([
                        self.aerodeck_results['aoa'][i],
                        self.aerodeck_results['mach'][i],
                        self.aerodeck_results['CL'][i],
                        self.aerodeck_results['CD'][i],
                        self.aerodeck_results['Cm'][i],
                        self.aerodeck_results['CL_CD'][i]
                    ])
            
            self.results_text.append(f"\n📁 Results saved to: {filename}")
            
        except Exception as e:
            self.results_text.append(f"\n⚠️ Could not save CSV: {str(e)}")
    
    def plot_aerodeck_results(self):
        """Plot AeroDeck sweep results in a separate Qt window"""
        if not hasattr(self, 'aerodeck_results') or not self.aerodeck_results['aoa']:
            QMessageBox.warning(self, "No Data", "Run a sweep first!")
            return
        
        try:
            # Use the Qt-compatible plot window
            if AERODECK_PLOT_AVAILABLE:
                # Create and show the plot window (store reference to prevent garbage collection)
                self.aerodeck_plot_window = AerodeckPlotWindow(self.aerodeck_results, parent=self)
                self.aerodeck_plot_window.show()
            else:
                # Fallback: try to use matplotlib with Qt5Agg backend
                import matplotlib
                matplotlib.use('Qt5Agg')
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                
                aoa = np.array(self.aerodeck_results['aoa'])
                mach = np.array(self.aerodeck_results['mach'])
                CL = np.array(self.aerodeck_results['CL'])
                CD = np.array(self.aerodeck_results['CD'])
                CL_CD = np.array(self.aerodeck_results['CL_CD'])
                
                unique_aoa = np.unique(aoa)
                unique_mach = np.unique(mach)
                
                fig = plt.figure(figsize=(14, 9))
                
                can_surface = (len(unique_aoa) > 1 and len(unique_mach) > 1 and 
                              len(aoa) == len(unique_aoa) * len(unique_mach))
                
                if can_surface:
                    AOA, MACH = np.meshgrid(unique_aoa, unique_mach, indexing='ij')
                    CL_grid = CL.reshape(len(unique_aoa), len(unique_mach))
                    CL_CD_grid = CL_CD.reshape(len(unique_aoa), len(unique_mach))
                    
                    ax1 = fig.add_subplot(221, projection='3d')
                    ax1.plot_surface(AOA, MACH, CL_CD_grid, cmap='viridis', alpha=0.9)
                    ax1.set_xlabel('AoA (°)')
                    ax1.set_ylabel('Mach')
                    ax1.set_zlabel('CL/CD')
                    ax1.set_title('CL/CD Ratio')
                    
                    ax2 = fig.add_subplot(222, projection='3d')
                    ax2.plot_surface(AOA, MACH, CL_grid, cmap='coolwarm', alpha=0.9)
                    ax2.set_xlabel('AoA (°)')
                    ax2.set_ylabel('Mach')
                    ax2.set_zlabel('CL')
                    ax2.set_title('Lift Coefficient')
                    
                    ax3 = fig.add_subplot(223)
                    for m in unique_mach:
                        mask = mach == m
                        ax3.plot(aoa[mask], CL[mask], 'o-', label=f'M={m:.1f}')
                    ax3.set_xlabel('AoA (°)')
                    ax3.set_ylabel('CL')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    
                    ax4 = fig.add_subplot(224)
                    for m in unique_mach:
                        mask = mach == m
                        ax4.plot(aoa[mask], CL_CD[mask], 's-', label=f'M={m:.1f}')
                    ax4.set_xlabel('AoA (°)')
                    ax4.set_ylabel('CL/CD')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                else:
                    ax1 = fig.add_subplot(221, projection='3d')
                    ax1.scatter(aoa, mach, CL_CD, c=CL_CD, cmap='viridis')
                    ax1.set_xlabel('AoA (°)')
                    ax1.set_ylabel('Mach')
                    ax1.set_zlabel('CL/CD')
                    
                    ax2 = fig.add_subplot(222, projection='3d')
                    ax2.scatter(aoa, mach, CL, c=CL, cmap='coolwarm')
                    ax2.set_xlabel('AoA (°)')
                    ax2.set_ylabel('Mach')
                    ax2.set_zlabel('CL')
                    
                    ax3 = fig.add_subplot(223)
                    ax3.scatter(aoa, CL, c=mach, cmap='plasma')
                    ax3.set_xlabel('AoA (°)')
                    ax3.set_ylabel('CL')
                    
                    ax4 = fig.add_subplot(224)
                    ax4.scatter(aoa, CL_CD, c=mach, cmap='plasma')
                    ax4.set_xlabel('AoA (°)')
                    ax4.set_ylabel('CL/CD')
                
                fig.suptitle('AeroDeck Sweep Results', fontsize=14, fontweight='bold')
                fig.tight_layout()
                
                # Use non-blocking show
                plt.ion()
                plt.show(block=False)
                plt.pause(0.1)
            
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Could not create plots:\n{str(e)}")
            import traceback
            traceback.print_exc()

    # ========== END AERO ANALYSIS METHODS ==========

# -------------------- Entry point -------------------- #

    # ========== END AERO ANALYSIS METHODS ==========

def main():
    """Main application entry point"""
    # Required for Windows multiprocessing support
    import multiprocessing as mp
    mp.freeze_support()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Force dot as decimal separator (not comma)
    from PyQt5.QtCore import QLocale
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
    
    gui = WaveriderGUI()
    gui.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()