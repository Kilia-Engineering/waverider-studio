#!/usr/bin/env python3
"""
Leading Edge Blunting Tab for Waverider GUI

Provides an interactive interface for applying and visualizing leading edge
blunting on waverider geometries, including:
- Real-time 3D preview of blunted geometry
- Parameter controls for blunting profile
- Thermal analysis visualization
- Export options

Author: Angelos & Claude
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QSlider, QScrollArea, QSplitter, QMessageBox,
    QTabWidget, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

# Import blunting modules - try simple version first (more robust)
BLUNTING_AVAILABLE = False
SIMPLE_BLUNTING = False

try:
    from simple_blunting import simple_blunt_waverider, SimpleBluntingParams
    BLUNTING_AVAILABLE = True
    SIMPLE_BLUNTING = True
    print("Using simple blunting module (recommended)")
except ImportError:
    pass

# Fallback to complex blunting if simple not available
if not BLUNTING_AVAILABLE:
    try:
        from leading_edge_blunting import (
            blunt_waverider, BluntingParameters, BluntedWaverider,
            estimate_stagnation_heating_from_mach, recommend_nose_radius,
            compute_heating_vs_radius, get_atmosphere_properties
        )
        BLUNTING_AVAILABLE = True
        SIMPLE_BLUNTING = False
        print("Using complex blunting module")
    except ImportError as e:
        print(f"Blunting module not available: {e}")

# Import thermal functions (may be in either module)
try:
    from leading_edge_blunting import (
        estimate_stagnation_heating_from_mach, recommend_nose_radius,
        compute_heating_vs_radius, get_atmosphere_properties
    )
    THERMAL_AVAILABLE = True
except ImportError:
    THERMAL_AVAILABLE = False
    # Define dummy functions
    def estimate_stagnation_heating_from_mach(mach, radius, altitude):
        return 1e6  # 1 MW/m² placeholder
    def recommend_nose_radius(mach, max_heating, altitude):
        return 0.01  # 10mm placeholder
    def compute_heating_vs_radius(radii, mach, altitude):
        return np.ones_like(radii) * 1e6
    def get_atmosphere_properties(altitude):
        return {'temperature': 220, 'pressure': 2500, 'density': 0.04}


class BluntingCanvas3D(FigureCanvas):
    """3D canvas for visualizing blunted waverider geometry."""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Initialize plot
        self.ax.set_xlabel('X (Streamwise) [m]')
        self.ax.set_ylabel('Y (Vertical) [m]')
        self.ax.set_zlabel('Z (Spanwise) [m]')
        self.ax.set_title('Blunted Waverider Preview')
    
    def plot_blunted_waverider(self, blunted_waverider, original_waverider=None,
                                show_original=True, show_blunted=True,
                                show_le=True, focus_le=False):
        """
        Plot the blunted waverider geometry.
        
        Parameters:
        -----------
        blunted_waverider : BluntedWaverider
            The blunted waverider object
        original_waverider : waverider, optional
            Original sharp waverider for comparison
        show_original : bool
            Show original geometry (in blue, transparent)
        show_blunted : bool
            Show blunted geometry (in red)
        show_le : bool
            Show leading edge curves
        focus_le : bool
            Zoom in on leading edge region
        """
        self.ax.clear()
        
        # Plot original if available and requested
        if show_original and original_waverider is not None:
            self._plot_waverider_surfaces(original_waverider, 
                                          color='blue', alpha=0.2, 
                                          label='Original')
            if show_le:
                le = original_waverider.leading_edge
                # Plot leading edge: X, Y, Z (matching main 3D view)
                self.ax.plot(le[:, 0], le[:, 1], le[:, 2], 
                            'b-', linewidth=2, alpha=0.5, label='Original LE')
                # Mirror
                self.ax.plot(le[:, 0], le[:, 1], -le[:, 2], 
                            'b-', linewidth=2, alpha=0.5)
        
        # Plot blunted geometry
        if show_blunted and blunted_waverider is not None:
            self._plot_waverider_surfaces(blunted_waverider,
                                          color='red', alpha=0.6,
                                          label='Blunted')
            if show_le:
                le = blunted_waverider.leading_edge
                # Plot leading edge: X, Y, Z (matching main 3D view)
                self.ax.plot(le[:, 0], le[:, 1], le[:, 2],
                            'r-', linewidth=3, label='Blunted LE')
                # Mirror
                self.ax.plot(le[:, 0], le[:, 1], -le[:, 2],
                            'r-', linewidth=3)
        
        # Auto-scale axes based on geometry with EQUAL aspect ratio
        waverider_for_scaling = blunted_waverider if blunted_waverider is not None else original_waverider
        if waverider_for_scaling is not None:
            # Get bounds from waverider
            if hasattr(waverider_for_scaling, 'upper_surface_x') and waverider_for_scaling.upper_surface_x is not None:
                x_min = waverider_for_scaling.upper_surface_x.min()
                x_max = waverider_for_scaling.upper_surface_x.max()
                y_min = waverider_for_scaling.upper_surface_y.min()
                y_max = waverider_for_scaling.upper_surface_y.max()
                z_max = waverider_for_scaling.upper_surface_z.max()
            else:
                # Fallback to leading edge bounds
                le = waverider_for_scaling.leading_edge
                x_min, x_max = le[:, 0].min(), le[:, 0].max()
                y_min, y_max = le[:, 1].min(), le[:, 1].max()
                z_max = le[:, 2].max()
            
            # Compute ranges
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = 2 * z_max  # Full span (mirrored)
            
            # Find the maximum range for equal scaling
            max_range = max(x_range, y_range, z_range)
            
            # Center points
            x_mid = (x_max + x_min) / 2
            y_mid = (y_max + y_min) / 2
            z_mid = 0  # Symmetric about z=0
            
            if not focus_le:
                # Set equal aspect ratio by using the same range for all axes
                self.ax.set_xlim([x_mid - max_range/2, x_mid + max_range/2])
                self.ax.set_ylim([y_mid - max_range/2, y_mid + max_range/2])
                self.ax.set_zlim([z_mid - max_range/2, z_mid + max_range/2])
                
                # Use set_box_aspect for truly equal scaling (matplotlib 3.3+)
                try:
                    self.ax.set_box_aspect([1, 1, 1])
                except AttributeError:
                    pass  # Older matplotlib version
        
        # Set view (zoom to LE if requested)
        if focus_le and blunted_waverider is not None:
            # Zoom to leading edge region
            le = blunted_waverider.leading_edge
            x_min = le[:, 0].min()
            x_max = x_min + blunted_waverider.length * 0.2  # First 20% of length
            y_min = le[:, 1].min() - 0.1
            y_max = le[:, 1].max() + 0.1
            z_max = le[:, 2].max() * 1.1
            
            self.ax.set_xlim([x_min, x_max])
            self.ax.set_ylim([y_min, y_max])
            self.ax.set_zlim([-z_max, z_max])
        
        self.ax.set_xlabel('X (Streamwise) [m]')
        self.ax.set_ylabel('Y (Vertical) [m]')
        self.ax.set_zlabel('Z (Spanwise) [m]')
        self.ax.set_title('Blunted Waverider Preview')
        
        # Set view angle to match main 3D view
        # elev=20 looks from slightly above, azim=-60 for standard isometric
        self.ax.view_init(elev=20, azim=-60)
        
        # Only add legend if there are labeled artists
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc='upper right')
        
        self.draw()
    
    def _plot_waverider_surfaces(self, waverider_obj, color='cyan', alpha=0.6, label=None):
        """Plot waverider surfaces from streamlines."""
        from scipy.interpolate import interp1d
        
        # Handle different waverider types
        # Standard waverider has upper_surface_streams (list of 2D arrays)
        # SHADOW waverider has upper_surface (3D array: spanwise x streamwise x xyz)
        
        if hasattr(waverider_obj, 'lower_surface_streams') and waverider_obj.lower_surface_streams is not None:
            # Standard waverider format
            lower_streams = waverider_obj.lower_surface_streams
            upper_streams = waverider_obj.upper_surface_streams
        elif hasattr(waverider_obj, 'lower_surface') and waverider_obj.lower_surface is not None:
            # SHADOW waverider format - convert 3D arrays to list of streams
            lower_surface = waverider_obj.lower_surface  # shape: (n_span, n_streamwise, 3)
            upper_surface = waverider_obj.upper_surface
            
            # Convert to list of streams (each stream is one spanwise station)
            lower_streams = [lower_surface[i, :, :] for i in range(lower_surface.shape[0])]
            upper_streams = [upper_surface[i, :, :] for i in range(upper_surface.shape[0])]
        else:
            print(f"Warning: Unknown waverider format - cannot plot surfaces")
            return
        
        # Plot lower surface from streamlines
        n_streams = len(lower_streams)
        for i in range(n_streams - 1):
            stream1 = lower_streams[i]
            stream2 = lower_streams[i + 1]
            
            # Handle different lengths by interpolating
            n_points = min(len(stream1), len(stream2), 30)  # Limit for performance
            
            if len(stream1) > 1 and len(stream2) > 1:
                t1 = np.linspace(0, 1, len(stream1))
                t2 = np.linspace(0, 1, len(stream2))
                t_common = np.linspace(0, 1, n_points)
                
                try:
                    stream1_x = interp1d(t1, stream1[:, 0], fill_value='extrapolate')(t_common)
                    stream1_y = interp1d(t1, stream1[:, 1], fill_value='extrapolate')(t_common)
                    stream1_z = interp1d(t1, stream1[:, 2], fill_value='extrapolate')(t_common)
                    stream2_x = interp1d(t2, stream2[:, 0], fill_value='extrapolate')(t_common)
                    stream2_y = interp1d(t2, stream2[:, 1], fill_value='extrapolate')(t_common)
                    stream2_z = interp1d(t2, stream2[:, 2], fill_value='extrapolate')(t_common)
                    
                    # Create surface - right half (X, Y, Z order to match main view)
                    x_surf = np.array([stream1_x, stream2_x])
                    y_surf = np.array([stream1_y, stream2_y])
                    z_surf = np.array([stream1_z, stream2_z])
                    
                    self.ax.plot_surface(x_surf, y_surf, z_surf,
                                        color=color, alpha=alpha, 
                                        edgecolor='none', shade=True)
                    
                    # Mirror - left half (negative Z)
                    self.ax.plot_surface(x_surf, y_surf, -z_surf,
                                        color=color, alpha=alpha,
                                        edgecolor='none', shade=True)
                except:
                    pass  # Skip problematic stream pairs
        
        # Plot upper surface from streams
        n_upper_streams = len(upper_streams)
        for i in range(n_upper_streams - 1):
            stream1 = upper_streams[i]
            stream2 = upper_streams[i + 1]
            
            n_points = min(len(stream1), len(stream2), 30)
            
            if len(stream1) > 1 and len(stream2) > 1:
                t1 = np.linspace(0, 1, len(stream1))
                t2 = np.linspace(0, 1, len(stream2))
                t_common = np.linspace(0, 1, n_points)
                
                try:
                    stream1_x = interp1d(t1, stream1[:, 0], fill_value='extrapolate')(t_common)
                    stream1_y = interp1d(t1, stream1[:, 1], fill_value='extrapolate')(t_common)
                    stream1_z = interp1d(t1, stream1[:, 2], fill_value='extrapolate')(t_common)
                    stream2_x = interp1d(t2, stream2[:, 0], fill_value='extrapolate')(t_common)
                    stream2_y = interp1d(t2, stream2[:, 1], fill_value='extrapolate')(t_common)
                    stream2_z = interp1d(t2, stream2[:, 2], fill_value='extrapolate')(t_common)
                    
                    x_surf = np.array([stream1_x, stream2_x])
                    y_surf = np.array([stream1_y, stream2_y])
                    z_surf = np.array([stream1_z, stream2_z])
                    
                    self.ax.plot_surface(x_surf, y_surf, z_surf,
                                        color=color, alpha=alpha,
                                        edgecolor='none', shade=True)
                    # Mirror
                    self.ax.plot_surface(x_surf, y_surf, -z_surf,
                                        color=color, alpha=alpha,
                                        edgecolor='none', shade=True)
                except:
                    pass
    
    def plot_cross_section(self, blunted_waverider, original_waverider, station_idx):
        """Plot a cross-section comparison at a specific station."""
        self.ax.clear()
        
        if station_idx >= len(original_waverider.upper_surface_streams):
            return
        
        # Get original streams at station
        orig_upper = original_waverider.upper_surface_streams[station_idx]
        orig_lower = original_waverider.lower_surface_streams[station_idx]
        
        # Get blunted streams at station
        blunt_upper = blunted_waverider.upper_surface_streams[station_idx]
        blunt_lower = blunted_waverider.lower_surface_streams[station_idx]
        
        # Plot in 3D (showing first N points from LE)
        n_show = min(30, len(orig_upper), len(blunt_upper))
        
        # Original
        self.ax.plot(orig_upper[:n_show, 0], orig_upper[:n_show, 2], orig_upper[:n_show, 1],
                    'b-', linewidth=2, label='Original upper')
        self.ax.plot(orig_lower[:n_show, 0], orig_lower[:n_show, 2], orig_lower[:n_show, 1],
                    'b--', linewidth=2, label='Original lower')
        
        # Blunted
        n_show_blunt = min(50, len(blunt_upper))
        self.ax.plot(blunt_upper[:n_show_blunt, 0], blunt_upper[:n_show_blunt, 2], 
                    blunt_upper[:n_show_blunt, 1],
                    'r-', linewidth=2, label='Blunted upper')
        n_show_blunt_lower = min(50, len(blunt_lower))
        self.ax.plot(blunt_lower[:n_show_blunt_lower, 0], blunt_lower[:n_show_blunt_lower, 2],
                    blunt_lower[:n_show_blunt_lower, 1],
                    'r--', linewidth=2, label='Blunted lower')
        
        z = original_waverider.leading_edge[station_idx, 2]
        self.ax.set_title(f'Cross-section at Station {station_idx} (z = {z:.3f} m)')
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Z [m]')
        self.ax.set_zlabel('Y [m]')
        self.ax.legend()
        
        self.draw()


class ThermalAnalysisCanvas(FigureCanvas):
    """Canvas for thermal analysis plots."""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
    
    def plot_heating_vs_radius(self, mach: float, altitude_km: float,
                                current_radius: float = None,
                                max_heating: float = None):
        """
        Plot heating rate vs nose radius curve.
        
        Parameters:
        -----------
        mach : float
            Flight Mach number
        altitude_km : float
            Altitude in km
        current_radius : float, optional
            Current nose radius to mark on plot
        max_heating : float, optional
            Maximum allowable heating rate (MW/m²) to show as line
        """
        self.ax.clear()
        
        # Compute heating curve
        radii, heating = compute_heating_vs_radius(mach, altitude_km)
        
        # Plot curve
        self.ax.plot(radii * 1000, heating, 'b-', linewidth=2, label=f'Mach {mach}')
        
        # Mark current radius if provided
        if current_radius is not None:
            current_heating = estimate_stagnation_heating_from_mach(
                current_radius, mach, altitude_km) / 1e6
            self.ax.axvline(x=current_radius * 1000, color='r', linestyle='--',
                           linewidth=2, label=f'Current: {current_radius*1000:.1f} mm')
            self.ax.plot(current_radius * 1000, current_heating, 'ro', markersize=10)
            self.ax.annotate(f'{current_heating:.2f} MW/m²',
                            xy=(current_radius * 1000, current_heating),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=10, color='red')
        
        # Mark max heating limit if provided
        if max_heating is not None:
            self.ax.axhline(y=max_heating, color='orange', linestyle=':',
                           linewidth=2, label=f'Limit: {max_heating} MW/m²')
            
            # Find and mark recommended radius
            rec_radius = recommend_nose_radius(mach, altitude_km, max_heating * 1e6)
            self.ax.axvline(x=rec_radius * 1000, color='g', linestyle='-.',
                           linewidth=1.5, alpha=0.7)
            self.ax.annotate(f'Min: {rec_radius*1000:.1f} mm',
                            xy=(rec_radius * 1000, max_heating),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, color='green')
        
        self.ax.set_xlabel('Nose Radius [mm]')
        self.ax.set_ylabel('Stagnation Heating Rate [MW/m²]')
        self.ax.set_title(f'Stagnation Heating vs Nose Radius\n(Mach {mach}, Alt {altitude_km} km)')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim([0, 50])
        
        # Set reasonable y limit
        max_y = min(heating.max() * 1.1, 20)  # Cap at 20 MW/m²
        self.ax.set_ylim([0, max_y])
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_multi_mach_heating(self, mach_values: list, altitude_km: float,
                                 current_radius: float = None):
        """Plot heating curves for multiple Mach numbers."""
        self.ax.clear()
        
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        for i, mach in enumerate(mach_values):
            radii, heating = compute_heating_vs_radius(mach, altitude_km)
            color = colors[i % len(colors)]
            self.ax.plot(radii * 1000, heating, '-', linewidth=2,
                        color=color, label=f'Mach {mach}')
        
        if current_radius is not None:
            self.ax.axvline(x=current_radius * 1000, color='black', linestyle='--',
                           linewidth=2, label=f'Current: {current_radius*1000:.1f} mm')
        
        self.ax.set_xlabel('Nose Radius [mm]')
        self.ax.set_ylabel('Stagnation Heating Rate [MW/m²]')
        self.ax.set_title(f'Stagnation Heating Comparison (Alt {altitude_km} km)')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim([0, 50])
        self.ax.set_ylim([0, 15])  # Cap for visibility
        
        self.fig.tight_layout()
        self.draw()


class LEBluntingTab(QWidget):
    """
    Leading Edge Blunting Tab for the Waverider GUI.
    
    Provides controls for:
    - Blunting profile selection (ellipse/power-law)
    - Nose radius and blend parameters
    - Variable bluntness along span
    - Thermal analysis
    - Live 3D preview
    """
    
    # Signal emitted when blunting is applied
    blunting_applied = pyqtSignal(object)  # Emits the blunted waverider
    blunting_reset = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.blunted_waverider = None
        self._original_waverider = None  # Stores reference to the waverider being blunted
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QHBoxLayout(self)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls (in scroll area)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Add control groups
        left_layout.addWidget(self._create_profile_group())
        left_layout.addWidget(self._create_parameters_group())
        left_layout.addWidget(self._create_variable_bluntness_group())
        left_layout.addWidget(self._create_thermal_group())
        left_layout.addWidget(self._create_actions_group())
        left_layout.addStretch()
        
        # Wrap in scroll area
        scroll = QScrollArea()
        scroll.setWidget(left_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(450)
        
        splitter.addWidget(scroll)
        
        # Right panel - Visualization
        right_widget = self._create_visualization_panel()
        splitter.addWidget(right_widget)
        
        # Set initial sizes
        splitter.setSizes([350, 900])
        
        main_layout.addWidget(splitter)
    
    def _create_profile_group(self):
        """Create the profile type selection group."""
        group = QGroupBox("Blunting Profile")
        layout = QGridLayout()
        
        layout.addWidget(QLabel("Profile Type:"), 0, 0)
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["Ellipse", "Power-Law"])
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        self.profile_combo.setToolTip(
            "Ellipse: Classic nose blunting with aspect ratio control\n"
            "Power-Law: r = k·x^n profile for hypersonic applications"
        )
        layout.addWidget(self.profile_combo, 0, 1)
        
        # Profile-specific parameters
        # Ellipse aspect ratio
        layout.addWidget(QLabel("Aspect Ratio (a/b):"), 1, 0)
        self.aspect_spin = QDoubleSpinBox()
        self.aspect_spin.setRange(1.0, 5.0)
        self.aspect_spin.setValue(2.0)
        self.aspect_spin.setSingleStep(0.1)
        self.aspect_spin.setDecimals(2)
        self.aspect_spin.setToolTip("Ellipse aspect ratio: a (streamwise) / b (normal)")
        layout.addWidget(self.aspect_spin, 1, 1)
        
        # Power-law exponent
        layout.addWidget(QLabel("Exponent (n):"), 2, 0)
        self.exponent_spin = QDoubleSpinBox()
        self.exponent_spin.setRange(0.3, 0.9)
        self.exponent_spin.setValue(0.5)
        self.exponent_spin.setSingleStep(0.05)
        self.exponent_spin.setDecimals(2)
        self.exponent_spin.setToolTip(
            "Power-law exponent:\n"
            "0.5 = Parabolic (min drag)\n"
            "0.67 = Compromise\n"
            "0.75 = Blunter (better thermal)"
        )
        self.exponent_spin.setEnabled(False)  # Disabled by default (ellipse selected)
        layout.addWidget(self.exponent_spin, 2, 1)
        
        group.setLayout(layout)
        return group
    
    def _create_parameters_group(self):
        """Create the main blunting parameters group."""
        group = QGroupBox("Blunting Parameters")
        layout = QGridLayout()
        
        # Nose radius
        layout.addWidget(QLabel("Nose Radius [mm]:"), 0, 0)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(1.0, 100.0)
        self.radius_spin.setValue(10.0)
        self.radius_spin.setSingleStep(1.0)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setToolTip("Bluntness radius at nose tip")
        layout.addWidget(self.radius_spin, 0, 1)
        
        # Radius slider
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setRange(10, 500)  # 1mm to 50mm in 0.1mm steps
        self.radius_slider.setValue(100)
        self.radius_slider.valueChanged.connect(
            lambda v: self.radius_spin.setValue(v / 10.0))
        self.radius_spin.valueChanged.connect(
            lambda v: self.radius_slider.setValue(int(v * 10)))
        layout.addWidget(self.radius_slider, 0, 2)
        
        # Blend length
        layout.addWidget(QLabel("Blend Length [% chord]:"), 1, 0)
        self.blend_spin = QDoubleSpinBox()
        self.blend_spin.setRange(1.0, 20.0)
        self.blend_spin.setValue(5.0)
        self.blend_spin.setSingleStep(0.5)
        self.blend_spin.setDecimals(1)
        self.blend_spin.setToolTip("How far back the blend extends (% of local chord)")
        layout.addWidget(self.blend_spin, 1, 1)
        
        # Blend slider
        self.blend_slider = QSlider(Qt.Horizontal)
        self.blend_slider.setRange(10, 200)
        self.blend_slider.setValue(50)
        self.blend_slider.valueChanged.connect(
            lambda v: self.blend_spin.setValue(v / 10.0))
        self.blend_spin.valueChanged.connect(
            lambda v: self.blend_slider.setValue(int(v * 10)))
        layout.addWidget(self.blend_slider, 1, 2)
        
        # Resolution parameters
        layout.addWidget(QLabel("Nose Points:"), 2, 0)
        self.nose_points_spin = QSpinBox()
        self.nose_points_spin.setRange(10, 50)
        self.nose_points_spin.setValue(25)
        self.nose_points_spin.setToolTip("Number of points for nose profile")
        layout.addWidget(self.nose_points_spin, 2, 1)
        
        layout.addWidget(QLabel("Blend Points:"), 3, 0)
        self.blend_points_spin = QSpinBox()
        self.blend_points_spin.setRange(5, 30)
        self.blend_points_spin.setValue(15)
        self.blend_points_spin.setToolTip("Number of points for blend region")
        layout.addWidget(self.blend_points_spin, 3, 1)
        
        group.setLayout(layout)
        return group
    
    def _create_variable_bluntness_group(self):
        """Create the variable bluntness group."""
        group = QGroupBox("Variable Bluntness")
        layout = QGridLayout()
        
        self.variable_check = QCheckBox("Enable variable bluntness (root to tip)")
        self.variable_check.setToolTip(
            "Vary nose radius along span:\n"
            "Root (centerline) uses full radius\n"
            "Tip uses reduced radius"
        )
        self.variable_check.stateChanged.connect(self._on_variable_changed)
        layout.addWidget(self.variable_check, 0, 0, 1, 2)
        
        layout.addWidget(QLabel("Tip/Root Ratio:"), 1, 0)
        self.tip_ratio_spin = QDoubleSpinBox()
        self.tip_ratio_spin.setRange(0.1, 1.0)
        self.tip_ratio_spin.setValue(0.5)
        self.tip_ratio_spin.setSingleStep(0.1)
        self.tip_ratio_spin.setDecimals(2)
        self.tip_ratio_spin.setEnabled(False)
        self.tip_ratio_spin.setToolTip("Ratio of tip radius to root radius")
        layout.addWidget(self.tip_ratio_spin, 1, 1)
        
        group.setLayout(layout)
        return group
    
    def _create_thermal_group(self):
        """Create the thermal analysis group."""
        group = QGroupBox("Thermal Analysis")
        layout = QGridLayout()
        
        # Altitude
        layout.addWidget(QLabel("Altitude [km]:"), 0, 0)
        self.altitude_spin = QDoubleSpinBox()
        self.altitude_spin.setRange(0, 100)
        self.altitude_spin.setValue(25.0)
        self.altitude_spin.setSingleStep(5.0)
        self.altitude_spin.setDecimals(1)
        layout.addWidget(self.altitude_spin, 0, 1)
        
        # Max heating limit
        layout.addWidget(QLabel("Max Heating [MW/m²]:"), 1, 0)
        self.max_heating_spin = QDoubleSpinBox()
        self.max_heating_spin.setRange(0.1, 10.0)
        self.max_heating_spin.setValue(1.0)
        self.max_heating_spin.setSingleStep(0.1)
        self.max_heating_spin.setDecimals(1)
        self.max_heating_spin.setToolTip("Maximum allowable stagnation heating rate")
        layout.addWidget(self.max_heating_spin, 1, 1)
        
        # Results display
        self.thermal_info = QLabel("Generate waverider to see thermal analysis")
        self.thermal_info.setWordWrap(True)
        self.thermal_info.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        layout.addWidget(self.thermal_info, 2, 0, 1, 2)
        
        # Update thermal button
        self.update_thermal_btn = QPushButton("📊 Update Thermal Plot")
        self.update_thermal_btn.clicked.connect(self._update_thermal_plot)
        layout.addWidget(self.update_thermal_btn, 3, 0, 1, 2)
        
        group.setLayout(layout)
        return group
    
    def _create_actions_group(self):
        """Create the actions group."""
        group = QGroupBox("Actions")
        layout = QVBoxLayout()
        
        # Preview original button
        self.preview_btn = QPushButton("👁️ Preview Original Waverider")
        self.preview_btn.clicked.connect(self._show_original_waverider)
        self.preview_btn.setToolTip("Show the current waverider (before blunting) in 3D preview")
        layout.addWidget(self.preview_btn)
        
        # Apply blunting button
        self.apply_btn = QPushButton("✂️ Apply Blunting")
        self.apply_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px; font-size: 12px; }"
        )
        self.apply_btn.clicked.connect(self.apply_blunting)
        self.apply_btn.setToolTip("Apply blunting to the current waverider")
        layout.addWidget(self.apply_btn)
        
        # Reset button
        self.reset_btn = QPushButton("🔄 Reset to Sharp")
        self.reset_btn.clicked.connect(self.reset_blunting)
        self.reset_btn.setEnabled(False)
        layout.addWidget(self.reset_btn)
        
        # Preview options
        preview_layout = QHBoxLayout()
        self.show_original_check = QCheckBox("Show Original")
        self.show_original_check.setChecked(True)
        self.show_original_check.stateChanged.connect(self._update_preview)
        preview_layout.addWidget(self.show_original_check)
        
        self.focus_le_check = QCheckBox("Focus on LE")
        self.focus_le_check.setChecked(False)
        self.focus_le_check.stateChanged.connect(self._update_preview)
        preview_layout.addWidget(self.focus_le_check)
        layout.addLayout(preview_layout)
        
        # Status label
        self.status_label = QLabel("Ready - Generate a waverider first")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("QLabel { color: gray; }")
        layout.addWidget(self.status_label)
        
        group.setLayout(layout)
        return group
    
    def _create_visualization_panel(self):
        """Create the visualization panel with 3D view and thermal plot."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Tab widget for different views
        self.viz_tabs = QTabWidget()
        
        # 3D Preview tab
        tab_3d = QWidget()
        layout_3d = QVBoxLayout(tab_3d)
        
        self.canvas_3d = BluntingCanvas3D()
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, tab_3d)
        layout_3d.addWidget(self.toolbar_3d)
        layout_3d.addWidget(self.canvas_3d)
        
        self.viz_tabs.addTab(tab_3d, "3D Preview")
        
        # Thermal Analysis tab
        tab_thermal = QWidget()
        layout_thermal = QVBoxLayout(tab_thermal)
        
        self.canvas_thermal = ThermalAnalysisCanvas()
        self.toolbar_thermal = NavigationToolbar(self.canvas_thermal, tab_thermal)
        layout_thermal.addWidget(self.toolbar_thermal)
        layout_thermal.addWidget(self.canvas_thermal)
        
        self.viz_tabs.addTab(tab_thermal, "Thermal Analysis")
        
        layout.addWidget(self.viz_tabs)
        
        return widget
    
    def _on_profile_changed(self, index):
        """Handle profile type change."""
        is_ellipse = (index == 0)
        self.aspect_spin.setEnabled(is_ellipse)
        self.exponent_spin.setEnabled(not is_ellipse)
    
    def _on_variable_changed(self, state):
        """Handle variable bluntness checkbox change."""
        self.tip_ratio_spin.setEnabled(state == Qt.Checked)
    
    def _update_preview(self):
        """Update the 3D preview."""
        if self.blunted_waverider is None:
            return
        
        # Use stored original waverider if available, otherwise try to find it
        original = getattr(self, '_original_waverider', None)
        
        if original is None:
            # Try to find the original waverider
            if self.parent_gui and hasattr(self.parent_gui, 'waverider') and self.parent_gui.waverider is not None:
                original = self.parent_gui.waverider
            elif self.parent_gui and hasattr(self.parent_gui, 'shadow_waverider_tab'):
                cone_tab = self.parent_gui.shadow_waverider_tab
                if cone_tab and hasattr(cone_tab, 'waverider') and cone_tab.waverider is not None:
                    original = cone_tab.waverider
        
        self.canvas_3d.plot_blunted_waverider(
            self.blunted_waverider,
            original_waverider=original,
            show_original=self.show_original_check.isChecked(),
            show_blunted=True,
            show_le=True,
            focus_le=self.focus_le_check.isChecked()
        )
    
    def _update_thermal_plot(self):
        """Update the thermal analysis plot."""
        if self.parent_gui is None:
            return
        
        # Get Mach number - try from main GUI first, then use a default
        mach = 5.0  # Default
        if hasattr(self.parent_gui, 'm_inf_spin'):
            mach = self.parent_gui.m_inf_spin.value()
        elif hasattr(self.parent_gui, 'waverider') and self.parent_gui.waverider is not None:
            mach = self.parent_gui.waverider.M_inf
        elif hasattr(self, '_original_waverider') and self._original_waverider is not None:
            if hasattr(self._original_waverider, 'M_inf'):
                mach = self._original_waverider.M_inf
            elif hasattr(self._original_waverider, 'mach'):
                mach = self._original_waverider.mach
        altitude = self.altitude_spin.value()
        current_radius = self.radius_spin.value() / 1000  # Convert mm to m
        max_heating = self.max_heating_spin.value()
        
        self.canvas_thermal.plot_heating_vs_radius(
            mach, altitude, current_radius, max_heating
        )
        
        # Update thermal info
        heating = estimate_stagnation_heating_from_mach(current_radius, mach, altitude)
        rec_radius = recommend_nose_radius(mach, altitude, max_heating * 1e6)
        
        info_text = (
            f"<b>Current Configuration:</b><br>"
            f"  Mach: {mach}<br>"
            f"  Altitude: {altitude} km<br>"
            f"  Nose radius: {current_radius*1000:.1f} mm<br>"
            f"  Heating rate: <b>{heating/1e6:.2f} MW/m²</b><br><br>"
            f"<b>For {max_heating} MW/m² limit:</b><br>"
            f"  Min radius: {rec_radius*1000:.1f} mm"
        )
        
        if current_radius < rec_radius:
            info_text += "<br><font color='red'>⚠️ Current radius below recommended minimum!</font>"
            self.thermal_info.setStyleSheet("QLabel { background-color: #ffcccc; padding: 5px; }")
        else:
            self.thermal_info.setStyleSheet("QLabel { background-color: #ccffcc; padding: 5px; }")
        
        self.thermal_info.setText(info_text)
        
        # Switch to thermal tab
        self.viz_tabs.setCurrentIndex(1)
    
    def apply_blunting(self):
        """Apply blunting to the current waverider."""
        if not BLUNTING_AVAILABLE:
            QMessageBox.warning(self, "Module Not Available",
                              "Leading edge blunting module not available.")
            return
        
        # Find the waverider to blunt (standard or shadow-derived)
        waverider_to_blunt = None
        waverider_source = None
        
        if self.parent_gui is not None:
            # Check standard waverider first
            if hasattr(self.parent_gui, 'waverider') and self.parent_gui.waverider is not None:
                waverider_to_blunt = self.parent_gui.waverider
                waverider_source = "osculating cone"
            
            # If no standard waverider, check shadow waverider tab
            if waverider_to_blunt is None:
                if hasattr(self.parent_gui, 'shadow_waverider_tab') and self.parent_gui.shadow_waverider_tab is not None:
                    cone_tab = self.parent_gui.shadow_waverider_tab
                    if hasattr(cone_tab, 'waverider') and cone_tab.waverider is not None:
                        waverider_to_blunt = cone_tab.waverider
                        waverider_source = "shadow-derived"
        
        if waverider_to_blunt is None:
            QMessageBox.warning(self, "No Waverider",
                              "Generate a waverider first before applying blunting.\n\n"
                              "Use either:\n"
                              "- Main panel 'Generate Waverider' button\n"
                              "- SHADOW Waverider tab")
            return
        
        try:
            self.status_label.setText(f"Applying blunting to {waverider_source} waverider...")
            self.status_label.setStyleSheet("QLabel { color: orange; }")
            
            # Get parameters
            nose_radius = self.radius_spin.value() / 1000  # Convert mm to m
            blend_fraction = self.blend_spin.value() / 100  # Convert % to fraction
            
            # Create blunted waverider using appropriate method
            if SIMPLE_BLUNTING:
                # Use simple, robust blunting
                print(f"Using SIMPLE blunting: radius={nose_radius*1000:.1f}mm, blend={blend_fraction*100:.1f}%")
                self.blunted_waverider = simple_blunt_waverider(
                    waverider_to_blunt,
                    nose_radius=nose_radius,
                    blend_fraction=blend_fraction,
                    n_nose_points=self.nose_points_spin.value()
                )
                profile_type = "simple-ellipse"
            else:
                # Use complex blunting
                profile_type = "ellipse" if self.profile_combo.currentIndex() == 0 else "power_law"
                print(f"Using COMPLEX blunting: {profile_type}, radius={nose_radius*1000:.1f}mm")
                self.blunted_waverider = blunt_waverider(
                    waverider_to_blunt,
                    profile_type=profile_type,
                    nose_radius=nose_radius,
                    blend_length_fraction=blend_fraction,
                    ellipse_aspect_ratio=self.aspect_spin.value(),
                    power_law_exponent=self.exponent_spin.value(),
                    n_nose_points=self.nose_points_spin.value(),
                    n_blend_points=self.blend_points_spin.value(),
                    variable_bluntness=self.variable_check.isChecked(),
                    tip_to_root_ratio=self.tip_ratio_spin.value()
                )
            
            # Store reference to original waverider for preview
            self._original_waverider = waverider_to_blunt
            
            # Update preview
            self._update_preview()
            
            # Update thermal plot
            self._update_thermal_plot()
            
            # Enable reset button
            self.reset_btn.setEnabled(True)
            
            # Emit signal
            self.blunting_applied.emit(self.blunted_waverider)
            
            # Update status
            self.status_label.setText(
                f"✓ Blunting applied: {profile_type}, {nose_radius*1000:.1f}mm radius ({waverider_source})"
            )
            self.status_label.setStyleSheet("QLabel { color: green; }")
            
        except Exception as e:
            QMessageBox.critical(self, "Blunting Error",
                               f"Failed to apply blunting:\n\n{str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("QLabel { color: red; }")
    
    def reset_blunting(self):
        """Reset to original sharp waverider."""
        self.blunted_waverider = None
        self.reset_btn.setEnabled(False)
        
        # Clear preview
        self.canvas_3d.ax.clear()
        self.canvas_3d.ax.set_title('Blunted Waverider Preview')
        self.canvas_3d.draw()
        
        # Emit signal
        self.blunting_reset.emit()
        
        self.status_label.setText("Reset to sharp leading edge")
        self.status_label.setStyleSheet("QLabel { color: gray; }")
    
    def get_blunted_waverider(self):
        """Return the current blunted waverider (or None if not applied)."""
        return self.blunted_waverider
    
    def has_blunting(self):
        """Check if blunting has been applied."""
        return self.blunted_waverider is not None
    
    def refresh_from_waverider(self):
        """Refresh when a new waverider is generated."""
        # Reset blunting since the base waverider changed
        self.blunted_waverider = None
        self.reset_btn.setEnabled(False)
        
        if self.parent_gui and hasattr(self.parent_gui, 'waverider') and self.parent_gui.waverider is not None:
            self.status_label.setText("Ready - Click 'Apply Blunting' to blunt leading edge")
            self.status_label.setStyleSheet("QLabel { color: blue; }")
            
            # Update thermal info with design Mach
            self._update_thermal_plot()
            
            # Show the original waverider in the 3D preview
            self._show_original_waverider()
        else:
            self.status_label.setText("Ready - Generate a waverider first")
            self.status_label.setStyleSheet("QLabel { color: gray; }")
    
    def _show_original_waverider(self):
        """Show the original (sharp) waverider in the 3D preview."""
        if self.parent_gui is None:
            QMessageBox.warning(self, "No Parent GUI", "Cannot access waverider data.")
            return
        
        # Check for standard waverider first
        original = None
        waverider_source = None
        
        # Debug output
        print(f"\n_show_original_waverider DEBUG:")
        print(f"  parent_gui exists: {self.parent_gui is not None}")
        print(f"  has 'waverider' attr: {hasattr(self.parent_gui, 'waverider')}")
        if hasattr(self.parent_gui, 'waverider'):
            print(f"  waverider is not None: {self.parent_gui.waverider is not None}")
        print(f"  has 'shadow_waverider_tab' attr: {hasattr(self.parent_gui, 'shadow_waverider_tab')}")
        if hasattr(self.parent_gui, 'shadow_waverider_tab'):
            shadow_tab = self.parent_gui.shadow_waverider_tab
            print(f"  shadow_waverider_tab is not None: {shadow_tab is not None}")
            if shadow_tab is not None:
                print(f"  shadow_tab has 'waverider' attr: {hasattr(shadow_tab, 'waverider')}")
                if hasattr(shadow_tab, 'waverider'):
                    print(f"  shadow_tab.waverider is not None: {shadow_tab.waverider is not None}")
        
        if hasattr(self.parent_gui, 'waverider') and self.parent_gui.waverider is not None:
            original = self.parent_gui.waverider
            waverider_source = "osculating cone"
            print(f"  -> Using osculating cone waverider")
        
        # If no standard waverider, check for shadow waverider
        if original is None:
            if hasattr(self.parent_gui, 'shadow_waverider_tab') and self.parent_gui.shadow_waverider_tab is not None:
                shadow_tab = self.parent_gui.shadow_waverider_tab
                if hasattr(shadow_tab, 'waverider') and shadow_tab.waverider is not None:
                    original = shadow_tab.waverider
                    waverider_source = "SHADOW"
                    print(f"  -> Using SHADOW waverider")
        
        if original is None:
            print(f"  -> No waverider found!")
            QMessageBox.information(
                self, "No Waverider",
                "No waverider has been generated yet.\n\n"
                "Please go to the main panel and click 'Generate Waverider' first,\n"
                "or use the 'SHADOW Waverider' tab to generate a shadow-derived waverider."
            )
            return
        
        # Store reference
        self._original_waverider = original
        
        self.canvas_3d.plot_blunted_waverider(
            blunted_waverider=None,
            original_waverider=original,
            show_original=True,
            show_blunted=False,
            show_le=True,
            focus_le=False
        )
        self.canvas_3d.ax.set_title(f'Original Waverider ({waverider_source}, Sharp LE)')
        self.canvas_3d.draw()
        
        self.status_label.setText(f"Showing {waverider_source} waverider - Click 'Apply Blunting' to blunt")
        self.status_label.setStyleSheet("QLabel { color: blue; }")
