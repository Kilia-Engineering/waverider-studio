"""
Planar Waverider Design Tab

GUI tab implementing the 9-parameter planar waverider design method from:
  Jessen, Larsson, Brehm (2026) — "Comparative optimization of hypersonic
  waveriders using analytical and computational methods"
  Aerospace Science and Technology 172, 111703.
"""

import numpy as np
import traceback
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox, QCheckBox,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox, QSplitter,
    QScrollArea, QComboBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from planar_waverider import PlanarWaverider
from planar_waverider_aero import PlanarWaveriderAero, atmosphere

# Optional CAD export
try:
    from waverider_generator.cad_export import to_CAD
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
#  Background worker for geometry generation + aero analysis
# ──────────────────────────────────────────────────────────────────────

class StepExportWorker(QThread):
    """Exports STEP file in a background thread using B-spline surfaces."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str)   # filename
    error = pyqtSignal(str)

    def __init__(self, waverider, filename, parent=None):
        super().__init__(parent)
        self.waverider = waverider
        self.filename = filename

    def _build_blunted_streams(self, wr, scale, y_positions, n_pts):
        """Build upper and lower streamlines with Tincher & Burnett blunting.

        Uses a common normalized half-cosine parameterization across all
        stations: t ∈ [0, 1] mapped to [x_start, L] per station.  This
        gives consistent cross-station parameterization for B-spline surface
        fitting and avoids extreme spacing ratios that cause OCC interpolation
        failures for small R/chord values.

        Upper and lower streamlines share the SAME x values at each index —
        only z differs.

        Parameters
        ----------
        wr : PlanarWaverider
        scale : float  (m -> mm)
        y_positions : ndarray  -- spanwise y coordinates [m]
        n_pts : int  -- total streamwise points per streamline

        Returns
        -------
        upper_streams, lower_streams : list of ndarray (n_pts, 3)
        """
        theta_base = np.radians(wr.wedge_angle_deg)
        R = wr.R
        L = wr.length

        # Common normalized parameter: half-cosine clustering at LE
        # t=0 → LE (fine spacing), t=1 → TE (coarse spacing)
        t_param = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi / 2.0, n_pts)))
        # Rescale so t_param[-1] == 1.0 exactly
        t_param = t_param / t_param[-1]

        upper_streams = []
        lower_streams = []

        for y_j in y_positions:
            x_le_j = float(wr._leading_edge_x(y_j))
            z_le_j = float(wr._leading_edge_z(x_le_j))

            T_star_j = float(wr._angle_perturbation(np.array([abs(y_j)]))[0])
            theta_j = T_star_j * theta_base
            chord = L - x_le_j

            can_blunt = (R > 0 and theta_j > np.radians(0.5) and chord > 1e-6)
            R_eff = 0.0

            if can_blunt:
                half_theta = theta_j / 2.0
                offset_ideal = R / np.tan(half_theta)
                max_offset = 0.95 * chord
                R_eff = R if offset_ideal <= max_offset else (
                    max_offset * np.tan(half_theta))
                if R_eff < 1e-9:
                    can_blunt = False

            if can_blunt:
                # Skip blunting in STEP when arc is sub-resolution
                # (R_eff < 0.01% of chord → invisible in CAD)
                arc_fraction = R_eff / chord if chord > 0 else 0
                if arc_fraction < 1e-4:
                    can_blunt = False

            if can_blunt:
                xc = x_le_j + R_eff / np.tan(half_theta)
                zc = z_le_j - R_eff
                x_nose = xc - R_eff
                x_lt = xc - R_eff * np.sin(theta_j)
                z_lt = zc - R_eff * np.cos(theta_j)
                R2_exact = R_eff * R_eff

                # Map common parameter to this station's x range
                local_chord = L - x_nose
                x_local = x_nose + t_param * local_chord

                # Compute z at each x analytically
                z_upper = np.empty(n_pts)
                z_lower = np.empty(n_pts)

                for i, x in enumerate(x_local):
                    dx = x - xc
                    dx2 = dx * dx
                    # Tiny tolerance for the nose boundary (dx^2 == R^2)
                    if dx2 <= R2_exact * (1.0 + 1e-10):
                        sq = np.sqrt(max(0.0, R2_exact - dx2))
                        z_upper[i] = zc + sq     # top of circle
                        if x <= x_lt:
                            z_lower[i] = zc - sq  # bottom of circle
                        else:
                            # Past lower tangent — compression slope
                            z_lower[i] = z_lt - np.tan(theta_j) * (x - x_lt)
                    else:
                        # Past upper tangent — flat upper, sloped lower
                        z_upper[i] = z_le_j
                        z_lower[i] = z_le_j - np.tan(theta_j) * (x - x_le_j)

            else:
                # No blunting — half-cosine from LE to TE
                x_local = x_le_j + t_param * chord
                z_upper = np.full(n_pts, z_le_j)
                z_lower = z_le_j - np.tan(theta_j) * (x_local - x_le_j)

            y_arr = np.full(n_pts, y_j * scale)
            upper_streams.append(
                np.column_stack([x_local * scale, y_arr, z_upper * scale]))
            lower_streams.append(
                np.column_stack([x_local * scale, y_arr, z_lower * scale]))

        return upper_streams, lower_streams

    def run(self):
        try:
            import cadquery as cq
            from cadquery import exporters
            from waverider_generator.cad_export import (
                _make_bspline_face, _sew_faces_to_solid,
            )

            wr = self.waverider
            scale = 1000.0  # m -> mm
            L = wr.length
            w = wr.width
            half_w = w / 2.0

            # --- Adaptive streamwise resolution based on R/chord ---
            # Half-cosine first spacing ≈ chord * π²/(4*n²), so to resolve
            # the blunting arc (width ≈ R_eff) we need n ≥ (π/2)*√(chord/R).
            # Clamp to [50, 120] for tractable B-spline fitting.
            n_pts = 50  # default (increased from 30 for better LE resolution)
            if wr.R > 0:
                theta_base = np.radians(wr.wedge_angle_deg)
                if theta_base > np.radians(0.5):
                    arc_width = wr.R  # inscribed circle radius ≈ arc width
                    chord_center = L  # centerline chord ≈ L (x_le ≈ 0)
                    if arc_width > 0 and arc_width / chord_center >= 1e-4:
                        n_needed = int(np.ceil(
                            (np.pi / 2.0) * np.sqrt(chord_center / arc_width)))
                        n_pts = max(n_pts, min(n_needed, 120))
            print(f"[STEP] Using n_pts={n_pts} streamwise points")

            # --- Build custom y distribution with fine tip resolution ---
            # Core: ~21 uniform stations across 96% of the span
            # (odd count ensures y=0 is included for the centerline)
            n_main = 21
            y_uniform = np.linspace(-half_w * 0.96, half_w * 0.96, n_main)

            # Extra fine stations near wingtips for smooth geometric taper.
            # Fractional offsets from tip (relative to half-span):
            tip_fracs = np.array([0.005, 0.01, 0.02, 0.04, 0.08])
            tip_offsets = tip_fracs * half_w
            y_left_tip = -half_w + tip_offsets
            y_right_tip = half_w - tip_offsets[::-1]

            y_positions = np.unique(np.concatenate([
                y_left_tip, y_uniform, y_right_tip
            ]))

            # Build streamlines (works for both R>0 and R=0)
            self.progress.emit(
                f"Building surfaces ({len(y_positions)} stations, "
                f"R={wr.R*1000:.1f} mm)...")
            upper_streams, lower_streams = self._build_blunted_streams(
                wr, scale, y_positions, n_pts)

            self.progress.emit("Fitting B-spline surfaces...")
            upper_faces = _make_bspline_face(upper_streams)
            lower_faces = _make_bspline_face(lower_streams)

            from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
            from OCP.TColgp import TColgp_Array2OfPnt
            from OCP.gp import gp_Pnt
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

            # Base face (trailing edge, x = L)
            self.progress.emit("Building base face...")
            n_streams = len(upper_streams)

            # TE base face — ruled surface between upper TE and lower TE
            base_upper = np.array([s[-1] for s in upper_streams])
            base_lower = np.array([s[-1] for s in lower_streams])
            n_base = n_streams

            base_grid = TColgp_Array2OfPnt(1, n_base, 1, 2)
            for k in range(n_base):
                base_grid.SetValue(k + 1, 1,
                                   gp_Pnt(*base_upper[k].tolist()))
                base_grid.SetValue(k + 1, 2,
                                   gp_Pnt(*base_lower[k].tolist()))
            approx = GeomAPI_PointsToBSplineSurface()
            approx.Interpolate(base_grid)
            if not approx.IsDone():
                raise RuntimeError(
                    f"Base face B-spline interpolation failed "
                    f"on {n_base}x2 grid.")
            face_builder = BRepBuilderAPI_MakeFace(approx.Surface(), 1e-3)
            face_builder.Build()
            if not face_builder.IsDone():
                raise RuntimeError(
                    f"Base face MakeFace failed "
                    f"(error={face_builder.Error()}).")
            base_face = cq.Face(face_builder.Face())

            # No explicit wingtip caps needed — the fine-resolution tip
            # stations make upper/lower surfaces converge smoothly to a
            # near-point, closing the geometry naturally.

            # Sew into solid
            self.progress.emit("Sewing faces into solid...")
            all_faces = upper_faces + lower_faces + [base_face]
            solid = _sew_faces_to_solid(all_faces, tolerance=0.1 * scale)

            self.progress.emit("Writing STEP file...")
            compound = cq.Workplane("XY").newObject([solid])
            exporters.export(compound, self.filename)
            self.finished.emit(self.filename)

        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class PlanarWaveriderWorker(QThread):
    """Generates waverider geometry and runs aero analysis in a thread."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object, dict)  # (waverider, aero_results)
    error = pyqtSignal(str)

    def __init__(self, params, aero_params, parent=None):
        super().__init__(parent)
        self.params = params          # dict for PlanarWaverider constructor
        self.aero_params = aero_params  # dict with M_inf, alpha_deg, altitude_km, etc.

    def run(self):
        try:
            self.progress.emit("Generating geometry...")
            wr = PlanarWaverider(**self.params)
            nx = self.aero_params.get('nx', 60)
            ny = self.aero_params.get('ny', 40)
            wr.generate(nx=nx, ny=ny)

            self.progress.emit("Computing aerodynamic forces...")
            aero = PlanarWaveriderAero(gamma=self.params.get('gamma', 1.4))
            results = aero.compute_forces(
                wr,
                M_inf=self.aero_params['M_inf'],
                alpha_deg=self.aero_params['alpha_deg'],
                altitude_km=self.aero_params['altitude_km'],
                T_wall=self.aero_params.get('T_wall', None),
            )

            self.finished.emit(wr, results)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ──────────────────────────────────────────────────────────────────────
#  3D Canvas
# ──────────────────────────────────────────────────────────────────────

class PlanarWaveriderCanvas(FigureCanvas):
    """Matplotlib 3D canvas for planar waverider visualization."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 8), facecolor='#0A0A0A')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1A1A1A')
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_waverider(self, wr, show_upper=True, show_lower=True,
                       show_le=True, show_info=True, aero_results=None):
        """Plot the planar waverider surfaces."""
        self.ax.clear()
        self.ax.set_facecolor('#1A1A1A')

        if wr is None or wr.upper_surface_x is None:
            self.ax.text(0, 0, 0, "No geometry", color='white',
                         fontsize=14, ha='center')
            self.draw()
            return

        # Plot upper surface
        if show_upper:
            surf_u = self.ax.plot_surface(
                wr.upper_surface_x, wr.upper_surface_y, wr.upper_surface_z,
                color='#4488CC', alpha=0.5, edgecolor='#335577',
                linewidth=0.2, shade=True,
            )
            surf_u._facecolors2d = surf_u._facecolor3d
            surf_u._edgecolors2d = surf_u._edgecolor3d

        # Plot lower surface
        if show_lower:
            surf_l = self.ax.plot_surface(
                wr.lower_surface_x, wr.lower_surface_y, wr.lower_surface_z,
                color='#CC6644', alpha=0.6, edgecolor='#884422',
                linewidth=0.2, shade=True,
            )
            surf_l._facecolors2d = surf_l._facecolor3d
            surf_l._edgecolors2d = surf_l._edgecolor3d

        # Plot leading edge
        if show_le and wr.leading_edge is not None:
            le = wr.leading_edge
            self.ax.plot(le[:, 0], le[:, 1], le[:, 2],
                         color='#FFAA00', linewidth=2.5, label='Leading Edge')

        # Axis labels and styling
        self.ax.set_xlabel('X (streamwise)', color='#888888', fontsize=8)
        self.ax.set_ylabel('Y (spanwise)', color='#888888', fontsize=8)
        self.ax.set_zlabel('Z (vertical)', color='#888888', fontsize=8)
        self.ax.tick_params(colors='#666666', labelsize=7)

        # Legend (proxy artists for surfaces)
        import matplotlib.patches as mpatches
        handles = []
        if show_upper:
            handles.append(mpatches.Patch(color='#4488CC', label='Upper Surface'))
        if show_lower:
            handles.append(mpatches.Patch(color='#CC6644', label='Lower Surface'))
        if show_le:
            handles.append(self.ax.plot([], [], color='#FFAA00',
                                        linewidth=2, label='Leading Edge')[0])
        if handles:
            self.ax.legend(handles=handles, loc='upper right', fontsize=7,
                           facecolor='#2A2A2A', edgecolor='#555555',
                           labelcolor='#CCCCCC')

        # Equal aspect ratio
        self._set_axes_equal()

        # Info panel overlay
        if show_info and aero_results:
            self._draw_info(wr, aero_results)

        self.draw()

    def _set_axes_equal(self):
        """Make axes of 3D plot have equal scale."""
        ax = self.ax
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        centers = limits.mean(axis=1)
        max_range = (limits[:, 1] - limits[:, 0]).max() / 2.0
        ax.set_xlim3d([centers[0] - max_range, centers[0] + max_range])
        ax.set_ylim3d([centers[1] - max_range, centers[1] + max_range])
        ax.set_zlim3d([centers[2] - max_range, centers[2] + max_range])

    def _draw_info(self, wr, res):
        """Draw text info overlay on the figure (matches cone-derived style)."""
        # Remove old text annotations
        for txt in self.fig.texts:
            txt.remove()

        bw, bh = wr.base_dimensions()
        vol = wr.volume()
        S_ref = res.get('S_ref', 0)
        # Volumetric efficiency: V / S_ref^(3/2)
        vol_eff = vol / S_ref**1.5 if S_ref > 1e-8 else 0.0

        info = (
            "WAVERIDER INFO\n"
            f"  Method          Planar (Jessen 2026)\n"
            f"  Mach            {res.get('M_inf', 0):.2f}\n"
            f"  Shock \u03b2         {wr.beta_deg:.2f}\u00b0\n"
            f"  Wedge \u03b8         {wr.wedge_angle_deg:.4f}\u00b0\n"
            f"  Power-law n     {wr.n:.2f}\n"
            f"  Epsilon         {wr.epsilon:.3f}\n"
            f"  p1, p2, p3      {wr.p1:.2f}, {wr.p2:.2f}, {wr.p3:.2f}\n"
            f"  LE Radius       {wr.R:.4f} m\n"
            f"  Length           {wr.length:.4f} m\n"
            f"  Width            {wr.width:.4f} m\n"
            f"  Planform Area    {S_ref:.4f} m\u00b2\n"
            f"  Volume           {vol:.6f} m\u00b3\n"
            f"  Vol Efficiency   {vol_eff:.6f}\n"
            f"  Base             {bw:.4f} x {bh:.4f} m\n"
            f"  L/D              {res.get('L_over_D', 0):.4f}\n"
            f"  CL               {res.get('CL', 0):.6f}\n"
            f"  CD               {res.get('CD', 0):.6f}"
        )

        self.fig.text(
            0.02, 0.97, info, transform=self.fig.transFigure,
            fontsize=8, color='#CCCCCC', family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1A1A1A',
                      edgecolor='#FF8800', alpha=0.85),
        )


# ──────────────────────────────────────────────────────────────────────
#  Chebyshev preview canvas (small 2D plot for perturbation curve)
# ──────────────────────────────────────────────────────────────────────

class ChebyshevPreviewCanvas(FigureCanvas):
    """Small 2D plot showing the Chebyshev perturbation T*(y)."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(3.5, 1.8), facecolor='#1A1A1A')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#222222')
        self.fig.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.20)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMaximumHeight(140)

    def update_plot(self, p1, p2, p3, width):
        """Recompute and plot the Chebyshev perturbation curve."""
        self.ax.clear()
        self.ax.set_facecolor('#222222')

        try:
            tmp = PlanarWaverider(width=width, p1=p1, p2=p2, p3=p3)
            tmp._compute_chebyshev_coefficients()
            y = np.linspace(0, width / 2.0, 200)
            T_star = tmp._angle_perturbation(y)

            self.ax.plot(y, T_star, color='#66CCFF', linewidth=1.5)
            self.ax.axhline(1.0, color='#555555', linewidth=0.5, linestyle='--')

            # Mark control points
            y_p1 = width / 3.0
            y_p2 = width / 6.0
            y_p3 = 0.0
            self.ax.plot(y_p1, p1, 'o', color='#FF6644', markersize=5, label='p1')
            self.ax.plot(y_p2, p2, 's', color='#44FF66', markersize=5, label='p2')
            self.ax.plot(y_p3, p3, '^', color='#FFAA00', markersize=5, label='p3')

            self.ax.set_xlabel('y [m]', color='#888888', fontsize=7)
            self.ax.set_ylabel('T*(y)', color='#888888', fontsize=7)
            self.ax.set_title('Chebyshev Perturbation', color='#AAAAAA',
                              fontsize=8)
            self.ax.tick_params(colors='#666666', labelsize=6)
            self.ax.legend(fontsize=6, loc='upper right',
                           facecolor='#333333', edgecolor='#555555',
                           labelcolor='#CCCCCC')
        except Exception:
            self.ax.text(0.5, 0.5, 'Error', color='red', ha='center',
                         va='center', transform=self.ax.transAxes)

        self.draw()


# ──────────────────────────────────────────────────────────────────────
#  Main Tab Widget
# ──────────────────────────────────────────────────────────────────────

class PlanarWaveriderTab(QWidget):
    """GUI tab for designing planar waveriders (Jessen et al. 2026)."""

    waverider_generated = pyqtSignal(object)  # emits PlanarWaverider

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_gui = parent
        self.waverider = None
        self.aero_results = None
        self.worker = None
        self.init_ui()

    # ── UI Construction ─────────────────────────────────────────────

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Left panel (scrollable controls)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        left_layout.addWidget(self._create_flow_group())
        left_layout.addWidget(self._create_geometry_group())
        left_layout.addWidget(self._create_perturbation_group())
        left_layout.addWidget(self._create_blunting_group())
        left_layout.addWidget(self._create_mesh_group())
        left_layout.addWidget(self._create_viscous_group())
        left_layout.addWidget(self._create_generate_group())
        left_layout.addWidget(self._create_results_group())
        left_layout.addWidget(self._create_export_group())
        left_layout.addStretch()

        # Connect width to chebyshev preview (both widgets exist now)
        self.width_spin.valueChanged.connect(self._update_chebyshev_preview)

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_widget)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(420)

        # Right panel (3D view + results text)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 3D view options
        opts_layout = QHBoxLayout()
        self.show_upper_check = QCheckBox("Upper")
        self.show_upper_check.setChecked(True)
        self.show_lower_check = QCheckBox("Lower")
        self.show_lower_check.setChecked(True)
        self.show_le_check = QCheckBox("LE")
        self.show_le_check.setChecked(True)
        self.show_info_check = QCheckBox("Info")
        self.show_info_check.setChecked(True)
        for cb in [self.show_upper_check, self.show_lower_check,
                   self.show_le_check, self.show_info_check]:
            cb.stateChanged.connect(self._update_3d_view)
            opts_layout.addWidget(cb)
        opts_layout.addStretch()
        right_layout.addLayout(opts_layout)

        # 3D canvas + toolbar
        self.canvas_3d = PlanarWaveriderCanvas()
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, right_widget)
        right_layout.addWidget(self.toolbar_3d)
        right_layout.addWidget(self.canvas_3d, stretch=1)

        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 9))
        self.results_text.setMaximumHeight(200)
        right_layout.addWidget(self.results_text)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

    # ── Control Groups ──────────────────────────────────────────────

    def _create_flow_group(self):
        group = QGroupBox("Flow Conditions")
        layout = QGridLayout()

        layout.addWidget(QLabel("Mach:"), 0, 0)
        self.mach_spin = QDoubleSpinBox()
        self.mach_spin.setRange(1.5, 25.0)
        self.mach_spin.setValue(6.85)
        self.mach_spin.setSingleStep(0.5)
        self.mach_spin.setDecimals(2)
        layout.addWidget(self.mach_spin, 0, 1)

        layout.addWidget(QLabel("Altitude [km]:"), 1, 0)
        self.alt_spin = QDoubleSpinBox()
        self.alt_spin.setRange(0, 80)
        self.alt_spin.setValue(25.0)
        self.alt_spin.setSingleStep(1.0)
        self.alt_spin.setDecimals(1)
        layout.addWidget(self.alt_spin, 1, 1)

        layout.addWidget(QLabel("Alpha [deg]:"), 2, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(-5.0, 15.0)
        self.alpha_spin.setValue(0.0)
        self.alpha_spin.setSingleStep(0.5)
        self.alpha_spin.setDecimals(2)
        layout.addWidget(self.alpha_spin, 2, 1)

        group.setLayout(layout)
        return group

    def _create_geometry_group(self):
        group = QGroupBox("Geometry Parameters")
        layout = QGridLayout()

        layout.addWidget(QLabel("Length [m]:"), 0, 0)
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.1, 50.0)
        self.length_spin.setValue(1.0)
        self.length_spin.setSingleStep(0.1)
        self.length_spin.setDecimals(3)
        layout.addWidget(self.length_spin, 0, 1)

        layout.addWidget(QLabel("Width [m]:"), 1, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.05, 50.0)
        self.width_spin.setValue(12.0)
        self.width_spin.setSingleStep(0.5)
        self.width_spin.setDecimals(3)
        layout.addWidget(self.width_spin, 1, 1)

        layout.addWidget(QLabel("Power-law n:"), 2, 0)
        self.n_spin = QDoubleSpinBox()
        self.n_spin.setRange(0.1, 20.0)
        self.n_spin.setValue(0.5)
        self.n_spin.setSingleStep(0.1)
        self.n_spin.setDecimals(2)
        self.n_spin.setToolTip("LE power-law exponent (0.5 = parabolic)")
        layout.addWidget(self.n_spin, 2, 1)

        layout.addWidget(QLabel("Shock angle [deg]:"), 3, 0)
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(5.0, 85.0)
        self.beta_spin.setValue(9.0)
        self.beta_spin.setSingleStep(0.5)
        self.beta_spin.setDecimals(2)
        self.beta_spin.setToolTip("Planar shock angle beta")
        layout.addWidget(self.beta_spin, 3, 1)

        layout.addWidget(QLabel("LE perturbation:"), 4, 0)
        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(-1.0, 1.0)
        self.epsilon_spin.setValue(0.0)
        self.epsilon_spin.setSingleStep(0.05)
        self.epsilon_spin.setDecimals(3)
        self.epsilon_spin.setToolTip("Parabolic LE perturbation epsilon")
        layout.addWidget(self.epsilon_spin, 4, 1)

        group.setLayout(layout)
        return group

    def _create_perturbation_group(self):
        group = QGroupBox("Chebyshev Perturbations")
        layout = QVBoxLayout()

        grid = QGridLayout()
        grid.addWidget(QLabel("p1 (y=w/3):"), 0, 0)
        self.p1_spin = QDoubleSpinBox()
        self.p1_spin.setRange(0.1, 3.0)
        self.p1_spin.setValue(1.0)
        self.p1_spin.setSingleStep(0.05)
        self.p1_spin.setDecimals(3)
        self.p1_spin.setToolTip("Angle multiplier at y = w/3 (LE tip)")
        grid.addWidget(self.p1_spin, 0, 1)

        grid.addWidget(QLabel("p2 (y=w/6):"), 1, 0)
        self.p2_spin = QDoubleSpinBox()
        self.p2_spin.setRange(0.1, 3.0)
        self.p2_spin.setValue(1.0)
        self.p2_spin.setSingleStep(0.05)
        self.p2_spin.setDecimals(3)
        self.p2_spin.setToolTip("Angle multiplier at y = w/6")
        grid.addWidget(self.p2_spin, 1, 1)

        grid.addWidget(QLabel("p3 (y=0):"), 2, 0)
        self.p3_spin = QDoubleSpinBox()
        self.p3_spin.setRange(0.1, 3.0)
        self.p3_spin.setValue(1.0)
        self.p3_spin.setSingleStep(0.05)
        self.p3_spin.setDecimals(3)
        self.p3_spin.setToolTip("Angle multiplier at centerline y = 0")
        grid.addWidget(self.p3_spin, 2, 1)

        layout.addLayout(grid)

        # Chebyshev preview plot
        self.cheb_preview = ChebyshevPreviewCanvas()
        layout.addWidget(self.cheb_preview)

        # Connect spinboxes to live preview
        for spin in [self.p1_spin, self.p2_spin, self.p3_spin]:
            spin.valueChanged.connect(self._update_chebyshev_preview)

        group.setLayout(layout)
        return group

    def _create_blunting_group(self):
        group = QGroupBox("Leading Edge Blunting")
        layout = QGridLayout()

        layout.addWidget(QLabel("LE Radius [m]:"), 0, 0)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.0, 0.5)
        self.radius_spin.setValue(0.0)
        self.radius_spin.setSingleStep(0.001)
        self.radius_spin.setDecimals(4)
        self.radius_spin.setToolTip("Leading edge nose radius (0 = sharp)")
        layout.addWidget(self.radius_spin, 0, 1)

        # R/L percentage label
        self.r_over_l_label = QLabel("R/L = 0.00%")
        layout.addWidget(self.r_over_l_label, 1, 0, 1, 2)
        self.radius_spin.valueChanged.connect(self._update_r_over_l)
        self.length_spin.valueChanged.connect(self._update_r_over_l)

        group.setLayout(layout)
        return group

    def _create_mesh_group(self):
        group = QGroupBox("Mesh Resolution")
        layout = QGridLayout()

        layout.addWidget(QLabel("Streamwise:"), 0, 0)
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(10, 300)
        self.nx_spin.setValue(60)
        self.nx_spin.setSingleStep(10)
        layout.addWidget(self.nx_spin, 0, 1)

        layout.addWidget(QLabel("Spanwise:"), 1, 0)
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(10, 200)
        self.ny_spin.setValue(40)
        self.ny_spin.setSingleStep(10)
        layout.addWidget(self.ny_spin, 1, 1)

        group.setLayout(layout)
        return group

    def _create_viscous_group(self):
        group = QGroupBox("Viscous Model")
        layout = QGridLayout()

        layout.addWidget(QLabel("Wall temp:"), 0, 0)
        self.twall_combo = QComboBox()
        self.twall_combo.addItems(["Adiabatic", "Custom"])
        self.twall_combo.currentIndexChanged.connect(self._on_twall_mode_changed)
        layout.addWidget(self.twall_combo, 0, 1)

        layout.addWidget(QLabel("T_wall [K]:"), 1, 0)
        self.twall_spin = QDoubleSpinBox()
        self.twall_spin.setRange(200, 3000)
        self.twall_spin.setValue(300.0)
        self.twall_spin.setSingleStep(50)
        self.twall_spin.setDecimals(0)
        self.twall_spin.setEnabled(False)
        layout.addWidget(self.twall_spin, 1, 1)

        group.setLayout(layout)
        return group

    def _create_generate_group(self):
        group = QGroupBox("Generate")
        layout = QVBoxLayout()

        self.generate_btn = QPushButton("Generate Waverider")
        self.generate_btn.setStyleSheet(
            "QPushButton { background-color: #2B5B2B; color: white; "
            "padding: 8px; font-weight: bold; }"
            "QPushButton:hover { background-color: #3B7B3B; }"
        )
        self.generate_btn.clicked.connect(self.generate_waverider)
        layout.addWidget(self.generate_btn)

        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        # Preset buttons
        preset_layout = QHBoxLayout()
        btn_initial = QPushButton("Paper Initial")
        btn_initial.setToolTip("Initial guess from Jessen et al. Table 1")
        btn_initial.clicked.connect(self._load_preset_initial)
        preset_layout.addWidget(btn_initial)

        btn_opt_a = QPushButton("Paper Opt-A")
        btn_opt_a.setToolTip("Analytical optimized from Table 1")
        btn_opt_a.clicked.connect(self._load_preset_opt_analytical)
        preset_layout.addWidget(btn_opt_a)

        btn_opt_c = QPushButton("Paper Opt-C")
        btn_opt_c.setToolTip("CFD optimized from Table 1")
        btn_opt_c.clicked.connect(self._load_preset_opt_cfd)
        preset_layout.addWidget(btn_opt_c)

        layout.addLayout(preset_layout)

        group.setLayout(layout)
        return group

    def _create_results_group(self):
        group = QGroupBox("Results")
        layout = QGridLayout()

        self.result_labels = {}
        row = 0
        for key, label_text in [
            ('wedge_angle', 'Wedge angle:'),
            ('L_over_D', 'L/D:'),
            ('CL', 'CL:'),
            ('CD', 'CD:'),
            ('L', 'Lift [N]:'),
            ('D', 'Drag [N]:'),
            ('D_inv', 'D_inviscid [N]:'),
            ('D_visc', 'D_viscous [N]:'),
            ('D_base', 'D_base [N]:'),
            ('D_le', 'D_LE [N]:'),
            ('S_ref', 'S_ref [m2]:'),
            ('volume', 'Volume [m3]:'),
            ('base_w', 'Base width [m]:'),
            ('base_h', 'Base height [m]:'),
        ]:
            layout.addWidget(QLabel(label_text), row, 0)
            lbl = QLabel("—")
            lbl.setFont(QFont("Courier", 9))
            self.result_labels[key] = lbl
            layout.addWidget(lbl, row, 1)
            row += 1

        group.setLayout(layout)
        return group

    def _create_export_group(self):
        group = QGroupBox("Export")
        layout = QGridLayout()

        stl_btn = QPushButton("STL")
        stl_btn.clicked.connect(self.export_stl)
        layout.addWidget(stl_btn, 0, 0)

        step_btn = QPushButton("STEP")
        step_btn.clicked.connect(self.export_step)
        step_btn.setEnabled(CADQUERY_AVAILABLE)
        if not CADQUERY_AVAILABLE:
            step_btn.setToolTip("CadQuery not available")
        layout.addWidget(step_btn, 0, 1)

        send_btn = QPushButton("Send to Aero Tab")
        send_btn.setToolTip("Send mesh to main Aero Analysis tab")
        send_btn.clicked.connect(self._send_to_aero_tab)
        layout.addWidget(send_btn, 1, 0, 1, 2)

        group.setLayout(layout)
        return group

    # ── Slot Handlers ───────────────────────────────────────────────

    def _on_twall_mode_changed(self, idx):
        self.twall_spin.setEnabled(idx == 1)

    def _update_r_over_l(self):
        R = self.radius_spin.value()
        L = self.length_spin.value()
        pct = R / L * 100 if L > 0 else 0.0
        self.r_over_l_label.setText(f"R/L = {pct:.2f}%")

    def _update_chebyshev_preview(self):
        self.cheb_preview.update_plot(
            self.p1_spin.value(), self.p2_spin.value(),
            self.p3_spin.value(), self.width_spin.value(),
        )

    def _update_3d_view(self):
        """Refresh 3D canvas with current checkbox states."""
        self.canvas_3d.plot_waverider(
            self.waverider,
            show_upper=self.show_upper_check.isChecked(),
            show_lower=self.show_lower_check.isChecked(),
            show_le=self.show_le_check.isChecked(),
            show_info=self.show_info_check.isChecked(),
            aero_results=self.aero_results,
        )

    # ── Generation ──────────────────────────────────────────────────

    def generate_waverider(self):
        """Launch background worker to generate geometry + aero."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Generation already running.")
            return

        params = {
            'length': self.length_spin.value(),
            'width': self.width_spin.value(),
            'n': self.n_spin.value(),
            'beta_deg': self.beta_spin.value(),
            'epsilon': self.epsilon_spin.value(),
            'p1': self.p1_spin.value(),
            'p2': self.p2_spin.value(),
            'p3': self.p3_spin.value(),
            'R': self.radius_spin.value(),
            'M_inf': self.mach_spin.value(),
            'gamma': 1.4,
        }

        T_wall = None
        if self.twall_combo.currentIndex() == 1:
            T_wall = self.twall_spin.value()

        aero_params = {
            'M_inf': self.mach_spin.value(),
            'alpha_deg': self.alpha_spin.value(),
            'altitude_km': self.alt_spin.value(),
            'T_wall': T_wall,
            'nx': self.nx_spin.value(),
            'ny': self.ny_spin.value(),
        }

        self.generate_btn.setEnabled(False)
        self.progress_label.setText("Generating...")

        self.worker = PlanarWaveriderWorker(params, aero_params)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg):
        self.progress_label.setText(msg)

    def _on_finished(self, wr, results):
        self.waverider = wr
        self.aero_results = results
        self.generate_btn.setEnabled(True)
        self.progress_label.setText("Done")

        # Update 3D view
        self._update_3d_view()

        # Update result labels
        self._update_results(wr, results)

        # Update results text area
        self._update_results_text(wr, results)

        # Emit signal for inter-tab communication
        self.waverider_generated.emit(wr)

        # Update Chebyshev preview
        self._update_chebyshev_preview()

    def _on_error(self, msg):
        self.generate_btn.setEnabled(True)
        self.progress_label.setText("Error!")
        QMessageBox.critical(self, "Error", msg)

    def _update_results(self, wr, res):
        """Fill the results labels."""
        fmt = {
            'wedge_angle': f"{wr.wedge_angle_deg:.4f} deg",
            'L_over_D': f"{res['L_over_D']:.4f}",
            'CL': f"{res['CL']:.6f}",
            'CD': f"{res['CD']:.6f}",
            'L': f"{res['L']:.1f}",
            'D': f"{res['D']:.1f}",
            'D_inv': f"{res['D_inviscid']:.1f}",
            'D_visc': f"{res['D_viscous']:.1f}",
            'D_base': f"{res['D_base']:.1f}",
            'D_le': f"{res.get('D_le', 0):.1f}",
            'S_ref': f"{res['S_ref']:.4f}",
            'volume': f"{wr.volume():.6f}",
        }
        bw, bh = wr.base_dimensions()
        fmt['base_w'] = f"{bw:.4f}"
        fmt['base_h'] = f"{bh:.4f}"

        for key, val in fmt.items():
            if key in self.result_labels:
                self.result_labels[key].setText(val)

    def _update_results_text(self, wr, res):
        """Write detailed results to the text area."""
        T_inf, P_inf, rho_inf, a_inf = atmosphere(res['altitude_km'])
        V_inf = res['M_inf'] * a_inf

        lines = [
            "=" * 55,
            "  PLANAR WAVERIDER (Jessen et al. 2026)",
            "=" * 55,
            "",
            "  Flow Conditions",
            f"    Mach         = {res['M_inf']:.2f}",
            f"    Altitude     = {res['altitude_km']:.1f} km",
            f"    Alpha        = {res['alpha_deg']:.2f} deg",
            f"    T_inf        = {T_inf:.2f} K",
            f"    P_inf        = {P_inf:.1f} Pa",
            f"    V_inf        = {V_inf:.1f} m/s",
            f"    q_inf        = {res['q_inf']:.1f} Pa",
            "",
            "  Geometry",
            f"    Length       = {wr.length:.3f} m",
            f"    Width        = {wr.width:.3f} m",
            f"    n (power)    = {wr.n:.3f}",
            f"    beta (shock) = {wr.beta_deg:.2f} deg",
            f"    theta (wedge)= {wr.wedge_angle_deg:.4f} deg",
            f"    epsilon      = {wr.epsilon:.3f}",
            f"    p1, p2, p3   = {wr.p1:.3f}, {wr.p2:.3f}, {wr.p3:.3f}",
            f"    R (LE radius)= {wr.R:.4f} m",
            f"    S_ref        = {res['S_ref']:.4f} m2",
            f"    Volume       = {wr.volume():.6f} m3",
            "",
            "  Aerodynamic Performance",
            f"    L/D          = {res['L_over_D']:.4f}",
            f"    CL           = {res['CL']:.6f}",
            f"    CD           = {res['CD']:.6f}",
            f"    Lift         = {res['L']:.1f} N",
            f"    Drag (total) = {res['D']:.1f} N",
            f"      Inviscid   = {res['D_inviscid']:.1f} N",
            f"      Viscous    = {res['D_viscous']:.1f} N",
            f"      Base       = {res['D_base']:.1f} N",
            f"      LE         = {res.get('D_le', 0):.1f} N",
            "",
            "=" * 55,
        ]
        self.results_text.setText('\n'.join(lines))

    # ── Presets (Table 1 from the paper) ────────────────────────────

    def _load_preset_initial(self):
        """Initial guess: Table 1 row 1."""
        self.mach_spin.setValue(6.85)
        self.alt_spin.setValue(25.0)
        self.alpha_spin.setValue(0.0)
        self.length_spin.setValue(1.0)
        self.width_spin.setValue(12.0)
        self.n_spin.setValue(0.5)
        self.beta_spin.setValue(9.0)
        self.epsilon_spin.setValue(0.0)
        self.p1_spin.setValue(1.0)
        self.p2_spin.setValue(1.0)
        self.p3_spin.setValue(1.0)
        self.radius_spin.setValue(0.0025)  # R/L = 0.25%

    def _load_preset_opt_analytical(self):
        """Analytical optimized: Table 1 row 2."""
        self.mach_spin.setValue(6.85)
        self.alt_spin.setValue(25.0)
        self.alpha_spin.setValue(0.0)
        self.length_spin.setValue(1.0)
        self.width_spin.setValue(12.81)
        self.n_spin.setValue(0.90)
        self.beta_spin.setValue(5.42)
        self.epsilon_spin.setValue(-0.35)
        self.p1_spin.setValue(1.47)
        self.p2_spin.setValue(1.54)
        self.p3_spin.setValue(1.57)
        self.radius_spin.setValue(0.0025)

    def _load_preset_opt_cfd(self):
        """CFD optimized: Table 1 row 3."""
        self.mach_spin.setValue(6.85)
        self.alt_spin.setValue(25.0)
        self.alpha_spin.setValue(0.0)
        self.length_spin.setValue(1.0)
        self.width_spin.setValue(19.06)
        self.n_spin.setValue(0.90)
        self.beta_spin.setValue(9.00)
        self.epsilon_spin.setValue(-0.56)
        self.p1_spin.setValue(0.98)
        self.p2_spin.setValue(1.02)
        self.p3_spin.setValue(0.97)
        self.radius_spin.setValue(0.0025)

    # ── Parameter serialisation (JSON save / load) ─────────────────

    def get_params_dict(self):
        """Return all design parameters as a JSON-serialisable dict."""
        return {
            'mach': self.mach_spin.value(),
            'altitude': self.alt_spin.value(),
            'alpha': self.alpha_spin.value(),
            'length': self.length_spin.value(),
            'width': self.width_spin.value(),
            'n': self.n_spin.value(),
            'beta': self.beta_spin.value(),
            'epsilon': self.epsilon_spin.value(),
            'p1': self.p1_spin.value(),
            'p2': self.p2_spin.value(),
            'p3': self.p3_spin.value(),
            'le_radius': self.radius_spin.value(),
            'nx': self.nx_spin.value(),
            'ny': self.ny_spin.value(),
            'twall_mode': self.twall_combo.currentText(),
            'twall': self.twall_spin.value(),
        }

    def set_params_dict(self, d):
        """Restore parameters from a dict (e.g. loaded from JSON)."""
        from PyQt5.QtWidgets import QDoubleSpinBox, QSpinBox, QComboBox

        def _s(widget, value):
            if value is None:
                return
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                widget.setCurrentIndex(idx if idx >= 0 else 0)

        _s(self.mach_spin, d.get('mach'))
        _s(self.alt_spin, d.get('altitude'))
        _s(self.alpha_spin, d.get('alpha'))
        _s(self.length_spin, d.get('length'))
        _s(self.width_spin, d.get('width'))
        _s(self.n_spin, d.get('n'))
        _s(self.beta_spin, d.get('beta'))
        _s(self.epsilon_spin, d.get('epsilon'))
        _s(self.p1_spin, d.get('p1'))
        _s(self.p2_spin, d.get('p2'))
        _s(self.p3_spin, d.get('p3'))
        _s(self.radius_spin, d.get('le_radius'))
        _s(self.nx_spin, d.get('nx'))
        _s(self.ny_spin, d.get('ny'))
        _s(self.twall_combo, d.get('twall_mode'))
        _s(self.twall_spin, d.get('twall'))

    # ── Export ──────────────────────────────────────────────────────

    def export_stl(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate waverider first!")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save STL", "planar_waverider.stl", "STL (*.stl)")
        if not fn:
            return
        try:
            verts, faces = self.waverider.get_mesh()
            self._write_stl(fn, verts, faces)
            QMessageBox.information(self, "Success", f"Saved: {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _write_stl(self, filename, vertices, faces):
        """Write binary STL file."""
        import struct
        with open(filename, 'wb') as f:
            f.write(b'\0' * 80)  # header
            f.write(struct.pack('<I', len(faces)))
            for face in faces:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal /= norm
                f.write(struct.pack('<3f', *normal))
                f.write(struct.pack('<3f', *v0))
                f.write(struct.pack('<3f', *v1))
                f.write(struct.pack('<3f', *v2))
                f.write(struct.pack('<H', 0))

    def export_step(self):
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate waverider first!")
            return
        if not CADQUERY_AVAILABLE:
            QMessageBox.warning(self, "Warning", "CadQuery not available!")
            return
        if hasattr(self, '_step_worker') and self._step_worker is not None \
                and self._step_worker.isRunning():
            QMessageBox.warning(self, "Busy", "STEP export already running.")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save STEP", "planar_waverider.step", "STEP (*.step)")
        if not fn:
            return

        self.progress_label.setText("Exporting STEP...")
        self._step_worker = StepExportWorker(self.waverider, fn)
        self._step_worker.progress.connect(self._on_progress)
        self._step_worker.finished.connect(self._on_step_done)
        self._step_worker.error.connect(self._on_step_error)
        self._step_worker.start()

    def _on_step_done(self, filename):
        self.progress_label.setText("STEP export done")
        QMessageBox.information(self, "Success", f"Saved: {filename}")

    def _on_step_error(self, msg):
        self.progress_label.setText("STEP export failed")
        QMessageBox.critical(self, "Error", msg)

    def _send_to_aero_tab(self):
        """Send mesh data to the main aero analysis tab."""
        if self.waverider is None:
            QMessageBox.warning(self, "Warning", "Generate waverider first!")
            return
        if self.parent_gui and hasattr(self.parent_gui, 'imported_geometry'):
            verts, faces = self.waverider.get_mesh()
            self.parent_gui.imported_geometry = {
                'vertices': verts,
                'faces': faces,
                'source': 'planar_waverider',
                'params': self.waverider.to_dict(),
            }
            QMessageBox.information(
                self, "Sent",
                "Planar waverider mesh sent to Aero Analysis tab.")
        else:
            QMessageBox.warning(
                self, "Warning",
                "Parent GUI or imported_geometry not available.")
