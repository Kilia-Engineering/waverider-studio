#!/usr/bin/env python3
"""
Plot Windows for Optimization Visualization
Separate windows that update in real-time during optimization
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QMainWindow, QHBoxLayout, 
                             QPushButton, QFileDialog, QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import os


class ConvergencePlotWindow(QMainWindow):
    """Window showing convergence of objectives over generations"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.objectives = config['objectives']
        
        # Data storage
        self.generation_data = {}  # {gen: {obj_name: [values]}}
        self.best_per_gen = {}  # {obj_name: [best_value_per_gen]}
        
        for obj in self.objectives:
            self.best_per_gen[obj['name']] = []
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Optimization Convergence")
        self.resize(800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Initialize plot
        self.init_plot()
        
    def init_plot(self):
        """Initialize the convergence plot"""
        self.figure.clear()
        
        n_objectives = len(self.objectives)
        
        # Create subplots (one per objective)
        self.axes = []
        for i, obj in enumerate(self.objectives):
            ax = self.figure.add_subplot(n_objectives, 1, i+1)
            ax.set_xlabel('Generation')
            ax.set_ylabel(f"{obj['name']} ({obj['mode']})")
            ax.set_title(f"{obj['name']} Convergence")
            ax.grid(True, alpha=0.3)
            self.axes.append(ax)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def add_design(self, design_id, metrics):
        """Add a design evaluation to the plot"""
        generation = metrics.get('generation', 1)
        
        # Initialize generation data if needed
        if generation not in self.generation_data:
            self.generation_data[generation] = {}
            for obj in self.objectives:
                self.generation_data[generation][obj['name']] = []
        
        # Add values
        for obj in self.objectives:
            obj_name = obj['name']
            if obj_name in metrics:
                value = metrics[obj_name]
                # Filter out penalty values
                if abs(value) < 1e5:
                    self.generation_data[generation][obj_name].append(value)
        
        # Update plot periodically (every 5 designs to reduce overhead)
        if design_id % 5 == 0:
            self.update_plot()
    
    def update_plot(self):
        """Update the convergence plot"""
        for i, (ax, obj) in enumerate(zip(self.axes, self.objectives)):
            ax.clear()
            ax.set_xlabel('Generation')
            ax.set_ylabel(f"{obj['name']} ({obj['mode']})")
            ax.set_title(f"{obj['name']} Convergence")
            ax.grid(True, alpha=0.3)
            
            obj_name = obj['name']
            obj_mode = obj['mode']
            
            generations = sorted(self.generation_data.keys())
            if not generations:
                continue
            
            # Calculate best value per generation
            best_values = []
            for gen in generations:
                if obj_name in self.generation_data[gen]:
                    values = self.generation_data[gen][obj_name]
                    if values:
                        if obj_mode == 'Minimize':
                            best = min(values)
                        else:
                            best = max(values)
                        best_values.append(best)
                    else:
                        best_values.append(None)
                else:
                    best_values.append(None)
            
            # Filter out None values
            valid_gens = [g for g, v in zip(generations, best_values) if v is not None]
            valid_bests = [v for v in best_values if v is not None]
            
            if valid_gens:
                ax.plot(valid_gens, valid_bests, 'o-', linewidth=2, markersize=6,
                       label=f'Best {obj_mode}')
                ax.legend()
            
            # Format axes
            ax.ticklabel_format(style='plain', axis='y')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def load_final_plot(self, results_folder):
        """Load the final saved convergence plot"""
        plot_path = os.path.join(results_folder, 'plots', 'convergence.png')
        if os.path.exists(plot_path):
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            img = plt.imread(plot_path)
            ax.imshow(img)
            ax.axis('off')
            self.figure.tight_layout()
            self.canvas.draw()


class ParetoPlotWindow(QMainWindow):
    """Window showing Pareto front for 2-objective problems"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.objectives = config['objectives']
        
        if len(self.objectives) != 2:
            raise ValueError("ParetoPlotWindow requires exactly 2 objectives")
        
        # Data storage
        self.all_designs = []  # List of {obj1: val, obj2: val, success: bool}
        self.pareto_designs = []
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Pareto Front")
        self.resize(800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Initialize plot
        self.init_plot()
        
    def init_plot(self):
        """Initialize the Pareto front plot"""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        obj1 = self.objectives[0]
        obj2 = self.objectives[1]
        
        self.ax.set_xlabel(f"{obj1['name']} ({obj1['mode']})")
        self.ax.set_ylabel(f"{obj2['name']} ({obj2['mode']})")
        self.ax.set_title("Pareto Front")
        self.ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def add_design(self, design_id, metrics):
        """Add a design to the Pareto plot"""
        obj1_name = self.objectives[0]['name']
        obj2_name = self.objectives[1]['name']
        
        if obj1_name in metrics and obj2_name in metrics:
            obj1_val = metrics[obj1_name]
            obj2_val = metrics[obj2_name]
            success = metrics.get('success', True)
            
            # Filter out penalty values
            if abs(obj1_val) < 1e5 and abs(obj2_val) < 1e5:
                self.all_designs.append({
                    'design_id': design_id,
                    obj1_name: obj1_val,
                    obj2_name: obj2_val,
                    'success': success
                })
        
        # Update plot periodically (every 5 designs)
        if design_id % 5 == 0:
            self.update_plot()
    
    def update_plot(self):
        """Update the Pareto front plot"""
        if not self.all_designs:
            return
        
        self.ax.clear()
        
        obj1 = self.objectives[0]
        obj2 = self.objectives[1]
        obj1_name = obj1['name']
        obj2_name = obj2['name']
        
        # Extract data
        obj1_vals = [d[obj1_name] for d in self.all_designs]
        obj2_vals = [d[obj2_name] for d in self.all_designs]
        
        # Plot all designs
        self.ax.scatter(obj1_vals, obj2_vals, c='lightblue', s=50, alpha=0.6,
                       edgecolors='gray', linewidths=0.5, label='All designs')
        
        # Calculate and plot Pareto front (simple version for live update)
        pareto_mask = self.is_pareto_efficient(np.array([obj1_vals, obj2_vals]).T,
                                               [obj1['mode'] == 'Minimize', 
                                                obj2['mode'] == 'Minimize'])
        
        pareto_obj1 = [v for v, m in zip(obj1_vals, pareto_mask) if m]
        pareto_obj2 = [v for v, m in zip(obj2_vals, pareto_mask) if m]
        
        if pareto_obj1:
            self.ax.scatter(pareto_obj1, pareto_obj2, c='red', s=100, marker='*',
                           edgecolors='black', linewidths=1.5,
                           label=f'Pareto front ({len(pareto_obj1)})', zorder=10)
        
        self.ax.set_xlabel(f"{obj1_name} ({obj1['mode']})")
        self.ax.set_ylabel(f"{obj2_name} ({obj2['mode']})")
        self.ax.set_title(f"Pareto Front ({len(self.all_designs)} designs)")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.ticklabel_format(style='plain')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    @staticmethod
    def is_pareto_efficient(costs, minimize):
        """Simple Pareto efficiency check"""
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Determine dominance based on minimize flags
                if minimize[0] and minimize[1]:
                    is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                elif minimize[0] and not minimize[1]:
                    dominated = (costs[is_efficient, 0] < c[0]) | (costs[is_efficient, 1] > c[1])
                    is_efficient[is_efficient] = dominated
                elif not minimize[0] and minimize[1]:
                    dominated = (costs[is_efficient, 0] > c[0]) | (costs[is_efficient, 1] < c[1])
                    is_efficient[is_efficient] = dominated
                else:  # Both maximize
                    is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
                is_efficient[i] = True
        return is_efficient
    
    def load_final_plot(self, results_folder):
        """Load the final saved Pareto front plot"""
        plot_path = os.path.join(results_folder, 'plots', 'pareto_front.png')
        if os.path.exists(plot_path):
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            img = plt.imread(plot_path)
            ax.imshow(img)
            ax.axis('off')
            self.figure.tight_layout()
            self.canvas.draw()


class ConstraintPlotWindow(QMainWindow):
    """
    Window showing geometric constraint vs objective (Figure 11 style from paper).
    
    Plots: X2/(1-X1)^4 vs Objective
    Shows the design space boundary and feasible/infeasible regions.
    
    From the paper (Equation 8):
    X2 / (1 - X1)^4 < (7/64) * (width/height)^4
    """
    
    def __init__(self, config, width=3.0, height=1.34, parent=None):
        super().__init__(parent)
        self.config = config
        self.objectives = config['objectives']
        self.width = width
        self.height = height
        
        # Calculate constraint boundary
        # Boundary: (7/64) * (width/height)^4
        self.boundary = (7.0 / 64.0) * (width / height) ** 4
        self.boundary_safe = 0.90 * self.boundary  # 10% safety margin as in paper
        
        # Data storage
        self.all_designs = []  # List of {X1, X2, obj_values, is_pareto}
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        from PyQt5.QtWidgets import QLabel, QHBoxLayout, QComboBox
        
        self.setWindowTitle("Geometric Constraint Analysis (Paper Fig. 11)")
        self.resize(900, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Info label
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel(f"<b>Geometry:</b> w={self.width:.2f}m, h={self.height:.2f}m"))
        info_layout.addWidget(QLabel(f"<b>Boundary:</b> {self.boundary:.4f}"))
        info_layout.addWidget(QLabel(f"<b>Safe (90%):</b> {self.boundary_safe:.4f}"))
        info_layout.addStretch()
        layout.addLayout(info_layout)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 7))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Objective selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Plot objective:"))
        self.obj_combo = QComboBox()
        for obj in self.objectives:
            self.obj_combo.addItem(obj['name'])
        self.obj_combo.currentTextChanged.connect(self.update_plot)
        selector_layout.addWidget(self.obj_combo)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)
        
        # Initialize plot
        self.init_plot()
        
    def init_plot(self):
        """Initialize the constraint plot"""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        self.ax.set_xlabel('Objective Value')
        self.ax.set_ylabel('Geometric Constraint Value\n$X_2 / (1-X_1)^4$')
        self.ax.set_title('Geometric Constraint vs Objective (Paper Figure 11 Style)')
        self.ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def add_design(self, X1, X2, obj_values, is_pareto=False):
        """
        Add a design to the constraint plot.
        
        Parameters
        ----------
        X1 : float
            Design variable X1
        X2 : float
            Design variable X2
        obj_values : dict
            Dictionary of objective values {obj_name: value}
        is_pareto : bool
            Whether this design is Pareto optimal
        """
        # Calculate constraint value
        constraint_value = X2 / ((1 - X1) ** 4 + 1e-10)
        
        self.all_designs.append({
            'X1': X1,
            'X2': X2,
            'constraint': constraint_value,
            'obj_values': obj_values,
            'is_pareto': is_pareto
        })
        
        # Update plot periodically (every 10 designs to reduce overhead)
        if len(self.all_designs) % 10 == 0:
            self.update_plot()
    
    def update_pareto_status(self, pareto_design_ids):
        """Update Pareto status for designs"""
        for i, design in enumerate(self.all_designs):
            design['is_pareto'] = i in pareto_design_ids
        self.update_plot()
    
    def update_plot(self):
        """Update the constraint plot"""
        if not self.all_designs:
            return
        
        self.ax.clear()
        
        # Get selected objective
        obj_name = self.obj_combo.currentText()
        
        # Find minimize flag for this objective
        minimize = True
        for obj in self.objectives:
            if obj['name'] == obj_name:
                minimize = obj['mode'].lower() == 'minimize'
                break
        
        # Extract data
        constraint_values = []
        obj_values = []
        is_pareto = []
        
        for design in self.all_designs:
            cv = design['constraint']
            if obj_name in design['obj_values']:
                ov = design['obj_values'][obj_name]
                # Filter out penalty values
                if abs(ov) < 1e5 and cv < self.boundary * 3:
                    constraint_values.append(cv)
                    obj_values.append(ov)
                    is_pareto.append(design['is_pareto'])
        
        if not obj_values:
            self.ax.text(0.5, 0.5, "No valid data to display",
                        ha='center', va='center', fontsize=12,
                        transform=self.ax.transAxes)
            self.canvas.draw()
            return
        
        constraint_values = np.array(constraint_values)
        obj_values = np.array(obj_values)
        is_pareto = np.array(is_pareto)
        
        # Plot infeasible region (shaded area above boundary)
        x_min, x_max = obj_values.min() * 0.95, obj_values.max() * 1.05
        if x_min == x_max:
            x_min, x_max = x_min - 0.1, x_max + 0.1
        
        self.ax.fill_between([x_min, x_max], 
                            [self.boundary, self.boundary], 
                            [self.boundary * 2.5, self.boundary * 2.5],
                            color='lightcoral', alpha=0.3, 
                            label='Infeasible region', zorder=1)
        
        # Plot constraint boundary line
        self.ax.axhline(y=self.boundary, color='darkred', linestyle='--', linewidth=2,
                       label=f'Design space boundary ({self.boundary:.3f})', zorder=3)
        
        # Plot 90% safety margin line
        self.ax.axhline(y=self.boundary_safe, color='orange', linestyle=':', linewidth=1.5,
                       label=f'90% safety margin ({self.boundary_safe:.3f})', zorder=3)
        
        # Plot non-Pareto designs
        non_pareto_mask = ~is_pareto
        if np.any(non_pareto_mask):
            self.ax.scatter(obj_values[non_pareto_mask], constraint_values[non_pareto_mask],
                          c='steelblue', s=40, alpha=0.6, 
                          edgecolors='white', linewidths=0.3,
                          label='Evaluated designs', zorder=4)
        
        # Plot Pareto designs
        if np.any(is_pareto):
            self.ax.scatter(obj_values[is_pareto], constraint_values[is_pareto],
                          c='gold', s=150, marker='*', 
                          edgecolors='darkorange', linewidths=1.5,
                          label='Pareto optimal', zorder=6)
        
        # Highlight designs near boundary (within 5% of safe boundary)
        near_boundary_mask = (constraint_values > self.boundary_safe * 0.95) & \
                            (constraint_values <= self.boundary)
        if np.any(near_boundary_mask):
            self.ax.scatter(obj_values[near_boundary_mask], 
                          constraint_values[near_boundary_mask],
                          facecolors='none', edgecolors='red', s=80, linewidths=1.5,
                          label='Near boundary', zorder=5)
        
        # Formatting
        direction = "â†“ minimize" if minimize else "â†‘ maximize"
        self.ax.set_xlabel(f'{obj_name} ({direction})', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Geometric Constraint\n$X_2 / (1-X_1)^4$', 
                          fontsize=12, fontweight='bold')
        self.ax.set_title(f'Geometric Constraint vs {obj_name}\n(Paper Figure 11 Style)',
                         fontsize=14, fontweight='bold')
        
        # Set axis limits
        self.ax.set_xlim(x_min, x_max)
        y_max = min(constraint_values.max() * 1.3, self.boundary * 2.5)
        self.ax.set_ylim(0, y_max)
        
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=9)
        
        # Statistics box
        n_total = len(constraint_values)
        n_feasible = np.sum(constraint_values <= self.boundary)
        n_pareto = np.sum(is_pareto)
        n_near = np.sum(near_boundary_mask)
        
        stats_text = (
            f"Designs: {n_total}\n"
            f"Feasible: {n_feasible} ({100*n_feasible/n_total:.0f}%)\n"
            f"Near boundary: {n_near}\n"
            f"Pareto: {n_pareto}"
        )
        
        props = dict(boxstyle='round,pad=0.4', facecolor='white', 
                    edgecolor='gray', alpha=0.9)
        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace', bbox=props)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def load_final_plot(self, results_folder):
        """Load the final saved constraint plot or generate from CSV"""
        plot_path = os.path.join(results_folder, 'plots', 'constraint_plot.png')
        if os.path.exists(plot_path):
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            img = plt.imread(plot_path)
            ax.imshow(img)
            ax.axis('off')
            self.figure.tight_layout()
            self.canvas.draw()
        else:
            # Try to load from CSV and regenerate
            self.load_from_csv(results_folder)
    
    def load_from_csv(self, results_folder):
        """Load design data from CSV and update plot"""
        import pandas as pd
        
        csv_path = os.path.join(results_folder, 'designs.csv')
        pareto_csv = os.path.join(results_folder, 'pareto_front.csv')
        
        if not os.path.exists(csv_path):
            return
        
        df = pd.read_csv(csv_path)
        df_success = df[df['Success'] == True]
        
        # Load Pareto IDs
        pareto_ids = set()
        if os.path.exists(pareto_csv):
            pareto_df = pd.read_csv(pareto_csv)
            pareto_ids = set(pareto_df['Design_ID'].values)
        
        # Clear existing data
        self.all_designs = []
        
        for _, row in df_success.iterrows():
            X1 = row['X1']
            X2 = row['X2']
            obj_vals = {}
            for obj in self.objectives:
                if obj['name'] in row:
                    obj_vals[obj['name']] = row[obj['name']]
            is_pareto = row['Design_ID'] in pareto_ids
            
            constraint_value = X2 / ((1 - X1) ** 4 + 1e-10)
            
            self.all_designs.append({
                'X1': X1,
                'X2': X2,
                'constraint': constraint_value,
                'obj_values': obj_vals,
                'is_pareto': is_pareto
            })
        
        self.update_plot()


class AerodeckPlotWindow(QMainWindow):
    """Window for displaying AeroDeck sweep results in a proper Qt window."""
    
    def __init__(self, aerodeck_results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AeroDeck Sweep Results")
        self.setGeometry(100, 100, 1200, 800)
        
        # Store results
        self.results = aerodeck_results
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(14, 9), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Button bar
        btn_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save Figure")
        save_btn.clicked.connect(self.save_figure)
        btn_layout.addWidget(save_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        # Create the plots
        self.create_plots()
    
    def save_figure(self):
        """Save the figure to a file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", "aerodeck_results.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )
        if filepath:
            try:
                self.figure.savefig(filepath, dpi=150, bbox_inches='tight')
                QMessageBox.information(self, "Saved", f"Figure saved to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
    
    def create_plots(self):
        """Create the aerodeck plots."""
        from mpl_toolkits.mplot3d import Axes3D
        
        aoa = np.array(self.results['aoa'])
        mach = np.array(self.results['mach'])
        CL = np.array(self.results['CL'])
        CD = np.array(self.results['CD'])
        CL_CD = np.array(self.results['CL_CD'])
        
        # Get unique values
        unique_aoa = np.unique(aoa)
        unique_mach = np.unique(mach)
        
        # Clear figure
        self.figure.clear()
        
        # Check if we have a proper grid for surface plots
        can_surface = (len(unique_aoa) > 1 and len(unique_mach) > 1 and 
                      len(aoa) == len(unique_aoa) * len(unique_mach))
        
        if can_surface:
            # Reshape for surface plots
            AOA, MACH = np.meshgrid(unique_aoa, unique_mach, indexing='ij')
            CL_grid = CL.reshape(len(unique_aoa), len(unique_mach))
            CD_grid = CD.reshape(len(unique_aoa), len(unique_mach))
            CL_CD_grid = CL_CD.reshape(len(unique_aoa), len(unique_mach))
            
            # 3D Surface: CL/CD vs Mach vs AoA
            ax1 = self.figure.add_subplot(221, projection='3d')
            surf1 = ax1.plot_surface(AOA, MACH, CL_CD_grid, cmap='viridis', 
                                    edgecolors='k', linewidth=0.2, alpha=0.9)
            ax1.set_xlabel('AoA (°)', fontsize=10)
            ax1.set_ylabel('Mach', fontsize=10)
            ax1.set_zlabel('CL/CD', fontsize=10)
            ax1.set_title('CL/CD Ratio', fontsize=12, fontweight='bold')
            self.figure.colorbar(surf1, ax=ax1, shrink=0.5, label='CL/CD')
            
            # 3D Surface: CL vs Mach vs AoA
            ax2 = self.figure.add_subplot(222, projection='3d')
            surf2 = ax2.plot_surface(AOA, MACH, CL_grid, cmap='coolwarm',
                                    edgecolors='k', linewidth=0.2, alpha=0.9)
            ax2.set_xlabel('AoA (°)', fontsize=10)
            ax2.set_ylabel('Mach', fontsize=10)
            ax2.set_zlabel('CL', fontsize=10)
            ax2.set_title('Lift Coefficient', fontsize=12, fontweight='bold')
            self.figure.colorbar(surf2, ax=ax2, shrink=0.5, label='CL')
            
            # 2D: CL vs AoA for each Mach
            ax3 = self.figure.add_subplot(223)
            for m in unique_mach:
                mask = mach == m
                ax3.plot(aoa[mask], CL[mask], 'o-', label=f'M={m:.1f}', markersize=5)
            ax3.set_xlabel('AoA (°)', fontsize=11)
            ax3.set_ylabel('CL', fontsize=11)
            ax3.set_title('CL vs AoA', fontsize=12, fontweight='bold')
            ax3.legend(loc='best', fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # 2D: CL/CD vs AoA for each Mach
            ax4 = self.figure.add_subplot(224)
            for m in unique_mach:
                mask = mach == m
                ax4.plot(aoa[mask], CL_CD[mask], 's-', label=f'M={m:.1f}', markersize=5)
            ax4.set_xlabel('AoA (°)', fontsize=11)
            ax4.set_ylabel('CL/CD', fontsize=11)
            ax4.set_title('CL/CD vs AoA', fontsize=12, fontweight='bold')
            ax4.legend(loc='best', fontsize=8)
            ax4.grid(True, alpha=0.3)
            
        else:
            # Scatter plots if not a complete grid
            ax1 = self.figure.add_subplot(221, projection='3d')
            sc1 = ax1.scatter(aoa, mach, CL_CD, c=CL_CD, cmap='viridis', s=50)
            ax1.set_xlabel('AoA (°)')
            ax1.set_ylabel('Mach')
            ax1.set_zlabel('CL/CD')
            ax1.set_title('CL/CD Ratio')
            self.figure.colorbar(sc1, ax=ax1, shrink=0.5)
            
            ax2 = self.figure.add_subplot(222, projection='3d')
            sc2 = ax2.scatter(aoa, mach, CL, c=CL, cmap='coolwarm', s=50)
            ax2.set_xlabel('AoA (°)')
            ax2.set_ylabel('Mach')
            ax2.set_zlabel('CL')
            ax2.set_title('Lift Coefficient')
            self.figure.colorbar(sc2, ax=ax2, shrink=0.5)
            
            ax3 = self.figure.add_subplot(223)
            sc3 = ax3.scatter(aoa, CL, c=mach, cmap='plasma', s=50)
            ax3.set_xlabel('AoA (°)')
            ax3.set_ylabel('CL')
            ax3.set_title('CL vs AoA (color=Mach)')
            self.figure.colorbar(sc3, ax=ax3, label='Mach')
            
            ax4 = self.figure.add_subplot(224)
            sc4 = ax4.scatter(aoa, CL_CD, c=mach, cmap='plasma', s=50)
            ax4.set_xlabel('AoA (°)')
            ax4.set_ylabel('CL/CD')
            ax4.set_title('CL/CD vs AoA (color=Mach)')
            self.figure.colorbar(sc4, ax=ax4, label='Mach')
        
        self.figure.suptitle('AeroDeck Sweep Results', fontsize=14, fontweight='bold')
        self.figure.tight_layout()
        self.canvas.draw()
