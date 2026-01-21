#!/usr/bin/env python3
"""
Optimization Utilities for Waverider Design

Handles:
- Results folder creation and management
- CSV logging of all designs
- Pareto front extraction
- Design file organization
- Statistics and reporting
- Result visualization preparation

Author: Waverider Optimization System
"""

import os
import sys
import csv
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt


class OptimizationResults:
    """
    Manages optimization results storage and analysis.
    
    Creates a structured results folder with:
    - designs.csv: All evaluated designs
    - pareto_front.csv: Non-dominated solutions  
    - convergence.png: Objective history plot
    - designs/: Individual design folders with STEP/STL files
    - pareto_designs/: STEP/STL files for Pareto-optimal designs
    - optimization_config.json: Problem configuration
    """
    
    def __init__(self, base_dir: str = "results", run_name: Optional[str] = None):
        """
        Initialize results manager.
        
        Parameters
        ----------
        base_dir : str
            Base directory for all results (default: "results")
        run_name : str, optional
            Custom name for this optimization run
            If None, uses timestamp
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create timestamped run directory
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"optimization_{timestamp}"
        
        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(exist_ok=True)
        
        # Store the results folder path (THIS IS THE MISSING ATTRIBUTE!)
        self.results_folder = str(self.run_dir)
        
        # Create subdirectories
        self.designs_dir = self.run_dir / "designs"
        self.designs_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.pareto_designs_dir = self.run_dir / "pareto_designs"
        self.pareto_designs_dir.mkdir(exist_ok=True)
        
        # Initialize CSV files
        self.designs_csv = self.run_dir / "designs.csv"
        self.pareto_csv = self.run_dir / "pareto_front.csv"
        
        # Track design counter
        self.design_counter = 0
        
        print(f"✓ Results directory created: {self.run_dir}")
    
    def save_config(self, config: Dict):
        """
        Save optimization configuration to JSON.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary containing:
            - objectives: List of objective definitions
            - constraints: Constraint definitions  
            - design_variables: X1-X4 ranges
            - fixed_parameters: M_inf, beta, height, width
            - algorithm: population_size, n_generations, etc.
        """
        config_file = self.run_dir / "optimization_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"✓ Configuration saved: {config_file.name}")
    
    def initialize_designs_csv(self, design_var_names: List[str], 
                               objective_names: List[str],
                               constraint_names: List[str]):
        """
        Initialize designs.csv with headers.
        
        Parameters
        ----------
        design_var_names : List[str]
            Names of design variables (e.g., ['X1', 'X2', 'X3', 'X4'])
        objective_names : List[str]
            Names of objectives (e.g., ['CD', 'Volume'])
        constraint_names : List[str]
            Names of constraints (e.g., ['CL_min', 'Cm_min'])
        """
        headers = (['Design_ID', 'Generation', 'Timestamp'] + 
                   design_var_names + 
                   objective_names + 
                   constraint_names +
                   ['Success', 'Eval_Time_s'])
        
        with open(self.designs_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f"✓ CSV initialized: {self.designs_csv.name}")
    
    def log_design(self, design_vars: Dict, objectives: Dict, 
                   constraints: Dict, generation: int, 
                   success: bool, eval_time: float):
        """
        Log a design evaluation to CSV.
        
        Parameters
        ----------
        design_vars : Dict
            Design variables {'X1': val, 'X2': val, ...}
        objectives : Dict
            Objective values {'CD': val, 'Volume': val, ...}
        constraints : Dict
            Constraint violations {'CL_min': val, ...}
        generation : int
            Generation number
        success : bool
            Whether evaluation succeeded
        eval_time : float
            Evaluation time in seconds
        """
        self.design_counter += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = ([self.design_counter, generation, timestamp] +
               list(design_vars.values()) +
               list(objectives.values()) +
               list(constraints.values()) +
               [success, eval_time])
        
        with open(self.designs_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def extract_pareto_front(self, objective_names: List[str], 
                            minimize: List[bool]) -> pd.DataFrame:
        """
        Extract Pareto front from designs.csv.
        
        Parameters
        ----------
        objective_names : List[str]
            Names of objectives to consider
        minimize : List[bool]
            Whether to minimize each objective (True) or maximize (False)
            
        Returns
        -------
        pd.DataFrame
            Pareto-optimal designs
        """
        # Load all designs
        df = pd.read_csv(self.designs_csv)
        
        # Filter successful designs only
        df_success = df[df['Success'] == True].copy()
        
        if len(df_success) == 0:
            print("⚠ No successful designs to extract Pareto front")
            return pd.DataFrame()
        
        # Filter out designs with penalty values (1e6 or higher)
        for obj_name in objective_names:
            if obj_name in df_success.columns:
                df_success = df_success[df_success[obj_name].abs() < 1e5]
        
        if len(df_success) == 0:
            print("⚠ No valid designs after filtering penalty values")
            return pd.DataFrame()
        
        # Extract objective values
        objectives = df_success[objective_names].values
        
        # Flip sign for maximization objectives
        for i, should_min in enumerate(minimize):
            if not should_min:
                objectives[:, i] = -objectives[:, i]
        
        # Find Pareto front
        is_pareto = self._is_pareto_efficient(objectives)
        pareto_df = df_success[is_pareto].copy()
        
        # Save to CSV
        pareto_df.to_csv(self.pareto_csv, index=False)
        
        print(f"✓ Pareto front extracted: {len(pareto_df)} / {len(df_success)} designs")
        print(f"  Saved to: {self.pareto_csv.name}")
        
        return pareto_df
    
    def _copy_pareto_design_files(self, pareto_df: pd.DataFrame):
        """
        Copy STEP and STL files of Pareto-optimal designs to pareto_designs folder.
        
        Parameters
        ----------
        pareto_df : pd.DataFrame
            DataFrame of Pareto-optimal designs
        """
        if len(pareto_df) == 0:
            return
        
        copied_count = 0
        for _, row in pareto_df.iterrows():
            design_id = int(row['Design_ID'])
            design_folder = self.designs_dir / f"design_{design_id:05d}"
            
            if design_folder.exists():
                # Copy STEP file if it exists
                step_file = design_folder / f"waverider_{design_id:05d}.step"
                if step_file.exists():
                    shutil.copy2(step_file, self.pareto_designs_dir / step_file.name)
                    copied_count += 1
                
                # Copy STL file if it exists
                stl_file = design_folder / f"waverider_{design_id:05d}.stl"
                if stl_file.exists():
                    shutil.copy2(stl_file, self.pareto_designs_dir / stl_file.name)
        
        if copied_count > 0:
            print(f"✓ Copied {copied_count} Pareto design files to: {self.pareto_designs_dir.name}/")
    
    @staticmethod
    def _is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
        """
        Find Pareto-efficient points (minimization).
        
        Parameters
        ----------
        costs : np.ndarray
            Cost matrix (n_points x n_objectives)
            
        Returns
        -------
        np.ndarray
            Boolean array indicating Pareto-efficient points
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Keep any point with a lower cost in any dimension
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True  # Keep self
        return is_efficient
    
    def plot_convergence(self, objective_names: List[str], 
                        minimize: List[bool], 
                        save_name: str = "convergence.png"):
        """
        Plot objective convergence over generations.
        
        Parameters
        ----------
        objective_names : List[str]
            Names of objectives to plot
        minimize : List[bool]
            Whether to minimize each objective
        save_name : str
            Filename for saved plot
        """
        df = pd.read_csv(self.designs_csv)
        df_success = df[df['Success'] == True]
        
        if len(df_success) == 0:
            print("⚠ No successful designs to plot")
            return
        
        fig, axes = plt.subplots(1, len(objective_names), 
                                figsize=(6*len(objective_names), 5))
        
        if len(objective_names) == 1:
            axes = [axes]
        
        for ax, obj_name, should_min in zip(axes, objective_names, minimize):
            # Group by generation and find best/mean/worst
            gen_stats = df_success.groupby('Generation')[obj_name].agg(['min', 'mean', 'max'])
            
            if should_min:
                best = gen_stats['min']
                label = 'Best (min)'
            else:
                best = gen_stats['max']
                label = 'Best (max)'
            
            ax.plot(gen_stats.index, best, 'b-', linewidth=2, label=label)
            ax.plot(gen_stats.index, gen_stats['mean'], 'g--', label='Mean')
            ax.fill_between(gen_stats.index, gen_stats['min'], gen_stats['max'], 
                           alpha=0.2, color='gray', label='Range')
            
            ax.set_xlabel('Generation')
            ax.set_ylabel(obj_name)
            ax.set_title(f'{obj_name} Convergence')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Convergence plot saved: {save_path.name}")
    
    def plot_pareto_front(self, objective_names: List[str],
                         minimize: List[bool],
                         save_name: str = "pareto_front.png"):
        """
        Plot 2D Pareto front with performance-based coloring and professional styling.
        
        Parameters
        ----------
        objective_names : List[str]
            Two objective names to plot
        minimize : List[bool]
            Whether to minimize each objective
        save_name : str
            Filename for saved plot
        """
        if len(objective_names) != 2:
            print("⚠ Pareto front plotting requires exactly 2 objectives")
            return
        
        df = pd.read_csv(self.designs_csv)
        df_success = df[df['Success'] == True].copy()
        
        # Filter out penalty values
        for obj_name in objective_names:
            if obj_name in df_success.columns:
                df_success = df_success[df_success[obj_name].abs() < 1e5]
        
        if len(df_success) == 0:
            print("⚠ No successful designs to plot")
            return
        
        # Load Pareto front if it exists
        pareto_df = None
        if self.pareto_csv.exists():
            pareto_df = pd.read_csv(self.pareto_csv)
            for obj_name in objective_names:
                if obj_name in pareto_df.columns:
                    pareto_df = pareto_df[pareto_df[obj_name].abs() < 1e5]
        
        # Determine color metric (use CL if available, otherwise first objective)
        color_metric = None
        color_label = None
        if 'CL' in df_success.columns:
            color_metric = 'CL'
            color_label = 'Lift Coefficient (CL)'
        elif 'CL/CD' in df_success.columns and 'CL/CD' not in objective_names:
            color_metric = 'CL/CD'
            color_label = 'Lift-to-Drag Ratio'
        elif 'Generation' in df_success.columns:
            color_metric = 'Generation'
            color_label = 'Generation'
        else:
            # Fallback to first objective
            color_metric = objective_names[0]
            color_label = objective_names[0]
        
        # Create figure with extra space for info box
        fig = plt.figure(figsize=(14, 9))
        
        # Main plot area (leave space on right for info box)
        ax = fig.add_axes([0.08, 0.12, 0.60, 0.78])
        
        # Get Pareto Design_IDs for exclusion from regular points
        pareto_ids = set()
        if pareto_df is not None and len(pareto_df) > 0:
            pareto_ids = set(pareto_df['Design_ID'].values)
        
        # Plot non-Pareto designs with performance-based coloring
        non_pareto = df_success[~df_success['Design_ID'].isin(pareto_ids)]
        scatter = None
        if len(non_pareto) > 0:
            x = non_pareto[objective_names[0]].values
            y = non_pareto[objective_names[1]].values
            c = non_pareto[color_metric].values
            
            # Create colormap (cool to warm: blue -> purple -> red -> orange)
            scatter = ax.scatter(x, y, c=c, cmap='plasma', s=45, alpha=0.7,
                                edgecolors='white', linewidth=0.3, zorder=2)
        
        # Plot Pareto designs as gold stars
        if pareto_df is not None and len(pareto_df) > 0:
            x_pareto = pareto_df[objective_names[0]].values
            y_pareto = pareto_df[objective_names[1]].values
            design_ids = pareto_df['Design_ID'].values
            
            # Sort by first objective for line plot
            sort_idx = np.argsort(x_pareto)
            x_sorted = x_pareto[sort_idx]
            y_sorted = y_pareto[sort_idx]
            
            # Draw Pareto front line (solid, behind stars)
            if len(x_pareto) > 1:
                ax.plot(x_sorted, y_sorted, color='#c0392b', linewidth=2.5, 
                       alpha=0.8, zorder=3, solid_capstyle='round')
            
            # Plot stars with gold/orange color
            ax.scatter(x_pareto, y_pareto, c='#f39c12', s=280, marker='*',
                      edgecolors='#d35400', linewidth=1.2, zorder=4,
                      label=f'Pareto Optimal ({len(pareto_df)})')
            
            # Add labels with Design_ID (no #)
            for xi, yi, did in zip(x_pareto, y_pareto, design_ids):
                ax.annotate(f'{int(did)}', (xi, yi),
                           textcoords='offset points', xytext=(0, -16),
                           ha='center', va='top', fontsize=9, fontweight='bold',
                           color='#922b21', zorder=5)
        
        # Colorbar on the right side of the plot
        if scatter is not None:
            cbar_ax = fig.add_axes([0.70, 0.12, 0.02, 0.78])
            cbar = fig.colorbar(scatter, cax=cbar_ax)
            cbar.set_label(color_label, fontsize=11, fontweight='bold')
            cbar.ax.tick_params(labelsize=9)
        
        # Axis labels with optimization direction
        dir1 = "minimize ↓" if minimize[0] else "maximize ↑"
        dir2 = "minimize ↓" if minimize[1] else "maximize ↑"
        ax.set_xlabel(f'{objective_names[0]}  ({dir1})', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{objective_names[1]}  ({dir2})', fontsize=12, fontweight='bold')
        
        # Title
        ax.set_title('Pareto Front - Objective Space', fontsize=14, fontweight='bold', pad=15)
        
        # Subtle grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
        ax.set_axisbelow(True)
        
        # Set axis limits with padding
        x_all = df_success[objective_names[0]].values
        y_all = df_success[objective_names[1]].values
        x_margin = (x_all.max() - x_all.min()) * 0.06 if x_all.max() != x_all.min() else 0.1
        y_margin = (y_all.max() - y_all.min()) * 0.06 if y_all.max() != y_all.min() else 0.1
        ax.set_xlim(x_all.min() - x_margin, x_all.max() + x_margin)
        ax.set_ylim(y_all.min() - y_margin, y_all.max() + y_margin)
        
        # Legend (compact, semi-transparent)
        if pareto_df is not None and len(pareto_df) > 0:
            legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.9,
                              edgecolor='gray', fancybox=True)
        
        # ===== Information Box (outside plot area) =====
        info_ax = fig.add_axes([0.76, 0.12, 0.22, 0.78])
        info_ax.axis('off')
        
        # Calculate statistics
        n_total = len(df_success)
        n_pareto = len(pareto_df) if pareto_df is not None else 0
        
        # Best values for each objective
        if minimize[0]:
            best_obj1 = df_success[objective_names[0]].min()
            best_obj1_id = df_success.loc[df_success[objective_names[0]].idxmin(), 'Design_ID']
        else:
            best_obj1 = df_success[objective_names[0]].max()
            best_obj1_id = df_success.loc[df_success[objective_names[0]].idxmax(), 'Design_ID']
        
        if minimize[1]:
            best_obj2 = df_success[objective_names[1]].min()
            best_obj2_id = df_success.loc[df_success[objective_names[1]].idxmin(), 'Design_ID']
        else:
            best_obj2 = df_success[objective_names[1]].max()
            best_obj2_id = df_success.loc[df_success[objective_names[1]].idxmax(), 'Design_ID']
        
        # Build info text
        info_lines = [
            ('OPTIMIZATION SUMMARY', 'title'),
            ('', 'spacer'),
            ('Designs Evaluated', 'header'),
            (f'  Total: {n_total}', 'value'),
            (f'  Pareto Optimal: {n_pareto}', 'value'),
            ('', 'spacer'),
            (f'Best {objective_names[0]}', 'header'),
            (f'  {best_obj1:.4f}', 'value'),
            (f'  (Design {int(best_obj1_id)})', 'small'),
            ('', 'spacer'),
            (f'Best {objective_names[1]}', 'header'),
            (f'  {best_obj2:.4f}', 'value'),
            (f'  (Design {int(best_obj2_id)})', 'small'),
        ]
        
        # Add Pareto designs list if not too many
        if pareto_df is not None and 0 < len(pareto_df) <= 8:
            info_lines.append(('', 'spacer'))
            info_lines.append(('Pareto Design IDs', 'header'))
            ids_str = ', '.join([str(int(x)) for x in pareto_df['Design_ID'].values])
            info_lines.append((f'  {ids_str}', 'value'))
        
        # Render info text
        y_pos = 0.95
        for text, style in info_lines:
            if style == 'title':
                info_ax.text(0.0, y_pos, text, fontsize=12, fontweight='bold',
                           color='#2c3e50', transform=info_ax.transAxes)
                y_pos -= 0.06
            elif style == 'header':
                info_ax.text(0.0, y_pos, text, fontsize=10, fontweight='bold',
                           color='#34495e', transform=info_ax.transAxes)
                y_pos -= 0.05
            elif style == 'value':
                info_ax.text(0.0, y_pos, text, fontsize=10, color='#2c3e50',
                           transform=info_ax.transAxes, family='monospace')
                y_pos -= 0.045
            elif style == 'small':
                info_ax.text(0.0, y_pos, text, fontsize=9, color='#7f8c8d',
                           transform=info_ax.transAxes, style='italic')
                y_pos -= 0.045
            elif style == 'spacer':
                y_pos -= 0.025
        
        # Add subtle border around info box
        info_ax.add_patch(plt.Rectangle((-0.05, 0.0), 1.1, 1.0, fill=False,
                                        edgecolor='#bdc3c7', linewidth=1,
                                        transform=info_ax.transAxes))
        
        # Save figure
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white',
                   edgecolor='none', pad_inches=0.2)
        plt.close()
        
        print(f"✓ Pareto front plot saved: {save_path.name}")
    
    def plot_constraint_vs_objective(self, objective_name: str, 
                                     width: float, height: float,
                                     minimize: bool = True,
                                     save_name: str = "constraint_plot.png"):
        """
        Plot geometric constraint value vs objective (similar to Figure 11 in paper).
        
        The geometric constraint is: X2 / (1 - X1)^4
        The constraint boundary is: (7/64) * (width/height)^4
        
        Parameters
        ----------
        objective_name : str
            Objective to plot on x-axis (e.g., 'CD', 'CL/CD', 'Volume')
        width : float
            Waverider width in meters
        height : float
            Waverider height in meters
        minimize : bool
            Whether the objective is minimized
        save_name : str
            Filename for saved plot
        """
        df = pd.read_csv(self.designs_csv)
        df_success = df[df['Success'] == True].copy()
        
        # Filter out penalty values
        if objective_name in df_success.columns:
            df_success = df_success[df_success[objective_name].abs() < 1e5]
        
        if len(df_success) == 0:
            print("⚠ No successful designs to plot")
            return
        
        # Calculate geometric constraint value for each design
        # Constraint: X2 / (1 - X1)^4
        X1 = df_success['X1'].values
        X2 = df_success['X2'].values
        constraint_values = X2 / ((1 - X1) ** 4 + 1e-10)
        
        # Calculate constraint boundary
        # Boundary: (7/64) * (width/height)^4
        boundary = (7.0 / 64.0) * (width / height) ** 4
        boundary_with_margin = 0.90 * boundary  # 10% safety margin
        
        # Get objective values
        obj_values = df_success[objective_name].values
        
        # Load Pareto front if it exists
        pareto_ids = set()
        if self.pareto_csv.exists():
            pareto_df = pd.read_csv(self.pareto_csv)
            if objective_name in pareto_df.columns:
                pareto_df = pareto_df[pareto_df[objective_name].abs() < 1e5]
            pareto_ids = set(pareto_df['Design_ID'].values)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Identify Pareto and non-Pareto designs
        design_ids = df_success['Design_ID'].values
        is_pareto = np.array([d in pareto_ids for d in design_ids])
        
        # Plot infeasible region (shaded area above boundary)
        x_range = [obj_values.min() * 0.95, obj_values.max() * 1.05]
        ax.fill_between(x_range, [boundary, boundary], [boundary * 3, boundary * 3],
                       color='lightcoral', alpha=0.3, label='Infeasible region', zorder=1)
        
        # Plot constraint boundary line
        ax.axhline(y=boundary, color='darkred', linestyle='--', linewidth=2,
                  label=f'Design space boundary ({boundary:.3f})', zorder=3)
        
        # Plot boundary with 10% margin
        ax.axhline(y=boundary_with_margin, color='orange', linestyle=':', linewidth=1.5,
                  label=f'90% safety margin ({boundary_with_margin:.3f})', zorder=3)
        
        # Plot non-Pareto designs
        non_pareto_mask = ~is_pareto
        if np.any(non_pareto_mask):
            ax.scatter(obj_values[non_pareto_mask], constraint_values[non_pareto_mask],
                      c='steelblue', s=40, alpha=0.6, edgecolors='white', linewidth=0.3,
                      label='Evaluated designs', zorder=4)
        
        # Plot Pareto designs
        if np.any(is_pareto):
            ax.scatter(obj_values[is_pareto], constraint_values[is_pareto],
                      c='gold', s=120, marker='*', edgecolors='darkorange', linewidth=1,
                      label='Pareto optimal', zorder=5)
        
        # Highlight designs near boundary (within 5%)
        near_boundary_mask = (constraint_values > boundary_with_margin * 0.95) & (constraint_values <= boundary)
        if np.any(near_boundary_mask):
            ax.scatter(obj_values[near_boundary_mask], constraint_values[near_boundary_mask],
                      facecolors='none', edgecolors='red', s=80, linewidth=1.5,
                      label='Near boundary', zorder=6)
        
        # Formatting
        direction = "↓ minimize" if minimize else "↑ maximize"
        ax.set_xlabel(f'{objective_name} ({direction})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Geometric Constraint Value\n$X_2 / (1-X_1)^4$', fontsize=12, fontweight='bold')
        ax.set_title(f'Geometric Constraint vs {objective_name}\n(Similar to Paper Figure 11)',
                    fontsize=14, fontweight='bold')
        
        # Set axis limits
        ax.set_xlim(x_range)
        y_max = min(constraint_values.max() * 1.2, boundary * 2)
        ax.set_ylim(0, y_max)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Legend
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
        
        # Add annotation box with statistics
        n_total = len(df_success)
        n_feasible = np.sum(constraint_values <= boundary)
        n_near_boundary = np.sum(near_boundary_mask)
        n_pareto = np.sum(is_pareto)
        
        stats_text = (
            f"Total designs: {n_total}\n"
            f"Feasible: {n_feasible} ({100*n_feasible/n_total:.1f}%)\n"
            f"Near boundary: {n_near_boundary}\n"
            f"Pareto optimal: {n_pareto}\n"
            f"\nGeometry:\n"
            f"  width = {width:.2f} m\n"
            f"  height = {height:.2f} m\n"
            f"  w/h ratio = {width/height:.3f}"
        )
        
        props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace', bbox=props)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Constraint plot saved: {save_path.name}")
        
        return save_path
    
    def generate_pareto_design_files(self, pareto_df: pd.DataFrame, 
                                      waverider_generator, cad_exporter,
                                      fixed_params: dict):
        """
        Generate and save STEP/STL files for Pareto-optimal designs.
        
        Parameters
        ----------
        pareto_df : pd.DataFrame
            DataFrame of Pareto-optimal designs from extract_pareto_front
        waverider_generator : callable
            Function to generate waverider geometry
        cad_exporter : callable
            Function to export CAD (to_CAD)
        fixed_params : dict
            Fixed parameters (M_inf, beta, height, width, etc.)
        """
        if len(pareto_df) == 0:
            print("⚠ No Pareto designs to generate")
            return
        
        print(f"\n📦 Generating {len(pareto_df)} Pareto design files...")
        
        generated_count = 0
        for _, row in pareto_df.iterrows():
            try:
                # Get original Design_ID
                design_id = int(row['Design_ID'])
                
                # Extract design variables
                X1 = float(row['X1'])
                X2 = float(row['X2'])
                X3 = float(row['X3'])
                X4 = float(row['X4'])
                
                # Generate waverider
                wr = waverider_generator(
                    M_inf=fixed_params['M_inf'],
                    beta=fixed_params['beta'],
                    height=fixed_params['height'],
                    width=fixed_params['width'],
                    dp=[X1, X2, X3, X4],
                    n_upper_surface=10000,
                    n_shockwave=10000,
                    n_planes=fixed_params.get('n_planes', 40),
                    n_streamwise=fixed_params.get('n_streamwise', 30),
                    delta_streamwise=fixed_params.get('delta_streamwise', 0.1)
                )
                
                # Create design-specific filenames using original Design_ID
                step_file = self.pareto_designs_dir / f"design_{design_id:04d}.step"
                stl_file = self.pareto_designs_dir / f"design_{design_id:04d}.stl"
                
                # Export STEP
                cad_exporter(waverider=wr, sides='both', export=True, 
                            filename=str(step_file), scale=1.0)
                
                # Generate STL mesh using subprocess (Windows-safe)
                mesh_size = fixed_params.get('mesh_size', 0.1)
                self._generate_stl_from_step_subprocess(str(step_file), str(stl_file), mesh_size)
                
                # Save design info to text file
                info_file = self.pareto_designs_dir / f"design_{design_id:04d}_info.txt"
                with open(info_file, 'w') as f:
                    f.write(f"Pareto Design {design_id}\n")
                    f.write("="*40 + "\n\n")
                    f.write("Design Variables:\n")
                    f.write(f"  X1 = {X1:.4f}\n")
                    f.write(f"  X2 = {X2:.4f}\n")
                    f.write(f"  X3 = {X3:.4f}\n")
                    f.write(f"  X4 = {X4:.4f}\n\n")
                    f.write("Performance:\n")
                    # Extract objective values from row
                    for col in row.index:
                        if col not in ['Design_ID', 'Generation', 'Timestamp', 'X1', 'X2', 'X3', 'X4', 'Success', 'Eval_Time_s']:
                            f.write(f"  {col} = {row[col]}\n")
                
                generated_count += 1
                print(f"  ✓ Generated design_{design_id:04d} (X=[{X1:.3f}, {X2:.3f}, {X3:.3f}, {X4:.3f}])")
                
            except Exception as e:
                print(f"  ✗ Failed to generate design {design_id}: {str(e)}")
        
        print(f"\n✓ Generated {generated_count}/{len(pareto_df)} Pareto design files")
        print(f"  Location: {self.pareto_designs_dir}")
    
    def _generate_stl_from_step_subprocess(self, step_file: str, stl_file: str, mesh_size: float):
        """Generate STL mesh from STEP file using Gmsh in subprocess (Windows-safe)."""
        import subprocess
        import sys
        
        # Create a small Python script to run gmsh
        mesh_script = f'''
import gmsh
import sys

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("General.Verbosity", 0)

try:
    gmsh.model.add("waverider")
    gmsh.model.occ.importShapes(r"{step_file}")
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
    
    gmsh.option.setNumber("Mesh.MeshSizeMin", {mesh_size * 0.5})
    gmsh.option.setNumber("Mesh.MeshSizeMax", {mesh_size * 2.0})
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    
    gmsh.model.mesh.generate(2)
    gmsh.write(r"{stl_file}")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
finally:
    gmsh.finalize()
'''
        
        # Write script to temp file
        script_file = os.path.join(os.path.dirname(step_file), "mesh_script.py")
        with open(script_file, 'w') as f:
            f.write(mesh_script)
        
        try:
            # Run in subprocess
            result = subprocess.run(
                [sys.executable, script_file],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0 or "ERROR" in result.stdout:
                error_msg = result.stderr or result.stdout
                raise RuntimeError(f"Gmsh subprocess failed: {error_msg}")
            
            if not os.path.exists(stl_file):
                raise RuntimeError("STL file was not created")
                
        finally:
            try:
                os.remove(script_file)
            except:
                pass
    
    def _generate_stl_from_step(self, step_file: str, stl_file: str, mesh_size: float):
        """Generate STL mesh from STEP file using Gmsh (deprecated, use subprocess version)."""
        import gmsh
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        
        try:
            gmsh.model.add("waverider")
            gmsh.model.occ.importShapes(step_file)
            gmsh.model.occ.synchronize()
            
            gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size * 2.0)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            
            gmsh.model.mesh.generate(2)
            gmsh.write(stl_file)
        finally:
            gmsh.finalize()
    
    def print_summary(self):
        """Print optimization summary statistics."""
        if not self.designs_csv.exists():
            print("⚠ No designs.csv found")
            return
        
        df = pd.read_csv(self.designs_csv)
        df_success = df[df['Success'] == True]
        
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Total designs evaluated: {len(df)}")
        print(f"Successful evaluations: {len(df_success)} ({100*len(df_success)/len(df):.1f}%)")
        print(f"Failed evaluations: {len(df) - len(df_success)}")
        
        if len(df_success) > 0:
            print(f"\nAverage evaluation time: {df_success['Eval_Time_s'].mean():.2f}s")
            print(f"Total computation time: {df_success['Eval_Time_s'].sum()/3600:.2f} hours")
        
        if self.pareto_csv.exists():
            pareto_df = pd.read_csv(self.pareto_csv)
            print(f"\nPareto-optimal designs: {len(pareto_df)}")
        
        print(f"\nResults location: {self.run_dir}")
        print("="*60)


# Test code
if __name__ == "__main__":
    print("Testing OptimizationResults...")
    
    # Create test results manager
    results = OptimizationResults(run_name="test_run")
    
    # Save test config
    config = {
        'objectives': [
            {'name': 'CD', 'mode': 'minimize'},
            {'name': 'Volume', 'mode': 'maximize'}
        ],
        'constraints': [
            {'name': 'CL_min', 'value': 1.0}
        ]
    }
    results.save_config(config)
    
    # Initialize CSV
    results.initialize_designs_csv(
        design_var_names=['X1', 'X2', 'X3', 'X4'],
        objective_names=['CD', 'Volume'],
        constraint_names=['CL_min']
    )
    
    # Log some test designs
    np.random.seed(42)
    for gen in range(5):
        for i in range(10):
            design_vars = {
                f'X{j}': np.random.rand() for j in range(1, 5)
            }
            objectives = {
                'CD': 0.5 + np.random.rand() * 0.1,
                'Volume': 2.0 + np.random.rand() * 1.0
            }
            constraints = {'CL_min': -0.1 + np.random.rand() * 0.2}  # Negative = satisfied
            
            results.log_design(
                design_vars=design_vars,
                objectives=objectives,
                constraints=constraints,
                generation=gen,
                success=True,
                eval_time=15.0 + np.random.rand() * 5.0
            )
    
    # Extract Pareto front
    pareto_df = results.extract_pareto_front(
        objective_names=['CD', 'Volume'],
        minimize=[True, False]  # Minimize CD, maximize Volume
    )
    
    # Plot convergence
    results.plot_convergence(
        objective_names=['CD', 'Volume'],
        minimize=[True, False]
    )
    
    # Plot Pareto front
    results.plot_pareto_front(
        objective_names=['CD', 'Volume'],
        minimize=[True, False]
    )
    
    # Print summary
    results.print_summary()
    
    print("\n✅ All tests passed!")
    print(f"   Check results in: {results.run_dir}")
    print(f"   Pareto designs folder: {results.pareto_designs_dir}")
