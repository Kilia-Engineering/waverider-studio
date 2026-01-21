#!/usr/bin/env python3
"""
Mesh Canvas for STL Visualization
Displays STL mesh in the Analysis tab
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class MeshCanvas(FigureCanvas):
    """Canvas for displaying STL mesh"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = None
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Initialize empty 3D plot
        self.init_plot()
        
    def init_plot(self):
        """Initialize empty 3D axes"""
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Z [m]')
        self.ax.set_zlabel('Y [m]')
        self.ax.set_title('STL Mesh Preview')
        self.ax.grid(True, alpha=0.3)
        self.draw()
    
    def plot_mesh(self, vertices, faces, info_text=""):
        """
        Plot STL mesh
        
        Parameters:
        -----------
        vertices : ndarray (N, 3)
            Vertex coordinates
        faces : ndarray (M, 3)
            Triangle face indices
        info_text : str
            Information text to display
        """
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Create mesh
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Build triangle collection
        triangles = vertices[faces]
        
        # Create collection with face colors
        mesh = Poly3DCollection(triangles, alpha=0.7, edgecolor='black', 
                               linewidths=0.1, facecolors='cyan')
        self.ax.add_collection3d(mesh)
        
        # Set limits based on vertices
        self.ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        self.ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        self.ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
        
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Z [m]')
        self.ax.set_zlabel('Y [m]')
        
        # Add info text
        title = 'STL Mesh Preview'
        if info_text:
            title += f'\n{info_text}'
        self.ax.set_title(title, fontsize=10)
        
        # Equal aspect ratio
        self._set_axes_equal()
        
        self.draw()
    
    def _set_axes_equal(self):
        """Make axes of 3D plot have equal scale"""
        if self.ax is None:
            return
            
        limits = np.array([
            self.ax.get_xlim3d(),
            self.ax.get_ylim3d(),
            self.ax.get_zlim3d(),
        ])
        
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        
        self.ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        self.ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        self.ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    
    def clear_mesh(self):
        """Clear the mesh and show empty axes"""
        self.init_plot()
