"""
Animation utilities for SHADOW waverider optimization.

Generates GIF animations showing waverider shape evolution
during optimization iterations, similar to the PySAGAS front-page GIF.
"""

import os
import numpy as np
from typing import List, Dict, Optional


def _extract_unique_designs(history: List[Dict], poly_order: int) -> List[Dict]:
    """Deduplicate history entries (SLSQP uses repeated evals for FD gradients)."""
    n_vars = 3 if poly_order == 3 else 2
    unique = []
    prev_x = None
    for entry in history:
        try:
            x = np.array([entry[f'x{i}'] for i in range(n_vars)])
        except KeyError:
            continue
        if prev_x is None or not np.allclose(x, prev_x, atol=1e-8):
            unique.append(entry)
            prev_x = x.copy()
    return unique


def generate_optimization_gif(
    history: List[Dict],
    mach: float,
    shock_angle: float,
    poly_order: int,
    output_path: str,
    n_le: int = 15,
    n_stream: int = 15,
    fps: int = 4,
    dpi: int = 120,
    hold_last_frames: int = 8,
    elev: float = 25.0,
    azim: float = -60.0,
    max_frames: int = 60,
) -> Optional[str]:
    """
    Generate a GIF animation of waverider shape evolution during optimization.

    Regenerates waveriders from the design variable history (no file I/O
    dependency on VTK files). Deduplicates SLSQP finite-difference evaluations
    to show only unique optimizer steps.

    Parameters
    ----------
    history : list of dict
        Optimization history entries with 'x0', 'x1' (and 'x2' for 3rd-order).
    mach, shock_angle : float
        Flow conditions for waverider creation.
    poly_order : int
        Polynomial order (2 or 3).
    output_path : str
        Full path for the output .gif file.
    n_le, n_stream : int
        Mesh resolution for regenerated waveriders.
    fps : int
        Frames per second in the GIF.
    dpi : int
        Resolution of each frame.
    hold_last_frames : int
        Extra frames holding the final shape for emphasis.
    elev, azim : float
        Camera elevation and azimuth angles.
    max_frames : int
        Maximum number of unique frames (subsamples if exceeded).

    Returns
    -------
    str or None
        Path to the generated GIF, or None if generation failed.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from shadow_waverider import (
            create_second_order_waverider,
            create_third_order_waverider,
        )
    except ImportError as e:
        print(f"Animation: missing dependency ({e}), skipping GIF generation.")
        return None

    # Extract unique design points
    unique_entries = _extract_unique_designs(history, poly_order)
    if len(unique_entries) < 2:
        print("Animation: fewer than 2 unique designs, skipping GIF.")
        return None

    # Subsample if too many frames
    if len(unique_entries) > max_frames:
        indices = np.linspace(0, len(unique_entries) - 1, max_frames, dtype=int)
        unique_entries = [unique_entries[i] for i in indices]

    # Regenerate waveriders from design variables
    n_vars = 3 if poly_order == 3 else 2
    waveriders = []
    for entry in unique_entries:
        try:
            if poly_order == 2:
                wr = create_second_order_waverider(
                    mach=mach, shock_angle=shock_angle,
                    A2=entry['x0'], A0=entry['x1'],
                    n_leading_edge=n_le, n_streamwise=n_stream)
            else:
                wr = create_third_order_waverider(
                    mach=mach, shock_angle=shock_angle,
                    A3=entry['x0'], A2=entry['x1'], A0=entry['x2'],
                    n_leading_edge=n_le, n_streamwise=n_stream)
            waveriders.append(wr)
        except Exception:
            continue  # Skip failed geometries

    if len(waveriders) < 2:
        print("Animation: fewer than 2 valid waveriders, skipping GIF.")
        return None

    # Compute global axis limits across all frames (prevents camera jumps)
    all_mins = np.full(3, np.inf)
    all_maxs = np.full(3, -np.inf)
    for wr in waveriders:
        for surf in [wr.upper_surface, wr.lower_surface]:
            # Plotting coordinate transform: Z(span)->X, X(stream)->Y, Y(vert)->Z
            plot_coords = surf[:, :, [2, 0, 1]]
            all_mins = np.minimum(all_mins, plot_coords.min(axis=(0, 1)))
            all_maxs = np.maximum(all_maxs, plot_coords.max(axis=(0, 1)))

    center = (all_mins + all_maxs) / 2
    radius = 0.5 * np.max(all_maxs - all_mins) * 1.15  # 15% padding

    # Build animation
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.clear()
        wr_idx = min(frame_idx, len(waveriders) - 1)
        wr = waveriders[wr_idx]

        upper = wr.upper_surface
        lower = wr.lower_surface

        # Same coordinate transform as ShadowWaveriderCanvas.plot_waverider
        ax.plot_surface(upper[:, :, 2], upper[:, :, 0], upper[:, :, 1],
                        color='steelblue', alpha=0.6, linewidth=0,
                        antialiased=True, shade=True)
        ax.plot_surface(lower[:, :, 2], lower[:, :, 0], lower[:, :, 1],
                        color='indianred', alpha=0.6, linewidth=0,
                        antialiased=True, shade=True)

        # Leading edge
        if hasattr(wr, 'leading_edge') and wr.leading_edge is not None:
            le = wr.leading_edge
            ax.plot(le[:, 2], le[:, 0], le[:, 1], 'k-', linewidth=1.5)

        # Fixed camera and limits
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        return []

    total_frames = len(waveriders) + hold_last_frames

    try:
        anim = FuncAnimation(fig, update, frames=total_frames, blit=False)
        writer = PillowWriter(fps=fps)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        anim.save(output_path, writer=writer, dpi=dpi)
        plt.close(fig)
        return output_path
    except Exception as e:
        plt.close(fig)
        print(f"Animation: failed to save GIF ({e})")
        return None


def generate_gif_from_history_file(
    history_json_path: str,
    mach: float,
    shock_angle: float,
    poly_order: int,
    output_path: str = None,
    **kwargs,
) -> Optional[str]:
    """
    Load convergence_history.json and generate evolution GIF.

    Parameters
    ----------
    history_json_path : str
        Path to the convergence_history.json file.
    mach, shock_angle : float
        Flow conditions.
    poly_order : int
        Polynomial order (2 or 3).
    output_path : str, optional
        Output GIF path. Defaults to same directory as history file.

    Returns
    -------
    str or None
        Path to the generated GIF, or None if generation failed.
    """
    import json
    with open(history_json_path) as f:
        history = json.load(f)

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(history_json_path),
            'waverider_evolution.gif')

    return generate_optimization_gif(
        history, mach, shock_angle, poly_order,
        output_path, **kwargs)
