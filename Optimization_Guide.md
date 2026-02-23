# Waverider Optimization Guide

This guide covers all optimization features in the Waverider Design GUI, including where results and geometries are saved.

---

## Table of Contents

1. [Overview of Optimization Methods](#overview-of-optimization-methods)
2. [SHADOW Gradient-Based Optimization](#shadow-gradient-based-optimization)
3. [NSGA-II Multi-Objective Optimization](#nsga-ii-multi-objective-optimization)
4. [Surrogate-Assisted Optimization](#surrogate-assisted-optimization)
5. [Neural Network Surrogate (Off-Design)](#neural-network-surrogate-off-design)
6. [Multi-Mach Hunter](#multi-mach-hunter)
7. [Design Space Exploration](#design-space-exploration)
8. [Where Results and Geometries Are Saved](#where-results-and-geometries-are-saved)
9. [Troubleshooting](#troubleshooting)

---

## Overview of Optimization Methods

The GUI provides multiple optimization approaches, each suited to different design goals:

| Method | Tab | Best For | Speed | Variables |
|--------|-----|----------|-------|-----------|
| **SHADOW Gradient-Based** | SHADOW Waverider → Optimization | Single-objective, fine-tuning | Fast (25-50 evals) | A2, A0 (or A3, A2, A0) |
| **NSGA-II Genetic Algorithm** | Optimization | Multi-objective Pareto fronts | Slow (1000+ evals) | X1, X2, X3, X4 |
| **Surrogate (Gaussian Process)** | Surrogate Opt | Multi-objective with few evals | Medium (50-200 evals) | X1, X2, X3, X4 |
| **Neural Network Hunter** | Off-Design Surrogate | Quick design screening | Very fast (seconds) | X1-X4 + Mach/beta |
| **Multi-Mach Hunter** | Multi-Mach Hunter | Robust multi-Mach performance | Very fast (seconds) | X1-X4 + Mach/beta |
| **Design Space Exploration** | SHADOW Waverider → Design Space | Understanding parameter landscape | Medium | A2, A0 (or A3, A2) |

---

## SHADOW Gradient-Based Optimization

### What It Does

Optimizes the leading-edge polynomial coefficients of a cone-derived (SHADOW) waverider using scipy gradient-based methods. Each evaluation generates a waverider geometry, meshes it with Gmsh, and runs PySAGAS Oblique Panel Method for aerodynamic analysis.

### Design Variables

**2nd Order Polynomial (default):**
- **A2**: Quadratic coefficient controlling leading-edge curvature (typical range: -15 to -0.5)
- **A0**: Constant offset controlling leading-edge height (typical range: -0.4 to -0.02)

**3rd Order Polynomial:**
- **A3**: Cubic coefficient (typical range: -30 to 30)
- **A2**: Quadratic coefficient (typical range: -15 to -0.5)
- **A0**: Constant offset (typical range: -0.4 to -0.02)

### Configuration

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| **Objective** | L/D, -CD, or CL | L/D (maximize lift-to-drag ratio) |
| **Method** | SLSQP, COBYLA, or Nelder-Mead | Nelder-Mead (most robust) or SLSQP (faster when it works) |
| **Max Iterations** | Maximum optimizer iterations | 50 |
| **Stability Constraints** | Enforce Cm_alpha<0, Cn_beta>0, Cl_beta<0 | Off (adds significant cost per evaluation) |
| **Gmsh Mesh Min/Max** | Mesh element size in meters | Medium preset (0.005/0.05) for optimization, Fine for final |
| **Save Pressure VTK** | Save pressure field at each iteration | On (for debugging; off for speed) |
| **Save Geometry VTK** | Save waverider mesh VTK per iteration | On (for ParaView animation) |

### How to Run

1. Go to the **SHADOW Waverider** tab
2. Set Mach number and shock angle for your design condition
3. Adjust A2, A0 values to a reasonable starting point (generate and inspect first)
4. Switch to the **Optimization** sub-tab
5. Choose objective, method, and bounds
6. Click **Run Optimization**

### Choosing a Method

- **SLSQP**: Uses finite-difference gradients. Fast convergence when the design space is smooth. Can fail with "Inequality constraints incompatible" if the optimizer encounters discontinuities (e.g., CL sign flips near boundary of feasible region).
- **COBYLA**: Gradient-free constrained optimizer. More robust than SLSQP but slower convergence.
- **Nelder-Mead**: Gradient-free simplex method. Most robust — handles noisy/discontinuous objectives well. **Recommended for first attempts.** Does not support explicit constraints (bounds are not enforced).

### Tips

- **Start with a valid design**: Generate the waverider first and verify it looks reasonable before optimizing.
- **Use Nelder-Mead first**: If SLSQP fails, switch to Nelder-Mead. It's slower but much more robust.
- **Narrow bounds carefully**: Wide bounds increase the chance of encountering degenerate geometries. Start narrow and widen if needed.
- **Use Coarse mesh for exploration**: Use the "Coarse" mesh preset during optimization, then re-evaluate the optimum with "Fine" mesh.
- **Check convergence plot**: The real-time L/D vs iteration plot shows whether the optimizer is making progress or oscillating.
- **Apply the result**: After optimization, a dialog asks whether to apply the optimal design to the main panel.
- **Even failed optimizations produce useful results**: If the optimizer doesn't converge, you'll be offered the best design it found along the way.

---

## NSGA-II Multi-Objective Optimization

### What It Does

Uses the NSGA-II genetic algorithm (via pymoo) to find a Pareto front of non-dominated designs. Supports multiple objectives simultaneously (e.g., maximize L/D while maximizing volume).

### Design Variables

The parametric waverider is defined by 4 normalized variables:

| Variable | Meaning | Range |
|----------|---------|-------|
| **X1** | Flat region of shockwave | 0.0 – 1.0 |
| **X2** | Height of shockwave (constrained by X1) | 0.0 – 1.0 |
| **X3** | Upper surface central control point | 0.0 – 1.0 |
| **X4** | Upper surface side control point | 0.0 – 1.0 |

**Important constraint (Kontogiannis et al., 2017):**
```
X2 / (1-X1)^4 < (7/64) * (width/height)^4
```
This geometric constraint ensures valid waverider shapes. The optimizer automatically repairs designs that violate it.

### Fixed Parameters

Set these before running:
- **Mach number (M∞)**: 3.0 – 8.0
- **Shock angle (beta)**: 10° – 25°
- **Height**: 0.5 – 5.0 m
- **Width**: 1.0 – 10.0 m
- **Angle of attack**: -10° to +10° (typically 0°)
- **Altitude** (or direct pressure/temperature)

### Objectives

Select one or more objectives. Each can be set to maximize or minimize:

| Objective | Typical Goal |
|-----------|-------------|
| **CD** (Drag coefficient) | Minimize |
| **CL** (Lift coefficient) | Maximize |
| **CL/CD** (Lift-to-drag ratio) | Maximize |
| **Cm** (Pitching moment) | Minimize or target a value |
| **Volume** | Maximize |

### Constraints

| Constraint | Description |
|------------|-------------|
| **Design Space** | Geometric discriminant formula (auto-repair) |
| **CL_min** | Minimum lift coefficient |
| **CD_max** | Maximum drag coefficient |
| **Cm_max** | Maximum pitching moment magnitude |
| **Volume_min** | Minimum internal volume |

### Algorithm Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| **Population Size** | Designs per generation | 40-100 |
| **Generations** | Number of GA generations | 20-50 |
| **Crossover Probability** | SBX crossover rate | 0.9 |
| **Mutation Probability** | Polynomial mutation rate | 0.1 |
| **CPU Cores** | Parallel evaluation threads | Number of physical cores |
| **Mesh Size** | Gmsh element size | 0.2 (coarser = faster) |

### Tips

- **Start with 2 objectives**: e.g., CL/CD + Volume. More objectives require exponentially larger populations.
- **Use parallel evaluation**: Set CPU cores to your core count for significant speedup.
- **Population size matters**: At least 10× the number of design variables (so >= 40 for 4 variables).
- **Monitor the Pareto front**: Watch for the front spreading and stabilizing across generations.
- **Export individual designs**: Select designs from the Pareto front table to export their CAD files.

---

## Surrogate-Assisted Optimization

A detailed tutorial is available in **`Surrogate_Optimization_Tutorial.md`**.

### Quick Summary

Uses Gaussian Process (GP) surrogate models to approximate the expensive PySAGAS evaluations, enabling efficient optimization with far fewer CFD calls.

**Two modes:**
- **Adaptive (EGO)**: Sequential single-design evaluation guided by acquisition functions (EI, LCB, PI)
- **Hybrid**: Combines GA optimization on the surrogate model with PySAGAS validation of promising designs

**Key settings:**
- Initial samples: 50 (Latin Hypercube)
- Kernel: Matérn 5/2
- Max evaluations: 100-200
- Objectives: Same as NSGA-II (CL/CD, Volume, etc.)

---

## Neural Network Surrogate (Off-Design)

### What It Does

Uses a pre-trained neural network ensemble for instant aerodynamic predictions without running PySAGAS. Supports off-design conditions (different flight Mach and AoA from the design point).

### Modes

- **Prediction Mode**: Enter design parameters and flight conditions, get instant CL, CD, CL/CD predictions with uncertainty estimates.
- **Hunter Mode**: Automatically searches for the best design given target flight conditions and geometry constraints. Returns top 10 candidates ranked by CL/CD.

### Requirements

A trained ensemble model must be loaded first. Models are trained from CSV datasets containing design parameters and their PySAGAS-computed aerodynamic coefficients.

---

## Multi-Mach Hunter

### What It Does

Finds waverider designs that perform well across a range of Mach numbers, not just at a single design point.

### Optimization Objectives

| Strategy | Description |
|----------|-------------|
| **Robust** | Maximize worst-case (minimum) CL/CD across the Mach range |
| **Mean** | Maximize average CL/CD |
| **Consistent** | Minimize variation in CL/CD across Mach numbers |
| **Balanced** | Combination of mean performance and consistency |

---

## Design Space Exploration

### What It Does

Performs a systematic parameter sweep over the SHADOW waverider design variables. Evaluates a grid of A2, A0 (or A3, A2) combinations and plots the results as a colored design space map.

### Configuration

- **A2 range**: Start, end, and number of steps
- **A0 range**: Start, end, and number of steps (2nd order) — or A3 range (3rd order)
- **Include aero**: Whether to run PySAGAS analysis at each point (slower but gives L/D data)
- **Color by**: L/D, CL, CD, Volume, or validity

### How to Read the Plot

- Each point represents one waverider geometry
- Color indicates the selected metric (e.g., L/D)
- Invalid designs (geometry failures) are shown in gray
- The best design is highlighted and its parameters displayed

### Exporting Results

Click **Export CSV** to save the full design space data for external analysis.

---

## Where Results and Geometries Are Saved

### SHADOW Gradient-Based Optimizer

All files saved in **`optimization_results/`** (relative to the GUI working directory):

```
optimization_results/
├── iter_0001.step              # STEP geometry at iteration 1
├── iter_0001.stl               # Gmsh surface mesh at iteration 1
├── iter_0001/
│   └── pressure.vtu            # PySAGAS pressure visualization (VTK)
├── geometry_0001.vtu           # Waverider mesh VTK (for ParaView animation)
├── iter_0002.step
├── iter_0002.stl
├── iter_0002/
│   └── pressure.vtu
├── geometry_0002.vtu
│   ...
├── optimized_waverider.stl     # Final optimized geometry (STL)
├── optimized_waverider.tri     # Final optimized geometry (TRI format)
├── convergence_history.json    # Full iteration log (JSON)
└── convergence_history.csv     # Full iteration log (CSV)
```

**ParaView animation**: Open the `geometry_*.vtu` files as a time series in ParaView to see the waverider shape evolving during optimization.

### NSGA-II Optimizer

All files saved in **`results/optimization_YYYYMMDD_HHMMSS/`**:

```
results/optimization_20260223_143022/
├── designs.csv                 # All evaluated designs with aero coefficients
├── pareto_front.csv            # Non-dominated (Pareto-optimal) solutions
├── optimization_config.json    # Full configuration (objectives, constraints, params)
├── designs/
│   └── design_00001/           # Individual design geometry files
│       ├── waverider.step
│       └── waverider.stl
├── pareto_designs/             # Geometry files for Pareto-optimal designs
│   ├── pareto_001.step
│   └── pareto_001.stl
└── plots/
    ├── convergence.png         # Objective convergence plot
    └── pareto_front.png        # Pareto front visualization
```

### Surrogate Optimizer

Results are saved to the surrogate tab's configured output directory, including:
- Evaluated designs CSV
- GP model quality metrics
- Pareto front plots

### Manual Export (All Tabs)

From any tab with a generated waverider:
- **STL**: Click "STL" button → choose save location (triangulated surface mesh)
- **TRI**: Click "TRI" button → choose save location (Cart3D format)
- **STEP**: Click "STEP" button → choose save location (NURBS CAD solid, mm scale)
- **CSV**: Click "Export CSV" for design space data (SHADOW tab)

### Configuration Data

The NSGA-II optimizer saves `optimization_config.json` containing:
- All objective definitions (name, mode, target values)
- All constraint settings (name, value, active flag)
- Design variable bounds
- Fixed parameters (Mach, beta, height, width)
- Algorithm parameters (population, generations, crossover/mutation rates)
- Simulation parameters (AoA, pressure, temperature, mesh size)

---

## Troubleshooting

### "Inequality constraints incompatible" (SLSQP)

**Cause**: SLSQP's internal QP subproblem cannot satisfy the linearized bound constraints. This typically happens when the finite-difference gradient estimation encounters a design where CL flips sign (e.g., L/D jumps from +6 to -37), creating a discontinuity that breaks the linear approximation.

**Solutions**:
1. **Switch to Nelder-Mead**: Gradient-free, handles discontinuities well
2. **Narrow bounds**: Reduce the A2/A0 range to stay in the well-behaved region
3. **Use the best-found design**: The optimizer now shows the best design it found before failing — use it as a new starting point with tighter bounds

### Negative L/D Values During Optimization

**Cause**: For certain polynomial coefficient combinations, the waverider geometry produces negative lift (CL < 0). The L/D ratio becomes negative and very large in magnitude.

**Solution**: The optimizer now detects negative CL designs and applies a smooth penalty instead of using the raw negative L/D. This prevents gradient corruption. If you see negative L/D values in the convergence history, the optimizer is exploring near the edge of the valid design space — narrowing bounds will help.

### Slow Optimization

**Cause**: Each evaluation involves geometry generation → Gmsh meshing → PySAGAS analysis.

**Solutions**:
1. Use "Coarse" mesh preset during optimization (refine the final result)
2. Disable "Save Pressure VTK" to skip VTK file writing
3. Reduce max iterations
4. For multi-objective problems, use the Surrogate optimizer (far fewer evaluations needed)

### "PySAGAS failed" / "Geometry failed"

**Cause**: The waverider geometry at that design point is degenerate (self-intersecting surfaces, zero area panels, etc.).

**Solutions**:
1. Check your Mach number and shock angle are compatible (shock angle must exceed Mach angle)
2. Generate the waverider manually first to verify the baseline is valid
3. Narrow the design variable bounds to avoid degenerate regions

### "Gmsh pipeline failed, falling back to direct mesh"

**Cause**: The STEP → Gmsh → STL meshing pipeline encountered an error (often due to complex geometry). The optimizer automatically falls back to direct triangulation from the waverider's internal mesh.

**Note**: The fallback mesh is lower quality but allows optimization to continue. Results may be slightly different from Gmsh-meshed evaluations. For the final design, manually re-evaluate with the Gmsh pipeline.

### Optimization Converges but L/D Is Low

**Possible causes**:
1. Initial point is in a local minimum — try multiple starting points
2. Bounds are too narrow — the optimal design is outside the search range
3. Mach/shock angle combination limits achievable L/D — try different flow conditions
4. Mesh is too coarse — re-evaluate the optimum with a finer mesh

### Design Space Constraint (NSGA-II)

If many designs in the NSGA-II optimization are being repaired or rejected:
1. Check that width/height ratio is reasonable for your X1, X2 bounds
2. The constraint `X2 / (1-X1)^4 < (7/64) * (w/h)^4` becomes very restrictive for X1 near 1.0
3. Consider constraining X1 upper bound to 0.5 or less
