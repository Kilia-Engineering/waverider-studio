# Surrogate-Assisted Optimization Tutorial for Waverider Design

This tutorial guides you through using the Surrogate Optimization tab in the Waverider Design GUI. Surrogate-assisted optimization uses Gaussian Process (GP) models to approximate the expensive aerodynamic simulations, enabling efficient exploration of the design space with fewer CFD evaluations.

---

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Introduction to Surrogate Optimization](#introduction-to-surrogate-optimization)
3. [Launching the GUI](#launching-the-gui)
4. [Setting Up Baseline Geometry](#setting-up-baseline-geometry)
5. [Configuring Surrogate Optimization](#configuring-surrogate-optimization)
   - [Mode Selection](#mode-selection)
   - [Sampling Settings](#sampling-settings)
   - [Surrogate Model Settings](#surrogate-model-settings)
   - [Acquisition Function Settings](#acquisition-function-settings)
   - [Budget Settings](#budget-settings)
   - [Objective Configuration](#objective-configuration)
   - [Design Variable Bounds](#design-variable-bounds)
6. [Running the Optimization](#running-the-optimization)
7. [Monitoring Progress](#monitoring-progress)
8. [Reviewing Results](#reviewing-results)
9. [Tips and Best Practices](#tips-and-best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start Guide

For users who want to get started immediately with recommended settings:

### Step 1: Launch the GUI
```bash
python waverider_gui.py
```

### Step 2: Set Baseline Geometry (Parameters Panel)
| Parameter | Recommended Value |
|-----------|------------------|
| Mach (M∞) | 5.0 |
| Shock Angle (β) | 15.0° |
| Height | 1.34 m |
| Width | 3.00 m |
| X1, X2, X3, X4 | 0.25, 0.50, 0.50, 0.50 |

### Step 3: Navigate to "🔮 Surrogate Opt" Tab

### Step 4: Apply Quick Start Settings
| Setting | Value |
|---------|-------|
| **Mode** | Hybrid |
| **Sampling Method** | Latin Hypercube (LHS) |
| **Initial Samples** | 50 |
| **Kernel** | Matérn 5/2 |
| **Max Evaluations** | 200 |
| **Surrogate Generations** | 10 |
| **Validate per Cycle** | 10 |
| **Objectives** | CL/CD (maximize) + Volume (maximize) |

### Step 5: Set Design Variable Bounds
| Variable | Min | Max |
|----------|-----|-----|
| X1 | 0.0 | 0.5 |
| X2 | 0.0 | 0.5 |
| X3 | 0.0 | 1.0 |
| X4 | 0.0 | 1.0 |

### Step 6: Click "▶ Start Optimization"

### Step 7: Monitor Progress
- Watch the console log for evaluation progress
- View live Pareto front and response surface plots
- Check surrogate model quality (R² > 0.7 is good)

---

## Introduction to Surrogate Optimization

### What is Surrogate-Assisted Optimization?

Traditional optimization of waverider designs requires hundreds or thousands of CFD simulations, each taking significant computational time. Surrogate-assisted optimization addresses this by:

1. **Building a surrogate model** (Gaussian Process) from a small set of initial CFD evaluations
2. **Using the surrogate** to predict aerodynamic performance quickly
3. **Strategically selecting** new designs to evaluate based on the surrogate predictions
4. **Updating the surrogate** with new data to improve accuracy

This approach can find near-optimal designs with 5-10x fewer CFD evaluations compared to traditional genetic algorithms.

### Two Optimization Modes

The GUI provides two surrogate optimization strategies:

#### Adaptive Mode (EGO - Efficient Global Optimization)
- Evaluates **one design at a time**
- Uses acquisition functions (Expected Improvement, etc.) to balance exploration vs exploitation
- Best for: Fine-tuning, single-objective optimization, when you want maximum control

#### Hybrid Mode (Recommended)
- Combines surrogate modeling with **NSGA-II genetic algorithm**
- Runs GA on the fast surrogate, then validates promising designs with real CFD
- Best for: Multi-objective optimization, robust exploration, production runs

---

## Launching the GUI

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install numpy pandas matplotlib PyQt5 scikit-learn scipy pymoo
```

For aerodynamic analysis (PySAGAS):
```bash
pip install pysagas
```

### Starting the Application

Navigate to your project directory and run:

```bash
python waverider_gui.py
```

The main window will open with several tabs. The surrogate optimization is in the **"🔮 Surrogate Opt"** tab.

---

## Setting Up Baseline Geometry

Before running optimization, configure the baseline waverider geometry in the **Parameters** panel on the left side of the GUI.

### Flow Conditions

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| **Mach Number (M∞)** | Freestream Mach number | 4.0 - 8.0 |
| **Shock Angle (β)** | Oblique shock angle in degrees | 10° - 25° |

### Geometric Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| **Height** | Waverider height in meters | 1.0 - 2.0 m |
| **Width** | Waverider span in meters | 2.0 - 4.0 m |

### Design Variables (X1 - X4)

These control the waverider shape:

| Variable | Controls | Range | Notes |
|----------|----------|-------|-------|
| **X1** | Leading edge curvature | 0.0 - 0.5 | Higher = more curved |
| **X2** | Upper surface profile | 0.0 - 0.5 | Constrained by X1 |
| **X3** | Spanwise distribution | 0.0 - 1.0 | **Strong influence on volume** |
| **X4** | Thickness distribution | 0.0 - 1.0 | **Strong influence on volume** |

> ⚠️ **Important**: X1 and X2 are linked by a geometric constraint (Equation 8 from the paper). The GUI automatically enforces this constraint, but be aware that increasing X1 reduces the maximum allowable X2.

---

## Configuring Surrogate Optimization

Navigate to the **"🔮 Surrogate Opt"** tab to configure your optimization run.

### Mode Selection

Choose between two optimization strategies:

#### Adaptive (EGO)
```
○ Adaptive (EGO)
```
- Sequential, one-at-a-time evaluation
- Uses acquisition functions to select next point
- Good for: Single-objective, fine-tuning, small budgets

#### Hybrid (Recommended)
```
● Hybrid
```
- Runs NSGA-II on surrogate, validates best designs
- More robust for multi-objective problems
- Good for: Production runs, multi-objective optimization

**Recommendation**: Start with **Hybrid** mode for multi-objective waverider optimization.

### Sampling Settings

Configure how initial samples are generated:

| Setting | Options | Recommendation |
|---------|---------|----------------|
| **Method** | LHS, Sobol, Random | **LHS** (Latin Hypercube) |
| **Initial Samples** | 10 - 100 | **50** for 4 variables |

**Why LHS?** Latin Hypercube Sampling ensures good coverage of the design space with fewer samples than random sampling.

**How many initial samples?** A rule of thumb is 10× the number of design variables. For 4 variables (X1-X4), 40-50 samples is a good starting point.

### Surrogate Model Settings

Configure the Gaussian Process surrogate:

| Setting | Options | Recommendation |
|---------|---------|----------------|
| **Kernel** | Matérn 5/2, Matérn 3/2, RBF | **Matérn 5/2** |
| **Normalize** | Yes/No | **Yes** |
| **Restarts** | 1 - 20 | **10** |

**Why Matérn 5/2?** This kernel is twice-differentiable, making it suitable for smooth aerodynamic responses while still being flexible enough to capture local variations.

### Acquisition Function Settings

**(Adaptive mode only)**

The acquisition function determines how the next evaluation point is selected:

| Function | Description | When to Use |
|----------|-------------|-------------|
| **EI** (Expected Improvement) | Balances exploration and exploitation | Default choice |
| **LCB** (Lower Confidence Bound) | More explorative with high κ | When you want broader search |
| **PI** (Probability of Improvement) | More exploitative | When refining known good regions |

**κ (kappa) parameter**: Controls exploration vs exploitation trade-off.
- High κ (2.0+): More exploration
- Low κ (0.5): More exploitation

**Adaptive κ**: Enable to automatically reduce κ over time (explore early, exploit later).

### Budget Settings

Control how many evaluations to perform:

| Setting | Description | Recommendation |
|---------|-------------|----------------|
| **Max Evaluations** | Total CFD evaluations allowed | **200** |
| **Surrogate Generations** | GA generations on surrogate per cycle | **10** |
| **Validate per Cycle** | Designs to validate with CFD per cycle | **10** |

**Budget calculation for Hybrid mode**:
```
Total evaluations ≈ Initial samples + (Cycles × Validate per cycle)
Cycles = (Max evaluations - Initial samples) / Validate per cycle
```

Example with recommended settings:
```
Initial: 50 samples
Remaining: 150 evaluations
Cycles: 150 / 10 = 15 cycles
```

### Objective Configuration

Select which objectives to optimize:

| Objective | Direction | Description |
|-----------|-----------|-------------|
| **CD** | Minimize | Drag coefficient |
| **CL** | Maximize | Lift coefficient |
| **CL/CD** | Maximize | Lift-to-drag ratio (aerodynamic efficiency) |
| **Volume** | Maximize | Internal volume for payload |
| **Cm** | Target | Pitching moment (stability) |

**Recommended multi-objective setup**:
- ✅ CL/CD (Maximize) - Aerodynamic efficiency
- ✅ Volume (Maximize) - Payload capacity

This creates a Pareto front showing the trade-off between efficiency and volume.

### Design Variable Bounds

Set the search bounds for each design variable:

| Variable | Min | Max | Notes |
|----------|-----|-----|-------|
| **X1** | 0.0 | 0.5 | Keep max ≤ 0.5 to satisfy constraint |
| **X2** | 0.0 | 0.5 | Automatically limited by X1 |
| **X3** | 0.0 | 1.0 | Full range recommended |
| **X4** | 0.0 | 1.0 | Full range recommended |

> 💡 **Tip**: The GUI shows constraint status: ✓ Valid, ⚡ Near boundary, or ⚠️ Invalid

---

## Running the Optimization

### Pre-flight Checklist

Before clicking Start, verify:

- [ ] Baseline geometry is set (Mach, β, height, width)
- [ ] Optimization mode is selected (Hybrid recommended)
- [ ] Initial samples configured (50 recommended)
- [ ] Max evaluations set (200 recommended)
- [ ] At least one objective enabled
- [ ] Design variable bounds are reasonable
- [ ] Constraint status shows ✓ Valid

### Starting the Optimization

1. Click the **"▶ Start Optimization"** button
2. The console will show initialization messages
3. Phase 1 (Initial Sampling) begins automatically
4. Phase 2 (Optimization) follows after surrogate is built

### Stopping the Optimization

Click **"⏹ Stop"** to gracefully stop the optimization. Current results will be saved.

---

## Monitoring Progress

### Console Log

The console displays real-time progress:

```
============================================================
HYBRID SURROGATE OPTIMIZATION
============================================================
Time: 2025-01-15 10:30:00

📊 Initial samples: 50
📊 Max evaluations: 200
📊 Objectives: ['CL/CD', 'Volume']

========================================
PHASE 1: Initial Sampling
========================================
Generating 50 initial samples using lhs...
  Evaluating design 1: X=[0.210, 0.220, 0.844, 0.097]
  Evaluating design 2: X=[0.385, 0.074, 0.641, 0.031]
  ...
✓ 50/50 successful evaluations

Building initial surrogate model...
  CL/CD: R²=0.782, RMSE=0.156
  Volume: R²=0.891, RMSE=0.043

========================================
PHASE 2: Hybrid Optimization
========================================

==============================
CYCLE 1
==============================
🔄 Running NSGA-II on surrogate for 10 generations...
  Found 25 surrogate Pareto designs
🔬 Validating 10 designs with PySAGAS...
  ...
```

### Key Metrics to Watch

| Metric | Good Value | Meaning |
|--------|------------|---------|
| **R²** | > 0.7 | Surrogate explains 70%+ of variance |
| **RMSE** | Low | Prediction error (units of objective) |
| **Pareto designs** | Growing | Optimization finding trade-offs |

### Live Plots

Open the plot windows to visualize progress:

1. **Response Surface**: 3D view of surrogate predictions
2. **Pareto Front**: Trade-off between objectives
3. **Constraint Plot**: X1-X2 constraint vs objective (Figure 11 style)

---

## Reviewing Results

### Results Folder Structure

After optimization completes, results are saved to:

```
results/
└── surrogate_YYYYMMDD_HHMMSS/
    ├── designs.csv           # All evaluated designs
    ├── pareto_front.csv      # Pareto optimal designs
    ├── surrogate_model.pkl   # Saved GP model
    ├── optimization_config.json
    └── plots/
        ├── pareto_front.png
        ├── convergence.png
        └── constraint_plot.png
```

### Pareto Front CSV

The `pareto_front.csv` contains the non-dominated solutions:

| Design_ID | X1 | X2 | X3 | X4 | CL/CD | Volume | CD | CL |
|-----------|-----|-----|-----|-----|-------|--------|-----|-----|
| 47 | 0.12 | 0.35 | 0.92 | 0.78 | 3.82 | 2.65 | 0.189 | 0.72 |
| 89 | 0.08 | 0.28 | 0.88 | 0.91 | 3.95 | 2.71 | 0.185 | 0.73 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Selecting a Final Design

From the Pareto front, select a design based on your priorities:

- **Maximum efficiency**: Choose highest CL/CD
- **Maximum volume**: Choose highest Volume
- **Balanced**: Choose a design in the "knee" of the Pareto front

### Exporting Designs

Use the GUI to export selected designs:
1. Select a Pareto design from the results table
2. Click "Export STEP" or "Export STL"
3. Choose destination folder

---

## Tips and Best Practices

### Design Variable Insights

Based on the waverider parametrization:

> 💡 **X3 and X4 have the strongest influence on volume.** If maximizing internal volume is important, ensure these variables have wide bounds (0.0 - 1.0).

> ⚠️ **Watch the X1-X2 constraint boundary.** The geometric constraint (Equation 8) limits X2 based on X1. The GUI enforces a 90% safety margin. Designs near this boundary may have unusual shapes.

### Recommended Settings by Use Case

#### Quick Exploration (< 1 hour)
| Setting | Value |
|---------|-------|
| Initial samples | 30 |
| Max evaluations | 100 |
| Validate per cycle | 5 |

#### Standard Optimization (2-4 hours)
| Setting | Value |
|---------|-------|
| Initial samples | 50 |
| Max evaluations | 200 |
| Validate per cycle | 10 |

#### Thorough Search (4+ hours)
| Setting | Value |
|---------|-------|
| Initial samples | 80 |
| Max evaluations | 500 |
| Validate per cycle | 15 |

### Improving Surrogate Quality

If R² is low (< 0.6):

1. **Increase initial samples**: More data = better model
2. **Try different kernel**: RBF for smoother responses, Matérn 3/2 for rougher
3. **Increase restarts**: Better hyperparameter optimization
4. **Check for outliers**: Failed simulations can corrupt the model

### Multi-Objective Tips

- Start with 2 objectives (e.g., CL/CD + Volume)
- More objectives = harder problem, need more evaluations
- Look for the "knee" of the Pareto front for balanced solutions

---

## Troubleshooting

### Common Issues

#### "Too few successful evaluations"
**Cause**: Many initial samples failed CFD analysis.
**Solution**: 
- Check mesh quality settings
- Verify geometry parameters are valid
- Increase initial samples

#### Low R² values (< 0.5)
**Cause**: Surrogate model not fitting well.
**Solution**:
- Increase initial samples
- Try different kernel
- Check for extreme outliers in data

#### "Surrogate NSGA-II error"
**Cause**: Problem with optimization on surrogate.
**Solution**:
- Ensure surrogate model was built successfully
- Check that bounds are valid
- Restart the optimization

#### Optimization stuck / not progressing
**Cause**: Local optimum or poor exploration.
**Solution**:
- Increase κ for more exploration
- Use adaptive κ schedule
- Increase surrogate generations

### Getting Help

If you encounter issues:

1. Check the console log for error messages
2. Verify all dependencies are installed
3. Try with default/recommended settings first
4. Check that PySAGAS is working independently

---

## Appendix: Algorithm Details

### Hybrid Mode Algorithm

```
1. Generate initial samples using LHS
2. Evaluate all samples with PySAGAS (CFD)
3. Build GP surrogate model
4. REPEAT until budget exhausted:
   a. Run NSGA-II on surrogate (fast, many generations)
   b. Extract Pareto front from surrogate optimization
   c. Select diverse designs from surrogate Pareto
   d. Evaluate selected designs with PySAGAS (CFD)
   e. Add new data to training set
   f. Rebuild surrogate model
5. Return validated Pareto front
```

### Adaptive Mode (EGO) Algorithm

```
1. Generate initial samples using LHS
2. Evaluate all samples with PySAGAS (CFD)
3. Build GP surrogate model
4. REPEAT until budget exhausted:
   a. Compute acquisition function over design space
   b. Find point that maximizes acquisition
   c. Evaluate that point with PySAGAS (CFD)
   d. Add to training set
   e. Rebuild surrogate periodically
5. Return best designs found
```

### X1-X2 Geometric Constraint

From the paper (Equation 8):

```
X2 / (1 - X1)^4 < (7/64) × (width/height)^4
```

With 90% safety margin:
```
X2_max = 0.90 × (7/64) × (width/height)^4 × (1 - X1)^4
```

Example for width=3.0m, height=1.34m:
- X1 = 0.25 → X2_max = 1.42 (unconstrained)
- X1 = 0.50 → X2_max = 0.48
- X1 = 0.70 → X2_max = 0.078

---

*Document version: 1.0*
*Last updated: December 2025*
