# Waverider GUI

An interactive desktop application for designing, analyzing, and optimizing hypersonic waverider vehicles. Built with PyQt5.

## Features

- **Parametric waverider generation** from conical shock flow fields (Mach number, shock angle, height, width, design point)
- **3D visualization** of upper/lower surfaces, leading edge, base plane, and wireframe
- **Shadow waverider** generation with polynomial leading-edge parameterization
- **CAD export** to STEP format (via CadQuery) with STL mesh generation (via Gmsh)
- **Aerodynamic analysis** using PySAGAS oblique panel method (OPM) at multiple angles of attack
- **Multi-objective optimization** with genetic algorithms (NSGA-II) for L/D, volume, and drag
- **Surrogate modeling** using scikit-learn for fast aerodynamic prediction and design-space exploration
- **Multi-Mach optimization** for broadband performance across flight regimes
- **Claude AI assistant** integration for interactive design guidance (requires Anthropic API key)

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
git clone https://github.com/aklothakis/waverider_gui.git
cd waverider_gui
pip install -r requirements.txt
```

For optional features:

```bash
# High-quality mesh generation
pip install gmsh

# Claude AI assistant
pip install anthropic

# Report export
pip install python-docx

# PyTorch surrogate models
pip install torch
```

## Usage

```bash
python waverider_gui.py
```

### Quick start

1. Set freestream Mach number and shock angle (beta)
2. Adjust height, width, and design point parameters
3. Click **Generate Waverider** to create the geometry
4. Explore the 3D view, base plane, and leading edge tabs
5. Export to STEP for CAD or generate STL mesh for analysis

### Programmatic usage

```python
from waverider_generator.generator import waverider

wv = waverider(
    M_inf=5.0,
    beta=15.0,
    height=1.34,
    width=3.0,
    dp=[0.11, 0.63, 0.0, 0.46],
    n_upper_surface=10000,
    n_shockwave=10000,
)
```

## Project Structure

```
waverider_gui.py              # Main GUI application
waverider_generator/          # Core waverider generation
  generator.py                #   Parametric waverider from conical flow
  flowfield.py                #   Taylor-Maccoll conical flow solver
  cad_export.py               #   STEP export via CadQuery
shadow_waverider.py           # Shadow waverider with polynomial LE
optimization_tab.py           # Multi-objective optimization UI
optimization_engine.py        # GA optimization engine (NSGA-II)
surrogate_tab.py              # Surrogate model training UI
ai_surrogate.py               # Surrogate model training logic
claude_assistant_tab.py       # Claude AI assistant integration
pysagas/                      # Vendored PySAGAS aerodynamic solver
```

## Dependencies

See [requirements.txt](requirements.txt) for the full list. Core dependencies:

- **NumPy / SciPy / Pandas** -- numerical computing
- **PyQt5** -- GUI framework
- **Matplotlib** -- visualization
- **CadQuery** -- STEP CAD export
- **scikit-learn** -- surrogate models
- **pymoo / DEAP** -- optimization algorithms

## References

Kontogiannis, K., Sübester, A., & Taylor, N. (2017). Efficient Parameterization of Waverider Geometries. *J. Aircraft*, 54(3). [doi:10.2514/1.C033902](https://doi.org/10.2514/1.C033902)

Son, J., Son, C., & Yee, K. (2022). A Novel Direct Optimization Framework for Hypersonic Waverider Inverse Design Methods. *Aerospace*, 9(7), 348. [doi:10.3390/aerospace9070348](https://doi.org/10.3390/aerospace9070348)

Weaver, A. S. (2025). Investigating Stability of Cone-Derived Hypersonic Waverider Vehicles via Design Space Exploration. Master's Thesis, Utah State University. [digitalcommons.usu.edu/etd2023/485](https://digitalcommons.usu.edu/etd2023/485)

Mackle, Kieran & Jahn, Ingo. (2024). Efficient and Flexible Methodology for the Aerodynamic Shape Optimisation of Hypersonic Vehicle Concepts in a High-Dimensional Design Space. AIAA SCITECH 2024 Forum. p. 2838. [doi:10.2514/6.2024-2838](https://doi.org/10.2514/6.2024-2838)
## License

[MIT](LICENSE)
