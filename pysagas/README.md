<a name="readme-top"></a>

<h1 align="center">PySAGAS</h1>



<p align="center">
  <a href="https://pypi.org/project/hypysagas/">
    <img src="https://img.shields.io/pypi/v/hypysagas.svg?color=blue&style=plastic" alt="Latest version" width=95 height=20>
  </a>
  
  <a href="https://github.com/psf/black">
    <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
  </a>

  <a href="https://kieran-mackle.github.io/pysagas/pytest_report">
    <img src="https://github.com/kieran-mackle/pysagas/actions/workflows/tests.yml/badge.svg" alt="Test Status" >
  </a>

  <a href="https://kieran-mackle.github.io/pysagas/coverage">
    <img src="https://github.com/kieran-mackle/pysagas/raw/gh-pages/coverage.svg?raw=true" alt="Test Coverage" >
  </a>
  
</p>


<!-- start intro -->
**PySAGAS** is a Python package for the generation of **S**ensitivity **A**pproximations
for **G**eometric and **A**erodynamic **S**urface properties. It provides a 
computationally-efficient method for generating surface sensitivity approximations from 
existing flow solutions, to use in aerodynamic shape optimisation studies. The GIF below 
is an example of this, where a hypersonic waverider was optimised for maximum L/D at Mach
6.

<!-- end intro -->



![waverider-evolution-flipped](https://github.com/kieran-mackle/pysagas/assets/60687606/4c78a82c-8f20-4235-baf3-ad57bda4945d)



<!-- TABLE OF CONTENTS -->
<details>
  <summary><h2>Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#citing-pysagas">Citing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



## Getting Started
<!-- start getting started -->

### Prerequisites


#### ParaView
If you plan on using *PySAGAS* with Cart3D solutions, you should also install ParaView, or at 
least the ParaView Python bindings. If you are using an Anaconda environment, you can install 
the ParaView Python packages via
[Conda Forge](https://anaconda.org/conda-forge/paraview) using 
the command below:

```
conda install -c conda-forge paraview
```

If you already have ParaView installed, you can append the path to the binaries to the Python 
path using the snippet below.

```python
import sys
sys.path.insert(0, "/opt/ParaView-5.6.2-MPI-Linux-64bit/bin")

# Now the import should work
import paraview
```

For more information on ParaView's Python packages, see the 
[ParaView Wiki](https://www.paraview.org/Wiki/PvPython_and_PvBatch).


#### pyOptSparse

*PySAGAS* shape optimisation modules wrap around 
[pyOptSparse](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/index.html) to converge on optimal geometries. Follow the
[installation instructions](https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/install.html), noting that special optimisers require custom builds.

If using an Anaconda environment, you can also install PyOptSparse from Conda forge:

```
conda install -c conda-forge pyoptsparse
```

#### PyMesh

Having [PyMesh](https://github.com/PyMesh/PyMesh) installed can greatly enhance the capabilities
offered by `PySAGAS`. However, it can be difficult to install. Troubleshooting guide coming soon.


### Installation
After installing the dependencies above, clone this repo to your machine.

```
git clone https://github.com/kieran-mackle/pysagas
```

Next, use pip to install the `pysagas` package from the repo you 
just cloned.

```
cd pysagas
python3 -m pip install .
```

<!-- end getting started -->

<p align="right">[<a href="#readme-top">back to top</a>]</p>


## Usage

<!-- start usage -->

*PySAGAS* uses low-order methods to approximate sensitivities on the surface of aerodynamic 
geometries. The user must provide a nominal condition of the flow properties on the surface, along
with the sensitivity of the geometric vertices to the design parameters. From here, one of *PySAGAS* sensitivity calculators can be used.


<!-- end usage -->

<p align="right">[<a href="#readme-top">back to top</a>]</p>



## Citing PySAGAS
If you use PySAGAS in any published work, please cite it using the BibTex reference below.

```text
@inproceedings{Mackle2024,
  author    = {Mackle, Kieran and Jahn, Ingo},
  booktitle = {AIAA Science and Technology Forum and Exposition},
  title     = {Efficient and Flexible Methodology for the Aerodynamic Shape Optimisation of Hypersonic Vehicle Concepts in a High-Dimensional Design Space},
  year      = {2024},
}
```

<p align="right">[<a href="#readme-top">back to top</a>]</p>


## License
PySAGAS is licensed under [GPLv3](COPYING).

<p align="right">[<a href="#readme-top">back to top</a>]</p>
