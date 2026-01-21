import os
import glob
import shutil
import pickle
import numpy as np
import pandas as pd
from pysagas.cfd import OPM
from pysagas.flow import FlowState
from hypervehicle.generator import Generator
from pyoptsparse import Optimizer, Optimization
from pysagas.geometry.parsers import PyMesh, STL
from pysagas.optimisation.optimiser import ShapeOpt
from typing import List, Dict, Optional, Optional, Any
from pysagas.sensitivity import GenericSensitivityCalculator
from hypervehicle.utilities import SensitivityStudy, merge_stls


class OPMShapeOpt(ShapeOpt):
    def __init__(
        self,
        optimiser: Optimizer,
        vehicle_generator: Generator,
        objective_callback: callable,
        jacobian_callback: callable,
        freestream: FlowState,
        geometry_filename: str,
        A_ref: Optional[float] = 1.0,
        optimiser_options: Optional[Dict[str, Any]] = None,
        sensitivity_filename: str = "all_components_sensitivity.csv",
        working_dir_name: str = "working_dir",
        basefiles_dir_name: str = "basefiles",
        save_evolution: bool = True,
        verbosity: int = 1,
        sensitivity_kwargs: dict = None,
        parser_kwargs: dict = None,
    ) -> None:
        # Declare global variables
        global fs_flow
        global generator
        global working_dir
        global obj_cb, jac_cb
        global save_geom, evo_dir
        global sens_filename, basefiles_dir
        global geom_filename
        global _A_ref
        global verbose

        # Dynamics global variables
        global cells, solver
        cells = None
        solver = None

        # Construct paths
        # TODO - I dont think basefiles is needed.
        home_dir = os.getcwd()
        basefiles_dir = os.path.join(home_dir, basefiles_dir_name)
        working_dir = os.path.join(home_dir, working_dir_name)
        sens_filename = sensitivity_filename
        generator = vehicle_generator
        save_geom = save_evolution
        verbose = verbosity

        # Save sensitivity kwargs
        global _sens_kwargs, _parser_kwargs
        _sens_kwargs = sensitivity_kwargs if sensitivity_kwargs else {}
        _parser_kwargs = parser_kwargs if parser_kwargs else {}

        if save_geom:
            # Create evolution history directory
            evo_dir = os.path.join(home_dir, "evolution")
            if not os.path.exists(evo_dir):
                os.mkdir(evo_dir)

        # Save callback functions
        obj_cb = objective_callback
        jac_cb = jacobian_callback

        # Construct optimisation problem
        self.opt_problem = Optimization(
            name="Cart3D-PySAGAS Shape Optimisation",
            objFun=evaluate_objective,
            sens=evaluate_gradient,
        )

        # Other
        fs_flow = freestream
        geom_filename = geometry_filename
        _A_ref = A_ref

        # Complete initialisation
        optimiser_options = optimiser_options if optimiser_options else {}
        super().__init__(optimiser, working_dir, optimiser_options)


def evaluate_objective(x: dict):
    if verbose > 0:
        print("\n\033[1mEvaluating objective function\033[0m")

    # Pre-process parameters
    _process_parameters(x)

    # Run flow solver
    loads_dict = _run_simulation()

    # Load properties
    properties_dir = glob.glob("*_properties")
    if properties_dir:
        # Load volume and mass
        volmass = pd.read_csv(
            glob.glob(os.path.join(properties_dir[0], "*volmass.csv"))[0],
            index_col=0,
        )

        # Load COG data
        cog = np.loadtxt(
            glob.glob(os.path.join(properties_dir[0], "*cog.txt"))[0], delimiter=","
        )

        # Fetch user-defined properties
        properties_file = glob.glob(os.path.join(properties_dir[0], "*properties.csv"))
        if len(properties_file) > 0:
            # File exists, load it
            properties = pd.read_csv(
                properties_file[0],
                index_col=0,
            )["0"]
        else:
            properties = None

    else:
        # No properties data found
        volmass = None
        properties = None
        cog = None

    # Evaluate objective function
    funcs = obj_cb(
        loads_dict=loads_dict,
        volmass=volmass,
        properties=properties,
        parameters=x,
        cog=cog,
    )
    failed = False

    if verbose > 0:
        print("  Done evaluating objective function.")

    return funcs, failed


def evaluate_gradient(x: dict, objective: dict):
    if verbose > 0:
        print("\n\033[1mEvaluating gradient function\033[0m")

    # Pre-process parameters
    _process_parameters(x)

    # Run flow solver for gradient information
    loads_dict, coef_sens = _run_sensitivities()

    # Calculate Jacobian
    properties_dir = glob.glob("*_properties")
    if properties_dir:
        scalar_sens_dir = os.path.join("scalar_sensitivities")
        vm = pd.read_csv(
            glob.glob(os.path.join(properties_dir[0], "*volmass.csv"))[0],
            index_col=0,
        )
        vm_sens = pd.read_csv(
            os.path.join(scalar_sens_dir, "volmass_sensitivity.csv"),
            index_col=0,
        )[x.keys()]

        # Fetch user-defined properties and sensitivities
        properties_file = glob.glob(os.path.join(properties_dir[0], "*properties.csv"))
        if len(properties_file) > 0:
            # File exists, load it
            properties = pd.read_csv(
                properties_file[0],
                index_col=0,
            )["0"]

            # Also load sensitivity file
            property_sens = pd.read_csv(
                os.path.join(scalar_sens_dir, "property_sensitivity.csv"),
                index_col=0,
            )[x.keys()]
        else:
            properties = None
            property_sens = None

    else:
        # No properties data found
        vm = None
        vm_sens = None
        properties = None
        property_sens = None

    # Call function
    jac = jac_cb(
        parameters=x,
        coef_sens=coef_sens,
        loads_dict=loads_dict,
        volmass=vm,
        volmass_sens=vm_sens,
        properties=properties,
        property_sens=property_sens,
    )

    if verbose > 0:
        print("  Done evaluating gradient function.")

    return jac


def _process_parameters(x):
    # Move into working directory
    os.chdir(working_dir)

    # Load existing parameters to compare
    already_started = _compare_parameters(x)

    if already_started and verbose > 0:
        print("  Picking up solution from file...")

    if not already_started:
        # These parameters haven't run yet, prepare the working directory
        if save_geom:
            # Move components file into evolution directory
            geom_filepath = os.path.join(working_dir, geom_filename)

            # Check if file exists
            if os.path.exists(geom_filepath):
                # Determine new name
                suffix = geom_filename.split(".")[-1]
                new_filename = f"{len(os.listdir(evo_dir)):04d}.{suffix}"

                # Copy file over
                shutil.copyfile(
                    geom_filepath,
                    os.path.join(evo_dir, new_filename),
                )

        # Delete any files from previous run
        _clean_dir(working_dir)

    # Dump design parameters to file
    with open("parameters.pkl", "wb") as f:
        pickle.dump(x, f)

    # Generate vehicle and geometry sensitivities
    if len(glob.glob("*sensitivity*")) == 0 or not already_started:
        # No sensitivity files generated yet, or this is new geometry
        if verbose > 0:
            print("  Running geometric sensitivity study.")
        parameters = _unwrap_x(x)
        ss = SensitivityStudy(vehicle_constructor=generator, verbosity=0)
        ss.dvdp(parameter_dict=parameters, perturbation=2, write_nominal_stl=True)
        ss.to_csv()

        # Merge STLs
        vehicle = ss.nominal_vehicle_instance
        stl_files = [f"{c}.stl" for c in vehicle.get_non_ghost_components().keys()]
        if len(stl_files) > 1:
            geom_file = merge_stls(
                stl_files=stl_files, name=vehicle.name, verbosity=verbose
            )
        else:
            geom_file = stl_files[0]


def _run_simulation():
    """Prepare and run the CFD simulation with the OPM solver."""
    global cells, solver

    cells = None
    solver = None

    # Load cells from geometry
    if cells is None:
        try:
            cells = PyMesh.load_from_file(
                geom_filename, verbosity=verbose, **_parser_kwargs
            )
        except:
            cells = STL.load_from_file(
                geom_filename, verbosity=verbose, **_parser_kwargs
            )

    # Run OPM solver
    # if solver is None:
    solver = OPM(cells=cells, freestream=fs_flow, verbosity=verbose)

    if verbose > 0:
        print("  Running OPM flow solver.")

    sim_results = solver.solve()

    # Construct coefficient dictionary
    CL, CD, Cm = sim_results.coefficients()
    coefficients = {"CL": CL, "CD": CD, "Cm": Cm}

    return coefficients


def _run_sensitivities():
    global cells, solver

    cells = None
    solver = None

    # Load cells from geometry
    if cells is None:
        try:
            # cells = PyMesh.load_from_file(geom_filename, verbosity=verbose, **_parser_kwargs)
            cells = PyMesh.load_from_file(
                filepath=geom_filename,
                geom_sensitivities=sens_filename,
                verbosity=verbose,
                **_parser_kwargs,
            )
        except:
            cells = STL.load_from_file(
                geom_filename, verbosity=verbose, **_parser_kwargs
            )

    # Run OPM to generate nominal aero data on cells
    flow_solver = OPM(cells=cells, freestream=fs_flow, verbosity=verbose)
    aero = flow_solver.solve(freestream=fs_flow)

    # Solve for aerodynamic sensitivities
    sens_solver = GenericSensitivityCalculator(
        cells=cells,
        sensitivity_filepath=sens_filename,
        cells_have_sens_data=True,
        verbosity=verbose,
    )
    result = sens_solver.calculate(**_sens_kwargs)
    F_sense = result.f_sens

    # Non-dimensionalise
    coef_sens = F_sense / (fs_flow.q * _A_ref)

    # # Run OPM solver for coefficients
    # if solver is None:
    #     solver = OPM(cells=cells, freestream=fs_flow, verbosity=0)

    # # TODO - how will sens combining be handled? Need to use merged STL
    # # TODO - remove hard coding below
    # if verbose > 0:
    #     print("  Running OPM sensitivity flow solver.")
    # sens_results = solver.solve_sens(sensitivity_filepath="nose_sensitivity.csv")

    # # Non-dimensionalise
    # coef_sens = sens_results.f_sens / (fs_flow.q * _A_ref)

    # Construct coefficient dictionary
    CL, CD, Cm = aero.coefficients()
    coefficients = {"CL": CL, "CD": CD, "Cm": Cm}

    return coefficients, coef_sens


def _compare_parameters(x):
    """Compares the current parameters x to the last run simulation
    parameters."""
    try:
        with open("parameters.pkl", "rb") as f:
            try:
                xp = pickle.load(f)
            except EOFError:
                return False

        # Compare to current parameters
        already_run = x == xp

    except FileNotFoundError:
        # Simulation not run yet
        already_run = False

    return already_run


def _clean_dir(directory: str, keep: list = None):
    """Deletes everything in a directory except for what is specified
    in keep."""
    global cells, solver

    all_files = os.listdir(directory)
    if keep is None:
        # Convert to empty list
        keep = []

    # Define files to remove and remove them
    rm_files = set(all_files) - set(keep)
    for f in rm_files:
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)

    # Also reset dynamic global variables
    cells = None
    solver = None


def _unwrap_x(x: dict) -> dict:
    """Unwraps an ordered dictionary."""
    unwrapped = {}
    for key, val in x.items():
        if len(val) == 1:
            unwrapped[key] = val[0]
        else:
            unwrapped[key] = val
    return unwrapped
