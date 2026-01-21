import os
import time
import glob
import pickle
import shutil
import subprocess
import numpy as np
import pandas as pd
import multiprocess as mp
from pysagas.flow import FlowState
from pysagas.cfd.cart3d import Cart3D
from hypervehicle.generator import Generator
from pyoptsparse import Optimizer, Optimization
from pysagas.optimisation.optimiser import ShapeOpt
from hypervehicle.utilities import SensitivityStudy
from pysagas.sensitivity.cart3d import Cart3DSensitivityCalculator
from pysagas.optimisation.cart3d.utilities import C3DPrep, combine_sense_data


np.seterr(all="ignore")


class Cart3DShapeOpt(ShapeOpt):
    """A wrapper to perform shape optimisation with Cart3D."""

    C3D_errors = [
        "==> ADAPT failed",
        "Check cart3d.out in AD_A_J for more clues",
        "==> adjointErrorEst_quad failed again, status = 1",
        "ERROR: CUBES failed",
        "ERROR: ADAPT failed with status = 1",
        "ERROR",
    ]

    def __init__(
        self,
        freestream: FlowState,
        optimiser: Optimizer,
        vehicle_generator: Generator,
        objective_callback: callable,
        jacobian_callback: callable,
        A_ref: float = 1.0,
        l_ref: float = 1.0,
        optimiser_options: dict = None,
        sensitivity_filename: str = "all_components_sensitivity.csv",
        working_dir_name: str = "working_dir",
        sim_directory_name: str = "simulation",
        basefiles_dir_name: str = "basefiles",
        c3d_log_name: str = "C3D_log",
        c3d_info_file: str = None,
        matching_tolerance: float = 1e-5,
        save_evolution: bool = True,
        write_config_xml: bool = True,
        sensitivity_kwargs: dict = None,
    ) -> None:
        """Initialise Cart3D Shape Optimiser.

        Parameters
        ----------
        a_inf : float
            The freestream speed of sound (m/s).

        rho_inf : float
            The freestream density (kg/m^3).

        V_inf : float
            The freestream velocity (m/s).

        A_ref : float
            The aerodynamic reference area (m^2).

        optimiser : Optimizer
            The pyoptsparse Optimizer object of choice.

        optimiser_options : dict, optional
            The options to pass to the optimiser. See the PyOptSparse
            documentation for solver-specific options. The default is None.

        vehicle_generator : Generator
            The vehicle generator object.

        objective_callback : callable
            The callback function to compute and return the objective function
            value and constraint violation values.

        jacobian_callback : callable
            The callback function to compute and return the Jacobian of the
            objective function value and constraint violation values.

        sensitivity_filename : str, optional
            The filename of the combined components sensitivities. The default
            is 'all_components_sensitivity.csv'.

        working_dir_name : str, optional
            The name of the working directory. The default is 'working_dir'.

        sim_dir_name : str, optional
            The name of the simulation directory. The default is 'simulation'.

        basefiles_dir_name : str, optional
            The name of the base files directory. The default is 'basefiles'.

        c3d_log_name : str, optional
            The name to use for the Cart3D logfile. The default is C3D_log.

        save_evolution : bool, optional
            A boolean flag to save geometry files of the evolving geometry.
            If True, the files will be saved to the evolution directory. The
            default is True.

        write_config_xml : bool, optional
            A boolean flag to write the Cart3D Config.xml file when running
            comp2tri. If the geometry may be perturbed to a state where one
            component is no longer part of the wetted surface, writing the
            comp2tri file can cause set-up issues, and so it can be beneficial
            to turn it off. The default is True.
        """
        # Define global variable so that functions can access them
        global c3d_logname, _matching_tolerance, _max_matching_tol, _matching_target
        global sens_filename, basefiles_dir, _c3dprepper, sim_dir_name
        global _A_ref, _l_ref
        global _freestream
        global home_dir, working_dir, f_sense_filename
        global obj_cb, jac_cb
        global generator
        global save_comptri, evo_dir
        global moo
        moo = False

        _freestream = freestream

        save_comptri = save_evolution

        generator = vehicle_generator

        # Construct paths
        home_dir = os.getcwd()
        basefiles_dir = os.path.join(home_dir, basefiles_dir_name)
        working_dir = os.path.join(home_dir, working_dir_name)
        sim_dir_name = sim_directory_name
        sens_filename = sensitivity_filename
        f_sense_filename = "F_sensitivities.csv"
        c3d_logname = c3d_log_name

        if save_comptri:
            # Create evolution history directory
            evo_dir = os.path.join(home_dir, "evolution")
            if not os.path.exists(evo_dir):
                os.mkdir(evo_dir)

        # Save callback functions
        obj_cb = objective_callback
        jac_cb = jacobian_callback

        # Save reference area
        _A_ref = A_ref
        _l_ref = l_ref

        # Create instance of Cart3D prepper
        _c3dprepper = C3DPrep(
            logfile=c3d_logname, info_file=c3d_info_file, write_config=write_config_xml
        )

        # Save sensitivity kwargs
        global _sens_kwargs
        _sens_kwargs = sensitivity_kwargs if sensitivity_kwargs else {}

        # Other settings
        _matching_tolerance = matching_tolerance
        _max_matching_tol = 0.1
        _matching_target = 0.9

        # Construct optimisation problem
        self.opt_problem = Optimization(
            name="Cart3D-PySAGAS Shape Optimisation",
            objFun=evaluate_objective,
            sens=evaluate_gradient,
        )

        # Complete super initialisation
        super().__init__(
            optimiser=optimiser,
            working_dir=working_dir,
            optimiser_options=optimiser_options,
        )


class Cart3DMooShapeOpt(Cart3DShapeOpt):
    def __init__(
        self,
        a_inf: float,
        rho_inf: float,
        V_inf: float,
        A_ref: float,
        optimiser: Optimizer,
        vehicle_generator: Generator,
        objective_callback: callable,
        jacobian_callback: callable,
        mach_aoa_points: list[tuple[float, float]],
        optimiser_options: dict = None,
        sensitivity_filename: str = "all_components_sensitivity.csv",
        working_dir_name: str = "working_dir",
        sim_directory_name: str = "simulation",
        basefiles_dir_name: str = "basefiles",
        c3d_log_name: str = "C3D_log",
        c3d_info_file: str = None,
        matching_tolerance: float = 0.00001,
        save_evolution: bool = True,
        write_config_xml: bool = True,
    ) -> None:
        super().__init__(
            a_inf,
            rho_inf,
            V_inf,
            A_ref,
            optimiser,
            vehicle_generator,
            objective_callback,
            jacobian_callback,
            optimiser_options,
            sensitivity_filename,
            working_dir_name,
            sim_directory_name,
            basefiles_dir_name,
            c3d_log_name,
            c3d_info_file,
            matching_tolerance,
            save_evolution,
            write_config_xml,
        )
        global sim_points
        sim_points = mach_aoa_points

        global moo
        moo = True

        # Overload
        self.opt_problem = Optimization(
            name="Cart3D-PySAGAS Shape Optimisation",
            objFun=evaluate_moo_objective,
            sens=evaluate_moo_gradient,
        )

        self.opt_problem: Optimization
        self.optimiser = optimiser(options=optimiser_options)


def evaluate_moo_objective(x: dict) -> dict:
    print("Evaluating objective function.")

    # Pre-process parameters
    _process_parameters(x)

    print("Deploying simulation tasks to pool.")
    pool = mp.Pool()
    results = []
    for result in pool.starmap(aero_wrapper, sim_points):
        results.append(result)
    print("  Done.")

    # Load properties
    properties_dir = glob.glob("*_properties")
    if properties_dir:
        volmass = pd.read_csv(
            glob.glob(os.path.join(properties_dir[0], "*volmass.csv"))[0],
            index_col=0,
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

    # Objective function callback
    funcs = obj_cb(
        parameters=_unwrap_x(x), results=results, volmass=volmass, properties=properties
    )

    return funcs, False


def aero_wrapper(mach, aoa):
    freestream = FlowState(mach=mach, pressure=101e3, temperature=288, aoa=aoa)
    c3d = Cart3D(stl_files=stl_files, freestream=freestream, verbosity=0)
    flow_result = c3d.solve()
    return flow_result


def sens_wrapper(mach, aoa):
    freestream = FlowState(mach=mach, pressure=101e3, temperature=288, aoa=aoa)
    c3d = Cart3D(stl_files=stl_files, freestream=freestream, verbosity=0)

    # Append sensitivity data to tri file
    sim_dir = os.path.join(working_dir, f"M{mach}A{float(aoa)}")
    components_filepath = os.path.join(sim_dir, "Components.i.tri")
    combine_sense_data(
        components_filepath=components_filepath,
        sensitivity_files=sensitivity_files,
        match_target=_matching_target,
        tol_0=_matching_tolerance,
        max_tol=_max_matching_tol,
        outdir=sim_dir,
        verbosity=0,
    )

    # Run the sensitivity solver
    sens_file = os.path.join(sim_dir, "all_components_sensitivity.csv")
    sens_result = c3d.solve_sens(sensitivity_filepath=sens_file)
    flow_result = c3d.flow_result

    return flow_result, sens_result


def evaluate_moo_gradient(x: dict, objective: dict) -> dict:
    """Evaluates the gradient function at the parameter set `x`."""
    print("Evaluating gradient function.")

    # Pre-process parameters
    _process_parameters(x)

    print("Deploying sensitivity tasks to pool.")
    pool = mp.Pool()
    results = []
    for result in pool.starmap(sens_wrapper, sim_points):
        results.append(result)
    print("  Done.")

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
        parameters=_unwrap_x(x),
        results=results,
        volmass=vm,
        volmass_sens=vm_sens,
        properties=properties,
        property_sens=property_sens,
    )

    return jac


def evaluate_objective(x: dict) -> dict:
    """Evaluates the objective function at the parameter set `x`."""
    # Pre-process parameters
    _process_parameters(x)

    # Run Cart3D simulation
    sim_success, loads_dict, _ = _run_simulation()

    # Load properties
    properties_dir = glob.glob("*_properties")
    if properties_dir:
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

    # Call objective function
    failed = False
    if sim_success:
        # Evaluate objective function
        funcs: dict[str, dict[str, list]] = obj_cb(
            loads_dict=loads_dict,
            volmass=volmass,
            properties=properties,
            cog=cog,
        )

        # Check for nans
        for v1 in funcs.values():
            if any(np.isnan(v1)):
                failed = True

    else:
        # Simulation failed
        # funcs = {}
        loads_dict = {"C_L-entire": 0, "C_D-entire": 1}
        # funcs = obj_cb(
        #     loads_dict=loads_dict,
        #     volmass=volmass,
        #     properties=properties,
        #     cog=cog,
        # )
        failed = True

    if failed:
        funcs = {}

    return funcs, failed


def evaluate_gradient(x: dict, objective: dict) -> dict:
    """Evaluates the gradient function at the parameter set `x`."""
    # Pre-process parameters
    _process_parameters(x)

    # Run Cart3D simulation
    _, loads_dict, components_plt_filepath = _run_simulation()

    # Initialise filepaths
    components_filepath = components_plt_filepath
    sensitivity_filepath = os.path.join(working_dir, sens_filename)

    # Create PySAGAS wrapper and run
    try:
        wrapper = Cart3DSensitivityCalculator(
            freestream=_freestream,
            sensitivity_filepath=sensitivity_filepath,
            components_filepath=components_filepath,
            verbosity=0,
        )

    except ValueError:
        # The sensitivity data does not match the point data, regenerate it
        tri_components_filepath = os.path.join(
            working_dir, sim_dir_name, "Components.i.tri"
        )
        sensitivity_files = glob.glob(os.path.join("*sensitivity*"))
        combine_sense_data(
            tri_components_filepath,
            sensitivity_files=sensitivity_files,
            match_target=_matching_target,
            tol_0=_matching_tolerance,
            max_tol=_max_matching_tol,
            outdir=working_dir,
            verbosity=0,
        )

        # Re-instantiate the wrapper
        wrapper = Cart3DSensitivityCalculator(
            freestream=_freestream,
            sensitivity_filepath=sensitivity_filepath,
            components_filepath=components_filepath,
            verbosity=0,
        )

    result = wrapper.calculate(**_sens_kwargs)
    F_sense = result.f_sens
    M_sens = result.m_sens

    # Non-dimensionalise
    coef_sens = F_sense / (_freestream.q * _A_ref)
    moment_coef_sens = M_sens / (_freestream.q * _A_ref * _l_ref)

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

        # Load COG data
        cog = np.loadtxt(
            glob.glob(os.path.join(properties_dir[0], "*cog.txt"))[0], delimiter=","
        )
        cog_sens_files = glob.glob(
            os.path.join(scalar_sens_dir, "*cog_sensitivity.txt")
        )
        cog_sens = {}
        for file in cog_sens_files:
            param = file.split("_cog_sensitivity.txt")[0].split("/")[-1]
            cog_sens[param] = np.loadtxt(file, delimiter=",")

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
        cog = None
        cog_sens = None

    # Call function
    failed = False
    try:
        jac: dict[str, dict[str, list]] = jac_cb(
            parameters=x,
            coef_sens=coef_sens,
            moment_coef_sens=moment_coef_sens,
            loads_dict=loads_dict,
            volmass=vm,
            volmass_sens=vm_sens,
            properties=properties,
            property_sens=property_sens,
            cog=cog,
            cog_sens=cog_sens,
        )

        # Check for nans
        for v1 in jac.values():
            for v2 in v1.values():
                if any(np.isnan(v2)):
                    failed = True

    except Exception as e:
        print(f"Exception calling jacobian callback: {e}")
        failed = True

    if failed:
        jac = {}

    return jac, failed


def _process_parameters(x):
    # Move into working directory
    os.chdir(working_dir)

    # Load existing parameters to compare
    already_started = _compare_parameters(x)

    if not already_started:
        # These parameters haven't run yet, prepare the working directory
        if save_comptri:
            # Move components file into evolution directory
            comp_filepath = os.path.join(working_dir, sim_dir_name, "Components.i.tri")

            # Check if file exists
            if os.path.exists(comp_filepath):
                # Determine new name
                new_filename = f"{len(os.listdir(evo_dir)):04d}.tri"

                # Copy file over
                shutil.copyfile(
                    comp_filepath,
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
        print("Running sensitivity study.")
        parameters = _unwrap_x(x)
        ss = SensitivityStudy(vehicle_constructor=generator, verbosity=0)
        ss.dvdp(parameter_dict=parameters, perturbation=2, write_nominal_stl=True)
        ss.to_csv()
        print("  Done.")

    if moo:
        # Multi-objective optimisation
        global stl_files, sensitivity_files
        stl_files = glob.glob("*.stl")
        sensitivity_files = glob.glob("*sensitivity*")

        # Also copy aero.csh and input.cntl into working directory
        for filename in ["input.cntl", "aero.csh"]:
            shutil.copyfile(
                os.path.join(home_dir, filename),
                os.path.join(working_dir, filename),
            )
            shutil.copymode(
                os.path.join(home_dir, filename),
                os.path.join(working_dir, filename),
            )


def _run_simulation(no_attempts: int = 3):
    """Prepare and run the CFD simulation with Cart3D. The simulation will be
    run in the 'simulation' subdirectory of the iteration directory.
    """
    # Make simulation directory
    sim_dir = os.path.join(working_dir, sim_dir_name)
    run_intersect = False
    components_filepath = os.path.join(sim_dir, "Components.i.tri")
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)
        run_intersect = True
    else:
        # Check for intersected file
        run_intersect = not os.path.exists(components_filepath)
        intersected = True

    # Attempt component intersection
    sim_success = False
    for attempt in range(no_attempts):
        # Run intersect
        if run_intersect:
            _c3dprepper._log(f"SHAPEOPT INTERSECT ATTEMPT {attempt+1}")
            intersected = _c3dprepper.intersect_stls()

        # Check for intersection
        if intersected:
            # Prepare rest of simulation directory
            if not os.path.exists(os.path.join(sim_dir, "input.cntl")):
                # Move files to simulation directory (including Components.i.tri)
                _c3dprepper.run_autoinputs()
                os.system(
                    f"mv *.tri Config.xml input.c3d preSpec.c3d.cntl {sim_dir} >> {c3d_logname} 2>&1"
                )

                # Copy sim files and permissions
                for filename in ["input.cntl", "aero.csh"]:
                    shutil.copyfile(
                        os.path.join(basefiles_dir, filename),
                        os.path.join(sim_dir, filename),
                    )
                    shutil.copymode(
                        os.path.join(basefiles_dir, filename),
                        os.path.join(sim_dir, filename),
                    )

            # Create all_components_sensitivity.csv
            if not os.path.exists(sens_filename):
                combine_sense_data(
                    components_filepath,
                    sensitivity_files=glob.glob("*sensitivity*"),
                    match_target=_matching_target,
                    tol_0=_matching_tolerance,
                    max_tol=_max_matching_tol,
                    verbosity=0,
                )

            # Run Cart3D and await result
            os.chdir(sim_dir)
            target_adapt = _infer_adapt(sim_dir)
            c3d_donefile = os.path.join(sim_dir, target_adapt, "FLOW", "DONE")
            _c3dprepper._log(f"Waiting for DONE file: {c3d_donefile}")
            run_cmd = "./aero.csh restart"
            _restarts = 0
            if not os.path.exists(c3d_donefile):
                with open(c3d_logname, "a") as f:
                    subprocess.run(
                        run_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT
                    )
                while not os.path.exists(c3d_donefile):
                    # Wait...
                    time.sleep(5)

                    # Check for C3D failure
                    running, e = _c3d_running()

                    if not running:
                        # C3D failed, try restart it
                        if _restarts > 3:
                            return False

                        f = open(c3d_logname, "a")
                        subprocess.run(
                            run_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT
                        )
                        f.close()
                        _restarts += 1

            sim_success = True
            break

    if sim_success:
        # Sim finished successfully, read loads file
        loads_dict = _read_c3d_loads(
            os.path.join(working_dir, sim_dir_name, "BEST/FLOW/loadsCC.dat")
        )
        components_plt_filepath = os.path.join(
            working_dir, sim_dir_name, "BEST/FLOW/Components.i.plt"
        )

    else:
        loads_dict = None
        components_plt_filepath = None

    # Change back to working directory
    os.chdir(working_dir)

    return sim_success, loads_dict, components_plt_filepath


def _read_c3d_loads(
    loadsCC_filepath: str,
    b_frame: bool = True,
    v_frame: bool = True,
    moments: bool = True,
) -> dict:
    load_dict = {}
    with open(loadsCC_filepath, "r") as file:
        for line in file:
            if line[0] == "#":
                # Commented line, skip
                continue

            # Remove linebreaks and multiple spaces
            line = " ".join(line.split())
            words = line.split(":")

            if len(words) == 0 or len(words) == 1:  # skip if empty line
                continue

            text = words[0]
            number = float(words[1])
            word = text.split(" ")
            tag = word[0]
            coeff = word[-1]
            coeff = coeff[1:4]
            if b_frame is True:
                if coeff in ["C_A", "C_Y", "C_N"]:  # get force in b_frame
                    load_dict["{0}-{1}".format(coeff, tag)] = number
            if v_frame is True:
                if coeff in ["C_D", "C_S", "C_L"]:  # get force in v_frame
                    load_dict["{0}-{1}".format(coeff, tag)] = number
            if moments is True:
                if coeff in ["C_l", "C_m", "C_n", "C_M"]:  # get moment coeff
                    load_dict["{0}-{1}".format(coeff, tag)] = number

    return load_dict


def _c3d_running() -> bool:
    """Watches the Cart3D log file to check for errors and return False
    if Cart3D has stopped running."""
    with open(c3d_logname) as f:
        # Get last line in log file
        for line in f:
            pass

        # Check if if it is in the known errors
        for e in Cart3DShapeOpt.C3D_errors:
            if e in line:
                return False, e

    # No errors
    return True, None


def _infer_adapt(sim_dir) -> str:
    with open(f"{sim_dir}/aero.csh", "r") as f:
        lines = f.readlines()

        for line in lines:
            if line.find("set n_adapt_cycles") != -1:
                return f"adapt{int(line.split('=')[-1]):02d}"


def _compare_parameters(x):
    """Compares the current parameters x to the last run simulation
    parameters."""
    try:
        with open("parameters.pkl", "rb") as f:
            xp = pickle.load(f)

        # Compare to current parameters
        already_run = x == xp

    except FileNotFoundError:
        # Simulation not run yet
        already_run = False

    return already_run


def _clean_dir(directory: str, keep: list = None):
    """Deletes everything in a directory except for what is specified
    in keep."""
    all_files = os.listdir(directory)
    if keep is None:
        # Convert to empty list
        keep = []
    rm_files = set(all_files) - set(keep)
    for f in rm_files:
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)


def _unwrap_x(x: dict) -> dict:
    """Unwraps an ordered dictionary."""
    unwrapped = {}
    for key, val in x.items():
        if len(val) == 1:
            unwrapped[key] = val[0]
        else:
            unwrapped[key] = val
    return unwrapped
