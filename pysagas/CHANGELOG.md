## v0.15.3 (2024-10-30)

### Fix

- **OPM**: oblique_rho2_rho1 function

## v0.15.2 (2024-07-04)

### Fix

- merge conflict

## v0.15.1 (2024-06-29)

### Refactor

- migrate Vector into hypervehicle

## v0.15.0 (2024-06-29)

### Feat

- **examples**: added example of generating aerodeck and sensedeck from geometry

### Fix

- **SensitivityCalculator**: do not overwrite verbosity
- **_read_c3d_loads**: fix tag handling
- **Cart3DShapeOpt**: add extra checks for errors
- **van_dyke_sensitivity**: added supersonic mach conditional

## v0.14.0 (2024-02-27)

### Feat

- use meshio instead of pymesh when saving FlowSolver results
- **parsers.py**: added meshio parser

## v0.13.0 (2024-01-02)

### Feat

- **ShapeOpt**: added passing of cog data

### Fix

- **optimisation**: fixed stale import
- **OPMShapeOpt**: use generic sensitivity solver instead of finite diff
- **Cart3DShapeOpt**: add sim_success flag to recover from bad sim
- **van_dyke_dPdp**: remove temporary method
- **Cell**: fix face_ids type hint

### Refactor

- **opm**: convert opm module name to lowercase for PEP8
- renamed dPdp methods to sensitivity for clarity
- refactored sensitivity models into senstivity module
- **wrapper**: refactored wrapper module into sensitivity module
- **utilities.py**: integrate piston sens with mach_limited_piston function
- **utilities.py**: move test helper functions into relevant test module
- **cart3d.py**: update import to avoid pymesh dependency error

## v0.12.1 (2023-11-06)

### Refactor

- **optimisation.cartd.__init__.py**: remove imports to simplify

## v0.12.0 (2023-10-31)

### Feat

- **Parser**: STL load_from_file also appends sensitivity data
- **OPM**: added count of bad cells for diagnosing bad solve
- **OPM**: optionally provide parameters to avoid sens extraction
- **FlowState**: add Vector property
- **Cart3DShapeOpt**: provide kwargs for wrapper.calculate

### Fix

- **Wrapper**: _extract_parameters preserves parameter ordering
- **van_dyke_dPdp_ingo**: take kwargs

## v0.11.0 (2023-06-18)

### Feat

- **deck.py**: added multideck object to tidy sensitivity deck
- **SensitivityResults**: translate f_sens to aero frame for coefficients
- **Deck**: added from_csv methods
- **deck.py**: added aerodeck object
- **Wrapper**: return SensitivityResult from calculate
- **Wrapper**: add verbosity to calculate method
- **PyMesh-Parser**: optionally provide geom sens data to append to cells on load
- **GenericWrapper**: specify that cells already have geom sens data to prevent transcription
- **wrappers**: added generic wrapper to calculate sensitivities from provided cells

### Refactor

- **Sensdeck**: change type to sensdeck

## v0.10.0 (2023-06-13)

### Feat

- **C3DPrep**: improved docstrings
- **C3DPrep**: allow passing args to autoinputs
- **utilities.py**: implemented van dykes second order theory for sensitivities
- **Cart3DMooShapeOpt**: added multi-objective cart3d shape optimiser
- **combine_sense_data**: added verbosity control
- **Cart3D**: solve_sens method implemented
- **Cart3D**: improved running function
- **Cart3D**: added status and stop functionality
- **Cart3D**: added cfd wrapper for cart3d
- **FlowState**: define direction by aoa
- **OPMShapeOpt**: added verbosity
- **OPMShapeOpt**: pass parameters to objective callback
- **add_sens_data**: improved reporting
- **OPM**: added perturbation amount argument for sens
- **add_sens_data**: return point match fraction metric
- **FlowSolver**: added sweep method to generate aerodeck
- **OPM**: calculate net moment
- first implementation of OPM optimisation wrapper
- **FlowResults**: improved aero coefficient helper method
- **FlowSolver**: added aero coefficient helper
- **PyMesh**: implemented multiprocessing for parser
- **FlowSolver**: check if a flow state has already been solved before proceeding
- **OPM**: implemented sensitivity solver
- **add_sens_data**: function to append sensitivity data to cells
- **FlowSolver**: added method to convert body to wind axes
- **GasState**: added default flow direction and q property
- **FlowSolver**: added convenience inputs of aoa and mach for solver instance
- **OPM**: added finite difference sensitivities for PM and oblique
- **OPM**: use normal shock relations for theta > theta_max
- **TRI-Parser**: now includes connectivity information
- **FlowSolver**: added vtk file save method
- **OPM**: first implementation running to produce net force vector
- **Parser**: implemented stl and tri parsers
- **OPM**: added oblique theory methods

### Fix

- **van_dyke_dPdp**: skip subsonic cells
- **Cart3DMooShapeOpt**: combine sens data for each sim point
- **Cart3D**: path management
- **OPMShapeOpt**: force reloading cells and instantiating solver
- **OPMShapeOpt**: re-instantiate cells and solver on each evaluation
- **OPM-ShapeOpt**: chain rule theta through parameter
- **OPMShapeOpt**: try load with pymesh, except use STL
- **add_sens_data**: removed print line interfering with progress bar
- **OPM**: fixed sensitivity evaluation
- **OPM**: allow solving sens with cells that already have geom sens data
- **GasState**: dynamically return properties to allow updating state
- **OPM**: threshold expansion cells
- **FlowResults**: __str__ return type
- **Cell**: ambiguity of numpy array
- **geometry**: fixed namespace generation

### Refactor

- **Cart3DWrapper**: pass flowstate instead of individual properties
- **combine_sense_data**: moved to pysagas.optimisation.cartd.utilities
- **geometry**: tidied geometry module into submodule
- **FlowSolver**: started implemented prelimary flow solvers

### Perf

- **Cell**: delay calculation of geometric sensitivities until required

## v0.9.0 (2023-04-15)

### Feat

- **C3DPrep**: add option to not write config.xml file
- **Cart3DShapeOpt**: pass user-defined properties and sensitivities to callback functions
- **Cart3DShapeOpt**: convert tri files to dat
- **Cart3DShapeOpt**: option to copy components file to observe evolution
- **ShapeOpt**: check bounds on variables when added
- **ShapeOpt**: implemented constraint interface
- **ShapeOpt**: implemented hypercube bounds
- **parse_tri_file**: added progress bar and verbosity control
- **ShapeOpt**: read c3d commands from warmstart dir
- **ShapeOpt**: prepare new simulation with warmstart flag
- **utilities**: implemented Newtonian impact solver

### Fix

- **C3DPrep**: config.xml writing option
- paraview and vtk compatibility
- **Cart3DShapeOpt**: revert file copying without .dat conversion
- **ShapeOpt**: pass variable bounds
- **Cart3DShapeOpt**: working directory cleaning logic
- **ShapeOpt**: pass parameters to obj_jac_cb
- **ShapeOpt**: handle when a geom has no properties data
- **ShapeOpt**: raise exception if Cart3D commands cannot be found
- **ShapeOpt**: added verbosity to logging
- **ShapeOpt**: append cart3d stdout to log file
- **ShapeOpt**: open/close file when running main cmd with stdout/err pipe
- **ShapeOpt**: mgPrep filename for warmstarted iterations
- **ShapeOpt**: run c3d commands using subprocess.run
- **ShapeOpt**: use os.path.join on conditional
- **ShapeOpt**: rename checkpoint file when copying warmstart files
- **ShapeOpt**: distinguish preparaton warmstart flag from simulation warmstart flag
- **ShapeOpt**: handle FileNotFoundErrors in _infer_adapt method
- **ShapeOpt**: remove hardcoded run command
- **ShapeOpt**: allow restarting warmstarted iteration
- **ShapeOpt**: return line in _find_in_file method
- **ShapeOpt**: move *.tri files into simulation directory on warmstarts
- **ShapeOpt**: override warmstart flag on restarts
- **ShapeOpt**: warmstart bool ambiguity
- **ShapeOpt**: falsify warmstart flag on initial iteration
- **ShapeOpt**: pass warmstart flag to run_simulation method
- **ShapeOpt**: allow specifying max_iterations
- **newtonian_impact_solver**: return type hint

### Refactor

- **Cart3DShapeOpt**: tidied repeated code
- **Cart3DShapeOpt**: clean working directory before running any operations
- **Cart3DShapeOpt**: minimal working example complete
- **Cart3DShapeOpt**: moved methods to functions
- **Cart3DShapeOpt**: improved flow of evaluate_objective method
- **Cart3DShapeOpt**: refactored into unified interface
- tidied up cart3d optimisation methods
- **C3DPrep**: moved into utilities module for C3D optimisation
- set up inheritance structure

## v0.8.0 (2023-03-01)

### Feat

- **ShapeOpt**: handle bad combined sens data
- **ShapeOpt**: limit number of C3D restarts
- **ShapeOpt**: added general C3D handling for 'ERROR'
- **C3DPrep**: optionally specify number of rotation attempts before quitting
- **ShapeOpt**: objective and jacobian provided by user callback function

### Fix

- **ShapeOpt**: explicitly specify sensitivity files when combining
- **Cart3DWrapper**: raise exception when len(sensdata) != len(pointdata)
- **C3DPrep**: apply reverse rotations upon intersection
- **ShapeOpt**: added max_adapt arg back in after merge deletion

### Refactor

- **ShapeOpt**: attempt component intersection multiple times
- **ShapeOpt**: retrieve vol and mass from properties dir

## v0.7.0 (2023-02-05)

### Feat

- implemented moment sensitivities
- **_C3DPrep**: overwrite Config.xml with tri files prefix
- **ShapeOpt**: implemented c3d adapt cycle scheduling
- **ShapeOpt**: added cubes failure to c3d errors to catch
- **cell_dfdp**: added mechanism to calculate moment sensitivities
- **ShapeOpt**: optionally provide theoretical convergence limit to post process
- **_C3DPrep**: added control over jitter
- **ShapeOpt**: implemented max step size argument
- **ShapeOpt**: added C3D error to recovery
- **ShapeOpt**: added error handling for C3D crashes
- **ShapeOpt**: added iterative matching tol reduction
- **ShapeOpt**: catch zero norm jacobian and bail
- **Cart3D**: improved verbosity output
- **DegenerateCell**: implemented exception for degenerate cells
- option to show plot in post for c3d shapeopt
- **ShapeOpt**: implemented automated step size
- **optimisation**: implemented basic shape optimisation wrapper for cart3d

### Fix

- **cell_dfdp**: include dr/dp in moment sensitivity calculation
- **ShapeOpt**: updated output of wrapper.calculate and sensitivity notation
- **ShapeOpt**: warmstart continuity of x_older and jac_older
- **process_components**: enforce paraview to point data in spreadsheet view
- **Cart3DWrapper**: improved parameter extraction

### Refactor

- **Wrapper**: allow passing c.o.g. as arg to calculate

## v0.6.0 (2023-01-18)

### Feat

- **Wrapper**: return sensitivities as dataframe

## v0.5.0 (2022-12-20)

### Feat

- **Cart3DWrapper**: allow directly providing point and cell data frames
- added sensitivities to transcribed cells
- modularised dPdp method

### Fix

- **test_cart3d.py**: name of test function
- **isentropic_dPdp**: calculation of pressure sensitivity

### Refactor

- improved flow of cart3d wrapper
- improved wrapper interface

### Perf

- **Wrapper**: check if cells have already been transcribed

## v0.4.0 (2022-12-08)

### Feat

- **utilities.py**: modularised dPdp method

### Fix

- merge conflicts

### Refactor

- **wrappers**: added wrappers submodule heirarchy

## v0.3.1 (2022-12-08)

### Fix

- **pyproject.toml**: rename to pysagas

### Refactor

- rename package to pysagas

## v0.3.0 (2022-12-06)

### Feat

- **__init__.py**: expose key classes to top namespace
- **test_cart3d.py**: code running to completion
- assign FlowState to Cells
- **flow.py**: created GasState and FlowState classes
- **Vector**: added unit vector property
- **utilities.py**: added ParaView macro to process components.i.plt
- **geometry.py**: added class methods for Vector and Cell objects

### Fix

- **flow.py**: circular import
- fixed circular import error
- **utilities.py**: fixed dimensionality for parameters

## v0.2.0 (2022-11-30)

### Feat

- implemented use of Cells
- **Cell**: added sensitivity methods to cell
- **geometry.py**: implemented area and normal methods for Cell
- **geometry.py**: implemented operation methods for Vector
- **utilities.py**: added force sensitivity utilities
- **geometry.py**: added vec property to Vector object
- **geometry.py**: added geometry module
- added some utilities
- **flow.py**: added flow state object

### Refactor

- code tidy up
