from pysagas.flow import FlowState
from pysagas.geometry.parsers import MeshIO
from pysagas.cfd import OPM, AeroDeck, SensDeck
from hypervehicle.hangar import ParametricWedge
from hypervehicle.utilities import SensitivityStudy
from pysagas.sensitivity import GenericSensitivityCalculator


# Define vehicle generator object for wedge and the parameters of interest
generator = ParametricWedge
parameters = {"thickness": 0.1}

# Generate geometry and parameter sensitivities
ss = SensitivityStudy(vehicle_constructor=generator, verbosity=0)
ss.dvdp(
    parameter_dict=parameters,
    perturbation=2,
    write_nominal_stl=True,
)
geom_sensitivities = ss.to_csv()

# Load all cells from merged file
cells = MeshIO.load_from_file(
    filepath="Wedge-swept-0.stl",
    geom_sensitivities=geom_sensitivities,
)

# Instantiate flow solver and sensitivity solver
flow_solver = OPM(cells)
sens_solver = GenericSensitivityCalculator(
    cells=cells,
    sensitivity_filepath=geom_sensitivities,
    cells_have_sens_data=True,
)

# Extract input range
aoa_range = [-3, 0, 3]
machs = [5]

# Perform sweep
aerodeck = AeroDeck(inputs=["aoa", "mach"])
sensdeck = SensDeck(inputs=["aoa", "mach"], parameters=parameters.keys())
for aoa in aoa_range:
    for mach in machs:
        # Define freestream
        freestream = FlowState(mach=mach, pressure=101e3, temperature=288, aoa=aoa)

        # Run flow solver
        aero = flow_solver.solve(freestream=freestream)

        # Run sensitivity solver
        sens = sens_solver.calculate(flowstate=freestream)

        # Insert results from this sim point
        aerodeck.insert(result=aero, aoa=aoa, mach=mach)
        sensdeck.insert(result=sens, aoa=aoa, mach=mach)

# Save decks
aerodeck.to_csv()
sensdeck.to_csv()
