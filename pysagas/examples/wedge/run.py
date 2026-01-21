from pysagas.cfd import OPM
from pysagas.flow import FlowState
from pysagas.geometry.parsers import MeshIO


# Load STL file
cells = MeshIO.load_from_file("wedge.stl")
A_ref = 1

# Instantiate solver
freestream = FlowState(mach=8, pressure=101e3, temperature=288)
solver = OPM(cells, freestream)

# Run solver
result = solver.solve(aoa=5)

# Can also evaluate at other flow conditions, with:
# result = solver.solve(aoa=5, Mach=6) # Modified AoA/Mach
# result = solver.solve(freestream=new_freestream) # Updated freestream

# Save results to VTK file for visualisation of solution
solver.save("wedge")

# Get aero coefficients
CL, CD, Cm = solver.flow_result.coefficients()
