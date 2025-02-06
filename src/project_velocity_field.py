import ufl
import numpy         as np
import dolfinx       as dfx
import adios4dolfinx as a4d

from mpi4py            import MPI
from basix.ufl         import element
from dolfinx.fem.petsc import LinearProblem

# Velocity data
mesh_prefix = "medium"
velocity_input_filename = \
    f"../output/checkpoints/deforming-mesh-{mesh_prefix}/BDM_chp+cilia+defo_velocity"
mesh = a4d.read_mesh(filename=velocity_input_filename,
                     comm=MPI.COMM_WORLD,
                     engine="BP4",
                     ghost_mode=dfx.mesh.GhostMode.shared_facet)
timestamps = np.linspace(0, 1, 31)
cg1_el = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.topology.dim,))

discontinuous_input = True
if discontinuous_input:
    # Define the input velocity function in a DG1 space
    bdm1_el = element("BDM", mesh.basix_cell(), 1)
    BDM = dfx.fem.functionspace(mesh=mesh, element=bdm1_el)
    u_project = dfx.fem.Function(BDM)
else:
    # Define the input velocity function in a CG2 space
    cg2_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.topology.dim,))
    CG2 = dfx.fem.functionspace(mesh=mesh, element=cg2_el)
    u_project = dfx.fem.Function(CG2)
# u_project.name = "uh" # Needs to have the same name as the checkpoint function

# Define the output velocity function space: CG1
CG = dfx.fem.functionspace(mesh=mesh, element=cg1_el)
u, v = ufl.TrialFunction(CG), ufl.TestFunction(CG)

# Projection as a variational problem
a = ufl.inner(u, v) * ufl.dx
L = ufl.inner(u_project, v) * ufl.dx

# Problem structure
problem = LinearProblem(a, L, petsc_options={"ksp_type" : "preonly",
                                             "pc_type" : "lu",
                                             "pc_factor_mat_solver_type" : "mumps"})

output_filename = f"../output/checkpoints/projections/deforming-mesh-{mesh_prefix}/BDM_chp+cilia+defo_velocity"
a4d.write_mesh(output_filename, mesh)

for int_time, time in enumerate(timestamps):
    a4d.read_function(velocity_input_filename, u_project, time=time) # Read the input velocity
    u_cg = problem.solve() # Solve for the CG1 velocity
    a4d.write_function(output_filename, u_cg, time=int_time)