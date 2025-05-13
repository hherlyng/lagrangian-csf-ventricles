import ufl
import numpy         as np
import dolfinx       as dfx
import adios4dolfinx as a4d

from sys               import argv
from mpi4py            import MPI
from basix.ufl         import element
from dolfinx.fem.petsc import LinearProblem

# Velocity data
mesh_prefix = "coarse"
solver_type = "navier-stokes"
velocity_input_filename = \
    f"../output/{mesh_prefix}-mesh/flow/{solver_type}/checkpoints/BDM_deforming_velocity"
mesh = a4d.read_mesh(filename=velocity_input_filename,
                     comm=MPI.COMM_WORLD,
                     engine="BP4",
                     ghost_mode=dfx.mesh.GhostMode.shared_facet)
T = 1
dt = 0.02
N = int(T / dt)
timestamps = np.linspace(0, T, N+1)
cg1_el = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.topology.dim,))

element_type = "BDM"
if element_type=="BDM":
    # Define the input velocity function in a BDM1 space
    bdm1_el = element("BDM", mesh.basix_cell(), 1)
    BDM = dfx.fem.functionspace(mesh=mesh, element=bdm1_el)
    u_project = dfx.fem.Function(BDM)
elif element_type=="DG":
    # Define the input velocity function in a DG1 space
    dg1_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    DG = dfx.fem.functionspace(mesh=mesh, element=dg1_el)
    u_project = dfx.fem.Function(DG)
else:
    # Define the input velocity function in a CG2 space
    cg2_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.topology.dim,))
    CG2 = dfx.fem.functionspace(mesh=mesh, element=cg2_el)
    u_project = dfx.fem.Function(CG2)
u_project.name = "relative_velocity" # Needs to have the same name as the checkpoint function

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

if __name__=='__main__':
    steady = True if int(argv[1])==1 else False
    print("Number of timestamps: ", N)

    output_filename = velocity_input_filename + "_projection"
    a4d.write_mesh(output_filename, mesh)

    if steady:
        a4d.read_function(velocity_input_filename, u_project)
        u_cg = problem.solve()
        a4d.write_function(output_filename, u_cg)
    else:
        for int_time, time in enumerate(timestamps):
            a4d.read_function(velocity_input_filename, u_project, time=time) # Read the input velocity
            u_cg = problem.solve() # Solve for the CG1 velocity
            a4d.write_function(output_filename, u_cg, time=int_time)