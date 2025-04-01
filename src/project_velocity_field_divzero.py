import ufl
import numpy         as np
import dolfinx       as dfx
import adios4dolfinx as a4d

from sys               import argv
from mpi4py            import MPI
from petsc4py          import PETSc
from basix.ufl         import element, mixed_element
from dolfinx.fem.petsc import assemble_matrix, assemble_vector

# Velocity data
mesh_prefix = "zfish"
velocity_input_filename = \
    f"../output/{mesh_prefix}-mesh/flow/checkpoints/BDM_velocity"
mesh = a4d.read_mesh(filename=velocity_input_filename,
                     comm=MPI.COMM_WORLD,
                     engine="BP4",
                     ghost_mode=dfx.mesh.GhostMode.shared_facet)
timestamps = np.linspace(0, 1, 31)
cg1_el = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.topology.dim,))
dg0_el = element("DG", mesh.basix_cell(), 0)

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

M = dfx.fem.functionspace(mesh, mixed_element([cg1_el, dg0_el]))
u, p = ufl.TrialFunctions(M)
v, q = ufl.TestFunctions(M)

# Projection as a variational problem
a = ufl.inner(u, v) * ufl.dx
a += p * ufl.div(v) * ufl.dx + q * ufl.div(u) * ufl.dx # Enforce divergence free solution
L = ufl.inner(u_project, v) * ufl.dx

# Compile forms and assemble PETSc structures
a_cpp = dfx.fem.form(a)
L_cpp = dfx.fem.form(L)
A = assemble_matrix(a_cpp, bcs=[])
A.assemble()
b = assemble_vector(L_cpp)

# Configure linear solver
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A)
solver.setType("preonly")
pc = solver.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
pc.setFactorSetUpSolverType()
pc.getFactorMatrix().setMumpsIcntl(icntl=14, ival=80) # Increase MUMPS working memory
pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1) # Support solving a singular matrix (non-empty pressure nullspace)
pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0) # Support solving a singular matrix (non-empty pressure nullspace)

if __name__=='__main__':
    steady = True if int(argv[1])==1 else False

    output_filename = velocity_input_filename + "_projection"
    a4d.write_mesh(output_filename, mesh)
    mh = dfx.fem.Function(M) # Solution function

    if steady:
        a4d.read_function(velocity_input_filename, u_project)
        solver.solve(b, mh.x.petsc_vec)
        uh = mh.sub(0).collapse()
        u_cg = dfx.fem.Function(dfx.fem.functionspace(mesh, cg1_el))
        u_cg.interpolate(uh)
        a4d.write_function(output_filename, u_cg)
    else:
        for int_time, time in enumerate(timestamps):
            a4d.read_function(velocity_input_filename, u_project, time=time) # Read the input velocity
            solver.solve(b, mh.x.petsc_vec)
            u_cg, _ = mh.split()        
            a4d.write_function(output_filename, u_cg, time=int_time)