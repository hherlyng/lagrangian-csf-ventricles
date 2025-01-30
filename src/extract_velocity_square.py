import ufl
import pyvista as pv
import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from ufl    import inner, dot, div, grad
from mpi4py import MPI
from petsc4py  import PETSc
from basix.ufl import element
from utilities.mesh import create_square_mesh_with_tags
from utilities.fem import stabilization, eps, tangent
import dolfinx.fem.petsc as dfx_petsc

comm = MPI.COMM_WORLD
LEFT = 1; RIGHT = 2; BOT = 3; TOP = 4

def assemble_system(lhs_form, rhs_form, bcs):
    A = dfx_petsc.assemble_matrix_nest(lhs_form, bcs=bcs)
    A.assemble()

    b = dfx_petsc.assemble_vector_nest(rhs_form)
    dfx_petsc.apply_lifting_nest(b, lhs_form, bcs=bcs)
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    spaces = dfx.fem.extract_function_spaces(rhs_form)
    bcs0 = dfx.fem.bcs_by_block(spaces, bcs)
    dfx_petsc.set_bc_nest(b, bcs0)
    return A, b
def create_block_solver_direct(A):

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()

    return ksp
def solve_stokes():

    A, b = assemble_system(a, L, bcs_stokes)
    
    ksp = create_block_solver_direct(A)

    u, p = dfx.fem.Function(V), dfx.fem.Function(Q)
    w = PETSc.Vec().createNest([u.x.petsc_vec, p.x.petsc_vec])

    ksp.solve(b, w)
    assert ksp.getConvergedReason() > 0

    u.x.scatter_forward()
    p.x.scatter_forward()

    return u, p

mesh, ft = create_square_mesh_with_tags(N_cells=16)


# tau = ufl.as_vector((1.0, 0.0))

ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
dx = ufl.Measure('dx', mesh)
n = ufl.FacetNormal(mesh)

bdm_el = element("BDM", mesh.basix_cell(), 1)
dg_el  = element("DG", mesh.basix_cell(), 0)
V = dfx.fem.functionspace(mesh, bdm_el)
Q = dfx.fem.functionspace(mesh, dg_el)
tau = dfx.fem.Function(V)
# tau.interpolate(lambda x: np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1]))))

# Read stress directions from file
cpoint_filename = "../output/checkpoints/square/velocity/"
a4d.read_function(filename=cpoint_filename, u=tau)
tau.x.array[:] = tau.x.array.copy()

# Trial and test functions
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)                

mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(1.0))
penalty = dfx.fem.Constant(mesh, dfx.default_scalar_type(10.0))

# Stokes problem in reference domain accounting for the deformation
a00 = (2*mu*inner(eps(u), eps(v))*dx # Viscous dissipation
      + stabilization(u, v, mu, penalty) # BDM stabilization
      - mu*inner(dot(grad(u).T, n), v)*ds(RIGHT) # Parallel flow at inlet/outlet
      )
a01 = inner(p, div(v))*dx
a10 = inner(q, div(u))*dx
a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*dx

L0 = inner(dfx.fem.Function(V), v)*dx + dot(dfx.fem.Constant(mesh, 1.0)*n, v)*ds(RIGHT) + inner(tangent(tau, n), tangent(v, n))*ds(TOP)
L1 = inner(dfx.fem.Function(Q), q)*dx

a = dfx.fem.form([[a00, a01], [a10, a11]])
L = dfx.fem.form([L0, L1])

u_bc = dfx.fem.Function(V)
u_bc_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim-1, np.concatenate(([ft.find(tag) for tag in [LEFT, TOP, BOT]])))
bcs_stokes = [dfx.fem.dirichletbc(u_bc, u_bc_dofs)]

uh, ph = solve_stokes()

# # Normalize velocity vectors to unit length and write to file
# uh_reshaped = uh.x.array.copy().reshape((int(uh.x.array.__len__()/2), mesh.geometry.dim)) # Reshape into vector
# uh_norm = np.sqrt(uh_reshaped[:, 0]**2 + uh_reshaped[:, 1]**2) # Norm of velocity vector
# uh_norm_vec = dfx.fem.Function(V)
# # Avoid division by zero
# where_norm_is_zero = np.where(uh_norm < 1e-10)[0]
# uh_norm[where_norm_is_zero] = 1.0
# uh_reshaped[:, 0] /= uh_norm
# uh_reshaped[:, 1] /= uh_norm
# uh_norm_vec.x.array[:] = uh_reshaped.ravel()
cpoint_filename = "../output/checkpoints/square/velocity/"
# a4d.write_function_on_input_mesh(filename=cpoint_filename, u=uh_norm_vec)
a4d.write_function_on_input_mesh(filename=cpoint_filename, u=uh)

# Visualize
dg_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
DG_vec = dfx.fem.functionspace(mesh, dg_vec_el)
u_dg = dfx.fem.Function(DG_vec)
u_dg.interpolate(uh)

cells, types, x = dfx.plot.vtk_mesh(DG_vec)
grid = pv.UnstructuredGrid(cells, types, x)
reshaped = u_dg.x.array.copy().reshape((int(u_dg.x.array.__len__()/2), mesh.geometry.dim))
vectors = np.zeros((reshaped.shape[0], mesh.geometry.dim+1))
vectors[:, :2] = reshaped
grid["vectors"] = vectors
glyphs = grid.glyph(orient="vectors", factor=1.2)
pl = pv.Plotter()
pl.add_mesh(grid, style='wireframe', color='k')
pl.add_mesh(glyphs)
pl.view_xy()
pl.show()