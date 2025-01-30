import ufl
import pyvista as pv
import numpy   as np
import dolfinx as dfx
import matplotlib.pyplot as plt

from ufl    import inner, dot, div, grad, sym
from mpi4py import MPI
from petsc4py  import PETSc
from basix.ufl import element
import dolfinx.fem.petsc as dfx_petsc

comm = MPI.COMM_WORLD
LEFT = 1; RIGHT = 2; BOT = 3; TOP = 4

def create_square_mesh_with_tags(N_cells: int) -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags)):
        mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, N_cells, N_cells,
                                           cell_type=dfx.mesh.CellType.triangle,
                                           ghost_mode=dfx.mesh.GhostMode.shared_facet)

        def left(x): return np.isclose(x[0], 0.0)
        def right(x): return np.isclose(x[0], 1.0)
        def bottom(x): return np.isclose(x[1], 0.0)
        def top(x): return np.isclose(x[1], 1.0)

        # Facet tags
        bc_facet_indices, bc_facet_markers = [], []
        fdim = mesh.topology.dim - 1

        inlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, left)
        bc_facet_indices.append(inlet_BC_facets)
        bc_facet_markers.append(np.full_like(inlet_BC_facets, LEFT))

        outlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, right)
        bc_facet_indices.append(outlet_BC_facets)
        bc_facet_markers.append(np.full_like(outlet_BC_facets, RIGHT))

        bottom_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, bottom)
        bc_facet_indices.append(bottom_BC_facets)
        bc_facet_markers.append(np.full_like(bottom_BC_facets, BOT))

        top_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, top)
        bc_facet_indices.append(top_BC_facets)
        bc_facet_markers.append(np.full_like(top_BC_facets, TOP))

        bc_facet_indices = np.hstack(bc_facet_indices).astype(np.int32)
        bc_facet_markers = np.hstack(bc_facet_markers).astype(np.int32)

        sorted_facets = np.argsort(bc_facet_indices)

        facet_tags = dfx.mesh.meshtags(mesh, fdim, bc_facet_indices[sorted_facets], bc_facet_markers[sorted_facets])

        return mesh, facet_tags
def stabilization(u: ufl.TrialFunction, v: ufl.TestFunction, consistent: bool=True):
    """ Displacement/Flux Stabilization term from Krauss et al paper. 

    Parameters
    ----------
    u : ufl.TrialFunction
        The finite element trial function.
    
    v : ufl.TestFunction
        The finite element test function.
    
    consistent : bool
        Add symmetric gradient terms to the form if True.

    Returns
    -------
    ufl.Coefficient
        Stabilization term for the bilinear form.
    """

    n, hA = ufl.FacetNormal(mesh), ufl.avg(ufl.CellDiameter(mesh)) # Facet normal vector and average cell diameter
    dS = ufl.Measure('dS', domain=mesh) # Interior facet integral measure

    if consistent: # Add symmetrization terms
        return (-inner(Avg(2*mu*Eps(u), n), Jump(Tangent(v, n)))*dS
                -inner(Avg(2*mu*Eps(v), n), Jump(Tangent(u, n)))*dS
                + 2*mu*(penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS)

    # For preconditioning
    return 2*mu*(penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS
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

tau = ufl.as_vector((1.0, 0.0))

ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
dx = ufl.Measure('dx', mesh)

# Operators
# NOTE: these are the jump operators from Krauss, Zikatonov paper.
# Jump is just a difference and it preserves the rank 
Jump = lambda arg: arg('+') - arg('-')

# Average uses dot with normal and AGAIN MINUS; it reduces the rank
Avg = lambda arg, n: .5*(dot(arg('+'), n('+')) - dot(arg('-'), n('-')))

# Action of (1 - n x n) on a vector yields the tangential component
Tangent = lambda v, n: v - n*dot(v, n)

n = ufl.FacetNormal(mesh)

bdm_el = element("BDM", mesh.basix_cell(), 1)
dg_el  = element("DG", mesh.basix_cell(), 0)
V = dfx.fem.functionspace(mesh, bdm_el)
Q = dfx.fem.functionspace(mesh, dg_el)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)                

# Symmetric gradient
Eps = lambda arg: sym(grad(arg))

mu = 1
penalty = 10

# Stokes problem in reference domain accounting for the deformation
a00 = (2*mu*inner(Eps(u), Eps(v))*dx # Viscous dissipation
      + stabilization(u, v) # BDM stabilization
      - mu*inner(dot(grad(u).T, n), v)*ds(RIGHT) # Parallel flow at inlet/outlet
      )
a01 = inner(p, div(v))*dx
a10 = inner(q, div(u))*dx
a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*dx

L0 = inner(dfx.fem.Function(V), v)*dx + dot(dfx.fem.Constant(mesh, 1.0)*n, v)*ds(RIGHT) + inner(Tangent(tau, n), Tangent(v, n))*ds(TOP)
L1 = inner(dfx.fem.Function(Q), q)*dx

a = dfx.fem.form([[a00, a01], [a10, a11]])
L = dfx.fem.form([L0, L1])

u_bc = dfx.fem.Function(V)
u_bc_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim-1, np.concatenate(([ft.find(tag) for tag in [LEFT, TOP, BOT]])))
bcs_stokes = [dfx.fem.dirichletbc(u_bc, u_bc_dofs)]

uh, ph = solve_stokes()

# Visualize
dg_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
DG_vec = dfx.fem.functionspace(mesh, dg_vec_el)
u_dg = dfx.fem.Function(DG_vec)
u_dg.interpolate(uh)

cg_vec_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
CG_vec = dfx.fem.functionspace(mesh, cg_vec_el)
u_cg = dfx.fem.Function(CG_vec)
u_cg.interpolate(uh)

cells, types, x = dfx.plot.vtk_mesh(CG_vec)
grid = pv.UnstructuredGrid(cells, types, x)
reshaped = u_cg.x.array.copy().reshape((int(u_cg.x.array.__len__()/2), mesh.geometry.dim))
vectors = np.zeros((reshaped.shape[0], mesh.geometry.dim+1))
vectors[:, :2] = reshaped
grid["vectors"] = vectors
glyphs = grid.glyph(orient="vectors", factor=1.2)
pl = pv.Plotter()
pl.add_mesh(grid, style='wireframe', color='k')
pl.add_mesh(glyphs)
pl.view_xy()
pl.show()
