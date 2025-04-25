import ufl

import numpy   as np
import dolfinx as dfx

from ufl       import inner, dot, grad, det, inv, avg, jump, nabla_grad
from scifem    import assemble_scalar
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from utilities.fem import create_normal_contribution_bc, calculate_mean
from dolfinx.fem.petsc import LinearProblem

print = PETSc.Sys.Print

# Solve stationary Stokes in moving domain by ALE method. Wall motion is 
# prescribed in time.
comm = MPI.COMM_WORLD
nx = ny = 32
mesh = dfx.mesh.create_unit_square(comm, nx, ny)

def left_boundary(x): return np.isclose(x[0], 0.0)
def right_boundary(x): return np.isclose(x[0], 1.0)
def bot_boundary(x): return np.isclose(x[1], 0.0)
def top_boundary(x): return np.isclose(x[1], 1.0)

# Generate mesh entities
facet_dim = mesh.topology.dim-1
mesh.topology.create_entities(facet_dim) # Create facets
mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity
num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts # Total number of facets

DEFAULT=1; LEFT=2; RIGHT=3; BOT=4; TOP=5
facet_marker = np.full(num_facets, DEFAULT, dtype=np.int32) # Default facet marker value

facets_bot   = dfx.mesh.locate_entities_boundary(mesh, facet_dim, bot_boundary)
facets_top   = dfx.mesh.locate_entities_boundary(mesh, facet_dim, top_boundary)
facets_left  = dfx.mesh.locate_entities_boundary(mesh, facet_dim, left_boundary)
facets_right = dfx.mesh.locate_entities_boundary(mesh, facet_dim, right_boundary)

facet_marker[facets_bot]   = BOT
facet_marker[facets_top]   = TOP
facet_marker[facets_left]  = LEFT
facet_marker[facets_right] = RIGHT

ft = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype=np.int32), facet_marker) 
ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
dx = ufl.Measure('dx', mesh)

# Parameters of the displacement
T_left = 1.0
A_left = 0.05
T_right = 0.5
A_right = 0.01

# Displacement expression classes
class LeftBoundaryDeformation:
    def __init__(self):
        self.t = 0
        self.A = A_left
        self.T = T_left
        
    def __call__(self, x):
        return np.stack((self.A*4*x[1]*(1-x[1])*np.sin(2*np.pi*self.t/self.T),
                         np.zeros(x.shape[1])))

class RightBoundaryDeformation:
    def __init__(self):
        self.t = 0
        self.A = A_right
        self.T = T_right
    
    def __call__(self, x):
        return np.stack((self.A*4*x[1]*(1-x[1])*np.sin(2*np.pi*self.t/self.T),
                         np.zeros(x.shape[1])))

# Velocity expression classes
class LeftBoundaryVelocity:
    def __init__(self):
        self.t = 0
        self.A = A_left
        self.T = T_left
        
    def __call__(self, x):
        return np.stack((self.A*4*x[1]*(1-x[1])*np.cos(2*np.pi*self.t/self.T)*2*np.pi/self.T,
                         np.zeros(x.shape[1])))

class RightBoundaryVelocity:
    def __init__(self):
        self.t = 0
        self.A = A_right
        self.T = T_right
    
    def __call__(self, x):
        return np.stack((self.A*4*x[1]*(1-x[1])*np.cos(2*np.pi*self.t/self.T)*2*np.pi/self.T,
                         np.zeros(x.shape[1])))
# Operators
# NOTE: these are the jump operators from Krauss, Zikatonov paper.
# Jump is just a difference and it preserves the rank 
Jump = lambda arg: arg('+') - arg('-')

# Average uses dot with normal and AGAIN MINUS; it reduces the rank
Avg = lambda arg, n: .5*(dot(arg('+'), n('+')) - dot(arg('-'), n('-')))

# Action of (1 - n x n) on a vector yields the tangential component
Tangent = lambda v, n: v - n*dot(v, n)

# The ALE problem needs to extend boundary deformation to the entire domain
# to define mesh displacement field
vec_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
W = dfx.fem.functionspace(mesh, vec_el)

# Test and trial functions
w, dw = ufl.TrialFunction(W), ufl.TestFunction(W)

# BC functions
u_bdry_right = dfx.fem.Function(W)
u_bdry_left  = dfx.fem.Function(W)
u_bdry_bot   = dfx.fem.Function(W)
u_bdry_top   = dfx.fem.Function(W)
zero = dfx.fem.Function(W)

a_ale = inner(grad(w), grad(dw))*dx
L_ale = inner(zero, dw)*dx

# BCs
dofs_right = dfx.fem.locate_dofs_topological(W, facet_dim, ft.find(RIGHT))
dofs_left = dfx.fem.locate_dofs_topological(W, facet_dim, ft.find(LEFT))
dofs_bot = dfx.fem.locate_dofs_topological(W, facet_dim, ft.find(BOT))
dofs_top = dfx.fem.locate_dofs_topological(W, facet_dim, ft.find(TOP))
u_bdry_left_expr = LeftBoundaryDeformation()
u_bdry_right_expr = RightBoundaryDeformation()
u_bdry_left.interpolate(u_bdry_left_expr)
u_bdry_right.interpolate(u_bdry_right_expr)
bcs_ale = [dfx.fem.dirichletbc(u_bdry_left, dofs_left),
           dfx.fem.dirichletbc(u_bdry_right, dofs_right),
           dfx.fem.dirichletbc(u_bdry_bot, dofs_bot),
           dfx.fem.dirichletbc(u_bdry_top, dofs_top)]

# This serves to define the deformation of the mesh
wh = dfx.fem.Function(W)
w_ = dfx.fem.Function(W)
dw_dt = dfx.fem.Function(W)

# Now that we have that we can define the Stokes problem in the deformed coordinates
r = ufl.SpatialCoordinate(mesh)
chi = r + wh          
F = grad(chi) # Deformation gradient
J = det(F) # Jacobian 
n = ufl.FacetNormal(mesh)

scal_el = element("Lagrange", mesh.basix_cell(), 1)
bdm_el = element("BDM", mesh.basix_cell(), 1)
dg_el  = element("DG", mesh.basix_cell(), 0)
dg_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
V = dfx.fem.functionspace(mesh, bdm_el)
Q = dfx.fem.functionspace(mesh, dg_el)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
                  
Grad = lambda arg: dot(grad(arg), inv(F))
Div = lambda arg: inner(grad(arg), inv(F))
Nabla_Grad = lambda arg: dot(nabla_grad(arg), inv(F))

# Symmetric gradient
Eps = lambda arg: dot(ufl.sym(grad(arg)), inv(F))

mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(1e-1))
rho = dfx.fem.Constant(mesh, dfx.default_scalar_type(1e1))
penalty = dfx.fem.Constant(mesh, dfx.default_scalar_type(100.0))
dS = ufl.Measure('dS', domain=mesh) # Interior facet integral measure
u_ = dfx.fem.Function(V)
timestep = 0.01
final_time = 1
N = int(final_time / timestep)
dt = dfx.fem.Constant(mesh, dfx.default_scalar_type(timestep))

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

    if consistent: # Add symmetrization terms
        return (-inner(Avg(2*mu*Eps(u), n), Jump(Tangent(v, n)))*J('+')*dS
                -inner(Avg(2*mu*Eps(v), n), Jump(Tangent(u, n)))*J('+')*dS
                + 2*mu*(penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*J('+')*dS)

    # For preconditioning
    return 2*mu*(penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*J('+')*dS

p_bc = dfx.fem.Function(Q)
class PressureBC:
    def __init__(self):
        self.t = 0
        self.A = -10
    def __call__(self, x):
        return x[0]*self.A#*np.sin(2*np.pi*self.t)
p_bc_expr = PressureBC()

c_vel = u_ - dw_dt
# Navier-Stokes problem in reference domain accounting for the deformation
a00 = (rho/dt * inner(u, v)*J*dx # Time derivative
      + 2*mu*inner(Eps(u), Eps(v))*J*dx # Viscous dissipation
      + stabilization(u, v) # BDM stabilization
      - mu*inner(dot(Grad(u).T, n), v)*J*(ds(LEFT) + ds(RIGHT)) # Parallel flow at inlet/outlet
      )
a01 = inner(p, Div(v))*J*dx
a10 = inner(q, Div(u))*J*dx
a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*J*dx

L0 = inner(zero, v)*J*dx + dot(p_bc*n, v)*J*ds(LEFT)
L1 = inner(dfx.fem.Function(Q), q)*J*dx

# Navier-Stokes problem
a00 += rho*inner(dot(c_vel, Nabla_Grad(u)), v)*J*dx # Convective term

# Add convective term stabilization
zeta = ufl.conditional(ufl.lt(dot(c_vel, n), 0), 1, 0) # Upwind velocity operator (equals 1 on inflow boundary, 0 on outflow boundary)
a00 += (- rho*1/2*dot(jump(c_vel), n('+')) * avg(dot(u, v))*J('+')  * dS 
        - rho*dot(avg(c_vel), n('+')) * dot(jump(u), avg(v))*J('+') * dS 
        - zeta*rho*1/2*dot(c_vel, n) * dot(u, v)*J * (ds(LEFT)+ds(RIGHT))
)

a = dfx.fem.form([[a00, a01], [a10, a11]])
L = dfx.fem.form([L0, L1])

# Set boundary conditions on velocity
# Impermeability
v_bdry_bot = dfx.fem.Function(V)
v_bdry_top = dfx.fem.Function(V)
v_dofs_bot = dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(BOT))
v_dofs_top = dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(TOP))
bcs_stokes = [dfx.fem.dirichletbc(v_bdry_top, v_dofs_top),
              dfx.fem.dirichletbc(v_bdry_bot, v_dofs_bot)]

# Production flux
tot_prod = 1.0e-2
flux = create_normal_contribution_bc(V, -tot_prod*n, ft.find(RIGHT))
v_dofs_right = dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(RIGHT))
bcs_stokes.append(dfx.fem.dirichletbc(flux, v_dofs_right))

# Define deforming mesh and reference coordinates (coordinates of mesh at t=0)
out_mesh = dfx.mesh.create_unit_square(comm, nx, ny)
x_reference = out_mesh.geometry.x.copy()
facet_marker = np.full(num_facets, DEFAULT, dtype=np.int32) # Default facet marker value

facets_bot   = dfx.mesh.locate_entities_boundary(out_mesh, facet_dim, bot_boundary)
facets_top   = dfx.mesh.locate_entities_boundary(out_mesh, facet_dim, top_boundary)
facets_left  = dfx.mesh.locate_entities_boundary(out_mesh, facet_dim, left_boundary)
facets_right = dfx.mesh.locate_entities_boundary(out_mesh, facet_dim, right_boundary)

facet_marker[facets_bot]   = BOT
facet_marker[facets_top]   = TOP
facet_marker[facets_left]  = LEFT
facet_marker[facets_right] = RIGHT

ft_out_mesh = dfx.mesh.meshtags(out_mesh, facet_dim, np.arange(num_facets, dtype=np.int32), facet_marker) 
ds_out = ufl.Measure('ds', domain=out_mesh, subdomain_data=ft_out_mesh)

# Compute cells for point evaluation of the deformation function wh
cells = []
bb_tree = dfx.geometry.bb_tree(out_mesh, mesh.topology.dim)
cell_candidates = dfx.geometry.compute_collisions_points(bb_tree, x_reference)
colliding_cells = dfx.geometry.compute_colliding_cells(out_mesh, cell_candidates, x_reference)
for i, point in enumerate(x_reference):
    if len(colliding_cells.links(i)>0):
        cc = colliding_cells.links(i)[0]
        cells.append(cc)
cells = np.array(cells)

DG1 = dfx.fem.functionspace(out_mesh, dg_vec_el)
uh = dfx.fem.Function(DG1); uh.name = 'uh'
ph = dfx.fem.Function(dfx.fem.functionspace(out_mesh, dg_el)); ph.name = 'ph'

velocity_output = dfx.io.VTKFile(comm, "../output/square-mesh/flow/navier-stokes/deforming_square_velocity.pvd", "w")
pressure_output = dfx.io.VTKFile(comm, "../output/square-mesh/flow/navier-stokes/deforming_square_pressure.pvd", "w")

# Define the mesh deformation problem
ale_problem = LinearProblem(a_ale, L_ale, bcs_ale, wh, petsc_options={"ksp_type" : "preonly",
                                                                      "pc_type" : "lu",
                                                                      "pc_factor_mat_solver_type" : "mumps"})

def assemble_system(lhs_form, rhs_form, bcs):
    A = dfx.fem.petsc.assemble_matrix_nest(lhs_form, bcs=bcs)
    A.assemble()

    b = dfx.fem.petsc.assemble_vector_nest(rhs_form)
    dfx.fem.petsc.apply_lifting_nest(b, lhs_form, bcs=bcs)
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    spaces = dfx.fem.extract_function_spaces(rhs_form)
    bcs0 = dfx.fem.bcs_by_block(spaces, bcs)
    dfx.fem.petsc.set_bc_nest(b, bcs0)
    return A, b

def create_preconditioner(Q, a, bcs):
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    a_p11 = dfx.fem.form(inner(p, q) * dx)
    a_p = dfx.fem.form([[a[0][0], None], [None, a_p11]])
    P = dfx.fem.petsc.assemble_matrix_nest(a_p, bcs)
    P.assemble()

    return P

def create_block_solver_direct(A):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()

    return ksp

def create_block_solver_iterative(A, P):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A, P)
    ksp.setType("minres")
    ksp.setTolerances(rtol=1e-9)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

    # Set the preconditioners for each block
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # Monitor the convergence of the KSP
    ksp.setFromOptions()
    return ksp

iterative = False
def solve_stokes(P=None):

    A, b = assemble_system(a, L, bcs_stokes)
    if iterative:
        ksp = create_block_solver_iterative(A, P)
    else:
        ksp = create_block_solver_direct(A)

    u, p = dfx.fem.Function(V), dfx.fem.Function(Q)
    w = PETSc.Vec().createNest([u.x.petsc_vec, p.x.petsc_vec])

    ksp.solve(b, w)
    assert ksp.getConvergedReason() > 0
    u.x.scatter_forward()
    p.x.scatter_forward()

    return u, p

if iterative: P = create_preconditioner(Q, a, bcs_stokes)

for time in np.linspace(0, final_time, N+1):
    # Update time variable of boundary conditions 
    # Boundary deformation
    u_bdry_left_expr.t = time
    u_bdry_right_expr.t = time

    # Pressure BC
    p_bc_expr.t = time
    p_bc.interpolate(p_bc_expr)

    # Interpolate BC expressions into BC functions
    u_bdry_left.interpolate(u_bdry_left_expr)
    u_bdry_right.interpolate(u_bdry_right_expr)
    
    ale_problem.solve() # Solve the mesh motion problem
    dw_dt.x.array[:] = (wh.x.array.copy() - w_.x.array.copy())/timestep
    w_.x.array[:] = wh.x.array.copy()

    if iterative:
        uh_, ph_ = solve_stokes(P) # Solve the Stokes equations
    else:
        uh_, ph_ = solve_stokes() # Solve the Stokes equations

    # Update output functions
    uh.interpolate(uh_)
    ph.interpolate(ph_)
    u_.x.array[:] = uh_.x.array.copy()

    wh_x_reference = wh.eval(x=x_reference, cells=cells) # Evaluate the deformed coordinates at the reference coordinates

    # Update output mesh
    out_mesh.geometry.x[:, :out_mesh.geometry.dim] = x_reference[:, :out_mesh.geometry.dim] + wh_x_reference

    # Make pressure mean = 0
    ph.x.array[:] -= calculate_mean(out_mesh, ph, ufl.dx(out_mesh))
    
    # Write output
    velocity_output.write_mesh(out_mesh, time)
    velocity_output.write_function(uh, time)
    pressure_output.write_mesh(out_mesh, time)
    pressure_output.write_function(ph, time)

    # Calculate mean pressure
    vol = comm.allreduce(assemble_scalar(1*ufl.dx(out_mesh)), op=MPI.SUM)
    print("Mean pressure: ", 1/vol*comm.allreduce(assemble_scalar(ph*ufl.dx(out_mesh)), op=MPI.SUM))

    # Calculate boundary flux at production site
    print("Flux: ", comm.allreduce(assemble_scalar(dot(uh, ufl.FacetNormal(out_mesh))*ds_out(RIGHT)), op=MPI.SUM))

# Close output files
velocity_output.close()
pressure_output.close()