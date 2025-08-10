import ufl
import sys 

import numpy   as np
import dolfinx as dfx

from ufl       import inner, dot, grad, det, inv, avg, jump, nabla_grad
from scifem    import assemble_scalar
from scipy.fft import fft, ifft, fftfreq
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from utilities.fem import calculate_mean, create_normal_contribution_bc
from dolfinx.fem.petsc import (LinearProblem,
                                assemble_matrix_block, 
                                assemble_vector_block, 
                                create_matrix_block, 
                                create_vector_block,
                                create_matrix,
                                assemble_matrix)

print = PETSc.Sys.Print
PETSc.Options().getAll()
# Solve stationary Stokes in moving domain by ALE method. Wall motion is 
# prescribed in time.
comm = MPI.COMM_WORLD
nx = ny = int(sys.argv[1])
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
T_left = 1.1
A_left = 0.05
T_right = 0.4
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

class TopBoundaryDeformation:
    def __init__(self):
        self.t = 0
        self.A = A_right
        self.T = T_right
    
    def __call__(self, x):
        return np.stack((np.zeros(x.shape[1]),
                         self.A*4*x[0]*(1-x[0])*np.sin(2*np.pi*self.t/self.T)
                         ))

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

class TopBoundaryVelocity:
    def __init__(self):
        self.t = 0
        self.A = A_right
        self.T = T_right
    
    def __call__(self, x):
        return np.stack((np.zeros(x.shape[1]),
                        self.A*4*x[0]*(1-x[0])*np.cos(2*np.pi*self.t/self.T)*2*np.pi/self.T
                         ))
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
u_bdry_top_expr = TopBoundaryDeformation()
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
n_hat = ufl.FacetNormal(mesh)

k = 2 # element degree
bdm_el = element("BDM", mesh.basix_cell(), k)
dg_el  = element("DG", mesh.basix_cell(), k-1)
dg_vec_el = element("DG", mesh.basix_cell(), k, shape=(mesh.geometry.dim,))
V = dfx.fem.functionspace(mesh, bdm_el)
Q = dfx.fem.functionspace(mesh, dg_el)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
                  
Grad = lambda arg: dot(grad(arg), inv(F))
Div = lambda arg: ufl.tr(dot(grad(arg), inv(F)))
Nabla_Grad = lambda arg: dot(nabla_grad(arg), inv(F))

# Symmetric gradient
Eps = lambda arg: ufl.sym(Grad(arg))

mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(1))
rho = dfx.fem.Constant(mesh, dfx.default_scalar_type(1))
xi = dfx.fem.Constant(mesh, dfx.default_scalar_type(25.0 * k))

dS_hat = ufl.Measure('dS', domain=mesh) # Interior facet integral measure
u_ = dfx.fem.Function(V)
timestep = 0.01
final_time = 1
N = int(final_time / timestep)
dt = dfx.fem.Constant(mesh, dfx.default_scalar_type(timestep))
n = J*inv(F.T)*n_hat # Physical domain facet normal vector
hA = ufl.avg(ufl.CellDiameter(mesh)) # Facet normal vector and average cell diameter
hB = 2*ufl.Circumradius(mesh)

uh_ = dfx.fem.Function(V)
ph_ = dfx.fem.Function(Q)
u_rel = dfx.fem.Function(V)
u_mesh = dfx.fem.Function(V)

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


    # if consistent: # Add symmetrization terms
    #     return (-inner(Avg(2*mu*Eps(u), n), Jump(Tangent(v, n)))*dS_hat
    #             -inner(Avg(2*mu*Eps(v), n), Jump(Tangent(u, n)))*dS_hat
    #             + 2*mu*(xi/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS_hat)
    if consistent: # Add symmetrization terms
        return (-inner(Avg(2*mu*Eps(u), n), Jump(v))*dS_hat
                -inner(Avg(2*mu*Eps(v), n), Jump(u))*dS_hat
                + 2*mu*(xi/hA)*inner(Jump(u), Jump(v))*dS_hat)

    # For preconditioning
    return 2*mu*(xi/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS_hat

c_vel = u_ - u_mesh
# Navier-Stokes problem in reference domain accounting for the deformation
a00 = (rho/dt * inner(u, v)*J*dx # Time derivative
      + 2*mu*inner(Eps(u), Eps(v))*J*dx # Viscous dissipation
      + stabilization(u, v) # BDM stabilization
    #   - mu*inner(dot(Grad(u).T, n), v)*(ds(LEFT)) # Parallel flow at inlet/outlet

      )
a01 = inner(p, Div(v))*J*dx
a10 = inner(q, Div(u))*J*dx
a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*J*dx

a_stokes = dfx.fem.form([[a00, a01], [a10, a11]])

L0 = rho/dt * inner(u_, v)*J*dx

# Traction BC with Navier slip friction
alpha = dfx.fem.Constant(mesh, dfx.default_scalar_type(1)) # Navier slip friction coefficient
a00 += alpha * inner(Tangent(u, n), Tangent(v, n)) * (ds(TOP) + ds(BOT) + ds(RIGHT)) # Navier slip term

# Weakly impose tangential traction to represent cilia
tau_val = 0#7.89e-0 # Tangential traction force density [Pa]
tau_mag = dfx.fem.Constant(mesh, dfx.default_scalar_type(tau_val))
tau_vec = dfx.fem.Function(V)
vel = lambda x: np.ones((2, x.shape[1]))
tau_vec.interpolate(vel)
tau = tau_mag * tau_vec / ufl.sqrt(inner(tau_vec, tau_vec))

L0 += inner(tau, Tangent(v, n))*ds(BOT)

L1 = inner(dfx.fem.Function(Q), q)*J*dx

# Navier-Stokes problem
navier_stokes = False
if navier_stokes:
    a00 += rho*inner(dot(c_vel, Nabla_Grad(u)), v)*J*dx # Convective term

    # Add convective term stabilization
    zeta = ufl.conditional(ufl.lt(dot(c_vel, n), 0), 1, 0) # Upwind velocity operator (equals 1 on inflow boundary, 0 on outflow boundary)
    a00 += (- rho*1/2*dot(jump(c_vel), n('+')) * avg(dot(u, v))*dS_hat 
            - rho*dot(avg(c_vel), n('+')) * dot(jump(u), avg(v))*dS_hat
            - zeta*rho*1/2*dot(c_vel, n) * dot(u, v)*(ds(TOP) + ds(LEFT) + ds(RIGHT))
    )
# else:
#     # a00 += rho*inner(dot(u_mesh, Nabla_Grad(u)), v)*J*dx

a = dfx.fem.form([[a00, a01], [a10, a11]])
L = dfx.fem.form([L0, L1])

# Set boundary conditions on velocity
# Impermeability
v_bdry_bot = dfx.fem.Function(V)
v_bdry_top = dfx.fem.Function(V)
v_bdry_top_expr = TopBoundaryVelocity()
v_dofs_bot = dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(BOT))
v_dofs_top = dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(TOP))
bcs_stokes = [dfx.fem.dirichletbc(v_bdry_top, v_dofs_top),
              dfx.fem.dirichletbc(v_bdry_bot, v_dofs_bot)]

v_bdry_left = dfx.fem.Function(V)
v_bdry_right = dfx.fem.Function(V)
v_bdry_left_expr = LeftBoundaryVelocity()
v_bdry_right_expr = RightBoundaryVelocity()
v_dofs_left = dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(LEFT))
v_dofs_right = dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(RIGHT))
# bcs_stokes.append(dfx.fem.dirichletbc(v_bdry_left, v_dofs_left))
# bcs_stokes.append(dfx.fem.dirichletbc(v_bdry_right, v_dofs_right))

# Production flux
tot_prod = 5.0e-3
normal_bc = dfx.fem.Function(V)
# prod_func.x.array[:] = tot_prod/1.0
# facet_midpoint = np.array([[0.5]], dtype=np.float64)
# normal_bc_expr = dfx.fem.Expression(n_hat*tot_prod, facet_midpoint)
# normal_bc.interpolate(normal_bc_expr)
v_dofs_right = dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(RIGHT))
bcs_stokes.append(dfx.fem.dirichletbc(normal_bc, v_dofs_right))

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

DG = dfx.fem.functionspace(out_mesh, dg_vec_el)
uh = dfx.fem.Function(DG); uh.name = 'uh'
ph = dfx.fem.Function(dfx.fem.functionspace(out_mesh, dg_el)); ph.name = 'ph'

velocity_output = dfx.io.VTKFile(comm, "../output/square-mesh/flow/navier-stokes/deforming_square_velocity.pvd", "w")
pressure_output = dfx.io.VTKFile(comm, "../output/square-mesh/flow/navier-stokes/deforming_square_pressure.pvd", "w")

# Define the mesh deformation problem
ale_problem = LinearProblem(a_ale, L_ale, bcs_ale, wh, petsc_options={"ksp_type" : "preonly",
                                                                      "pc_type" : "lu",
                                                                      "pc_factor_mat_solver_type" : "mumps",
                                                                      "mat_mumps_icntl_14" : 80,
                                                                      "mat_mumps_icntl_24" : 1,
                                                                      "mat_mumps_icntl_25" : 0})

# Set up Navier-Stokes problem linear system
A = create_matrix_block(a) # System matrix
xh = A.createVecRight() # Solution vector
b = create_vector_block(L) # RHS vector

# Calculate offsets

V_map = V.dofmap.index_map
Q_map = Q.dofmap.index_map
offset_u = V_map.size_local * V.dofmap.index_map_bs
offset_p = offset_u + Q_map.size_local*Q.dofmap.index_map_bs

def solve_blocked_system(ksp):
        
    A.zeroEntries()
    assemble_matrix_block(A, a, bcs=bcs_stokes)
    A.assemble()

    with b.localForm() as b_loc: b_loc.set(0)
    assemble_vector_block(b, L, a, bcs=bcs_stokes)

    ksp.solve(b, xh)
    assert ksp.getConvergedReason() > 0

    # Update and MPI communcation
    uh_.x.array[:offset_u] = xh.array[:offset_u]
    uh_.x.scatter_forward()
    ph_.x.array[:(offset_p-offset_u)] = xh.array[offset_u:offset_p]
    ph_.x.scatter_forward()

    return uh_, ph_

def create_preconditioner(Q, a, bcs):
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    a_p11 = dfx.fem.form(inner(p, q) * dx)
    a_p00 = (rho/dt * inner(u, v)*J*dx # Time derivative
            + 2*mu*inner(Eps(u), Eps(v))*J*dx # Viscous dissipation
            + rho*inner(dot(c_vel, Nabla_Grad(u)), v)*J*dx
            + 2*mu*(xi/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS_hat
            # - rho*1/2*dot(jump(c_vel), n('+')) * avg(dot(u, v))*dS_hat 
            # - rho*dot(avg(c_vel), n('+')) * dot(jump(u), avg(v))*dS_hat
    )
    a_p = dfx.fem.form([[a_p00, None], [None, a_p11]])
    P = assemble_matrix_block(a_p, bcs)
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
    opts = PETSc.Options() 
    opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
    opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
    opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
    opts["ksp_error_if_not_converged"] = 1 # Throw an error if KSP solver does not converge
    ksp.setFromOptions()

    return ksp
def build_monitor(prefix):
    """Creates a simple monitor function that prints residual info."""
    def monitor(ksp, it, rnorm):
        print(f"    {prefix} Iteration: {it}, Residual norm: {rnorm:.3e}")
    return monitor
def create_block_solver_iterative(A, P):
    # Create a MINRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A, P)
    ksp.setType("minres")
    ksp.setMonitor(build_monitor("Main")) # Equivalent to -main_ksp_monitor
    ksp.setTolerances(rtol=1e-7)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    # ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.SCHUR) # Use SCHUR instead of ADDITIVE
    # ksp.getPC().setFieldSplitSchurFactType(PETSc.PC.SchurFactType.LOWER) # Common factorization type

    ksp.setOptionsPrefix("main_")

    # Build PETSc index sets for each field (global dof indices for each
    # field)
    is_offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + Q_map.local_range[0]
    is_offset_p = is_offset_u + V_map.size_local * V.dofmap.index_map_bs
    print(offset_u)
    print(offset_p)
    is_u = PETSc.IS().createStride(
        V_map.size_local * V.dofmap.index_map_bs, is_offset_u, 1, comm=comm
    )
    is_p = PETSc.IS().createStride(Q_map.size_local, is_offset_p, 1, comm=comm)

    ksp.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))

    # Configure velocity and pressure sub-solvers
    # ksp.setUp() # This builds the internal structures, including sub-KSPs
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()   
    ksp_u.setOptionsPrefix("u_")
    ksp_p.setOptionsPrefix("p_")

    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_u.getPC().setGAMGType("classical") # Equivalent to -u_pc_gamg_type classical
    ksp_p.setType("gmres")
    ksp_p.setTolerances(rtol=1e-10) # Inner solve can be less precise
    ksp_p.setGMRESRestart(50)
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")
    ksp_p.setMonitor(build_monitor("  Pressure")) # Monitor the inner pressure solve
    # ksp_p.getPC().setType("hypre") # Use HYPRE's AMG
    # ksp_p.getPC().setHYPREType("boomeramg") # Equivalent to -p_pc_hypre_type boomeramg

    # p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    # mu = dfx.fem.Constant(mesh, 1e-2) # Example viscosity

    # # The UFL form for our custom pressure preconditioner P_p
    # a_p = (1 / dt) * inner(p, q) * dx + mu * inner(ufl.grad(p), ufl.grad(q)) * dx
    # P_p_form = dfx.fem.form(a_p)

    # # Assemble the operator into a PETSc matrix
    # P_p_mat = create_matrix(P_p_form)
    # assemble_matrix(P_p_mat, P_p_form)
    # P_p_mat.assemble()

    # # --- Step 5: Assign the custom operator to the pressure preconditioner ---
    # # Instead of a weak 'jacobi', we use our custom P_p_mat matrix.
    # # We'll use a direct solver (LU) to "invert" this preconditioner matrix.
    # # This is efficient because P_p_mat is much smaller and better-conditioned.
    # # ksp_p.getPC().setType("lu") # The PC action is just applying the operator P_p_mat
    # # ksp_p.getPC().setOperators(P_p_mat)
    # # Optional: If P_p_mat is large, you could use an AMG solver instead of preonly+LU
    # ksp_p.getPC().setType("gamg")
    # ksp_p.getPC().setOperators(P_p_mat)

    # Monitor the convergence of the KSP
    ksp.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, residual: {rnorm}"))
    
    # Set a non-zero initial guess; use previous timestep's solution
    # as initial guess for the next step
    ksp.setInitialGuessNonzero(True)

    return ksp

iterative = False

if iterative:
    P = create_preconditioner(Q, a, bcs_stokes)
    ksp = create_block_solver_iterative(A, P)
else:
    ksp = create_block_solver_direct(A)

times = np.linspace(0, final_time, N+1)
displacements = np.zeros((len(wh.x.array), len(times)), wh.x.array.dtype)
for i, time in enumerate(times):
    # Update time variable of boundary conditions
    # u_bdry_left_expr.t = time
    u_bdry_right_expr.t = time
    u_bdry_top_expr.t = time

    # Interpolate BC expressions into BC functions
    # u_bdry_left.interpolate(u_bdry_left_expr)
    u_bdry_right.interpolate(u_bdry_right_expr)
    u_bdry_top.interpolate(u_bdry_top_expr)
    
    ale_problem.solve() # Solve the mesh motion problem
    w_.x.array[:] = wh.x.array.copy()

    wh_x_reference = wh.eval(x=x_reference, cells=cells) # Evaluate the deformed coordinates at the reference coordinates

    # Update output mesh
    out_mesh.geometry.x[:, :out_mesh.geometry.dim] = x_reference[:, :out_mesh.geometry.dim] + wh_x_reference

    displacements[:, i] = wh.x.array.copy() # Store the displacements to be differentiated with FFT

# Calculate the displacement velocity with the Fast Fourier Transform (FFT)
# Get the frequencies of the signal
n_steps = displacements.shape[1]
freqs = fftfreq(n_steps, dt.value)

# Compute the Fast Fourier Transform of the displacement
wh_fft = fft(displacements, axis=1)

# Differentiate in frequency space by multiplying by (i * omega)
# where omega = 2*pi*f and i is the imaginary unit
wh_dot_fft = (1j*2*np.pi*freqs) * wh_fft

# Apply a Gaussian filter to filter out high-frequency noise
sigma = 10.0 # a.k.a. sigma in a Gaussian
filter = np.exp(-(freqs**2) / (2 * sigma**2))
wh_dot_fft = filter*wh_dot_fft

# Compute the inverse FFT to get the velocity in the time domain
velocity = ifft(wh_dot_fft, axis=1).real # Take only the real part

# Solve stokes for initial condition
v_bdry_top.interpolate(v_bdry_top_expr)
v_bdry_right.interpolate(v_bdry_right_expr)
assemble_matrix_block(A, a_stokes, bcs=bcs_stokes)
A.assemble()
assemble_vector_block(b, L, a_stokes, bcs=bcs_stokes)

# Solve and update solution functions
ksp.solve(b, xh)
u_.x.array[:offset_u] = xh.array[:offset_u]
u_.x.scatter_forward()

nn_form = dfx.fem.form(dot(n_hat, n)*ds(RIGHT))
wn_form = dfx.fem.form(dot(u_mesh, n)*ds(RIGHT))
tangent_correction_form = dfx.fem.form(dot(Tangent(uh_, n_hat), n_hat)*ds(RIGHT))

area_form = dfx.fem.form(ufl.sqrt(dot(n, n))*ds(RIGHT))
import time as time_module
tic = time_module.perf_counter()
vel = lambda x: np.vstack((x[0], np.zeros(x.shape[1])))
for i, time in enumerate(times):
    
    dw_dt.x.array[:] = velocity[:, i] # Update mesh velocity
    u_mesh.interpolate(dw_dt)

    v_bdry_top_expr.t = time
    v_bdry_top.interpolate(v_bdry_top_expr)

    v_bdry_right_expr.t = time
    v_bdry_right.interpolate(v_bdry_right_expr)
    
    wh.x.array[:] = displacements[:, i]
    wh_x_reference = wh.eval(x=x_reference, cells=cells) # Evaluate the deformed coordinates at the reference coordinates
    
    # Update output mesh
    out_mesh.geometry.x[:, :out_mesh.geometry.dim] = x_reference[:, :out_mesh.geometry.dim] + wh_x_reference
    
    nn_term = dfx.fem.assemble_scalar(nn_form)
    wn_term = dfx.fem.assemble_scalar(wn_form)
    tangent_correction = dfx.fem.assemble_scalar(tangent_correction_form)
    area = dfx.fem.assemble_scalar(area_form)
    nn = dfx.fem.assemble_scalar(nn_form)
    wn = dfx.fem.assemble_scalar(wn_form)
    print(f"{wn=}")
    print(f"{area=}\t{nn=}\t{nn_term}")
    prod_func = create_normal_contribution_bc(V, (-tot_prod/dot(n_hat, n) + dot(u_mesh, n)/dot(n_hat, n))*n_hat, ft.find(RIGHT)) #(-tot_prod + wn_term)/nn_term*n_hat
    normal_bc.interpolate(prod_func) 
    # normal_bc.x.array[:] += u_mesh.x.array.copy()

    uh_, ph_ = solve_blocked_system(ksp) # Solve the Stokes equations

    u_rel.x.array[:] = uh_.x.array.copy() - u_mesh.x.array.copy()
    # Update output functions
    uh.interpolate(u_rel)
    ph.interpolate(ph_)
    u_.x.array[:] = uh_.x.array.copy()

    # Make pressure mean = 0
    ph.x.array[:] -= calculate_mean(out_mesh, ph, ufl.dx(out_mesh))
    
    # Write output
    velocity_output.write_mesh(out_mesh, time)
    velocity_output.write_function(uh, time)
    pressure_output.write_mesh(out_mesh, time)
    pressure_output.write_function(ph, time)

    # Calculate mean pressure
    vol = assemble_scalar(1*ufl.dx(out_mesh))
    print("Mean pressure: ", 1/vol*assemble_scalar(ph*ufl.dx(out_mesh)))

    # Calculate boundary flux at production site
    print("Flux: ", assemble_scalar(dot(u_rel, n)*ds(RIGHT)))

    e_div_u = assemble_scalar(inner(Div(uh_), Div(uh_))*J*dx)
    print(f"e_div_u = {e_div_u}")
    if i==50: break
print("Time elapsed = ", time_module.perf_counter() - tic)
# Close output files
velocity_output.close()
pressure_output.close()