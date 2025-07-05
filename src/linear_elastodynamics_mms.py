import ufl

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from sys       import argv
from ufl       import inner, grad, sym, div
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from dolfinx.fem.petsc    import create_matrix, create_vector, assemble_vector, apply_lifting
from utilities.fem        import assemble_system
from utilities.mesh       import create_unit_cube_mesh, create_unit_square_mesh

print = PETSc.Sys.Print

# Solve linear elasticity equation on a unit square.
# Boundary conditions:
# - Prescribed motion on top and bottom walls
# - Zero traction on front, back and right walls
# - Anchored (clamped) left wall
comm = MPI.COMM_WORLD
N = int(argv[1])
dim = 2
if dim==2:
    mesh, ft = create_unit_square_mesh(N=N, comm=comm)
else:
    mesh, ft = create_unit_cube_mesh(N=N, comm=comm)
facet_dim = mesh.topology.dim-1
mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity

dx = ufl.Measure('dx', domain=mesh) # Volume integral measure
eps = lambda arg: sym(grad(arg)) # The symmetric gradient

# Material parameters
E = 1500 #3156 # Modulus of elasticity [Pa]
nu = 0.479 # Poisson's ratio [-]
eta_value = 2*E/(1+nu) # First Lamé parameter value
lam_value = nu*E/((1+nu)*(1-2*nu)) # Second Lamé parameter value
eta = dfx.fem.Constant(mesh, eta_value) # First Lamé parameter
lam = dfx.fem.Constant(mesh, lam_value) # Second Lamé parameter
rho = dfx.fem.Constant(mesh, 1000.0) # Density

# Exact solution constants
phi = dfx.fem.Constant(mesh, 0.5)
zeta = dfx.fem.Constant(mesh, 2.0)
theta = dfx.fem.Constant(mesh, 0.1)

# Define Generalized-alpha method parameters
gamma = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.5))
beta  = dfx.fem.Constant(mesh, dfx.default_scalar_type(1/4*(gamma.value + 0.5)**2))

print(f"Gamma = {gamma.value}, Beta = {beta.value}")

# Temporal parameters
timestep = 1e-2
dt = dfx.fem.Constant(mesh, dfx.default_scalar_type(timestep)) 
T = timestep*10
period = T
N = int(T / timestep)
times = np.linspace(0, T, N+1)
final_period_start = int(T - period)
write_time = 0
tt = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))

# Finite elements
degree = 2
vec_el = element("Lagrange", mesh.basix_cell(), degree, shape=(mesh.geometry.dim,))
W = dfx.fem.functionspace(mesh, vec_el)
wh = dfx.fem.Function(W) # Displacement function
wh_n = dfx.fem.Function(W) # Displacement function at previous timestep
wh_dot = dfx.fem.Function(W) # Velocity function
wh_dot_n = dfx.fem.Function(W) # Velocity function at previous timestep
wh_ddot  = dfx.fem.Function(W) # Acceleration function
wh_ddot_n = dfx.fem.Function(W) # Acceleration function at previous timestep
zero = dfx.fem.Function(W)

# Create exact solution expression and function
if dim==2:
    xx, yy = ufl.SpatialCoordinate(mesh)
    u_exact = ufl.as_vector((1+ufl.sin(xx) + ufl.cos(xx)*theta*tt**4,
                             1+phi*ufl.cos(yy) + ufl.sin(yy)*theta*tt**4))
    v_exact = ufl.diff(u_exact, ufl.variable(tt))
    a_exact = ufl.diff(v_exact, ufl.variable(tt))
    theta_vector = ufl.as_vector((1.0, 1.0))
else:
    xx, yy, zz = ufl.SpatialCoordinate(mesh)
    u_exact = ufl.as_vector((1+ufl.sin(xx**2) + theta*tt**4,
                             1+phi*ufl.cos(yy**3) + theta*tt**4,
                             1+zeta*ufl.sin(zz**2) + theta*tt**4))
    theta_vector = ufl.as_vector((1.0, 1.0, 1.0))

u_exact_expr = dfx.fem.Expression(u_exact, W.element.interpolation_points())
u_exact_bc = dfx.fem.Function(W)
u_exact_bc.interpolate(u_exact_expr)
print(f"\nNumber of degrees of freedom: {W.dofmap.index_map.size_global*W.dofmap.index_map_bs}")

# Set initial conditions
wh_n.x.array[:] = u_exact_bc.x.array.copy()
wh_dot_n.interpolate(lambda x: 4*theta*tt.value**3*np.stack((np.ones(x.shape[1]), np.ones(x.shape[1]))))
wh_ddot_n.interpolate(lambda x: 12*theta*tt.value**2*np.stack((np.ones(x.shape[1]), np.ones(x.shape[1]))))

sigma = lambda w: 2.0*eta*eps(w) + lam*ufl.tr(eps(w))*ufl.Identity(dim)

# Damping parameters
eta_M = dfx.fem.Constant(mesh, 0.0) # Damping proportional to inertia
eta_K = dfx.fem.Constant(mesh, 5e-3) # Damping proportional to stiffness

# Test and trial functions
w, dw = ufl.TrialFunction(W), ufl.TestFunction(W)

# Compute forcing term
f = rho*a_exact - div(sigma(u_exact)) + eta_M*rho*v_exact - eta_K*div(sigma(v_exact))
# f = - div(sigma(u_exact))


###############

acc = 1 / beta / dt**2 * (wh - wh_n - dt * wh_dot_n) + wh_ddot_n * (1 - 1 / 2 / beta)
acc_expr = dfx.fem.Expression(acc, W.element.interpolation_points())

vel = wh_dot_n + dt * ((1 - gamma) * wh_ddot_n + gamma * wh_ddot)
vel_expr = dfx.fem.Expression(vel, W.element.interpolation_points())

##################

# The weak form
# The weak form
a = rho/(beta*dt**2)*inner(w, dw)*dx + inner(sigma(w), eps(dw)) * dx
L = rho*inner(wh_n/(beta*dt**2) + wh_dot_n/(beta*dt) + (1-2*beta)/(2*beta)*wh_ddot_n, dw) * dx 

# Add Rayleigh damping: eta_M * M(v) + eta_K * K(v), where M = mass matrix, K = stiffness matrix, v = velocity
a += gamma/(beta*dt) * (eta_M*rho*inner(w, dw)*dx + eta_K*inner(sigma(w), eps(dw))*dx)
v_res = wh_dot_n + dt*wh_ddot_n*(1-gamma) - gamma/(beta*dt)*(wh_n + dt*wh_dot_n) + dt*gamma*wh_ddot_n*(2*beta-1)/(2*beta)
L -= (eta_M*rho* inner(v_res, dw)*dx + eta_K * inner(sigma(v_res), eps(dw))*dx)

mesh.topology.create_connectivity(facet_dim, mesh.topology.dim)
exterior_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dfx.fem.locate_dofs_topological(W, facet_dim, exterior_facets)
bcs = [dfx.fem.dirichletbc(u_exact_bc, boundary_dofs)]

# Create linear system
a_cpp, L_cpp = dfx.fem.form(a), dfx.fem.form(L)
A = create_matrix(a_cpp)
b = create_vector(L_cpp)

# # Configure linear solver based on
# # conjugate gradient with algebraic multigrid preconditioning
opts = PETSc.Options()
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
opts["ksp_error_if_not_converged"] = 1 # Throw an error if KSP solver does not converge

# Create the solver object, set options and enable convergence monitoring
solver = PETSc.KSP().create(comm)
solver.setOperators(A)
solver.setFromOptions()
solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, residual: {rnorm}"))

xdmf = dfx.io.XDMFFile(comm, f"../output/square-mesh/deformation/displacement_dt={timestep:.4g}_T={T:.4g}.xdmf", "w")
print(f"../output/square-mesh/deformation/displacement_dt={timestep:.4g}_T={T:.4g}.xdmf")
xdmf_vel = dfx.io.XDMFFile(comm, f"../output/square-mesh/deformation/displacement_velocity_dt={timestep:.4g}_T={T:.4g}.xdmf", "w")
xdmf.write_mesh(mesh)
xdmf_vel.write_mesh(mesh)
CG1_vector_space = dfx.fem.functionspace(mesh,
                                         element=element("Lagrange",
                                                         mesh.basix_cell(),
                                                         degree=1,
                                                         shape=(mesh.geometry.dim,)))

wh_out = dfx.fem.Function(CG1_vector_space)
vh_out = dfx.fem.Function(CG1_vector_space)
dw_dt = dfx.fem.Function(W)
k = 1 # BDM element degree
bdm_el = element("BDM", mesh.basix_cell(), k)
BDM = dfx.fem.functionspace(mesh, bdm_el)
dw_dt_bdm = dfx.fem.Function(BDM)
dw_dt_bdm.name = "defo_velocity"
wh.name = "defo_displacement"

vh_cpoint_filename = f"../output/square-mesh/deformation/checkpoints/displacement_velocity_dt={timestep:.4g}_T={T:.4g}/"
a4d.write_mesh(filename=vh_cpoint_filename, mesh=mesh)
a4d.write_meshtags(vh_cpoint_filename, mesh, ft, meshtag_name='ft')

# projection_problem = projection_problem_CG2_to_BDM1(dw_dt, dw_dt_bdm)

L2_error_form = dfx.fem.form(inner(wh - u_exact, wh - u_exact)*dx)
H1_error_form = dfx.fem.form(inner(grad(wh - u_exact), grad(wh - u_exact))*dx)

A, b = assemble_system(A, b, a_cpp, L_cpp, bcs)

for t in times:
    
    print(f"\nTime t = {t:.5g}")
    tt.value = t

    # Update displacement BCs
    u_exact_bc.interpolate(u_exact_expr)

    # Assemble linear system 
    with b.localForm() as local_b_vec: local_b_vec.set(0.0)
    assemble_vector(b, L_cpp)
    apply_lifting(b, [a_cpp], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dfx.fem.set_bc(b, bcs)

    # Solve
    solver.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward() # MPI communication

    # Update functions
    wh_ddot.interpolate(acc_expr)
    wh_dot.interpolate(vel_expr)
    wh_ddot_n.x.array[:] = wh_ddot.x.array.copy()
    wh_dot_n.x.array[:]  = wh_dot.x.array.copy() 
    wh_n.x.array[:] = wh.x.array.copy()

    # Project velocity if in the final period
    if t >= final_period_start:
        wh_out.interpolate(wh)
        xdmf.write_function(wh_out, t)
        
        # Calculate deformation velocity by a backward difference in time
        dw_dt.x.array[:] = wh_dot.x.array.copy()
        
        # Write checkpoints
        a4d.write_function(filename=vh_cpoint_filename, u=wh, time=write_time)
        a4d.write_function(filename=vh_cpoint_filename, u=dw_dt_bdm, time=write_time)
        
        # Interpolate the velocity into CG1 and write XDMF output
        vh_out.interpolate(dw_dt) 
        xdmf_vel.write_function(vh_out, t) 

        write_time += 1
    
    # Calculate error
    L2_error = np.sqrt(comm.allreduce(dfx.fem.assemble_scalar(L2_error_form), op=MPI.SUM))
    H1_error = np.sqrt(comm.allreduce(dfx.fem.assemble_scalar(H1_error_form), op=MPI.SUM))
    print(f"L2 error: {L2_error:.2e}")
    print(f"H1 error: {H1_error:.2e}")

xdmf.close()
xdmf_vel.close()