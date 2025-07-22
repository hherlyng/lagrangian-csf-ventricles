import ufl

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from sys       import argv
from ufl       import inner, grad, sym, dot
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from dolfinx.fem.petsc    import create_matrix, create_vector, assemble_vector, apply_lifting
from utilities.fem        import assemble_system
from utilities.mesh       import create_unit_cube_mesh, create_unit_square_mesh
from utilities.projection import projection_problem_CG_to_BDM

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
LEFT=1; RIGHT=2; BOT=3; TOP=4
tdim = mesh.topology.dim
facet_dim = tdim-1
mesh.topology.create_connectivity(facet_dim, tdim) # Create facet-cell connectivity

dx = ufl.Measure('dx', domain=mesh) # Volume integral measure
n = ufl.FacetNormal(mesh)
eps = lambda arg: sym(grad(arg)) # The symmetric gradient

# Material parameters
E = 1500 #3156 # Modulus of elasticity [Pa]
nu = 0.479 # Poisson's ratio [-]
eta_value = 2*E/(1+nu) # First Lamé parameter value
lam_value = nu*E/((1+nu)*(1-2*nu)) # Second Lamé parameter value
eta = dfx.fem.Constant(mesh, eta_value) # First Lamé parameter
lam = dfx.fem.Constant(mesh, lam_value) # Second Lamé parameter
rho = dfx.fem.Constant(mesh, 1000.0) # Density

# Define Generalized-alpha method parameters
gamma = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.5))
beta  = dfx.fem.Constant(mesh, dfx.default_scalar_type(1/4*(gamma.value + 0.5)**2))

print(f"Gamma = {gamma.value}, Beta = {beta.value}")

# Temporal parameters
timestep = 1e-3
dt = dfx.fem.Constant(mesh, dfx.default_scalar_type(timestep)) 
T = 5
period = 1
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

print(f"\nNumber of degrees of freedom: {W.dofmap.index_map.size_global*W.dofmap.index_map_bs}")


sigma = lambda w: 2.0*eta*eps(w) + lam*ufl.tr(eps(w))*ufl.Identity(dim)

# Damping parameters
eta_M = dfx.fem.Constant(mesh, 5e-1) # Damping proportional to inertia
eta_K = dfx.fem.Constant(mesh, 5e-1) # Damping proportional to stiffness

# Test and trial functions
w, dw = ufl.TrialFunction(W), ufl.TestFunction(W)

# Compute forcing term
f = dfx.fem.Function(W)

###############

acc = 1 / beta / dt**2 * (wh - wh_n - dt * wh_dot_n) + wh_ddot_n * (1 - 1 / 2 / beta)
acc_expr = dfx.fem.Expression(acc, W.element.interpolation_points())

vel = wh_dot_n + dt * ((1 - gamma) * wh_ddot_n + gamma * wh_ddot)
vel_expr = dfx.fem.Expression(vel, W.element.interpolation_points())

##################

# The weak form
# The weak form
# a = rho*inner(w, dw) * dx + inner(sigma(w), eps(dw)) * dx # Static
# L = inner(f, dw) * dx # Static
a = rho/(beta*dt**2)*inner(w, dw)*dx + inner(sigma(w), eps(dw)) * dx
L = rho*inner(wh_n/(beta*dt**2) + wh_dot_n/(beta*dt) + (1-2*beta)/(2*beta)*wh_ddot_n, dw) * dx 

# Add Rayleigh damping: eta_M * M(v) + eta_K * K(v), where M = mass matrix, K = stiffness matrix, v = velocity
a += gamma/(beta*dt) * (eta_M*rho*inner(w, dw)*dx + eta_K*inner(sigma(w), eps(dw))*dx)
v_res = wh_dot_n + dt*wh_ddot_n*(1-gamma) - gamma/(beta*dt)*(wh_n + dt*wh_dot_n) + dt*gamma*wh_ddot_n*(2*beta-1)/(2*beta)
L -= (eta_M*rho* inner(v_res, dw)*dx + eta_K * inner(sigma(v_res), eps(dw))*dx)


# Impose normal displacement weakly with Nitsche
# h = dfx.cpp.mesh.h(mesh._cpp_object, tdim, np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)).max()
h = ufl.CellDiameter(mesh)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
t = lambda w: dot(sigma(w), n)
w_normal = dfx.fem.Constant(mesh, 0.0)
phi = dfx.fem.Constant(mesh, 200000.0)

tags = (LEFT, RIGHT)
a += -inner(dot(t(w), n), dot(dw, n))*ds(tags)
a += -inner(dot(w, n), dot(t(dw),n))*ds(tags) \
    + phi/h*inner(dot(w, n), dot(dw, n))*ds(tags)

L += -inner(w_normal, dot(t(dw), n))*ds(tags) \
    + phi/h*inner(w_normal, dot(dw, n))*ds(tags)

anchor_dofs = dfx.fem.locate_dofs_topological(W, tdim-1, ft.find(BOT))
# anchor_dofs = dfx.fem.locate_dofs_topological(W, tdim-1, np.concatenate((ft.find(LEFT), ft.find(TOP), ft.find(BOT))))
bcs = [dfx.fem.dirichletbc(zero, anchor_dofs)]
# bcs = []

r = ufl.SpatialCoordinate(mesh)
chi = r + wh
F = ufl.grad(chi)
J = ufl.det(F)

area_form = dfx.fem.form(1*J*dx)
print("Initial area: ", dfx.fem.assemble_scalar(area_form))
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

xdmf = dfx.io.XDMFFile(comm, f"../output/square-mesh/deformation_nitsche/displacement_dt={timestep:.4g}_T={T:.4g}.xdmf", "w")
print(f"Output dir: ../output/square-mesh/deformation_nitsche/displacement_dt={timestep:.4g}_T={T:.4g}.xdmf")
xdmf_vel = dfx.io.XDMFFile(comm, f"../output/square-mesh/deformation_nitsche/displacement_velocity_dt={timestep:.4g}_T={T:.4g}.xdmf", "w")
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

vh_cpoint_filename = f"../output/square-mesh/deformation_nitsche/checkpoints/displacement_velocity_dt={timestep:.4g}_T={T:.4g}/"
a4d.write_mesh(filename=vh_cpoint_filename, mesh=mesh)
a4d.write_meshtags(vh_cpoint_filename, mesh, ft, meshtag_name='ft')

A, b = assemble_system(A, b, a_cpp, L_cpp, bcs)

from scipy.signal.windows import tukey
from scipy.fft import fft, ifft, fftfreq
period_times = np.linspace(0, period, int(period/dt.value)+1)
window = tukey(len(period_times), alpha=0.25)
right_disp_ = np.copy(window) * np.sin(2*np.pi*period_times)
half = int(len(right_disp_)/2)
right_disp_[:half] = window[:half]*np.sin(2*np.pi*period_times[:half])
right_disp_[half:] = np.sin(2*np.pi*period_times[half:])
right_disp_second = np.sin(2*np.pi*period_times)
right_disp_ = 5e-2*np.concatenate((right_disp_,
                                   right_disp_second[1:],
                                   right_disp_second[1:],
                                   right_disp_second[1:],
                                   right_disp_second[1:]))

displacements = np.zeros((len(wh.x.array), len(period_times)), wh.x.array.dtype)

for i, t in enumerate(times):
    
    tt.value = t

    w_normal.value = -right_disp_[i]

    # Assemble linear system 
    with b.localForm() as local_b_vec: local_b_vec.set(0.0)
    assemble_vector(b, L_cpp)
    apply_lifting(b, [a_cpp], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dfx.fem.set_bc(b, bcs)

    # Solve
    solver.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward() # MPI communication
    if i % 5 == 0:
        print(f"\nTime t = {t:.5g}")
        print("Area: ", dfx.fem.assemble_scalar(area_form))

    # Update functions
    wh_ddot.interpolate(acc_expr)
    wh_dot.interpolate(vel_expr)
    wh_ddot_n.x.array[:] = wh_ddot.x.array.copy()
    wh_dot_n.x.array[:]  = wh_dot.x.array.copy() 
    wh_n.x.array[:] = wh.x.array.copy()

    # Project velocity if in the final period
    if t >= final_period_start:
        # Write XDMF output and checkpoint
        wh_out.interpolate(wh)
        xdmf.write_function(wh_out, write_time)
        a4d.write_function(filename=vh_cpoint_filename, u=wh, time=write_time)

        displacements[:, write_time] = wh.x.array.copy()

        write_time += 1

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

# 4. Compute the Inverse FFT to get the velocity back in the time domain
# .real is used to discard any tiny imaginary parts from numerical error
velocity = ifft(wh_dot_fft, axis=1).real

u_dg = dfx.fem.Function(dfx.fem.functionspace(mesh, element("DG", mesh.basix_cell(), k, shape=(mesh.geometry.dim,))))
dg_vtx = dfx.io.VTXWriter(comm, "../output/square-mesh/deformation_nitsche/dg_vel.bp", [u_dg], "BP5")
projection_problem = projection_problem_CG_to_BDM(wh_dot, dw_dt_bdm, dx)
write_time = 0
for i, t in enumerate(period_times):
    
    # Interpolate the velocity into CG1 and write XDMF output
    wh_dot.x.array[:] = velocity[:, write_time]
    projection_problem.solve()
    u_dg.interpolate(dw_dt_bdm)
    dg_vtx.write(write_time)
    if i==0:
        initial_velocity = dw_dt_bdm.x.array.copy()
    
    vh_out.interpolate(wh_dot)
    xdmf_vel.write_function(vh_out, write_time) 

    write_time += 1

print(initial_velocity)
print(dw_dt_bdm.x.array.copy())
print(np.allclose(dw_dt_bdm.x.array.copy(), initial_velocity))

xdmf.close()
xdmf_vel.close()
dg_vtx.close()