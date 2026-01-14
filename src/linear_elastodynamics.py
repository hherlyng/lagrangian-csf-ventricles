import gc
import ufl
import sys

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from ufl       import inner, grad, sym, dot
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from pathlib   import Path
from scipy.fft import fft, ifft, fftfreq
from dolfinx.fem.petsc    import create_matrix, create_vector, assemble_vector, apply_lifting
from utilities.fem        import assemble_system
from utilities.projection import projection_problem_CG_to_BDM
from utilities.deformation_data import (DisplacementCorpusCallosumCephalocaudal, 
                                        DisplacementCaudateNucleusHeadLateral,
                                        DisplacementThirdVentricleLateral,
                                        DisplacementLateralVentricleHorns,
                                        DisplacementVentricleFloorCephalocaudal)

""" Solve the time-dependent equations of (damped) linear elasticity with
    a finite element method based on continuous Lagrange elements.

    Time-stepping scheme: Newmark beta-method.
"""

print = PETSc.Sys.Print # Only print from rank 0

# Form compilation optimization options
jit_options = {"cffi_extra_compile_args": ["-O3", "-march=native"]}

# Facet tags
CANAL_WALL = 13 
CANAL_OUT = 23
THIRD_VENTRICLE_WALL = 14
THIRD_VENTRICLE_FORAMINA = 46
AQUEDUCT_WALL = 15
AQUEDUCT_TOP = 45
AQUEDUCT_BOT = 58
FORAMINA_34_WALL = 16
LATERAL_VENTRICLES_FORAMINA = 67
LATERAL_VENTRICLES_WALL = 17
FOURTH_VENTRICLE_WALL = 18
FOURTH_VENTRICLE_OUT = 38
CHOROID_PLEXUS_LATERAL = 101
CHOROID_PLEXUS_THIRD = 103
CHOROID_PLEXUS_FOURTH = 104
LATERAL_APERTURES = 28
CORPUS_CALLOSUM = 110
THIRD_RIGHT = 111
THIRD_LEFT = 112
LATERAL_RIGHT = 113
LATERAL_LEFT = 114
THIRD_ANTERIOR = 115
THIRD_POSTERIOR = 116
THIRD_FLOOR = 117
LATERAL_FLOOR = 118
ZERO_SOLID_TRACTION = 1000
CHOROID_PLEXUS_LATERAL_ZERO_TRACTION = 1001

# Cell tags
CANAL = 3
THIRD_VENTRICLE = 4
AQUEDUCT = 5
FORAMINA_34 = 6
LATERAL_VENTRICLES = 7
FOURTH_VENTRICLE = 8

write_output = int(sys.argv[1])
p = int(sys.argv[2]) # CG (displacement) element degree
k = int(sys.argv[6]) # BDM element degree
comm = MPI.COMM_WORLD
mesh_suffix = int(sys.argv[5]) # Refinement degree of mesh
with dfx.io.XDMFFile(comm, f"../geometries/ventricles_{mesh_suffix}.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    gdim = mesh.geometry.dim
    
    # Generate mesh entities    
    facet_dim = mesh.topology.dim-1
    mesh.topology.create_entities(facet_dim) # Create facets
    mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity
    ft = xdmf.read_meshtags(mesh, "ft")
    ct = xdmf.read_meshtags(mesh, "ct")

ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft) # Boundary integral measure
dx = ufl.Measure('dx', domain=mesh, subdomain_data=ct) # Volume integral measure
n = ufl.FacetNormal(mesh)

# Material parameters
E = float(sys.argv[3]) # Modulus of elasticity [Pa]
nu = 0.479 # Poisson's ratio [-]
eta_value = 2*E / (1+nu) # First Lamé parameter value
lam_value = nu*E / ((1+nu) * (1-2*nu)) # Second Lamé parameter value
eta = dfx.fem.Constant(mesh, eta_value) # First Lamé parameter
lam = dfx.fem.Constant(mesh, lam_value) # Second Lamé parameter
rho = dfx.fem.Constant(mesh, 1000.0) # Ventricular wall density [kg/m^3]
print("Value of Lamé parameters:")
print(f"eta \t= {eta_value:.2f}\nlambda \t= {lam_value:.2f}")

# Define Generalized-alpha method parameters
gamma = dfx.fem.Constant(mesh, dfx.default_scalar_type(1/2))
beta  = dfx.fem.Constant(mesh, dfx.default_scalar_type(1/4*(gamma.value + 1/2)**2))

# Temporal parameters
timestep = 0.001
dt = dfx.fem.Constant(mesh, timestep) 
period = 1
num_periods = int(sys.argv[4])
T = period*num_periods
N = int(T / timestep)
times = np.linspace(0, T, N+1)
period_times = np.linspace(0, period, int(period/timestep)+1) # One period of timesteps for the output
final_period_start = int(T - period)
write_time = 0

# Definte finite element and function space
vec_el  = element("Lagrange", mesh.basix_cell(), p, shape=(mesh.geometry.dim,))
scal_el = element("Lagrange", mesh.basix_cell(), p)
W = dfx.fem.functionspace(mesh, vec_el)
V = dfx.fem.functionspace(mesh, scal_el)

# Extract subspaces
W_x = W.sub(0) # x displacement space
W_y = W.sub(1) # y displacement space
W_z = W.sub(2) # z displacement space

# Create functions for storing solutions and previous timestep values
wh = dfx.fem.Function(W) # Displacement function
wh_n = dfx.fem.Function(W) # Displacement function at previous timestep
wh_dot = dfx.fem.Function(W) # Velocity function
wh_dot_n = dfx.fem.Function(W) # Velocity function at previous timestep
wh_ddot  = dfx.fem.Function(W) # Acceleration function
wh_ddot_n = dfx.fem.Function(W) # Acceleration function at previous timestep
zero = dfx.fem.Function(W)

# Create expressions for updating the velocity and acceleration
vel = wh_dot_n + dt * ((1 - gamma) * wh_ddot_n + gamma * wh_ddot)
vel_expr = dfx.fem.Expression(vel, W.element.interpolation_points())

acc = 1 / beta / dt**2 * (wh - wh_n - dt * wh_dot_n) + wh_ddot_n * (1 - 1 / 2 / beta)
acc_expr = dfx.fem.Expression(acc, W.element.interpolation_points())

print(f"\nNumber of degrees of freedom: {W.dofmap.index_map.size_global*W.dofmap.index_map_bs}")

# Test and trial functions
w, dw = ufl.TrialFunction(W), ufl.TestFunction(W)

eps   = lambda arg: sym(grad(arg)) # The symmetric gradient
sigma = lambda w: 2.0*eta*eps(w) + lam*ufl.tr(eps(w))*ufl.Identity(mesh.geometry.dim) # Stress tensor

# Damping parameters
eta_M = dfx.fem.Constant(mesh, 0.20) # Damping proportional to inertia
eta_K = dfx.fem.Constant(mesh, 0.10) # Damping proportional to stiffness 

# The weak form
a = rho/(beta*dt**2)*inner(w, dw)*dx + inner(sigma(w), eps(dw)) * dx
L = rho*inner(wh_n/(beta*dt**2) + wh_dot_n/(beta*dt) + (1-2*beta)/(2*beta)*wh_ddot_n, dw) * dx 

# Add Rayleigh damping: eta_M * M(v) + eta_K * K(v), where M = mass matrix, K = stiffness matrix, v = velocity
a += gamma/(beta*dt) * (eta_M*rho*inner(w, dw)*dx + eta_K*inner(sigma(w), eps(dw))*dx)
v_res = wh_dot_n + dt*wh_ddot_n*(1-gamma) - gamma/(beta*dt)*(wh_n + dt*wh_dot_n) + dt*gamma*wh_ddot_n*(2*beta-1)/(2*beta)
L -= (eta_M*rho* inner(v_res, dw)*dx + eta_K * inner(sigma(v_res), eps(dw))*dx)

# Impose normal displacement weakly with Nitsche
h = ufl.CellDiameter(mesh)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
t = lambda w: dot(sigma(w), n)
w_normal_LV_lat = dfx.fem.Constant(mesh, 0.0)
w_normal_LV_cec = dfx.fem.Function(V)
w_normal_LV_horns = dfx.fem.Constant(mesh, 0.0)
w_normal_3V = dfx.fem.Constant(mesh, 0.0)
phi = dfx.fem.Constant(mesh, 20000000.0)

lateral_tags_cec = (CORPUS_CALLOSUM)
lateral_tags_horns = (LATERAL_VENTRICLES_WALL)
lateral_tags_lat = (LATERAL_LEFT, LATERAL_RIGHT)
third_tags = (THIRD_LEFT, THIRD_RIGHT, THIRD_ANTERIOR,
              THIRD_POSTERIOR, CHOROID_PLEXUS_THIRD, THIRD_VENTRICLE_WALL)
              
a += -inner(dot(t(w), n), dot(dw, n))*(ds(lateral_tags_lat) + ds(lateral_tags_cec) + ds(lateral_tags_horns) + ds(third_tags))

a += -inner(dot(w, n), dot(t(dw),n))*(ds(lateral_tags_lat) + ds(lateral_tags_cec) + ds(lateral_tags_horns) + ds(third_tags)) \
    + phi/h*inner(dot(w, n), dot(dw, n))*(ds(lateral_tags_lat) + ds(lateral_tags_cec) + ds(lateral_tags_horns) + ds(third_tags))

L += -inner(w_normal_LV_lat, dot(t(dw), n))*ds(lateral_tags_lat) \
    + phi/h*inner(w_normal_LV_lat, dot(dw, n))*ds(lateral_tags_lat) \
    - inner(w_normal_LV_cec, dot(t(dw), n))*ds(lateral_tags_cec) \
    + phi/h*inner(w_normal_LV_cec, dot(dw, n))*ds(lateral_tags_cec) \
    - inner(w_normal_LV_horns, dot(t(dw), n))*ds(lateral_tags_horns) \
    + phi/h*inner(w_normal_LV_horns, dot(dw, n))*ds(lateral_tags_horns) \
    - inner(w_normal_3V, dot(t(dw), n))*ds(third_tags) \
    + phi/h*inner(w_normal_3V, dot(dw, n))*ds(third_tags)
    
# Create linear system
a_cpp = dfx.fem.form(a, jit_options=jit_options)
L_cpp = dfx.fem.form(L, jit_options=jit_options)
A = create_matrix(a_cpp)
b = create_vector(L_cpp)

# Set BCs:
# Corpus callosum (roof of LV): normal displacement based on data
# Caudate nucleus (lateral sides of LV): normal displacement based on data
# Lateral ventricle horns (occipital and inferior): normal displacement inferred from data
# Lateral walls 3V: normal displacement based on data
# Spinal canal outlet and lateral apertures outlets: anchor (w_x = w_y = w_z = 0)
# Rest of the boundary is traction-free

# Get displacement expressions
cc_disp_expr = DisplacementCorpusCallosumCephalocaudal(period=period, timestep=timestep, final_time=T)
cc_disp_expr.y_max = comm.allreduce(mesh.geometry.x[:, 1].max(), op=MPI.MAX)
tv_disp_expr = DisplacementThirdVentricleLateral(period=period, timestep=timestep, final_time=T)
lv_disp_expr = DisplacementCaudateNucleusHeadLateral(period=period, timestep=timestep, final_time=T)
horns_expr   = DisplacementLateralVentricleHorns(period=period, timestep=timestep, final_time=T)

bcs = []

w_LV_floor = dfx.fem.Function(W)
w_3V_floor = dfx.fem.Function(W)
floor_expr = DisplacementVentricleFloorCephalocaudal(period=period, timestep=timestep, final_time=T)
dofs_LV_floor = dfx.fem.locate_dofs_topological((W_z, W), facet_dim, np.concatenate((ft.find(LATERAL_FLOOR), ft.find(CHOROID_PLEXUS_LATERAL))))
dofs_3V_floor = dfx.fem.locate_dofs_topological((W_z, W), facet_dim, ft.find(THIRD_FLOOR))
bcs.append(dfx.fem.dirichletbc(w_LV_floor, dofs_LV_floor, W_z))
bcs.append(dfx.fem.dirichletbc(w_3V_floor, dofs_3V_floor, W_z))

outlet_facets = np.concatenate((ft.find(CANAL_OUT), ft.find(LATERAL_APERTURES)))
outlet_dofs = dfx.fem.locate_dofs_topological(W, facet_dim, outlet_facets)
bcs.append(dfx.fem.dirichletbc(zero, outlet_dofs))

# Assemble the system matrix and the RHS vector
A, b = assemble_system(A, b, a_cpp, L_cpp, bcs)

# Configure linear solver
solver = PETSc.KSP().create(comm)
solver.setOperators(A)
opts = PETSc.Options()

iterative_solver = True
if iterative_solver:
    # Configure iterative solver using conjugate gradient
    # with hypre boomeramg algebraic multigrid preconditioning
    opts["ksp_type"] = "fgmres"
    opts["pc_type"] = "hypre"
    opts["ksp_rtol"] = 1e-7
    opts["ksp_atol"] = 1e-7
    opts["ksp_initial_guess_nonzero"] = True
else:
    # Configure direct solver MUMPS with exact preconditioner LU
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["pc_factor_mat_solver_type"] = "mumps"
    opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
    opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
    opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
    opts["ksp_error_if_not_converged"] = 1 # Throw an error if KSP solver does not converge

# Enable convergence monitoring and set options
solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, residual: {rnorm}"))
solver.setFromOptions()

# Prepare output functions and files
CG1_vector_space = dfx.fem.functionspace(mesh,
                                         element=element("Lagrange",
                                                         mesh.basix_cell(),
                                                         degree=1,
                                                         shape=(mesh.geometry.dim,)))
wh_out = dfx.fem.Function(CG1_vector_space)
vh_out = dfx.fem.Function(CG1_vector_space)
wh.name = "defo_displacement"
output_dir = f"../output/mesh_{mesh_suffix}/deformation_p={p}_E={E:.0f}_k={k}_T={T:.0f}/"
data_dir = output_dir+"data/"
Path(data_dir).mkdir(parents=True, exist_ok=True)

xdmf = dfx.io.XDMFFile(comm, output_dir+f"displacement_dt={timestep:.4g}.xdmf", "w")
xdmf_vel = dfx.io.XDMFFile(comm, output_dir+f"displacement_velocity_dt={timestep:.4g}.xdmf", "w")
xdmf.write_mesh(mesh)
xdmf_vel.write_mesh(mesh)

if write_output:
    vh_cpoint_filename = output_dir+f"checkpoints/displacement_velocity_dt={timestep:.4g}"
    a4d.write_mesh(filename=vh_cpoint_filename, mesh=mesh)
    a4d.write_meshtags(vh_cpoint_filename, mesh, ft, meshtag_name='ft')
    a4d.write_meshtags(vh_cpoint_filename, mesh, ct, meshtag_name='ct')

# Define some quantities that will be calculated during
# the solution loop.
# Energy norms
E_kinetic = dfx.fem.form(1/2*rho * inner(wh_dot_n, wh_dot_n) * dx, jit_options=jit_options)
E_elastic = dfx.fem.form(1/2 * inner(sigma(wh_n), eps(wh_n)) * dx, jit_options=jit_options)

# Data arrays for energy norms, max displacement magnitudes
# and the displacements at a point
energy = np.zeros((len(times), 2))
max_disp = np.zeros((len(times), 4))
point_disp = np.zeros((len(times), 3))
point_vel  = np.zeros((len(times), 3))
point_wh_dot = np.zeros((len(times), 3))

# Compute cells for point evaluation of the deformation function wh
cell = []
point_on_proc = []
# random_point = np.array([-0.00271071, -0.02997235, -0.04644623])
random_point = np.array([0.0066761 , 0.01322405, 0.01233037])
bb_tree = dfx.geometry.bb_tree(mesh, mesh.topology.dim)
cell_candidates = dfx.geometry.compute_collisions_points(bb_tree, random_point)
colliding_cells = dfx.geometry.compute_colliding_cells(mesh, cell_candidates, random_point)
if len(colliding_cells.links(0))>0:
    cc = colliding_cells.links(0)[0]
    cell.append(cc)
    cell = np.array(cell)

applied_bc1 = []
applied_bc2 = []
applied_bc3 = []
applied_bc4 = []
applied_bc5 = []

displacements = np.zeros((len(wh.x.array), len(period_times)), wh.x.array.dtype)

for i, t in enumerate(times[1:], 1):
    
    print(f"\nTime t = {t:.5g}")

    # Update displacement BCs
    cc_disp_expr.increment_index(t)
    w_normal_LV_cec.interpolate(cc_disp_expr)

    lv_disp_expr.increment_index(t)
    lv_disp_expr()
    w_normal_LV_lat.value = lv_disp_expr.amplitude

    tv_disp_expr.increment_index(t)
    tv_disp_expr()
    w_normal_3V.value = tv_disp_expr.amplitude

    floor_expr.increment_index(t)
    floor_expr()
    w_LV_floor.x.array[:] = floor_expr.amplitude
    w_3V_floor.x.array[:] = floor_expr.amplitude

    horns_expr.increment_index(t)
    horns_expr()
    w_normal_LV_horns.value = horns_expr.amplitude
    
    # Assemble right-hand side
    with b.localForm() as local_b_vec: local_b_vec.set(0.0)
    assemble_vector(b, L_cpp)
    apply_lifting(b, [a_cpp], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dfx.fem.set_bc(b, bcs)

    # Solve and MPI communicate
    solver.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()
    
    # Update functions
    wh_ddot.interpolate(acc_expr)
    wh_dot.interpolate(vel_expr)
    wh_ddot_n.x.array[:] = wh_ddot.x.array.copy()
    wh_dot_n.x.array[:]  = wh_dot.x.array.copy() 
    wh_n.x.array[:] = wh.x.array.copy()

    # Calculate energies
    energy[i, 0] = comm.allreduce(dfx.fem.assemble_scalar(E_kinetic), op=MPI.SUM)
    energy[i, 1] = comm.allreduce(dfx.fem.assemble_scalar(E_elastic), op=MPI.SUM)

    if len(cell)>0:
        point_disp[i, :] = wh.eval(x=random_point, cells=cell)
        point_wh_dot[i, :] = wh_dot.eval(x=random_point, cells=cell)

    max_disp[i, 0] = comm.allreduce(wh.sub(0).collapse().x.array.max(), op=MPI.MAX)
    max_disp[i, 1] = comm.allreduce(wh.sub(1).collapse().x.array.max(), op=MPI.MAX)
    max_disp[i, 2] = comm.allreduce(wh.sub(2).collapse().x.array.max(), op=MPI.MAX)
    max_disp[i, 3] = comm.allreduce(np.sqrt(wh.sub(0).collapse().x.array**2
                                + wh.sub(1).collapse().x.array**2
                                + wh.sub(2).collapse().x.array**2).max(),
                                op=MPI.MAX)
    print(f"Maximum displacement magnitude: {max_disp[i, 3]:.1e}")
    
    # Store the applied BCs (equal on all processes)
    applied_bc1.append(3/4*cc_disp_expr.amplitude) # Add the mean value of the BC
    applied_bc2.append(tv_disp_expr.amplitude)
    applied_bc3.append(lv_disp_expr.amplitude)
    applied_bc4.append(floor_expr.amplitude)
    applied_bc5.append(horns_expr.amplitude)

    if t >= final_period_start and write_output:
        
        # Write displacement checkpoint
        a4d.write_function(filename=vh_cpoint_filename, u=wh, time=write_time)
        
        # Interpolate P1 displacement function and write XDMF output
        wh_out.interpolate(wh)
        xdmf.write_function(wh_out, t)

        displacements[:, write_time] = wh.x.array.copy() # Store the displacements to be differentiated with FFT

        write_time += 1 # Increment write index

# Close displacement xdmf
xdmf.close()

# Free memory
del A
del b
del solver
gc.collect()

wh_project = dfx.fem.Function(W)
bdm_el = element("BDM", mesh.basix_cell(), k)
BDM = dfx.fem.functionspace(mesh, bdm_el)
dw_dt_bdm = dfx.fem.Function(BDM)
dw_dt_bdm.name = "defo_velocity"

# Create projection problem to project the Lagrange
# displacement velocity into a Brezzi-Douglas-Marini
# finite element space
projection_problem = projection_problem_CG_to_BDM(wh_project, dw_dt_bdm,
                                                  dx, jit_options=jit_options)

if write_output:

    # Calculate the displacement velocity with the Fast Fourier Transform (FFT)
    # Get the frequencies of the signal
    n_steps = displacements.shape[1]
    freqs = fftfreq(n_steps, dt.value)

    # Compute the Fast Fourier Transform of the displacement
    wh_fft = fft(displacements, axis=1)

    del displacements # Free memory
    gc.collect()

    # Differentiate in frequency space by multiplying by (i * omega)
    # where omega = 2*pi*f and i is the imaginary unit
    wh_dot_fft = (1j*2*np.pi*freqs) * wh_fft

    # Apply a Gaussian filter to filter out high-frequency noise
    sigma = 10.0 # a.k.a. sigma in a Gaussian
    filter = np.exp(-(freqs**2) / (2 * sigma**2))
    wh_dot_fft = filter*wh_dot_fft

    # Compute the inverse FFT to get the velocity in the time domain
    velocity = ifft(wh_dot_fft, axis=1).real # Take only the real part

    # Reset write time and loop over one period
    for write_time, _ in enumerate(period_times):
        # Update the finite element function for the velocity
        # and project the CG velocity into a BDM space
        wh_project.x.array[:] = velocity[:, write_time]
        wh_project.x.scatter_forward()
        projection_problem.solve()
        a4d.write_function(filename=vh_cpoint_filename, u=dw_dt_bdm, time=write_time)
        vh_out.interpolate(wh_project)
        xdmf_vel.write_function(vh_out, write_time) 

        if len(cell)>0:
            point_vel[write_time, :] = wh_project.eval(x=random_point, cells=cell)
    
    # Close output file
    xdmf_vel.close()

    # Perform parallell communication
    point_disp = comm.gather(point_disp, root=0)
    point_vel  = comm.gather(point_vel , root=0)
    point_wh_dot = comm.gather(point_wh_dot, root=0)

    if comm.rank==0:
        
        # Get the point displacements and velocities that were gathered at this rank
        for array in point_disp:
            if np.sum(np.abs(array)) > 0.0:
                point_disp = array
                break
        for array in point_vel:
            if np.sum(np.abs(array)) > 0.0:
                point_vel = array
                break
        for array in point_wh_dot:
            if np.sum(np.abs(array)) > 0.0:
                point_wh_dot = array
                break
        
        # Plot and save data arrays
        import matplotlib.pyplot as plt
        fig_dir = f"../output/illustrations/verification/mesh_{mesh_suffix}/"
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=([12, 8]))
        ax.plot(times[1:], np.array(applied_bc1), 'r', label="Corpus callosum")
        ax.plot(times[1:], np.array(applied_bc2), 'b', label="3V wall")
        ax.plot(times[1:], np.array(applied_bc3), 'g', label="Caudate nucleus")
        ax.plot(times[1:], np.array(applied_bc4), 'm', label="LV/3V floor")
        ax.plot(times[1:], np.array(applied_bc5), 'k', label="Inf./occ. horns")
        ax.legend()
        fig.savefig(fig_dir+f"applied_BCs_deformation_p={p}_E={E:.0f}_k={k}_T={T}_dt={timestep}.png")
        plt.show()
        np.save(data_dir+f"applied_bc_corpus_callosum_dt={timestep}.npy", np.array(applied_bc1))
        np.save(data_dir+f"applied_bc_3V_wall_dt={timestep}.npy", np.array(applied_bc2))
        np.save(data_dir+f"applied_bc_caudate_nucleus_dt={timestep}.npy", np.array(applied_bc3))
        np.save(data_dir+f"applied_bc_LV_3V_floor_dt={timestep}.npy", np.array(applied_bc4))
        np.save(data_dir+f"applied_bc_inf_occ_horns_dt={timestep}.npy", np.array(applied_bc5))

        fig2, ax2 = plt.subplots(figsize=([12, 8]))
        ax2_ = ax2.twinx()
        ax2.plot(times, energy[:, 0], 'r', label="Kinetic energy")
        ax2_.plot(times, energy[:, 1], color='b', label="Elastic energy", marker='^', markevery=int(len(times)/25))
        ax2.legend()
        ax2_.legend()
        fig2.savefig(fig_dir+f"MPI={comm.size}_energies_p={p}_E={E:.0f}_k={k}_T={T}_dt={timestep}.png")
        plt.show()
        np.save(data_dir+f"energies_dt={timestep}.npy", energy)

        fig3, ax3 = plt.subplots(figsize=([12, 8]))
        ax3.plot(times, max_disp[:, 0], 'k', label="Max x disp")
        ax3.plot(times, max_disp[:, 1], 'r', label="Max y disp")
        ax3.plot(times, max_disp[:, 2], 'b', label="Max z disp")
        ax3.plot(times, max_disp[:, 3], color='k', linestyle='-.', label="Max mag.")
        ax3.plot(times, point_disp[:, 0], 'g', label="Point x disp")
        ax3.plot(times, point_disp[:, 1], 'c', label="Point y disp")
        ax3.plot(times, point_disp[:, 2], 'm', label="Point z disp")
        ax3.legend()
        fig3.savefig(fig_dir+f"MPI={comm.size}_displacements_p={p}_E={E:.0f}_k={k}_T={T}_dt={timestep}.png")
        plt.show()
        np.save(data_dir+f"max_displacements_dt={timestep}.npy", max_disp)
        np.save(data_dir+f"point_displacements_dt={timestep}.npy", point_disp)

        fig4, ax4 = plt.subplots(figsize=([12, 8]))
        ax4.plot(times, point_vel[:, 0], 'g', label="Point x vel")
        ax4.plot(times, point_vel[:, 1], 'c', label="Point y vel")
        ax4.plot(times, point_vel[:, 2], 'm', label="Point z vel")
        ax4.plot(times, point_wh_dot[:, 2], 'k', label="Point z vel (Newmark)")
        ax4.legend()
        fig4.savefig(fig_dir+f"MPI={comm.size}_velocities_p={p}_E={E:.0f}_k={k}_T={T}_dt={timestep}.png")
        np.save(data_dir+f"point_velocities_dt={timestep}.npy", point_vel)
        np.save(data_dir+f"point_wh_dots_dt={timestep}.npy", point_wh_dot)