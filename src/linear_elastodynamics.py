import ufl
import sys

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from ufl       import inner, grad, sym
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from pathlib   import Path
from dolfinx.fem.petsc    import create_matrix, create_vector, assemble_vector, apply_lifting
from utilities.fem        import assemble_system
from utilities.projection import projection_problem_CG_to_BDM
from utilities.deformation_data import (DisplacementCorpusCallosumCephalocaudal, 
                                        DisplacementCaudateNucleusHeadLateral,
                                        DisplacementThirdVentricleLateral)

""" Solve the time-dependent equations of linear elasticity with
    a finite element method based on continuous Lagrange elements.
    Time-stepping scheme: Newmark beta-method.
"""

print = PETSc.Sys.Print # Only print from rank 0

# Mesh tags
CANAL_WALL = 13
FOURTH_VENTRICLE_WALL = 18
CANAL_OUT  = 23
LATERAL_APERTURES = 28
CORPUS_CALLOSUM = 110
THIRD_RIGHT = 111
THIRD_LEFT = 112
LATERAL_RIGHT = 113
LATERAL_LEFT = 114
THIRD_ANTERIOR = 115
THIRD_POSTERIOR = 116

write_output = int(sys.argv[1])
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

# Material parameters
E = float(sys.argv[3]) #3156 # Modulus of elasticity [Pa]
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
final_period_start = int(T - period)
write_time = 0

# Definte finite element and function space
p = int(sys.argv[2])
vec_el = element("Lagrange", mesh.basix_cell(), p, shape=(mesh.geometry.dim,))
W = dfx.fem.functionspace(mesh, vec_el)

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
eta_M = dfx.fem.Constant(mesh, 0.50) # Damping proportional to inertia
eta_K = dfx.fem.Constant(mesh, 0.25) # Damping proportional to stiffness 

# The weak form
a = rho/(beta*dt**2)*inner(w, dw)*dx + inner(sigma(w), eps(dw)) * dx
L = rho*inner(wh_n/(beta*dt**2) + wh_dot_n/(beta*dt) + (1-2*beta)/(2*beta)*wh_ddot_n, dw) * dx 

# Add Rayleigh damping: eta_M * M(v) + eta_K * K(v), where M = mass matrix, K = stiffness matrix, v = velocity
a += gamma/(beta*dt) * (eta_M*rho*inner(w, dw)*dx + eta_K*inner(sigma(w), eps(dw))*dx)
v_res = wh_dot_n + dt*wh_ddot_n*(1-gamma) - gamma/(beta*dt)*(wh_n + dt*wh_dot_n) + dt*gamma*wh_ddot_n*(2*beta-1)/(2*beta)
L -= (eta_M*rho* inner(v_res, dw)*dx + eta_K * inner(sigma(v_res), eps(dw))*dx)
    
# Create linear system
a_cpp, L_cpp = dfx.fem.form(a), dfx.fem.form(L)
A = create_matrix(a_cpp)
b = create_vector(L_cpp)

# Set BCs:
# Corpus callosum (roof of LV): impose z displacement, x and y traction-free
# Caudate nucleus (lateral sides of LV): impose x displacement, y and z traction-free
# Lateral walls 3V: impose x displacement, y and z traction-free
# Perimeter of spinal canal outlet (foramen magnum): anchor x and y, z traction-free
# Rest of the boundary is traction-free

# Get displacement expressions
cc_disp_expr = DisplacementCorpusCallosumCephalocaudal(period=period, timestep=timestep, final_time=T)
tv_disp_expr = DisplacementThirdVentricleLateral(period=period, timestep=timestep, final_time=T)
lv_disp_expr = DisplacementCaudateNucleusHeadLateral(period=period, timestep=timestep, final_time=T)

# Create finite element functions for the BCs
cc_disp_func = dfx.fem.Function(W)
tv_disp_func_right = dfx.fem.Function(W)
tv_disp_func_left  = dfx.fem.Function(W)
lv_disp_func_right = dfx.fem.Function(W)
lv_disp_func_left  = dfx.fem.Function(W)
bcs = []

cc_dofs = dfx.fem.locate_dofs_topological((W_z, W), facet_dim, ft.find(CORPUS_CALLOSUM))
bcs.append(dfx.fem.dirichletbc(cc_disp_func, cc_dofs, W_z))

lv_dofs_right = dfx.fem.locate_dofs_topological((W_x, W), facet_dim, ft.find(LATERAL_RIGHT))
bcs.append(dfx.fem.dirichletbc(lv_disp_func_right, lv_dofs_right, W_x))

lv_dofs_left  = dfx.fem.locate_dofs_topological((W_x, W), facet_dim, ft.find(LATERAL_LEFT))
bcs.append(dfx.fem.dirichletbc(lv_disp_func_left, lv_dofs_left, W_x))

tv_dofs_right = dfx.fem.locate_dofs_topological((W_x, W), facet_dim, ft.find(THIRD_RIGHT))
bcs.append(dfx.fem.dirichletbc(tv_disp_func_right, tv_dofs_right, W_x))

tv_dofs_left  = dfx.fem.locate_dofs_topological((W_x, W), facet_dim, ft.find(THIRD_LEFT))
bcs.append(dfx.fem.dirichletbc(tv_disp_func_left, tv_dofs_left, W_x))

# Anchoring the perimeter of the canal outlet requires
# finding the vertices on this perimeter, which
# are the vertices that neighbor facets of
# the two different boundaries (canal wall and canal outlet)

# First get the facets of the outlet and the canal wall
outlet_facets = ft.indices[ft.values==CANAL_OUT]
wall_facets = ft.indices[ft.values==CANAL_WALL]

# Get the connectivity from facets (dim-1) to vertices (dim=0)
mesh.topology.create_connectivity(facet_dim, 0)
f_to_v = mesh.topology.connectivity(facet_dim, 0)

# Find the outlet and wall vertices in a manner that
# is consistent for parallel communication
if len(outlet_facets) > 0:
    local_outlet_vertices = np.unique(np.concatenate([f_to_v.links(f) for f in outlet_facets]))
else:
    local_outlet_vertices = np.array([], dtype=np.int32)

if len(wall_facets) > 0:
    local_wall_vertices = np.unique(np.concatenate([f_to_v.links(f) for f in wall_facets]))
else:
    local_wall_vertices = np.array([], dtype=np.int32)

# The perimeter vertices are the intersection of the two sets of vertices.
# Gather all local vertex arrays from all processes and
# create a single global array of unique vertices.
# Then use the global arrays to compute intersection globally
all_outlet_vertices_list = comm.allgather(local_outlet_vertices) 
global_outlet_vertices = np.unique(np.concatenate(all_outlet_vertices_list)) 
all_wall_vertices_list = comm.allgather(local_wall_vertices)
global_wall_vertices = np.unique(np.concatenate(all_wall_vertices_list))
outlet_perimeter_vertices = np.intersect1d(global_outlet_vertices, global_wall_vertices)

# Find which outlet_perimeter_vertices are owned locally on the process
owned_vertex_range = mesh.topology.index_map(0).local_range # Get the start and end+1 of the owned vertex range on this process

# Create a boolean mask to filter the vertices
is_owned = (outlet_perimeter_vertices >= owned_vertex_range[0]) & \
           (outlet_perimeter_vertices < owned_vertex_range[1])

# Apply the mask to get only the vertices owned by this process
local_perimeter_vertices = outlet_perimeter_vertices[is_owned]

# Now find the outlet perimeter dofs that are located at the vertices found
mesh.topology.create_connectivity(0, mesh.topology.dim) # Create connectivity from vertices (dim=0) to cells (dim)
outlet_perimeter_dofs_x = dfx.fem.locate_dofs_topological((W_x, W), 0, local_perimeter_vertices)
outlet_perimeter_dofs_y = dfx.fem.locate_dofs_topological((W_y, W), 0, local_perimeter_vertices)

# Set the BCs
bcs.append(dfx.fem.dirichletbc(zero, outlet_perimeter_dofs_x, W_x))
bcs.append(dfx.fem.dirichletbc(zero, outlet_perimeter_dofs_y, W_y))

# Assemble the system matrix and the RHS vector
A, b = assemble_system(A, b, a_cpp, L_cpp, bcs)

# Configure direct linear solver using MUMPS.
# The system matrix is singular due to rigid body motions
# not being sufficiently constrained by the BCs, so
# we set some options that enable MUMPS to handle the nullspace of A
opts = PETSc.Options()
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (rigid motions nullspace)
opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (rigid motions nullspace)
opts["ksp_error_if_not_converged"] = 1 # Throw an error if KSP solver does not converge

# Create the solver object, set options and enable convergence monitoring
solver = PETSc.KSP().create(comm)
solver.setOperators(A)
solver.setFromOptions()

# Prepare output functions and files
CG1_vector_space = dfx.fem.functionspace(mesh,
                                         element=element("Lagrange",
                                                         mesh.basix_cell(),
                                                         degree=1,
                                                         shape=(mesh.geometry.dim,)))
wh_out = dfx.fem.Function(CG1_vector_space)
vh_out = dfx.fem.Function(CG1_vector_space)
k = 1 # BDM element degree
bdm_el = element("BDM", mesh.basix_cell(), k)
BDM = dfx.fem.functionspace(mesh, bdm_el)
dw_dt_bdm = dfx.fem.Function(BDM)
dw_dt_bdm.name = "defo_velocity"
wh.name = "defo_displacement"
output_dir = f"../output/mesh_{mesh_suffix}/deformation_p={p}_E={E:.0f}_T={T:.0f}/"
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

# Create projection problem to project the Lagrange
# displacement velocity into a Brezzi-Douglas-Marini
# finite element space
projection_problem = projection_problem_CG_to_BDM(wh_dot, dw_dt_bdm, dx)

# Define some quantities that will be calculated during
# the solution loop.
# Energy norms
E_kinetic = dfx.fem.form(1/2*rho * inner(wh_n, wh_n) * dx)
E_elastic = dfx.fem.form(1/2 * inner(sigma(wh_n), eps(wh_n)) * dx)

# Data arrays for energy norms, max displacement magnitudes
# and the displacements at a point
energy = np.zeros((len(times), 2))
max_disp = np.zeros((len(times), 4))
point_disp = np.zeros((len(times), 3))

# Compute cells for point evaluation of the deformation function wh
cell = []
point_on_proc = []
random_point = np.array([-0.00271071, -0.02997235, -0.04644623])
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

for i, t in enumerate(times[1:], 1):
    
    print(f"\nTime t = {t:.5g}")

    # Update displacement BCs
    cc_disp_expr.increment_index(t)
    cc_disp_expr()
    cc_disp_func.x.array[:] = cc_disp_expr.amplitude

    tv_disp_expr.increment_index(t)
    tv_disp_expr()
    tv_disp_func_right.x.array[:] = tv_disp_expr.amplitude
    tv_disp_func_left.x.array[:] = -1.0*tv_disp_expr.amplitude

    lv_disp_expr.increment_index(t)
    lv_disp_expr()
    lv_disp_func_right.x.array[:] = lv_disp_expr.amplitude
    lv_disp_func_left.x.array[:] = -1.0*lv_disp_expr.amplitude
    
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

    max_disp[i, 0] = comm.allreduce(wh.sub(0).collapse().x.array.max(), op=MPI.MAX)
    max_disp[i, 1] = comm.allreduce(wh.sub(1).collapse().x.array.max(), op=MPI.MAX)
    max_disp[i, 2] = comm.allreduce(wh.sub(2).collapse().x.array.max(), op=MPI.MAX)
    max_disp[i, 3] = comm.allreduce(np.sqrt(wh.sub(0).collapse().x.array**2
                                + wh.sub(1).collapse().x.array**2
                                + wh.sub(2).collapse().x.array**2).max(),
                                op=MPI.MAX)
    
    # Store the applied BCs (equal on all processes)
    applied_bc1.append(cc_disp_expr.amplitude)
    applied_bc2.append(tv_disp_expr.amplitude)
    applied_bc3.append(lv_disp_expr.amplitude)

    if t >= final_period_start and write_output:
        # Project velocity and write output
        projection_problem.solve() # Project deformation velocity into BDM 1 space for checkpointing
        
        # Write checkpoints
        a4d.write_function(filename=vh_cpoint_filename, u=wh, time=write_time)
        a4d.write_function(filename=vh_cpoint_filename, u=dw_dt_bdm, time=write_time)
        
        # Interpolate P1 functions and write XDMF output
        wh_out.interpolate(wh)
        vh_out.interpolate(wh_dot) 
        xdmf.write_function(wh_out, t)
        xdmf_vel.write_function(vh_out, t) 

        write_time += 1 # Increment write index

if write_output:
    xdmf.close()
    xdmf_vel.close()

# Perform parallell communication
point_disp = comm.gather(point_disp, root=0)

if comm.rank==0:
    
    # Get the point displacements that were gathered at this rank
    for array in point_disp:
        if np.sum(np.abs(array)) > 0.0:
            point_disp = array
            break
    
    # Plot and save data arrays
    import matplotlib.pyplot as plt
    fig_dir = f"../output/illustrations/verification/mesh_{mesh_suffix}/"
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=([12, 8]))
    ax.plot(times[1:], np.array(applied_bc1), 'r', label="Corpus callosum")
    ax.plot(times[1:], np.array(applied_bc2), 'b', label="3V wall")
    ax.plot(times[1:], np.array(applied_bc3), 'g', label="Caudate nucleus")
    ax.legend()
    fig.savefig(fig_dir+f"applied_BCs_deformation_p={p}_E={E:.0f}_dt={timestep}.png")
    plt.show()
    np.save(data_dir+f"applied_bc_corpus_callosum_dt={timestep}.npy", np.array(applied_bc1))
    np.save(data_dir+f"applied_bc_3V_wall_dt={timestep}.npy", np.array(applied_bc2))
    np.save(data_dir+f"applied_bc_caudate_nucleus_dt={timestep}.npy", np.array(applied_bc3))

    fig2, ax2 = plt.subplots(figsize=([12, 8]))
    ax2.plot(times, energy[:, 0], 'r', label="Kinetic energy")
    ax2.plot(times, energy[:, 1], 'b', label="Elastic energy")
    ax2.legend()
    fig2.savefig(fig_dir+f"MPI={comm.size}_energies_p={p}_E={E:.0f}_dt={timestep}.png")
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
    fig3.savefig(fig_dir+f"MPI={comm.size}_displacements_p={p}_E={E:.0f}_dt={timestep}.png")
    plt.show()
    np.save(data_dir+f"max_displacements_dt={timestep}.npy", max_disp)
    np.save(data_dir+f"point_displacements_dt={timestep}.npy", point_disp)