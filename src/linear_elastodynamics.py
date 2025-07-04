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
                                        DisplacementCanalAndFourthVentricleAnteroposterior,
                                        DisplacementThirdVentricleLateral)

# print = PETSc.Sys.Print

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

# Solve linear elasticity equation on the ventricles. Wall motion is 
# prescribed in time at a single point (close to corpus callosum).
write_output = int(sys.argv[1])
comm = MPI.COMM_WORLD
mesh_prefix = 'medium'
with dfx.io.XDMFFile(comm, f"../geometries/{mesh_prefix}_ventricles_mesh_tagged.xdmf", "r") as xdmf:
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
eps = lambda arg: sym(grad(arg)) # The symmetric gradient

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
T = 0.005
period = 1
N = int(T / timestep)
times = np.linspace(0, T, N+1)
final_period_start = int(T - period)
write_time = 0

# Finite elements
p = int(sys.argv[2])
vec_el = element("Lagrange", mesh.basix_cell(), p, shape=(mesh.geometry.dim,))
W = dfx.fem.functionspace(mesh, vec_el)
W_x = W.sub(0) # x displacement space
W_y = W.sub(1) # y displacement space
W_z = W.sub(2) # z displacement space
wh = dfx.fem.Function(W) # Displacement function
wh_n = dfx.fem.Function(W) # Displacement function at previous timestep
wh_dot = dfx.fem.Function(W) # Velocity function
wh_dot_n = dfx.fem.Function(W) # Velocity function at previous timestep
wh_ddot  = dfx.fem.Function(W) # Acceleration function
wh_ddot_n = dfx.fem.Function(W) # Acceleration function at previous timestep
zero = dfx.fem.Function(W)

acc = 1 / beta / dt**2 * (wh - wh_n - dt * wh_dot_n) + wh_ddot_n * (1 - 1 / 2 / beta)
acc_expr = dfx.fem.Expression(acc, W.element.interpolation_points())

vel = wh_dot_n + dt * ((1 - gamma) * wh_ddot_n + gamma * wh_ddot)
vel_expr = dfx.fem.Expression(vel, W.element.interpolation_points())

print(f"\nNumber of degrees of freedom: {W.dofmap.index_map.size_global*W.dofmap.index_map_bs}")

# Test and trial functions
w, dw = ufl.TrialFunction(W), ufl.TestFunction(W)

# Stress tensor
sigma = lambda w: 2.0*eta*eps(w) + lam*ufl.tr(eps(w))*ufl.Identity(mesh.geometry.dim)

# The weak form
a = rho/(beta*dt**2)*inner(w, dw)*dx + inner(sigma(w), eps(dw)) * dx
L = rho*inner(wh_n/(beta*dt**2) + wh_dot_n/(beta*dt) + (1-2*beta)/(2*beta)*wh_ddot_n, dw) * dx 

# Dirichlet BCs on corpus callosum and canal wall
cc_disp_expr = DisplacementCorpusCallosumCephalocaudal(period=period, timestep=timestep)
cwfv_disp_expr = DisplacementCanalAndFourthVentricleAnteroposterior(period=period, timestep=timestep)
tv_disp_expr = DisplacementThirdVentricleLateral(period=period, timestep=timestep)
lv_disp_expr = DisplacementCaudateNucleusHeadLateral(period=period, timestep=timestep)
cc_disp_func = dfx.fem.Function(W)
cwfv_disp_func = dfx.fem.Function(W)
tv_disp_func_right = dfx.fem.Function(W)
tv_disp_func_left  = dfx.fem.Function(W)
lv_disp_func_right = dfx.fem.Function(W)
lv_disp_func_left  = dfx.fem.Function(W)
bcs = []

# Set BCs
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

cwfv_facets = np.concatenate((ft.find(FOURTH_VENTRICLE_WALL), ft.find(CANAL_WALL)))
cwfv_antpost_dofs = dfx.fem.locate_dofs_topological((W_y, W), facet_dim, cwfv_facets)
bcs.append(dfx.fem.dirichletbc(cwfv_disp_func, cwfv_antpost_dofs, W_y))

# Zero displacement in x direction on the lower part of the geometry
cwfv_lateral_dofs = dfx.fem.locate_dofs_topological((W_x, W), facet_dim, cwfv_facets)
bcs.append(dfx.fem.dirichletbc(zero, cwfv_lateral_dofs, W_x))
    
# Create linear system
a_cpp, L_cpp = dfx.fem.form(a), dfx.fem.form(L)
A = create_matrix(a_cpp)
b = create_vector(L_cpp)

# Configure linear solver based on
# conjugate gradient with algebraic multigrid preconditioning
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
output_dir = f"../output/{mesh_prefix}-mesh/deformation_p={p}_E={E:.0f}/"
data_dir = output_dir+"data/"
Path(data_dir).mkdir(parents=True, exist_ok=True)

xdmf = dfx.io.XDMFFile(comm, output_dir+f"displacement_p={p}_E={E:.0f}_dt={timestep:.4g}_T={T:.0f}.xdmf", "w")
xdmf_vel = dfx.io.XDMFFile(comm, output_dir+f"displacement_p={p}_E={E:.0f}_velocity_dt={timestep:.4g}_T={T:.0f}.xdmf", "w")
xdmf.write_mesh(mesh)
xdmf_vel.write_mesh(mesh)

if write_output:
    vh_cpoint_filename = output_dir+f"checkpoints/displacement_velocity_dt={timestep:.4g}_T={T:.0f}/"
    a4d.write_mesh(filename=vh_cpoint_filename, mesh=mesh)
    a4d.write_meshtags(vh_cpoint_filename, mesh, ft, meshtag_name='ft')
    a4d.write_meshtags(vh_cpoint_filename, mesh, ct, meshtag_name='ct')

projection_problem = projection_problem_CG_to_BDM(wh_dot, dw_dt_bdm, dx)

A, b = assemble_system(A, b, a_cpp, L_cpp, bcs)

# Energy norms
E_kinetic = dfx.fem.form(1/2*rho * inner(wh_n, wh_n) * dx)
E_elastic = dfx.fem.form(1/2 * inner(sigma(wh_n), eps(wh_n)) * dx)

# Prepare data arrays
energy = np.zeros((len(times), 2))
max_disp = np.zeros((len(times), 4))
point_disp = np.zeros((len(times), 3))

# Compute cells for point evaluation of the deformation function wh
cell = []
point_on_proc = []
random_point = np.array([0.000435316, 0.0346574, 0.0150221])
# random_point = np.array([-0.00013767, 0.00550183, 0.00806552])
bb_tree = dfx.geometry.bb_tree(mesh, mesh.topology.dim)
cell_candidates = dfx.geometry.compute_collisions_points(bb_tree, random_point)
colliding_cells = dfx.geometry.compute_colliding_cells(mesh, cell_candidates, random_point)
if len(colliding_cells.links(0)>0):
    cc = colliding_cells.links(0)[0]
    cell.append(cc)
    cell = np.array(cell)
    point_on_proc.append(random_point)
    point_on_proc = np.array(point_on_proc)[0]

applied_bc1 = []
applied_bc2 = []
applied_bc3 = []
applied_bc4 = []

for i, t in enumerate(times[1:], 1):
    
    print(f"\nTime t = {t:.5g}")

    # Update displacement BCs
    cc_disp_expr.increment_index(t)
    cc_disp_expr()
    cc_disp_func.x.array[:] = cc_disp_expr.amplitude
    cwfv_disp_expr.increment_index(t)
    cwfv_disp_expr()
    cwfv_disp_func.x.array[:] = cwfv_disp_expr.amplitude
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
    
    applied_bc1.append(cc_disp_expr.amplitude)
    applied_bc2.append(cwfv_disp_expr.amplitude)
    applied_bc3.append(tv_disp_expr.amplitude)
    applied_bc4.append(lv_disp_expr.amplitude)

    # Project velocity if in the final period
    if t >= final_period_start and write_output:
        wh_out.interpolate(wh)
        xdmf.write_function(wh_out, t)

        # Project deformation velocity into BDM 1 space for checkpointing
        projection_problem.solve()
        
        # Write checkpoints
        a4d.write_function(filename=vh_cpoint_filename, u=wh, time=write_time)
        a4d.write_function(filename=vh_cpoint_filename, u=dw_dt_bdm, time=write_time)
        
        # Interpolate the velocity into CG1 and write XDMF output
        vh_out.interpolate(wh_dot) 
        xdmf_vel.write_function(vh_out, t) 

        write_time += 1

if write_output:
    xdmf.close()
    xdmf_vel.close()

# Perform parallell communication
point_disp = comm.gather(point_disp, root=0)

if comm.rank==0:
    # Get the point displacements
    for array in point_disp:
        if np.sum(np.abs(array)) > 0.0:
            point_disp = array
            break

    import matplotlib.pyplot as plt
    fig_dir = "../output/illustrations/verification/"
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=([12, 8]))
    ax.plot(times[1:], np.array(applied_bc1), 'r', label="Corpus callosum")
    ax.plot(times[1:], np.array(applied_bc2), 'b', label="Canal & 4V walls")
    ax.plot(times[1:], np.array(applied_bc3), 'g', label="3V wall")
    ax.plot(times[1:], np.array(applied_bc4), 'm', label="Caudate nucleus")
    ax.legend()
    fig.savefig(fig_dir+f"applied_BCs_deformation_p={p}_E={E:.0f}_dt={timestep}.png")
    plt.show()
    np.save(data_dir+f"applied_bc_corpus_callosum_dt={timestep}.npy", np.array(applied_bc1))
    np.save(data_dir+f"applied_bc_canal_wall__dt={timestep}.npy", np.array(applied_bc2))
    np.save(data_dir+f"applied_bc_3V_wall_dt={timestep}.npy", np.array(applied_bc3))

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