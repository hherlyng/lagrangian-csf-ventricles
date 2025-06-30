import ufl

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from ufl       import inner, grad, sym, div
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from dolfinx.fem.petsc    import create_matrix, create_vector, assemble_vector, apply_lifting
from utilities.fem        import assemble_system
from utilities.projection import projection_problem_CG_to_BDM
from utilities.deformation_data import WallDeformationCorpusCallosum, WallDeformationCanalWall, WallDeformationThirdVentricle

print = PETSc.Sys.Print

# Mesh tags
CANAL_WALL = 13
CANAL_OUT  = 23
LATERAL_APERTURES = 28
CORPUS_CALLOSUM = 110
THIRD_LATERAL_RIGHT = 111
THIRD_LATERAL_LEFT = 112

# Solve linear elasticity equation on the ventricles. Wall motion is 
# prescribed in time at a single point (close to corpus callosum).
comm = MPI.COMM_WORLD
mesh_prefix = 'medium'
with dfx.io.XDMFFile(comm, f"../geometries/{mesh_prefix}_ventricles_mesh_tagged.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    gdim = mesh.geometry.dim
    
    # Generate mesh entities    
    facet_dim = mesh.topology.dim-1
    mesh.topology.create_entities(facet_dim) # Create facets
    ft = xdmf.read_meshtags(mesh, "ft")
    ct = xdmf.read_meshtags(mesh, "ct")

mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity

output_dir = f"../output/{mesh_prefix}-mesh/deformation/"

ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft) # Boundary integral measure
dx = ufl.Measure('dx', domain=mesh, subdomain_data=ct) # Volume integral measure
eps = lambda arg: sym(grad(arg)) # The symmetric gradient

# Material parameters
E = 1500 #3156 # Modulus of elasticity [Pa]
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
T = 3.0
period = 1
N = int(T / timestep)
times = np.linspace(0, T, N+1)
final_period_start = int(T - period)
write_time = 0

# Finite elements
p = 2
vec_el = element("Lagrange", mesh.basix_cell(), p, shape=(mesh.geometry.dim,))
W = dfx.fem.functionspace(mesh, vec_el)
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
cc_disp_expr = WallDeformationCorpusCallosum(period=period, timestep=timestep)
cc_disp_func = dfx.fem.Function(W)
cw_disp_func = dfx.fem.Function(W)
cw_disp_expr = WallDeformationCanalWall(period=period, timestep=timestep)
tw_disp_expr = WallDeformationThirdVentricle(period=period, timestep=timestep)
tw_disp_func_right = dfx.fem.Function(W)
tw_disp_func_left  = dfx.fem.Function(W)
bcs = []

cc_dofs = dfx.fem.locate_dofs_topological((W_z, W), facet_dim, ft.find(CORPUS_CALLOSUM))
# cc_dofs = np.arange(cc_dofs*gdim, (cc_dofs+1)*gdim)
# num_cc_dofs = comm.allreduce(len(cc_dofs), op=MPI.SUM)
# print("Number of corpus callosum dofs: ", num_cc_dofs)
# assert num_cc_dofs>0, print("No corpus callosum dofs located.")
bcs.append(dfx.fem.dirichletbc(cc_disp_func, cc_dofs, W_z))


cw_dofs = dfx.fem.locate_dofs_topological((W_z, W), facet_dim, ft.find(CANAL_WALL))
# cw_dofs = np.arange(cw_dofs*gdim, (cw_dofs+1)*gdim)
# num_cw_dofs = comm.allreduce(len(cw_dofs), op=MPI.SUM)
# print("Number of canal wall dofs: ", num_cw_dofs)
bcs.append(dfx.fem.dirichletbc(cw_disp_func, cw_dofs, W_z))

tw_dofs_right = dfx.fem.locate_dofs_topological((W_y, W), facet_dim, ft.find(THIRD_LATERAL_RIGHT))
# tw_dofs_right = np.arange(tw_dofs_right*gdim, (tw_dofs_right+1)*gdim)
# num_tw_dofs_right = comm.allreduce(len(tw_dofs_right), op=MPI.SUM)
# print("Number of third ventricle wall (right) dofs: ", num_tw_dofs_right)
bcs.append(dfx.fem.dirichletbc(tw_disp_func_right, tw_dofs_right, W_y))

tw_dofs_left = dfx.fem.locate_dofs_topological((W_y, W), facet_dim, ft.find(THIRD_LATERAL_LEFT))
# tw_dofs_left = np.arange(tw_dofs_left*gdim, (tw_dofs_left+1)*gdim)
# num_tw_dofs_left = comm.allreduce(len(tw_dofs_left), op=MPI.SUM)
# print("Number of third ventricle wall (left) dofs: ", num_tw_dofs_left)
bcs.append(dfx.fem.dirichletbc(tw_disp_func_left, tw_dofs_left, W_y))

# Anchor spinal canal

# canal_out_dofs = dfx.fem.locate_dofs_topological(W, facet_dim, ft.find(CANAL_OUT))
# bcs.append(dfx.fem.dirichletbc(zero, canal_out_dofs))
# apertures_dofs = dfx.fem.locate_dofs_topological(W, facet_dim, ft.find(LATERAL_APERTURES))
# bcs.append(dfx.fem.dirichletbc(zero, apertures_dofs))
    
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
solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, residual: {rnorm}"))

xdmf = dfx.io.XDMFFile(comm, output_dir+f"displacement_dt={timestep:.4g}_T={T:.4g}.xdmf", "w")
xdmf_vel = dfx.io.XDMFFile(comm, output_dir+f"displacement_velocity_dt={timestep:.4g}_T={T:.4g}.xdmf", "w")
xdmf.write_mesh(mesh)
xdmf_vel.write_mesh(mesh)
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

vh_cpoint_filename = output_dir+f"checkpoints/displacement_velocity_dt={timestep:.4g}_T={T:.4g}/"
a4d.write_mesh(filename=vh_cpoint_filename, mesh=mesh)
a4d.write_meshtags(vh_cpoint_filename, mesh, ft, meshtag_name='ft')
a4d.write_meshtags(vh_cpoint_filename, mesh, ct, meshtag_name='ct')

projection_problem = projection_problem_CG_to_BDM(wh_dot, dw_dt_bdm, dx)

A, b = assemble_system(A, b, a_cpp, L_cpp, bcs)

applied_bc1 = []
applied_bc2 = []
applied_bc3 = []

# Energy norms
E_kinetic = dfx.fem.form(1/2*rho * inner(wh_n, wh_n) * dx)
E_elastic = dfx.fem.form(1/2 * inner(sigma(wh_n), eps(wh_n)) * dx)

energy = np.zeros((len(times), 2))
max_disp = np.zeros((len(times), 4))
point_disp = np.zeros((len(times), 3))
point_dof = dfx.fem.locate_dofs_topological(W, mesh.topology.dim-1, ft.find(CORPUS_CALLOSUM))[0]
point_dofs = np.arange(point_dof * (mesh.topology.dim), (point_dof + 1) * (mesh.topology.dim))

for i, t in enumerate(times[1:], 1):
    
    print(f"\nTime t = {t:.5g}")

    # Update displacement BCs
    cc_disp_expr.increment_index(t)
    cc_disp_func.x.array[:] = cc_disp_expr()
    # cc_disp_func.interpolate(cc_disp_expr)
    applied_bc1.append(cc_disp_expr.applied_bc)
    cw_disp_expr.increment_index(t)
    cw_disp_func.x.array[:] = cw_disp_expr()
    # cw_disp_func.interpolate(cw_disp_expr)
    applied_bc2.append(cw_disp_expr.applied_bc)
    tw_disp_expr.increment_index(t)
    tw_disp_func_right.x.array[:] = tw_disp_expr()
    # tw_disp_func_right.interpolate(tw_disp_expr)
    tw_disp_func_left.x.array[:] = -1.0*tw_disp_func_right.x.array.copy()
    applied_bc3.append(tw_disp_expr.applied_bc)

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

    point_disp[i, :] = wh.x.array[point_dofs]
    max_disp[i, 0] = wh.sub(0).collapse().x.array.max()
    max_disp[i, 1] = wh.sub(1).collapse().x.array.max()
    max_disp[i, 2] = wh.sub(2).collapse().x.array.max()
    max_disp[i, 3] = np.sqrt(wh.sub(0).collapse().x.array**2
                           + wh.sub(1).collapse().x.array**2
                           + wh.sub(2).collapse().x.array**2).max()

    # Project velocity if in the final period
    if t >= final_period_start:
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

xdmf.close()
xdmf_vel.close()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=([12, 8]))
ax.plot(times[1:], np.array(applied_bc1), 'r', label="Corpus callosum")
ax.plot(times[1:], np.array(applied_bc2), 'b', label="Canal wall")
ax.plot(times[1:], np.array(applied_bc3), 'g', label="3V wall")
ax.legend()
fig.savefig(f"../output/illustrations/applied_BCs_deformation_p={p}_E={E}_dt={timestep}.png")
plt.show()

np.save(output_dir+"applied_bc_corpus_callosum", np.array(applied_bc1))
np.save(output_dir+"applied_bc_canal_wall", np.array(applied_bc2))
np.save(output_dir+"applied_bc_3V_wall", np.array(applied_bc3))

fig2, ax2 = plt.subplots(figsize=([12, 8]))
ax2.plot(times, energy[:, 0], 'r', label="Kinetic energy")
ax2.plot(times, energy[:, 1], 'b', label="Elastic energy")
ax2.legend()
fig2.savefig(f"../output/illustrations/verification/energies_p={p}_E={E}_dt={timestep}.png")
plt.show()

fig3, ax3 = plt.subplots(figsize=([12, 8]))
ax3.plot(times, max_disp[:, 0], 'k', label="Max x disp")
ax3.plot(times, max_disp[:, 1], 'r', label="Max y disp")
ax3.plot(times, max_disp[:, 2], 'b', label="Max z disp")
ax3.plot(times, max_disp[:, 3], color='k', linestyle='-.', label="Max mag.")
ax3.plot(times, point_disp[:, 0], 'g', label="Point x disp")
ax3.plot(times, point_disp[:, 1], 'c', label="Point y disp")
ax3.plot(times, point_disp[:, 2], 'm', label="Point z disp")
ax3.legend()
fig3.savefig(f"../output/illustrations/verification/displacements_p={p}_E={E}_dt={timestep}.png")
plt.show()

fig4, ax4 = plt.subplots(figsize=([12, 8]))
ax4.plot(times[1:], np.array(applied_bc1), 'k', label="Applied BC")
ax4.plot(times[1:], point_disp[1:, 2], 'm', label="Point z disp")
ax4.legend()
fig4.savefig(f"../output/illustrations/verification/corpus_callosum_p={p}_E={E}_dt={timestep}.png")
plt.show()