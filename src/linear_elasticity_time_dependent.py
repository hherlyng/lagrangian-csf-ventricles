import ufl

import numpy   as np
import pyvista as pv
import dolfinx as dfx
import colormaps as cm
import adios4dolfinx as a4d

from ufl       import inner, grad, sym, div
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from dolfinx.fem.petsc    import create_matrix, create_vector
from utilities.fem        import assemble_system
from utilities.projection import projection_problem_CG2_to_BDM1
from utilities.deformation_data import WallDeformationCorpusCallosum, WallDeformationSpinalCanal, WallDeformationCanalWall

print = PETSc.Sys.Print

# Mesh tags
CANAL_WALL = 13
CANAL_OUT  = 23
LATERAL_APERTURES = 28

# Solve linear elasticity equation on the ventricles. Wall motion is 
# prescribed in time at a single point (close to corpus callosum).
comm = MPI.COMM_WORLD
mesh_prefix = 'medium'
with dfx.io.XDMFFile(comm, f"../geometries/{mesh_prefix}_ventricles_mesh_tagged.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    
    # Generate mesh entities    
    facet_dim = mesh.topology.dim-1
    mesh.topology.create_entities(facet_dim) # Create facets
    ft = xdmf.read_meshtags(mesh, "ft")
    ct = xdmf.read_meshtags(mesh, "ct")

mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity

ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft) # Boundary integral measure
dx = ufl.Measure('dx', domain=mesh, subdomain_data=ct) # Volume integral measure
eps = lambda arg: sym(grad(arg)) # The symmetric gradient

# Material parameters
E = 1500 # Modulus of elasticity [Pa]
nu = 0.35 # Poisson's ratio [-]
mu_value = 2*E/(1+nu) # First Lamé parameter value
lam_value = nu*E/((1+nu)*(1-2*nu)) # Second Lamé parameter value
mu = dfx.fem.Constant(mesh, mu_value) # First Lamé parameter
lam = dfx.fem.Constant(mesh, lam_value) # Second Lamé parameter
rho = dfx.fem.Constant(mesh, 1000.0) # Ventricular wall density [kg/m^3]
print("Value of Lamé parameters:")
print(f"mu \t= {mu_value:.2f}\nlambda \t= {lam_value:.2f}")

# Timestep size [s]
timestep = 0.2
dt = dfx.fem.Constant(mesh, timestep) 

# Finite elements
vec_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
W = dfx.fem.functionspace(mesh, vec_el)
wh = dfx.fem.Function(W) # Solution function
wh_n = dfx.fem.Function(W) # Solution function at n
wh_nmin = dfx.fem.Function(W) # Solution function at n-1
zero = dfx.fem.Function(W)
print(f"\nNumber of degrees of freedom: {W.dofmap.index_map.size_global*W.dofmap.index_map_bs}")

# Test and trial functions
w, dw = ufl.TrialFunction(W), ufl.TestFunction(W)

# The weak form
a = rho*inner(w, dw)*dx + dt**2*(2*mu * inner(eps(w), eps(dw))*dx + lam * div(w)*div(dw)*dx)
L = rho*inner(2*wh_n-wh_nmin, dw)*dx

# Dirichlet BC on corpus callosum
cc_disp_expr = WallDeformationCorpusCallosum(derivative=False)
cc_disp_func = dfx.fem.Function(W)

if mesh_prefix=='coarse':
    cc_facet = 7340
elif mesh_prefix=='medium':
    cc_facet = 51378

cc_dofs = dfx.fem.locate_dofs_topological(W, facet_dim, np.array([cc_facet], dtype=np.int32))
num_cc_dofs = comm.allreduce(len(cc_dofs), op=MPI.SUM)
print("Number of corpus callosum dofs: ", num_cc_dofs)
assert num_cc_dofs>0, print("No corpus callosum dofs located.")
bcs = [dfx.fem.dirichletbc(cc_disp_func, cc_dofs)]

cw_dofs = dfx.fem.locate_dofs_topological(W, facet_dim, ft.find(CANAL_WALL))
num_cw_dofs = comm.allreduce(len(cw_dofs), op=MPI.SUM)
print("Number of canal wall dofs: ", num_cw_dofs)
cw_disp_expr = WallDeformationCanalWall(derivative=False)
cw_disp_func = dfx.fem.Function(W)
bcs.append(dfx.fem.dirichletbc(cw_disp_func, cw_dofs))


anchor_spinal_canal = True
if anchor_spinal_canal: # Anchor spinal cord and lateral apertures
    canal_out_dofs = dfx.fem.locate_dofs_topological(W, facet_dim, ft.find(CANAL_OUT))
    apertures_dofs = dfx.fem.locate_dofs_topological(W, facet_dim, ft.find(LATERAL_APERTURES))
    anchor_dofs = np.concatenate((canal_out_dofs, apertures_dofs))
    bcs.append(dfx.fem.dirichletbc(zero, anchor_dofs))

else: # Impose deformation on spinal cord
    sc_disp_expr = WallDeformationSpinalCanal(derivative=False)
    sc_disp_func = dfx.fem.Function(W)
    sc_dofs = dfx.fem.locate_dofs_topological(W, facet_dim, ft.find(CANAL_OUT))
    num_sc_dofs = comm.allreduce(len(sc_dofs), op=MPI.SUM)
    assert num_sc_dofs>0, print("No spinal canal dofs located.")
    print("Number of spinal canal dofs: ", num_sc_dofs)
    bcs.append(dfx.fem.dirichletbc(sc_disp_func, sc_dofs))

# Create linear system
a_cpp, L_cpp = dfx.fem.form(a), dfx.fem.form(L)
A = create_matrix(a_cpp)
b = create_vector(L_cpp)

# Configure linear solver based on
# conjugate gradient with algebraic multigrid preconditioning
opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1e-6 # Relative tolerance
opts["pc_type"] = "gamg"

# Use Chebyshev smoothing for the multigrid PC
opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_pc_type"]  = "jacobi"

# Improve the estimate of eigenvalues for the Chebyshev smoothing
opts["mg_levels_ksp_chebyshev_esteig_steps"] = 10

# Create the solver object, set options and enable convergence monitoring
solver = PETSc.KSP().create(comm)
solver.setOperators(A)
solver.setFromOptions()
solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, residual: {rnorm}"))

T = 1
period = 1
N = int(T / timestep)
times = np.linspace(0, T, N+1)

xdmf = dfx.io.XDMFFile(comm, f"../output/{mesh_prefix}-mesh/deformation/time_dep_displacement_dt={timestep:.4g}_T={T:.4g}.xdmf", "w")
xdmf_vel = dfx.io.XDMFFile(comm, f"../output/{mesh_prefix}-mesh/deformation/time_dep_deformation_velocity_dt={timestep:.4g}_T={T:.4g}.xdmf", "w")
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
bdm_el = element("BDM", mesh.basix_cell(), 1)
BDM = dfx.fem.functionspace(mesh, bdm_el)
dw_dt_bdm = dfx.fem.Function(BDM)
dw_dt_bdm.name = "defo_velocity"
wh.name = "defo_displacement"

vh_cpoint_filename = f"../output/{mesh_prefix}-mesh/deformation/checkpoints/time_dep_deformation_velocity_dt={timestep:.4g}_T={T:.4g}/"
a4d.write_mesh(filename=vh_cpoint_filename, mesh=mesh, store_partition_info=True)
a4d.write_meshtags(vh_cpoint_filename, mesh, ft, meshtag_name='ft')

projection_problem = projection_problem_CG2_to_BDM1(dw_dt, dw_dt_bdm)

for idx, t in enumerate(times):
    
    print(f"\nTime t = {t:.4g}")
        
    if t > period:
        # Ensure BC function time value is within
        # the time interval [0, period]
        bc_time = t - int(t)
    else:
        bc_time = t

    # Update displacement BCs
    cc_disp_expr.t = bc_time
    cc_disp_func.interpolate(cc_disp_expr)
    cw_disp_expr.t = bc_time
    cw_disp_func.interpolate(cw_disp_expr)

    if not anchor_spinal_canal:
        sc_disp_expr.t = bc_time
        sc_disp_func.interpolate(sc_disp_expr)

    # Assemble linear system and solve the equations of linear elasticity
    A, b = assemble_system(A, b, a_cpp, L_cpp, bcs)
    solver.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward() # MPI communication

    wh_out.interpolate(wh)
    xdmf.write_function(wh_out, t)
    
    # Calculate deformation velocity by a central difference in time
    dw_dt.x.array[:] = \
        (wh.x.array.copy() - 2*wh_n.x.array.copy() + wh_nmin.x.array.copy())/timestep**2

    # Project deformation velocity into BDM 1 space for checkpointing
    projection_problem.solve()
    
    # Write checkpoints
    a4d.write_function(filename=vh_cpoint_filename, u=wh, time=float(f"{t:.4g}"))
    a4d.write_function(filename=vh_cpoint_filename, u=dw_dt_bdm, time=float(f"{t:.4g}"))
    
    # Interpolate the velocity into CG1 and write XDMF output
    vh_out.interpolate(dw_dt) 
    xdmf_vel.write_function(vh_out, t) 
    
    # Update deformation previous timesteps
    wh_nmin.x.array[:] = wh_n.x.array.copy()
    wh_n.x.array[:] = wh.x.array.copy() 

xdmf.close()
xdmf_vel.close()