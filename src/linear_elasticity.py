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
from dolfinx.fem.petsc import create_matrix, assemble_matrix_mat, create_vector, assemble_vector, apply_lifting
from utilities.wall_deformation_BC import WallDeformation
from utilities.create_nullspace import build_rigid_motions_nullspace

print = PETSc.Sys.Print

# Mesh tags
CANAL_OUT = 23

# Solve linear elasticity equation on the ventricles. Wall motion is 
# prescribed in time at a single point (close to corpus callosum).
comm = MPI.COMM_WORLD
mesh_prefix = 'coarse'
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
E = 10 # Modulus of elasticity [Pa]
nu = 0.1 # Poisson's ratio [-]
mu_value = 2*E/(1+nu) # First Lamé parameter value
lam_value = nu*E/((1+nu)*(1-2*nu)) # Second Lamé parameter value
mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(mu_value)) # First Lamé parameter
lam = dfx.fem.Constant(mesh, dfx.default_scalar_type(lam_value)) # Second Lamé parameter
print("Value of Lamé parameters:")
print(f"mu \t= {mu_value:.2f}\nlambda \t= {lam_value:.2f}")

# Finite elements
vec_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
W = dfx.fem.functionspace(mesh, vec_el)
wh = dfx.fem.Function(W) # Solution function
wh_ = dfx.fem.Function(W) # Solution function at previous timestep
zero = dfx.fem.Function(W)
print(f"\nNumber of degrees of freedom: {W.dofmap.index_map.size_global*W.dofmap.index_map_bs}")

# Test and trial functions
w, dw = ufl.TrialFunction(W), ufl.TestFunction(W)

# The weak form
a = 2*mu * inner(eps(w), eps(dw))*dx + lam * div(w)*div(dw)*dx
L = inner(zero, dw)*dx # Zero RHS because of zero traction or Dirichlet BC enforced on the whole boundary    

# Dirichlet BC on corpus callosum
cc_disp_expr = WallDeformation(derivative=False)
cc_disp_func = dfx.fem.Function(W)
cc_dofs = dfx.fem.locate_dofs_geometrical(W, lambda x: np.isclose(x[0], -0.012214))
assert comm.allreduce(len(cc_dofs), op=MPI.MAX)>0, print("No corpus callosum dofs located.")
bcs = [dfx.fem.dirichletbc(cc_disp_func, cc_dofs)]

anchor_spinal_canal = True
if anchor_spinal_canal:
    # Anchor spinal canal
    canal_out_dofs = dfx.fem.locate_dofs_topological(W, mesh.topology.dim-1, ft.find(CANAL_OUT))
    bcs.append(dfx.fem.dirichletbc(zero, canal_out_dofs))
else:
    # Build nullspace of rigid body deformations
    rm_nullspace = build_rigid_motions_nullspace(W)


bdry_dofs = dfx.fem.locate_dofs_topological(W, mesh.topology.dim-1, dfx.mesh.exterior_facet_indices(mesh.topology))

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

if not anchor_spinal_canal: A.setNearNullSpace(rm_nullspace) # Set near nullspace

N = 50
T = 1
times = np.linspace(0, T, N)
dt = T / N
fps = 5
#int(1/dt)//skip, skip = 50

# Create pyvista plotter object
cells, cell_types, x = dfx.plot.vtk_mesh(W)
grid = pv.UnstructuredGrid(cells, cell_types, x) 
pl = pv.Plotter()
pl.open_gif(f"../output/illustrations/linear_elasticity_deformation.gif", fps=fps)
pl.add_mesh(grid, style='wireframe', color='k') # Add initial mesh
pl.view_yz(negative=True)
pl.camera.zoom(1.25)
cmap = cm.matter
min_disp = 0.0
max_disp = 0.0

xdmf = dfx.io.XDMFFile(comm, f"../output/deforming-mesh-{mesh_prefix}/displacement.xdmf", "w")
xdmf_vel = dfx.io.XDMFFile(comm, f"../output/deforming-mesh-{mesh_prefix}/deformation_velocity.xdmf", "w")
xdmf.write_mesh(mesh)
xdmf_vel.write_mesh(mesh)
CG1_vector_space = dfx.fem.functionspace(mesh, element=element("Lagrange", mesh.basix_cell(), degree=1, shape=(mesh.geometry.dim,)))
wh_out = dfx.fem.Function(CG1_vector_space)
vh_out = dfx.fem.Function(CG1_vector_space)
dw_dt = dfx.fem.Function(W)

vh_cpoint_filename = f"../output/checkpoints/deforming-mesh-coarse/deformation_velocity/"
a4d.write_mesh(filename=vh_cpoint_filename, mesh=mesh, store_partition_info=True)

for idx, t in enumerate(times):
    # Update displacement BC
    cc_disp_expr.t = t
    cc_disp_func.interpolate(cc_disp_expr)

    # Assemble linear system and solve the equations of linear elasticity
    A.zeroEntries()
    assemble_matrix_mat(A, a_cpp, bcs=bcs)
    A.assemble()
    with b.localForm() as local_b_vec: local_b_vec.set(0.0)
    assemble_vector(b, L_cpp)
    apply_lifting(b, [a_cpp], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    for bc in bcs: bc.set(b.array_w)
    #rm_nullspace.remove(b) # Orthogonalize RHS vector

    solver.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward() # MPI communication    

    # Add pyvista grid for current timestep
    wh_reshaped = wh.x.array.copy().reshape((int(wh.x.array.__len__()/3), mesh.geometry.dim))
    grid[f"wh_{idx}"] = wh_reshaped

    # Update min/max displacements
    wh_magnitude = np.sqrt(wh_reshaped[:, 0]**2 + wh_reshaped[:, 1]**2 + wh_reshaped[:, 2]**2)
    min_this_t = comm.allreduce(wh_magnitude.min(), op=MPI.MIN)
    max_this_t = comm.allreduce(wh_magnitude.max(), op=MPI.MAX)

    min_disp = min(min_disp, min_this_t)
    max_disp = max(max_disp, max_this_t)

    wh_out.interpolate(wh)
    xdmf.write_function(wh_out, t)

    # Calculate deformation velocity
    dw_dt.x.array[:] = (wh.x.array.copy() - wh_.x.array.copy())/dt # Backward difference in time
    a4d.write_function(filename=vh_cpoint_filename, u=dw_dt, time=t)
    vh_out.interpolate(dw_dt) # Interpolate the velocity into CG1 for output
    xdmf_vel.write_function(vh_out, t) # Write to output file
    
    wh_.x.array[:] = wh.x.array.copy() # Update previous timestep deformation    

xdmf.close()
xdmf_vel.close()

# Plot
if comm.size==1:
    for idx, t in enumerate(times):
        warped = grid.warp_by_vector(f"wh_{idx}", factor=25)
        actor1 = pl.add_mesh(warped, cmap=cmap, clim=[min_disp, max_disp])
        actor2 = pl.add_text(f"time = {t:.2f} sec")

        pl.write_frame()
        pl.remove_actor(actor1)
        pl.remove_actor(actor2)

    pl.close()