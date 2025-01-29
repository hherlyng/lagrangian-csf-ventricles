import ufl
import time

import numpy   as np
import pyvista as pv
import dolfinx as dfx
import colormaps as cm

from ufl       import inner, dot, grad, sym, div
from scifem    import assemble_scalar
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from dolfinx.fem.petsc import LinearProblem
from utilities.wall_deformation_BC import WallDeformation

# Mesh tags
CANAL_OUT = 23

# Solve linear elasticity equation on the ventricles. Wall motion is 
# prescribed in time at a single point (close to corpus callosum).
comm = MPI.COMM_WORLD
mesh_prefix = 'coarse'
with dfx.io.XDMFFile(comm, f"../geometries/{mesh_prefix}_ventricles_mesh_tagged.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    out_mesh = xdmf.read_mesh()
    
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

# Anchor spinal canal
canal_out_dofs = dfx.fem.locate_dofs_topological(W, mesh.topology.dim-1, ft.find(CANAL_OUT))
bcs.append(dfx.fem.dirichletbc(zero, canal_out_dofs))

bdry_dofs = dfx.fem.locate_dofs_topological(W, mesh.topology.dim-1, dfx.mesh.exterior_facet_indices(mesh.topology))

# Create a direct linear solver
problem = LinearProblem(a, L, bcs=bcs, u=wh)


N = 25
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

for idx, t in enumerate(times):
    # Update displacement BC
    cc_disp_expr.t = t
    cc_disp_func.interpolate(cc_disp_expr)

    # Solve the equations of linear elasticity
    problem.solve()

    # Add pyvista grid for current timestep
    grid[f"wh_{idx}"] = wh.x.array.copy().reshape((int(wh.x.array.__len__()/3), mesh.geometry.dim))

    # Update min/max displacements
    min_this_t = comm.allreduce(wh.x.array[bdry_dofs].min(), op=MPI.MIN)
    max_this_t = comm.allreduce(wh.x.array[bdry_dofs].max(), op=MPI.MAX)

    min_disp = min(min_disp, min_this_t)
    max_disp = max(max_disp, max_this_t)

# Plot
for idx, t in enumerate(times):
    warped = grid.warp_by_vector(f"wh_{idx}", factor=25)
    actor1 = pl.add_mesh(warped, cmap=cmap, clim=[min_disp, max_disp])
    actor2 = pl.add_text(f"time = {t:.2f} sec")

    pl.write_frame()
    pl.remove_actor(actor1)
    pl.remove_actor(actor2)

pl.close()