import ufl

import numpy as np
import dolfinx as dfx
import adios4dolfinx as a4d

from mpi4py import MPI
from basix.ufl import element
from dolfinx.fem.petsc import LinearProblem

# Action of (1 - n x n) on a vector yields the tangential component
Tangent = lambda v, n: v - n*ufl.dot(v, n)

comm = MPI.COMM_WORLD
mesh_prefix = 'coarse'
cpoint_filename = f'../output/checkpoints/deforming-mesh-{mesh_prefix}/BDM_chp_velocity/'

# Read mesh and BDM velocity function
mesh = a4d.read_mesh(cpoint_filename, comm)
n = ufl.FacetNormal(mesh)
bdm_el = element("BDM", cell=mesh.basix_cell(), degree=1)
u_bdm = dfx.fem.Function(dfx.fem.functionspace(mesh, bdm_el))
a4d.read_function(cpoint_filename, u_bdm, name='uh')
dg_el = element("DG", cell=mesh.basix_cell(), degree=1, shape=(mesh.geometry.dim,))
DG1_vector = dfx.fem.functionspace(mesh, dg_el)
u_dg = dfx.fem.Function(DG1_vector)
u_dg.interpolate(u_bdm)
# u_dg_tangential_expr = dfx.fem.Expression(Tangent(u_dg, n), u_bdm.function_space.element.interpolation_points())
# u_dg.interpolate(u_dg_tangential_expr)

# Finite element functions for solving CG1 projection problem
cg_el = element("Lagrange", cell=mesh.basix_cell(), degree=1, shape=(mesh.geometry.dim,))
V = dfx.fem.functionspace(mesh, cg_el)
u_cg = dfx.fem.Function(V)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# Define the variational problem
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx  # Bilinear form
L = ufl.inner(u_bdm, v) * ufl.dx # Linear form

# Solve the variational problem with a direct solver
problem = LinearProblem(a, L, bcs=[], u=u_cg)
problem.solve()

from utilities.normals_and_tangents import facet_vector_approximation
nh = facet_vector_approximation(DG1_vector)
u_t_expr = dfx.fem.Expression((ufl.Identity(mesh.geometry.dim) - ufl.outer(nh, nh))*u_dg, DG1_vector.element.interpolation_points())
u_t = dfx.fem.Function(DG1_vector)
u_t.interpolate(u_t_expr)

u_cg.interpolate(u_t)

# Reshape vector and normalize
u_reshaped = u_cg.x.array.reshape((int(u_cg.x.array.__len__()/mesh.geometry.dim), mesh.geometry.dim))
u_reshaped_lengths = np.sqrt(u_reshaped[:, 0]**2 + u_reshaped[:, 1]**2 + u_reshaped[:, 2]**2)
for i in range(mesh.geometry.dim):
    u_reshaped[:, i] /= u_reshaped_lengths
u_cg.x.array[:] = u_reshaped.ravel()
with dfx.io.XDMFFile(comm, "../output/cilia-direction-vectors/scaled_cg1_velocity.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_cg)


# Compare with expression approach
u_bdm_scaled_ufl = u_bdm/ufl.inner(u_bdm, u_bdm)
u_bdm_expr = dfx.fem.Expression(u_bdm_scaled_ufl, DG1_vector.element.interpolation_points())
u_dg_scaled = dfx.fem.Function(DG1_vector)
u_dg_scaled.interpolate(u_bdm_expr)

with dfx.io.VTXWriter(comm, "../output/cilia-direction-vectors/u_dg_scaled.bp", [u_dg_scaled], "BP4") as vtx:
    vtx.write(t=0)