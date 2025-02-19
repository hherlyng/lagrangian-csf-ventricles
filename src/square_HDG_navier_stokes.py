from mpi4py   import MPI
from petsc4py import PETSc

import ufl
import numpy   as np
import dolfinx as dfx
from ufl    import div, dot, grad, inner, outer
from scifem import assemble_scalar
from utilities.fem     import create_normal_contribution_bc, compute_cell_boundary_int_entities, calculate_mean, calculate_norm_L2
from utilities.mesh    import create_square_mesh_with_tags
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block

# Parabolic velocity profile
def u_parabolic(x):
    return np.vstack((Re*(x[1]-x[1]**2),
                      np.zeros_like(x[1])
    ))

# Jump operator
jump = lambda phi, n: outer(phi('+'), n('+')) + outer(phi('-'), n('-'))

# Simulation parameters
comm = MPI.COMM_WORLD # MPI communicator
N = 32 # Mesh cells
t = 0.0
num_time_steps = 20
t_end = 2
delta_t = t_end/num_time_steps # Timestep size
Re = 1  # Reynolds Number
k = 1 # Polynomial degree

# Create mesh and boundary tags
mesh, ft = create_square_mesh_with_tags(N=N, comm=comm)
gdim = mesh.geometry.dim
LEFT=1; RIGHT=2; BOT=3; TOP=4

dirichlet_tags = (LEFT, BOT, TOP)
neumann_tags = [RIGHT]

# Create submesh in the facet dimension
facet_dim = mesh.topology.dim-1
num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts
facets = np.arange(num_facets, dtype=np.int32)
submesh, submesh_to_mesh = dfx.mesh.create_submesh(mesh, facet_dim, facets)[:2]
submesh.topology.create_connectivity(submesh.topology.dim, submesh.topology.dim)

# Generate integration entities for the submesh
cell_facet_map = compute_cell_boundary_int_entities(mesh)
cell_interior_bdry_tag = 0
facet_integral_entities = [(cell_interior_bdry_tag, cell_facet_map)]

# Add exterior facet integral entities
for tag in np.unique(ft.values):
    facet_integral_entities += [(int(tag), 
            dfx.fem.compute_integration_domains(
                dfx.fem.IntegralType.exterior_facet,
                mesh.topology,
                ft.find(int(tag)),
                facet_dim
                )
            )
        ]

# Function spaces for the velocity and for the pressure
V = dfx.fem.functionspace(mesh, ('Discontinuous Brezzi-Douglas-Marini', k+1))
Q = dfx.fem.functionspace(mesh, ('Discontinuous Lagrange', k))
Vbar = dfx.fem.functionspace(submesh, ('Discontinuous Lagrange', k+1, (gdim,)))
Qbar = dfx.fem.functionspace(submesh, ('Discontinuous Lagrange', k+1))
M = ufl.MixedFunctionSpace(V, Q, Vbar, Qbar)

# Function space for visualising the velocity field
W = dfx.fem.functionspace(mesh, ('Discontinuous Lagrange', k+1, (gdim,)))
print(f'Size of dofmap: {V.dofmap.index_map.size_global+Q.dofmap.index_map.size_global}')

# Define trial and test functions
u, p, ubar, pbar = ufl.TrialFunctions(M)
v, q, vbar, qbar = ufl.TestFunctions(M)

u_ = dfx.fem.Function(V)
ubar_ = dfx.fem.Function(Vbar)

dt = dfx.fem.Constant(mesh, dfx.default_real_type(delta_t))
alpha = dfx.fem.Constant(mesh, dfx.default_real_type(gdim*10 * k**2))
nu = dfx.fem.Constant(mesh, dfx.default_real_type(1.0))

# Mesh cell diameter and facet normal vector
h = ufl.CellDiameter(mesh)
n = ufl.FacetNormal(mesh)

# Integral measures
dx = ufl.Measure('dx', mesh) # Cell integral
ds = ufl.Measure('ds', mesh, subdomain_data=facet_integral_entities) # Facet integral

ds_interior = ds(cell_interior_bdry_tag)
ds_N = ds(neumann_tags[0])
ds_D1 = ds(LEFT)
ds_D2 = ds((BOT, TOP))

# Since the bilinear and linear forms are formulated on the mesh,
# we need a mapping between the facets in the mesh and the cells in the 
# submesh (these are the same entities). This mapping is simply
# the inverse mapping of submesh_to_mesh
mesh_to_submesh = np.zeros(num_facets, dtype=np.int32)
mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh), dtype=np.int32)
subentities_map = {submesh : mesh_to_submesh}

# Define the bilinear form
lmbda = ufl.conditional(ufl.gt(dot(u_, n), 0), 1, 0) # Function used to upwind the velocity
a = (
    # Time derivative 
    inner(u / dt, v) * dx 

    # Momentum terms, this is a(u_h, v_h) in Joe Dean's thesis
    + nu*inner(grad(u), grad(v)) * dx 
    - nu*inner(u - ubar, dot(grad(v), n)) * ds_interior
    - nu*inner(v - vbar, dot(grad(u), n)) * ds_interior
    + nu*alpha/h * inner(u - ubar, v - vbar) * ds_interior

    # b(v_h, p_h) terms
    - inner(p, div(v)) * dx
    + inner(dot(v, n), pbar) * ds_interior

    # c(u_h, u_h, v_h) terms
    - inner(outer(u_, u), grad(v)) * dx
    + inner(outer(u, u_) - outer(u - ubar, lmbda*u_), outer(v - vbar, n)) * ds_interior

    # Neumann BC terms
   + (1-lmbda) * dot(ubar_, n) * dot(ubar, vbar) * ds_N
   - inner(dot(ubar, n), qbar) * ds_N
   - inner(dot(vbar, n), pbar) * ds_N

    # b(u_h, q_h) terms
    - inner(q, div(u)) * dx
    + inner(dot(u, n), qbar) * ds_interior
)

a_blocked = dfx.fem.form(ufl.extract_blocks(a), entity_maps=subentities_map)

# Define the linear form
f = dfx.fem.Function(V)
u_D = dfx.fem.Function(Vbar)
u_D.interpolate(u_parabolic)
u_flux = dfx.fem.Function(Vbar)
L = inner(f, v) * dx + (
    # Time derivative
    inner(u_ / dt, v) * dx
    # Dirichlet BC terms
    + inner(dot(u_D, n), qbar) * ds_D1
    + inner(dot(u_flux, n), qbar) * ds_D2
    - inner(dfx.fem.Constant(mesh, 0.0)*n, vbar) * ds_N # Zero pressure
)
# Add zero block to pressure or else PETSc will complain
L += inner(dfx.fem.Constant(mesh, dfx.default_real_type(0.0)), q) * dx
L_blocked = dfx.fem.form(ufl.extract_blocks(L), entity_maps=subentities_map)

# Boundary conditions
# Prescribed profile
inflow_facets = mesh_to_submesh[ft.find(LEFT)]
inflow_dofs = dfx.fem.locate_dofs_topological(Vbar, facet_dim, inflow_facets)
bc_u = dfx.fem.dirichletbc(u_D, inflow_dofs)
bcs = [bc_u]

# Flux boundary condition: First
flux_facets_mesh = np.concatenate(([ft.find(tag) for tag in [BOT, TOP]]))
flux_facets_submesh = mesh_to_submesh[flux_facets_mesh]

# Create the flux function in a (non-broken) BDM space on the mesh
flux_space = dfx.fem.functionspace(mesh, ('Brezzi-Douglas-Marini', k+1))
flux_expr_mesh = create_normal_contribution_bc(flux_space, -n, flux_facets_mesh)
u_flux_mesh = dfx.fem.Function(flux_space)
u_flux_mesh.interpolate(flux_expr_mesh)

# Interpolate the function from the mesh onto the submesh
interpolation_data = dfx.fem.create_interpolation_data(V_to=Vbar,
                                                       V_from=flux_space,
                                                       cells=flux_facets_submesh)
u_flux.interpolate_nonmatching(u_flux_mesh,
                               cells=flux_facets_submesh,
                               interpolation_data=interpolation_data)

# Set the BC                               
flux_dofs = dfx.fem.locate_dofs_topological(Vbar, facet_dim, flux_facets_submesh)
bcs.append(dfx.fem.dirichletbc(u_flux, flux_dofs))                                                                                      

# Assemble the Navier-Stokes problem
A = assemble_matrix_block(a_blocked, bcs=bcs)
A.assemble()
b = assemble_vector_block(L_blocked, a_blocked, bcs=bcs)
x = A.createVecRight() # Solution vector

# Create and configure solver
ksp = PETSc.KSP().create(comm)  # type: ignore
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()  # type: ignore
opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
opts["ksp_error_if_not_converged"] = 1 # Throw an error if KSP solver does not converge
ksp.setFromOptions()

# Create solution functions
u_h = dfx.fem.Function(V)
p_h = dfx.fem.Function(Q)

u_vis = dfx.fem.Function(W)
u_vis.name = "u"
u_vis.interpolate(u_h)

# Write initial condition to file
u_file = dfx.io.VTXWriter(comm, "u.bp", u_vis)
p_file = dfx.io.VTXWriter(comm, "p.bp", p_h)
u_file.write(t)
p_file.write(t)

# Create offsets for the blocked functions
offset_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
offset_p = offset_u + Q.dofmap.index_map.size_local*Q.dofmap.index_map_bs
offset_ubar = offset_p + Vbar.dofmap.index_map.size_local*Vbar.dofmap.index_map_bs

# Time stepping loop
for _ in range(num_time_steps):
    t += dt.value

    A.zeroEntries()
    assemble_matrix_block(A, a_blocked, bcs=bcs)
    A.assemble()

    with b.localForm() as b_loc: b_loc.set(0)
    assemble_vector_block(b, L_blocked, a_blocked, bcs=bcs)

    # Compute solution
    ksp.solve(b, x)

    u_h.x.array[:offset_u] = x.array_r[:offset_u]
    u_h.x.scatter_forward()
    p_h.x.array[:(offset_p-offset_u)] = x.array_r[offset_u:offset_p]
    p_h.x.scatter_forward()
    p_h.x.array[:] -= calculate_mean(mesh, p_h)
    ubar_.x.array[:(offset_ubar-offset_p)] = x.array_r[offset_p:offset_ubar]
    ubar_.x.scatter_forward()

    u_vis.interpolate(u_h)

    # Write to file
    u_file.write(t)
    p_file.write(t)

    # Update u_n
    u_.x.array[:] = u_h.x.array

u_file.close()
p_file.close()


# Compute divergence L2 norm to check mass conservation
e_div_u = calculate_norm_L2(comm, div(u_h))
assert np.isclose(e_div_u, 0.0, atol=float(1.0e5 * np.finfo(dfx.default_real_type).eps))

# Compute flux on boundaries with flux BC
flux = assemble_scalar(dot(u_h, n)*ds((TOP)))
total_flux = assemble_scalar(dot(u_h, n)*ds)

if comm.rank == 0:
    print(f"e_div_u = {e_div_u}")
    print(f"Flux = {flux}")
    print(f"Total flux = {total_flux}")