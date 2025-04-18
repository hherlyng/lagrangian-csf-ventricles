from mpi4py   import MPI
from petsc4py import PETSc

import ufl
import numpy   as np
import dolfinx as dfx
from sys    import argv
from ufl    import div, dot, grad, inner, nabla_grad, jump, avg
from scifem import assemble_scalar
from utilities.fem     import calculate_mean, calculate_norm_L2, eps, stabilization, create_normal_contribution_bc, tangent
from utilities.mesh    import create_unit_square_mesh
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, create_matrix_block

# Simulation parameters
comm = MPI.COMM_WORLD # MPI communicator
N = int(argv[1]) # Mesh cells
t = 0.0
num_time_steps = 100
t_end = 1
delta_t = t_end/num_time_steps # Timestep size
mu_value  = 7e-4 # Dynamic viscosity
rho_value = 1e3 # Fluid density
Re = rho_value/mu_value  # Reynolds Number
k = 2 # Polynomial degree

# Create mesh and boundary tags
mesh, ft = create_unit_square_mesh(N=N, comm=comm)
gdim = mesh.geometry.dim
LEFT=1; RIGHT=2; BOT=3; TOP=4
facet_dim = mesh.topology.dim-1

# Function spaces for the velocity and for the pressure
V = dfx.fem.functionspace(mesh, ('Brezzi-Douglas-Marini', k))
Q = dfx.fem.functionspace(mesh, ('Discontinuous Lagrange', k-1))
M = ufl.MixedFunctionSpace(V, Q)

# Function space for visualising the velocity field
W = dfx.fem.functionspace(mesh, ('Discontinuous Lagrange', k, (gdim,)))
print(f'Size of dofmap: {V.dofmap.index_map.size_global+Q.dofmap.index_map.size_global}')

# Define trial and test functions
u, p = ufl.TrialFunctions(M)
v, q = ufl.TestFunctions(M)

u_  = dfx.fem.Function(V) # Velocity at timestep n
u__ = dfx.fem.Function(V) # Velocity at timestep n-1

dt = dfx.fem.Constant(mesh, dfx.default_real_type(delta_t)) # Timestep
gamma = dfx.fem.Constant(mesh, dfx.default_real_type(10.0)) # BDM penalty parameter
mu = dfx.fem.Constant(mesh, dfx.default_real_type(mu_value)) # Dynamic viscosity
rho = dfx.fem.Constant(mesh, dfx.default_scalar_type(rho_value)) # Fluid density

# Mesh cell diameter and facet normal vector
h = ufl.CellDiameter(mesh)
n = ufl.FacetNormal(mesh)

# Integral measures
dx = ufl.Measure('dx', mesh) # Cell integral
ds = ufl.Measure('ds', mesh, subdomain_data=ft) # Exterior facet integral
dS = ufl.Measure('dS', mesh) # Interior facet integral

# Tangential traction BC
tangential_traction = ufl.as_vector((1.0, 0.0))

# Create stokes problem to solve for initial condition
# Stokes equations bilinear form in block form
a00 = (3/2*rho/dt * inner(u, v) * dx # Time derivative
     + 2*mu*inner(eps(u), eps(v))*dx # Viscous dissipation
     + stabilization(u, v, mu, gamma) # BDM stabilization
     - mu*inner(dot(grad(u).T, n), v) * (ds(LEFT) + ds(RIGHT)) # Parallel flow at outlet
    )
a01 = -inner(p, div(v))*dx
a10 = -inner(q, div(u))*dx
a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*dx

# Linear form
L0 = 2*rho/dt * inner(u_, v) * dx - 1/2*rho/dt * inner(u__, v) * dx # Time derivative
L0 += inner(tangential_traction, tangent(v, n)) * ds(TOP)
        
L1 = inner(dfx.fem.Function(Q), q)*dx

a_stokes = dfx.fem.form([[a00, a01],
                         [a10, a11]])
L = dfx.fem.form([L0, L1])

# Navier-Stokes equations bilinear form in block form
a00 += rho*inner(dot(u_, nabla_grad(u)), v) * dx # Convective term

# Add convective term stabilization
zeta = ufl.conditional(ufl.lt(dot(u_, n), 0), 1, 0) # Upwind velocity operator (equals 1 on inflow boundary, 0 on outflow boundary)
a00 += (-rho*1/2*dot(jump(u_), n('+')) * avg(dot(u, v)) * dS 
      - rho*dot(avg(u_), n('+')) * dot(jump(u), avg(v)) * dS 
      - zeta*rho*1/2*dot(u_, n) * dot(u, v) * (ds(LEFT) + ds(RIGHT))
)

a = dfx.fem.form([[a00, a01],
                  [a10, a11]])

# Strong boundary conditions: Impermeability on top/bottom
impermeability_dofs = dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(TOP))
bc_impermeability = dfx.fem.dirichletbc(dfx.fem.Function(V), impermeability_dofs)
bcs = [bc_impermeability]               

influx = create_normal_contribution_bc(V, -1/4*n, ft.find(LEFT))
bc_inflow = dfx.fem.dirichletbc(influx, dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(LEFT)))
bcs.append(bc_inflow)

defo = dfx.fem.Function(V)
defo.interpolate(lambda x: (x[0], x[0]))
bc_defo = dfx.fem.dirichletbc(defo, dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(BOT)))
bcs.append(bc_defo)

# Assemble the Stokes problem
A_stokes = assemble_matrix_block(a_stokes,bcs=bcs)
A_stokes.assemble()
b = assemble_vector_block(L, a_stokes, bcs=bcs)

# Create the Navier-Stokes problem
A = create_matrix_block(a)
x = A.createVecRight() # Solution vector

# Create and configure solver
ksp = PETSc.KSP().create(comm)  # type: ignore
ksp.setOperators(A_stokes)
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

# Create velocity function for visualization
u_vis = dfx.fem.Function(W)
u_vis.name = "u"

# Create output files
u_file = dfx.io.VTXWriter(comm, "../output/square-mesh/flow/navier-stokes/square_BC_test_velocity.bp", u_vis)
p_file = dfx.io.VTXWriter(comm, "../output/square-mesh/flow/navier-stokes/square_BC_test_pressure.bp", p_h)

# Create offsets for the blocked functions
offset_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
offset_p = offset_u + Q.dofmap.index_map.size_local*Q.dofmap.index_map_bs

# Solve Stokes problem for the initial condition
ksp.solve(b, x)
u_.x.array[:offset_u] = x.array_r[:offset_u]
u_.x.scatter_forward()
u_vis.interpolate(u_) # Interpolate velocity into visualization function
u__.interpolate(u_) # Update timestep n-1 velocity
p_h.x.array[:(offset_p-offset_u)] = x.array_r[offset_u:offset_p]
p_h.x.scatter_forward()
p_h.x.array[:] -= calculate_mean(mesh, p_h, dX=dx)

# Write initial condition to file
u_file.write(t)
p_file.write(t)

# Set Navier-Stokes system matrix as KSP operator
ksp.setOperators(A)

# Time stepping loop
for _ in range(num_time_steps):
    t += dt.value

    A.zeroEntries()
    assemble_matrix_block(A, a, bcs=bcs)
    A.assemble()

    with b.localForm() as b_loc: b_loc.set(0)
    assemble_vector_block(b, L, a, bcs=bcs)

    # Compute solution
    ksp.solve(b, x)

    u_h.x.array[:offset_u] = x.array_r[:offset_u]
    u_h.x.scatter_forward()
    p_h.x.array[:(offset_p-offset_u)] = x.array_r[offset_u:offset_p]
    p_h.x.scatter_forward()
    p_h.x.array[:] -= calculate_mean(mesh, p_h, dX=dx)

    u_vis.interpolate(u_h)

    # Write to file
    u_file.write(t)
    p_file.write(t)

    # Update u_n and u_{n-1}
    u__.x.array[:] = u_.x.array.copy()
    u_.x.array[:] = u_h.x.array.copy()

u_file.close()
p_file.close()

# Compute divergence L2 norm to check mass conservation
e_div_u = calculate_norm_L2(comm, div(u_h), dX=dx)
assert np.isclose(e_div_u, 0.0, atol=float(1.0e5 * np.finfo(dfx.default_real_type).eps))

flux = assemble_scalar(
            dfx.fem.form(
                dot(u_h, n) * ds(LEFT)
            )
)

if comm.rank == 0:
    print(f"e_div_u = {e_div_u}")
    print(f"influx = {flux}")