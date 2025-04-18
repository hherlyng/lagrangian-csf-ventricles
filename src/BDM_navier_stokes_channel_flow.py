from mpi4py   import MPI
from petsc4py import PETSc

import ufl
import numpy   as np
import dolfinx as dfx
from sys    import argv
from ufl    import div, dot, grad, inner, jump, avg, nabla_grad, outer
from scifem import assemble_scalar
from utilities.fem     import calculate_mean, calculate_norm_L2, eps, stabilization
from utilities.mesh    import create_unit_square_mesh
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, create_matrix_block, create_vector_block

# Parabolic velocity profile
def u_parabolic(x):
    return np.vstack((4*Re*(x[1]-x[1]**2),
                      np.zeros_like(x[1])
    ))

# Simulation parameters
comm = MPI.COMM_WORLD # MPI communicator
N = int(argv[1]) # Mesh cells
t = 0.0
num_time_steps = 20
t_end = 4
delta_t = t_end/num_time_steps # Timestep size
mu_value  = 7e-2 # Dynamic viscosity
rho_value = 1e2 # Fluid density
Re = rho_value/mu_value  # Reynolds Number
k = 1 # Polynomial degree

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
u_.interpolate(u_parabolic) # Interpolate initial condition
u__.interpolate(u_parabolic) # Interpolate initial condition

dt = dfx.fem.Constant(mesh, dfx.default_real_type(delta_t)) # Timestep
gamma = dfx.fem.Constant(mesh, dfx.default_real_type(10.0)) # BDM penalty parameter
beta = dfx.fem.Constant(mesh, dfx.default_scalar_type(50.0)) # Nitsche penalty parameter
mu = dfx.fem.Constant(mesh, dfx.default_real_type(mu_value)) # Dynamic viscosity
rho = dfx.fem.Constant(mesh, dfx.default_scalar_type(rho_value)) # Fluid density

# Mesh cell diameter and facet normal vector
h = ufl.CellDiameter(mesh)
n = ufl.FacetNormal(mesh)

# Integral measures
dx = ufl.Measure('dx', mesh) # Cell integral
ds = ufl.Measure('ds', mesh, subdomain_data=ft) # Exterior facet integral
dS = ufl.Measure('dS', mesh) # Interior facet integral

# Upwind velocity operator (equals 1 on inflow boundary, 0 on outflow boundary)
zeta = ufl.conditional(ufl.lt(dot(u_, n), 0), 1, 0)

# Navier-Stokes equations bilinear form in block form
a00 = (
    3/2*rho/dt * inner(u, v) * dx # Time derivative
    + rho*inner(dot(u_, nabla_grad(u)), v) * dx # Convective term
    #  - rho*inner(outer(u_, u), grad(v)) * dx # Convective term
    #  - zeta*rho*dot(dot(outer(u_, u), n), v) * (ds(LEFT) + ds(RIGHT)) # Convective term
     + 2*mu*inner(eps(u), eps(v))*dx # Viscous dissipation
     + stabilization(u, v, mu, gamma) # BDM stabilization
     - mu*inner(dot(grad(u).T, n), v) * (ds(LEFT) + ds(RIGHT)) # Parallel flow at outlet
     + beta/h*inner(u, v)*(ds(TOP)+ds(BOT)) # Enforce noslip
    )
# Add convective term stabilization
a00 += (-rho*1/2*dot(jump(u_), n('+')) * avg(dot(u, v)) * dS 
      - rho*dot(avg(u_), n('+')) * dot(jump(u), avg(v)) * dS 
      - zeta*rho*1/2*dot(u_, n) * dot(u, v) * (ds(LEFT) + ds(RIGHT))
)
a01 = -inner(p, div(v))*dx
a10 = -inner(q, div(u))*dx
a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*dx

# Linear form
L0 = 2*rho/dt * inner(u_, v) * dx - 1/2*rho/dt * inner(u__, v) * dx # Time derivative
L0 -= dot(dfx.fem.Constant(mesh, dfx.default_scalar_type(8.0))*n, v) * ds(LEFT) # Impose pressure
        
L1 = inner(dfx.fem.Function(Q), q)*dx

a = dfx.fem.form([[a00, a01],
                  [a10, a11]])
L = dfx.fem.form([L0, L1])

# Strong boundary conditions: Impermeability on top/bottom
impermeability_dofs = dfx.fem.locate_dofs_topological(V, facet_dim, np.concatenate((ft.find(TOP), ft.find(BOT))))
bc_impermeability = dfx.fem.dirichletbc(dfx.fem.Function(V), impermeability_dofs)
bcs = [bc_impermeability]               

inflow = dfx.fem.Function(V)
inflow.interpolate(u_parabolic)
bc_inflow = dfx.fem.dirichletbc(inflow, dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(LEFT)))
# bcs.append(bc_inflow)

# Assemble the Navier-Stokes problem
A = create_matrix_block(a)
b = create_vector_block(L)
x = A.createVecRight() # Solution vector

# Create and configure solver
ksp = PETSc.KSP().create(comm)  # type: ignore
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()  # type: ignore
opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
opts["ksp_error_if_not_converged"] = 1 # Throw an error if KSP solver does not converge
ksp.setFromOptions()

# Create solution functions
u_h = dfx.fem.Function(V)
p_h = dfx.fem.Function(Q)

u_vis = dfx.fem.Function(W)
u_vis.name = "u"
u_vis.interpolate(u_)

# Write initial condition to file
u_file = dfx.io.VTXWriter(comm, "../output/square-mesh/flow/navier-stokes/channel_flow_velocity.bp", u_vis)
p_file = dfx.io.VTXWriter(comm, "../output/square-mesh/flow/navier-stokes/channel_flow_pressure.bp", p_h)
u_file.write(t)
p_file.write(t)

# Create offsets for the blocked functions
offset_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
offset_p = offset_u + Q.dofmap.index_map.size_local*Q.dofmap.index_map_bs

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
# assert np.isclose(e_div_u, 0.0, atol=float(1.0e5 * np.finfo(dfx.default_real_type).eps))

# Calculate error in velocity approximation
u_exact = dfx.fem.Function(V)
u_exact.interpolate(u_parabolic)
L2_error = assemble_scalar(
                dfx.fem.form(
                    inner(u_h-u_exact, u_h-u_exact) * dx
                )
            )
L2_error = np.sqrt(L2_error)
if comm.rank == 0:
    print(f"e_div_u = {e_div_u}")
    print(f"e_u_L2 = {L2_error}")