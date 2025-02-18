from mpi4py   import MPI
from petsc4py import PETSc

import ufl
import numpy   as np
import dolfinx as dfx
from ufl    import div, dot, grad, inner, outer, avg
from scifem import assemble_scalar
from utilities.fem     import create_normal_contribution_bc
from utilities.mesh    import create_square_mesh_with_tags
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block

# Helper functions
def norm_L2(comm: MPI.Comm, v: dfx.fem.Function):
    """Compute the L2(Ω)-norm of v"""
    return np.sqrt(
              comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(inner(v, v) * dx)),
              op=MPI.SUM)
              )

def domain_average(mesh: dfx.mesh.Mesh, v: dfx.fem.Function):
    """Compute the average of a function over the domain"""
    vol = comm.allreduce(
        dfx.fem.assemble_scalar(dfx.fem.form(
                                dfx.fem.Constant(mesh, dfx.default_real_type(1.0)) * dx)
                                ), op=MPI.SUM
    )
    return (1/vol) * comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(v * dx)), op=MPI.SUM)

def u_e_expr(x):
    """Expression for the exact velocity solution to Kovasznay flow"""
    return np.vstack(
        (
            1
            - np.exp((Re/2 - np.sqrt(Re**2 / 4 + 4 * np.pi**2)) * x[0])
            * np.cos(2 * np.pi * x[1]),
            (Re / 2 - np.sqrt(Re**2 / 4 + 4 * np.pi**2))
            / (2 * np.pi)
            * np.exp((Re / 2 - np.sqrt(Re**2 / 4 + 4 * np.pi**2)) * x[0])
            * np.sin(2 * np.pi * x[1]),
        )
    )

def u_quadratic(x, t):
    return np.vstack((Re*np.sin(np.pi*t)*(4*np.ones_like(x[1])),
                      np.zeros_like(x[1])
    ))

def p_e_expr(x):
    """Expression for the exact pressure solution to Kovasznay flow"""
    return (1 / 2) * (1 - np.exp(2 * (Re / 2 - np.sqrt(Re**2 / 4 + 4 * np.pi**2)) * x[0]))

def f_expr(x):
    """Expression for the applied force"""
    return np.vstack((np.zeros_like(x[0]), np.zeros_like(x[0])))

# Jump operator
jump = lambda phi, n: outer(phi('+'), n('+')) + outer(phi('-'), n('-'))

# Simulation parameters
comm = MPI.COMM_WORLD # MPI communicator
N = 16 # Mesh cells
t = 0.0
num_time_steps = 100
t_end = 10
delta_t = t_end/num_time_steps # Timestep size
Re = 25  # Reynolds Number
k = 1  # Polynomial degree

# Create mesh and boundary tags
mesh, ft = create_square_mesh_with_tags(N=N, comm=comm)
LEFT=1; RIGHT=2; BOT=3; TOP=4

# Function spaces for the velocity and for the pressure
V = dfx.fem.functionspace(mesh, ("Brezzi-Douglas-Marini", k+1))
# V = dfx.fem.functionspace(mesh, ("Raviart-Thomas", k+1))
Q = dfx.fem.functionspace(mesh, ("Discontinuous Lagrange", k))
VQ = ufl.MixedFunctionSpace(V, Q)

# Function space for visualising the velocity field
gdim = mesh.geometry.dim
W = dfx.fem.functionspace(mesh, ("Discontinuous Lagrange", k+1, (gdim,)))
print(f"Size of dofmap: {V.dofmap.index_map.size_global+Q.dofmap.index_map.size_global}")

# Define trial and test functions
u, p = ufl.TrialFunctions(VQ)
v, q = ufl.TestFunctions(VQ)

dt = dfx.fem.Constant(mesh, dfx.default_real_type(delta_t))
alpha = dfx.fem.Constant(mesh, dfx.default_real_type(2*6.0 * k**2))

# Mesh cell diameter and facet normal vector
h = ufl.CellDiameter(mesh)
n = ufl.FacetNormal(mesh)

# Integral measures
dx = ufl.Measure('dx', mesh) # Cell integral
ds = ufl.Measure('ds', mesh, subdomain_data=ft) # Boundary facet integral
dS = ufl.Measure('dS', mesh) # Interior facet integral

ds_N = ds(RIGHT)
ds_D = ds(LEFT)
ds_flux = ds((BOT, TOP))

# Get initial condition by solving the Stokes equations
a = (1.0 / Re) * (
    inner(grad(u), grad(v)) * dx
    - inner(avg(grad(u)), jump(v, n)) * dS
    - inner(jump(u, n), avg(grad(v))) * dS
    + (alpha / avg(h)) * inner(jump(u, n), jump(v, n)) * dS
    - inner(grad(u), outer(v, n)) * ds
    - inner(outer(u, n), grad(v)) * (ds_D + ds_flux)
    + (alpha / h) * inner(outer(u, n), outer(v, n)) * (ds_D + ds_flux)
)
a -= inner(p, div(v)) * dx
a -= inner(div(u), q) * dx

a_blocked = dfx.fem.form(ufl.extract_blocks(a))

f = dfx.fem.Function(W)
u_D = dfx.fem.Function(V)
u_D.interpolate(lambda x: u_quadratic(x, t))
u_flux = dfx.fem.Function(V)
L = inner(f, v) * dx + (1 / Re) * (
    -inner(outer(u_D, n), grad(v)) * ds_D + (alpha / h) * inner(outer(u_D, n), outer(v, n)) * ds_D
    -inner(outer(u_flux, n), grad(v)) * ds_flux + (alpha / h) * inner(outer(u_flux, n), outer(v, n)) * ds_flux
)
L += inner(dfx.fem.Constant(mesh, dfx.default_real_type(0.0)), q) * dx
L_blocked = dfx.fem.form(ufl.extract_blocks(L))

# Boundary conditions
# Prescribed profile
inflow_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim-1, ft.find(LEFT))
bc_u = dfx.fem.dirichletbc(u_D, inflow_dofs)
bcs = [bc_u]

# Flux
flux_facets = np.concatenate(([ft.find(tag) for tag in [BOT, TOP]]))
flux_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim-1, flux_facets)
flux_expr = create_normal_contribution_bc(V, -Re*n, flux_facets)
u_flux.interpolate(flux_expr)
bcs.append(dfx.fem.dirichletbc(u_flux, flux_dofs))                                              

# Assemble Stokes problem
A = assemble_matrix_block(a_blocked, bcs=bcs)
A.assemble()
b = assemble_vector_block(L_blocked, a_blocked, bcs=bcs)

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

# Solve Stokes for initial condition
x = A.createVecRight()
ksp.solve(b, x)

# Split the solution
u_h = dfx.fem.Function(V)
p_h = dfx.fem.Function(Q)
p_h.name = "p"
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u_h.x.array[:offset] = x.array_r[:offset]
u_h.x.scatter_forward()
p_h.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
p_h.x.scatter_forward()
# Subtract the average of the pressure since it is only determined up to
# a constant
p_h.x.array[:] -= domain_average(mesh, p_h)

u_vis = dfx.fem.Function(W)
u_vis.name = "u"
u_vis.interpolate(u_h)

# Write initial condition to file

u_file = dfx.io.VTXWriter(comm, "u.bp", u_vis)
p_file = dfx.io.VTXWriter(comm, "p.bp", p_h)
u_file.write(t)
p_file.write(t)

# Create function to store solution and previous time step
u_n = dfx.fem.Function(V)
u_n.x.array[:] = u_h.x.array

# Add the time stepping and convective terms to get the full Navier-Stokes equations

lmbda = ufl.conditional(ufl.gt(dot(u_n, n), 0), 1, 0)
u_uw = lmbda('+') * u('+') + lmbda('-') * u('-')
a += (
    inner(u / dt, v) * dx
    - inner(u, div(outer(v, u_n))) * dx
    + inner((dot(u_n, n))('+') * u_uw, v('+')) * dS
    + inner((dot(u_n, n))('-') * u_uw, v('-')) * dS
    + inner(dot(u_n, n) * lmbda * u, v) * ds
)
a_blocked = dfx.fem.form(ufl.extract_blocks(a))

L += (inner(u_n / dt, v) * dx - inner(dot(u_n, n) * (1 - lmbda) * u_D, v) * ds_D 
                              - inner(dot(u_n, n) * (1 - lmbda) * u_flux, v) * ds_flux) 
L_blocked = dfx.fem.form(ufl.extract_blocks(L))

# Time stepping loop
for _ in range(num_time_steps):
    t += dt.value

    u_D.interpolate(lambda x: u_quadratic(x, t))

    A.zeroEntries()
    assemble_matrix_block(A, a_blocked, bcs=bcs)
    A.assemble()

    with b.localForm() as b_loc: b_loc.set(0)
    assemble_vector_block(b, L_blocked, a_blocked, bcs=bcs)

    # Compute solution
    ksp.solve(b, x)

    u_h.x.array[:offset] = x.array_r[:offset]
    u_h.x.scatter_forward()
    p_h.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
    p_h.x.scatter_forward()
    p_h.x.array[:] -= domain_average(mesh, p_h)

    u_vis.interpolate(u_h)

    # Write to file
    u_file.write(t)
    p_file.write(t)

    # Update u_n
    u_n.x.array[:] = u_h.x.array

u_file.close()
p_file.close()


# Compute divergence L2 norm to check mass conservation
e_div_u = norm_L2(comm, div(u_h))
assert np.isclose(e_div_u, 0.0, atol=float(1.0e5 * np.finfo(dfx.default_real_type).eps))

# Compute flux on boundaries with flux BC
flux = assemble_scalar(dot(u_h, n)*ds_flux)
total_flux = assemble_scalar(dot(u_h, n)*ds)

if comm.rank == 0:
    print(f"e_div_u = {e_div_u}")
    print(f"Flux = {flux}")
    print(f"Total flux = {total_flux}")