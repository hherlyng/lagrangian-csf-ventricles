from mpi4py   import MPI
from petsc4py import PETSc

import ufl
import numpy   as np
import dolfinx as dfx
from sys    import argv
from ufl    import div, dot, inner, nabla_grad
from scifem import assemble_scalar
from utilities.fem     import calculate_mean, calculate_norm_L2, eps, stabilization
from utilities.mesh    import create_rectangle_mesh
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, create_matrix_block, create_vector_block

# Parabolic velocity profile
def u_taylor(t, x):
    """ Exact velocity expression for the Taylor vortex. """
    return np.vstack((
               -np.cos(x[0])*np.sin(x[1])*np.exp(-2*t/Re),
                np.sin(x[0])*np.cos(x[1])*np.exp(-2*t/Re)
            ))

def p_taylor(t, x):
    """ Exact pressure expression for the Taylor vortex. """
    return -1/4*(np.cos(2*x[0]) + np.cos(2*x[1]))*np.exp(-4*t/Re)

# Simulation parameters
comm = MPI.COMM_WORLD # MPI communicator
N = int(argv[1]) # Mesh cells
t = 0.0
num_time_steps = 64
t_end = 6
delta_t = t_end/num_time_steps # Timestep size
mu_value  = 1e-3 # Dynamic viscosity
rho_value = 1 # Fluid density
Re = rho_value/mu_value  # Reynolds Number
k = 3 # Polynomial degree

# Create mesh and boundary tags
lower_left  = [-np.pi/2, -np.pi/2]
upper_right = [np.pi/2, np.pi/2]
mesh, ft = create_rectangle_mesh(N=N,
                                 lower_left=lower_left,
                                 upper_right=upper_right,
                                 comm=comm)
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

u_ = dfx.fem.Function(V) # Velocity at previous timestep
u_.interpolate(lambda x: u_taylor(delta_t, x)) # Interpolate initial condition
u_bc_taylor = dfx.fem.Function(V)

dt = dfx.fem.Constant(mesh, dfx.default_real_type(delta_t)) # Timestep
gamma = dfx.fem.Constant(mesh, dfx.default_real_type(10.0)) # BDM penalty parameter
mu = dfx.fem.Constant(mesh, dfx.default_real_type(mu_value)) # Dynamic viscosity
rho = dfx.fem.Constant(mesh, dfx.default_scalar_type(rho_value)) # Fluid density

# Mesh cell diameter and facet normal vector
h = ufl.CellDiameter(mesh)
n = ufl.FacetNormal(mesh)

# Integral measures
dx = ufl.Measure('dx', mesh) # Cell integral
ds = ufl.Measure('ds', mesh, subdomain_data=ft) # Facet integral

# Navier-Stokes equations bilinear form in block form
a00 = (rho/dt * inner(u , v) * dx # Time derivative
     + rho*inner(dot(u_, nabla_grad(u)), v) * dx # Convective term
     + 2*mu*inner(eps(u), eps(v))*dx # Viscous dissipation
     + stabilization(u, v, mu, gamma) # BDM stabilization
     + gamma/h*inner(u, v) * ds
    )
a01 = -inner(p, div(v))*dx
a10 = -inner(q, div(u))*dx
a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*dx

# Linear form
L0 = rho/dt * inner(u_, v) * dx # Time derivative
L0 += gamma/h * inner(u_bc_taylor, v) * ds # Nitsche BC
        
L1 = inner(dfx.fem.Function(Q), q)*dx

a = dfx.fem.form([[a00, a01],
                  [a10, a11]])
L = dfx.fem.form([L0, L1])

# Strong boundary conditions: set the Taylor velocity
taylor_dofs = dfx.fem.locate_dofs_topological(V, facet_dim, ft.indices)
u_bc_taylor.interpolate(lambda x: u_taylor(t, x))
bc_taylor = dfx.fem.dirichletbc(u_bc_taylor, taylor_dofs)
bcs = [bc_taylor]                                                                                   

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
u_vis.interpolate(u_h)

# Write initial condition to file
u_file = dfx.io.VTXWriter(comm, "u.bp", u_vis)
p_file = dfx.io.VTXWriter(comm, "p.bp", p_h)
u_file.write(t)
p_file.write(t)

# Create offsets for the blocked functions
offset_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
offset_p = offset_u + Q.dofmap.index_map.size_local*Q.dofmap.index_map_bs
# Time stepping loop
for _ in range(num_time_steps):
    t += dt.value
    u_bc_taylor.interpolate(lambda x: u_taylor(t, x))

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

    # Update u_n
    u_.x.array[:] = u_h.x.array

u_file.close()
p_file.close()


# Compute divergence L2 norm to check mass conservation
e_div_u = calculate_norm_L2(comm, div(u_h), dX=dx)
# assert np.isclose(e_div_u, 0.0, atol=float(1.0e5 * np.finfo(dfx.default_real_type).eps))

# Calculate error in velocity approximation
u_exact = dfx.fem.Function(V)
u_exact.interpolate(lambda x: u_taylor(t=t, x=x))
L2_error = assemble_scalar(
                dfx.fem.form(
                    inner(u_h-u_exact, u_h-u_exact) * dx
                )
            )
L2_error = np.sqrt(L2_error)
if comm.rank == 0:
    print(f"e_div_u = {e_div_u}")
    print(f"e_u_L2 = {L2_error}")