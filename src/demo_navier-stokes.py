from mpi4py   import MPI
from petsc4py import PETSc

import numpy   as np
import dolfinx as dfx
from dolfinx import default_real_type, fem, io, mesh
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from ufl import (
    CellDiameter,
    FacetNormal,
    MixedFunctionSpace,
    TestFunctions,
    TrialFunctions,
    avg,
    conditional,
    div,
    dot,
    dS,
    ds,
    dx,
    extract_blocks,
    grad,
    gt,
    inner,
    outer,
)

# Helper functions

def norm_L2(comm: MPI.Comm, v: dfx.fem.Function):
    """Compute the L2(Ω)-norm of v"""
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(inner(v, v) * dx)), op=MPI.SUM))


def domain_average(msh: dfx.mesh.Mesh, v: dfx.fem.Function):
    """Compute the average of a function over the domain"""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, default_real_type(1.0)) * dx)), op=MPI.SUM
    )
    return (1 / vol) * msh.comm.allreduce(fem.assemble_scalar(fem.form(v * dx)), op=MPI.SUM)


def u_e_expr(x):
    """Expression for the exact velocity solution to Kovasznay flow"""
    return np.vstack(
        (
            1
            - np.exp((Re / 2 - np.sqrt(Re**2 / 4 + 4 * np.pi**2)) * x[0])
            * np.cos(2 * np.pi * x[1]),
            (Re / 2 - np.sqrt(Re**2 / 4 + 4 * np.pi**2))
            / (2 * np.pi)
            * np.exp((Re / 2 - np.sqrt(Re**2 / 4 + 4 * np.pi**2)) * x[0])
            * np.sin(2 * np.pi * x[1]),
        )
    )


def p_e_expr(x):
    """Expression for the exact pressure solution to Kovasznay flow"""
    return (1 / 2) * (1 - np.exp(2 * (Re / 2 - np.sqrt(Re**2 / 4 + 4 * np.pi**2)) * x[0]))


def f_expr(x):
    """Expression for the applied force"""
    return np.vstack((np.zeros_like(x[0]), np.zeros_like(x[0])))


# We define some simulation parameters

n = 16 # Mesh cells
num_time_steps = 25
t_end = 10
Re = 25  # Reynolds Number
k = 1  # Polynomial degree

msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)

# Function spaces for the velocity and for the pressure
V = fem.functionspace(msh, ("Raviart-Thomas", k + 1))
Q = fem.functionspace(msh, ("Discontinuous Lagrange", k))
VQ = MixedFunctionSpace(V, Q)

# Function space for visualising the velocity field
gdim = msh.geometry.dim
W = fem.functionspace(msh, ("Discontinuous Lagrange", k + 1, (gdim,)))

# Define trial and test functions

u, p = TrialFunctions(VQ)
v, q = TestFunctions(VQ)

delta_t = fem.Constant(msh, default_real_type(t_end / num_time_steps))
alpha = fem.Constant(msh, default_real_type(6.0 * k**2))

h = CellDiameter(msh)
n = FacetNormal(msh)


def jump(phi, n):
    return outer(phi("+"), n("+")) + outer(phi("-"), n("-"))


# Get initial condition by solving the Stokes equations
a = (1.0 / Re) * (
    inner(grad(u), grad(v)) * dx
    - inner(avg(grad(u)), jump(v, n)) * dS
    - inner(jump(u, n), avg(grad(v))) * dS
    + (alpha / avg(h)) * inner(jump(u, n), jump(v, n)) * dS
    - inner(grad(u), outer(v, n)) * ds
    - inner(outer(u, n), grad(v)) * ds
    + (alpha / h) * inner(outer(u, n), outer(v, n)) * ds
)
a -= inner(p, div(v)) * dx
a -= inner(div(u), q) * dx

a_blocked = fem.form(extract_blocks(a))

f = fem.Function(W)
u_D = fem.Function(V)
u_D.interpolate(u_e_expr)
L = inner(f, v) * dx + (1 / Re) * (
    -inner(outer(u_D, n), grad(v)) * ds + (alpha / h) * inner(outer(u_D, n), outer(v, n)) * ds
)
L += inner(fem.Constant(msh, default_real_type(0.0)), q) * dx
L_blocked = fem.form(extract_blocks(L))

# Boundary conditions
boundary_facets = mesh.exterior_facet_indices(msh.topology)
boundary_vel_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, boundary_facets)
bc_u = fem.dirichletbc(u_D, boundary_vel_dofs)
bcs = [bc_u]

# Assemble Stokes problem
A = assemble_matrix_block(a_blocked, bcs=bcs)
A.assemble()
b = assemble_vector_block(L_blocked, a_blocked, bcs=bcs)

# Create and configure solver
ksp = PETSc.KSP().create(msh.comm)  # type: ignore
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()  # type: ignore
opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
opts["ksp_error_if_not_converged"] = 1
ksp.setFromOptions()

# Solve Stokes for initial condition
x = A.createVecRight()
ksp.solve(b, x)

# Split the solution
u_h = fem.Function(V)
p_h = fem.Function(Q)
p_h.name = "p"
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u_h.x.array[:offset] = x.array_r[:offset]
u_h.x.scatter_forward()
p_h.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
p_h.x.scatter_forward()
# Subtract the average of the pressure since it is only determined up to
# a constant
p_h.x.array[:] -= domain_average(msh, p_h)

u_vis = fem.Function(W)
u_vis.name = "u"
u_vis.interpolate(u_h)

# Write initial condition to file
t = 0.0

u_file = io.VTXWriter(msh.comm, "u.bp", u_vis)
p_file = io.VTXWriter(msh.comm, "p.bp", p_h)
u_file.write(t)
p_file.write(t)

# Create function to store solution and previous time step
u_n = fem.Function(V)
u_n.x.array[:] = u_h.x.array

# Add the time stepping and convective terms to get the full Navier-Stokes equations

lmbda = conditional(gt(dot(u_n, n), 0), 1, 0)
u_uw = lmbda("+") * u("+") + lmbda("-") * u("-")
a += (
    inner(u / delta_t, v) * dx
    - inner(u, div(outer(v, u_n))) * dx
    + inner((dot(u_n, n))("+") * u_uw, v("+")) * dS
    + inner((dot(u_n, n))("-") * u_uw, v("-")) * dS
    + inner(dot(u_n, n) * lmbda * u, v) * ds
)
a_blocked = fem.form(extract_blocks(a))

L += inner(u_n / delta_t, v) * dx - inner(dot(u_n, n) * (1 - lmbda) * u_D, v) * ds
L_blocked = fem.form(extract_blocks(L))

# Time stepping loop
for n in range(num_time_steps):
    t += delta_t.value

    A.zeroEntries()
    fem.petsc.assemble_matrix_block(A, a_blocked, bcs=bcs)  # type: ignore
    A.assemble()

    with b.localForm() as b_loc:
        b_loc.set(0)
    fem.petsc.assemble_vector_block(b, L_blocked, a_blocked, bcs=bcs)  # type: ignore

    # Compute solution
    ksp.solve(b, x)

    u_h.x.array[:offset] = x.array_r[:offset]
    u_h.x.scatter_forward()
    p_h.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]
    p_h.x.scatter_forward()
    p_h.x.array[:] -= domain_average(msh, p_h)

    u_vis.interpolate(u_h)

    # Write to file
    u_file.write(t)
    p_file.write(t)

    # Update u_n
    u_n.x.array[:] = u_h.x.array

u_file.close()
p_file.close()

# Compare the computed solution to the exact solution
# Function spaces for exact velocity and pressure
V_e = fem.functionspace(msh, ("Lagrange", k + 3, (gdim,)))
Q_e = fem.functionspace(msh, ("Lagrange", k + 2))

u_e = fem.Function(V_e)
u_e.interpolate(u_e_expr)

p_e = fem.Function(Q_e)
p_e.interpolate(p_e_expr)

# Compute errors
e_u = norm_L2(msh.comm, u_h - u_e)
e_div_u = norm_L2(msh.comm, div(u_h))

# This scheme conserves mass exactly, so check this
assert np.isclose(e_div_u, 0.0, atol=float(1.0e5 * np.finfo(default_real_type).eps))
p_e_avg = domain_average(msh, p_e)
e_p = norm_L2(msh.comm, p_h - (p_e - p_e_avg))

if msh.comm.rank == 0:
    print(f"e_u = {e_u}")
    print(f"e_div_u = {e_div_u}")
    print(f"e_p = {e_p}")