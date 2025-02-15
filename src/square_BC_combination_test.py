from mpi4py import MPI
import ufl
import numpy   as np
import dolfinx as dfx
from petsc4py import PETSc
from ufl import inner, dot, grad, div
from scifem import create_real_functionspace, assemble_scalar
from basix.ufl import element
from utilities.fem  import eps, stabilization, create_normal_contribution_bc, assemble_nested_system
from utilities.mesh import create_square_mesh_with_tags


class LeftBoundaryDeformation:
    def __init__(self) -> None:
        self.t = 0
        self.A = 1/2
        self.T = 1/2
    
    def __call__(self, x):
        return self.A*np.sin(2*np.pi*self.t/self.T)*np.stack((np.ones(x.shape[1]),
                                                              np.zeros(x.shape[1])))
class RightBoundaryDeformation:
    def __init__(self) -> None:
        self.t = 0
        self.A = -1
        self.T = 1
    
    def __call__(self, x):
        return self.A*np.sin(2*np.pi*self.t/self.T)*np.cos(np.pi*x[1])*np.stack((np.ones(x.shape[1]),
                                                              np.zeros(x.shape[1])))

def create_direct_solver(A: PETSc.Mat, comm: MPI.Comm):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
    pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
    
    return ksp

def solve_stokes(a: dfx.fem.form, L: dfx.fem.form, bcs: list[dfx.fem.DirichletBC],
                 uh: dfx.fem.Function, ph: dfx.fem.Function):
    
    A, b = assemble_nested_system(a, L, bcs)

    ksp = create_direct_solver(A, comm)

    w = PETSc.Vec().createNest([uh.x.petsc_vec, ph.x.petsc_vec])
    ksp.solve(b, w)
    assert ksp.getConvergedReason() > 0, print(ksp.getConvergedReason())

    # MPI communcation
    uh.x.scatter_forward()
    ph.x.scatter_forward()

    return uh, ph
    
mesh, ft = create_square_mesh_with_tags(N_cells=16)
comm = mesh.comm

LEFT = 1; RIGHT = 2; BOT = 3; TOP = 4


facet_dim = mesh.topology.dim-1
mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity

ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
dx = ufl.Measure('dx', domain=mesh)

n = ufl.FacetNormal(mesh)

mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(7e-4)) #[kg/(m*s)] #*1e-2 # Dynamic viscosity [kg/(cm*s)]
penalty = dfx.fem.Constant(mesh, dfx.default_scalar_type(10.0))

bdm1_el = element("BDM", mesh.basix_cell(), 1)
dg0_el  = element("DG", mesh.basix_cell(), 0)
dg1_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
V = dfx.fem.functionspace(mesh, bdm1_el)
Q = dfx.fem.functionspace(mesh, dg0_el)
R = create_real_functionspace(mesh)
DG_vec = dfx.fem.functionspace(mesh, dg1_vec_el)
v_zero = dfx.fem.Function(V) # Zero velocity for BCs

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
lm, dlm = ufl.TrialFunction(R), ufl.TestFunction(R)

# Stokes problem in reference domain accounting for the deformation
a00 = (2*mu*inner(eps(u), eps(v))*dx # Viscous dissipation
        + stabilization(u, v, mu, penalty) # BDM stabilization
        - mu*inner(dot(grad(u).T, n), v)*ds(BOT) # Parallel flow at inlet/outlet
        )
a01 = p*div(v)*dx
a02 = None

a10 = q*div(u)*dx
a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*dx
a12 = lm*q*dx

a20 = None
a21 = dlm*p*dx
a22 = None

L0 = inner(v_zero, v)*dx \
    # + inner(tau_val*tangent(tau, n), tangent(v, n))*ds(cilia_tags) \
L1 = inner(dfx.fem.Function(Q), q)*dx
L2 = inner(dfx.fem.Function(R), dlm)*dx

# a = dfx.fem.form([[a00, a01, a02],
#                   [a10, a11, a12],
#                   [a20, a21, a22]])
a = dfx.fem.form([[a00, a01],
                  [a10, a11]])
L = dfx.fem.form([L0, L1])
# L = dfx.fem.form([L0, L1, L2])

# Set choroid plexus inflow velocity BC strongly
# Create expressions with positive and negative z-component of the velocity,
# and interpolate the expressions into finite element functions.
# chp_prod = 5.833e-9 # Corresponds to 504 ml production per day [Czosnyka et al.]
# chp_area = assemble_scalar(1*ds(TOP)) # The area of the choroid plexus boundary
# chp_velocity = chp_prod/chp_area
v_dofs_chp = dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(TOP))
v_chp = dfx.fem.Function(V)
v_chp_expr = create_normal_contribution_bc(V, -2*n, ft.find(TOP))
v_chp.interpolate(v_chp_expr)
bcs = [dfx.fem.dirichletbc(v_chp, v_dofs_chp)]

# Impose deformation velocity on the rest of the boundary
v_dofs_defo_left  = dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(LEFT))
v_dofs_defo_right = dfx.fem.locate_dofs_topological(V, facet_dim, ft.find(RIGHT))

v_defo_left  = dfx.fem.Function(V)
v_defo_right = dfx.fem.Function(V)

bcs.append(dfx.fem.dirichletbc(v_defo_left,  v_dofs_defo_left ))
bcs.append(dfx.fem.dirichletbc(v_defo_right, v_dofs_defo_right))

v_defo_left_expr = LeftBoundaryDeformation()
v_defo_right_expr = RightBoundaryDeformation()

uh = dfx.fem.Function(V)
ph = dfx.fem.Function(Q)
lmh = dfx.fem.Function(R)

uh_out = dfx.fem.Function(DG_vec)
velocity_out = dfx.io.VTKFile(comm, "square_velocity.pvd", "w")
pressure_out = dfx.io.VTKFile(comm, "square_pressure.pvd", "w")

dt = 0.01
T = 1
N = int(T / dt)
for t in np.linspace(dt, T, N):
    print(f'Time = {t:.4g}')
    v_defo_left_expr.t = t
    v_defo_right_expr.t = t
    v_defo_left.interpolate(v_defo_left_expr)
    v_defo_right.interpolate(v_defo_right_expr)
    
    uh, ph = solve_stokes(a, L, bcs, uh, ph)
    
    uh_out.interpolate(uh)
    velocity_out.write_mesh(mesh, t)
    velocity_out.write_function(uh_out, t)
    pressure_out.write_mesh(mesh, t)
    pressure_out.write_function(ph, t)

    print(f'Boundary flux: {assemble_scalar(dot(uh, n)*ds):.4g}')

velocity_out.close()