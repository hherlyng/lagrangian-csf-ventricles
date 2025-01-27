import ufl
import time

import numpy   as np
import dolfinx as dfx

from ufl       import inner, dot, grad, div
from scifem    import assemble_scalar
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from wall_deformation_BC import WallDeformation

print = PETSc.Sys.Print

# Facet tags
CANAL_WALL = 13
CANAL_OUT = 23
THIRD_VENTRICLE_WALL = 14
THIRD_VENTRICLE_FORAMINA = 46
AQUEDUCT_WALL = 15
AQUEDUCT_TOP = 45
AQUEDUCT_BOT = 58
FORAMINA_34_WALL = 16
LATERAL_VENTRICLES_FORAMINA = 67
LATERAL_VENTRICLES_WALL = 17
FOURTH_VENTRICLE_WALL = 18
FOURTH_VENTRICLE_OUT = 38
CHOROID_PLEXUS_LATERAL = 101
CHOROID_PLEXUS_THIRD = 103
CHOROID_PLEXUS_FOURTH = 104
LATERAL_APERTURES = 28

# Cell tags
CANAL = 3
THIRD_VENTRICLE = 4
AQUEDUCT = 5
FORAMINA_34 = 6
LATERAL_VENTRICLES = 7
FOURTH_VENTRICLE = 8

zero_displacement_tags = [CANAL_OUT]#(CANAL_OUT, CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD, CHOROID_PLEXUS_FOURTH)
# wall_displacement_tags = [LATERAL_VENTRICLES_WALL]
wall_displacement_tags = (AQUEDUCT_WALL, FORAMINA_34_WALL, LATERAL_VENTRICLES_WALL, FOURTH_VENTRICLE_WALL,
                          CANAL_WALL, THIRD_VENTRICLE_WALL)
cilia_tags = (AQUEDUCT_WALL, FORAMINA_34_WALL, LATERAL_VENTRICLES_WALL, FOURTH_VENTRICLE_WALL,
              CANAL_WALL, THIRD_VENTRICLE_WALL, CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD, CHOROID_PLEXUS_FOURTH)
choroid_plexus_tags = (CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD, CHOROID_PLEXUS_FOURTH)

# Solve stationary Stokes in moving domain by ALE method. Wall motion is 
# prescribed in time.
comm = MPI.COMM_WORLD
mesh_prefix = 'coarse'
with dfx.io.XDMFFile(comm, f"./geometries/{mesh_prefix}_ventricles_mesh_tagged.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    
    # Generate mesh entities    
    facet_dim = mesh.topology.dim-1
    mesh.topology.create_entities(facet_dim) # Create facets
    ft = xdmf.read_meshtags(mesh, "ft")
    ct = xdmf.read_meshtags(mesh, "ct")

mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity
num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts # Total number of facets

ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
dx = ufl.Measure('dx', domain=mesh, subdomain_data=ct)

# Parameters of the boundary conditions
f_cardiac = 62/60 # Heartbeats per second
f_respiratory = 0.15 # Breaths per second
T_cardiac = 1/f_cardiac
T_respiratory = 1/f_respiratory
A_cardiac = 12.3*1e2 # Amplitude of cardiac cycle pulsations pressure [from Liu, Baledent et al. 2024]
A_respiratory = 9.5*1e2 # # Amplitude of respiratory cycle pulsations pressure [from Liu, Baledent et al. 2024]
# 1e2 factor is to make pressure in unit of Pa*1e2, to match the Stokes eq. formulation in centimeters

# Choroid plexus BC
chp_area = assemble_scalar(1*ds(choroid_plexus_tags)) # The area of the choroid plexus boundary
v_in = 0.31 # Volumetric flux through ventricles [cm^3/s]
chp_velocity = v_in/chp_area # The choroid plexus influx velocity
class ChoroidPlexusFlux:
    def __init__(self): 
        self.velocity_value = chp_velocity

    def __call__(self, x):
        return self.velocity_value*np.stack((np.ones(x.shape[1]),
                                   np.ones(x.shape[1]),
                                   np.ones(x.shape[1])))

# Pressure expression class
class PressureBC:
    def __init__(self):
        self.t = 0
        self.A_1 = A_cardiac
        self.T_1 = T_cardiac
        self.A_2 = A_respiratory
        self.T_2 = T_respiratory

    def __call__(self, x):
        cardiac_term = self.A_1*np.cos(2*np.pi*self.t/self.T_1)
        respiratory_term = self.A_2*np.cos(2*np.pi*self.t/self.T_2)
        total = cardiac_term + respiratory_term
        return total*np.ones(x.shape[1])

# Operators
# NOTE: these are the jump operators from Krauss, Zikatonov paper.
# Jump is just a difference and it preserves the rank 
Jump = lambda arg: arg('+') - arg('-')

# Average uses dot with normal and AGAIN MINUS; it reduces the rank
Avg = lambda arg, n: .5*(dot(arg('+'), n('+')) - dot(arg('-'), n('-')))

# Action of (1 - n x n) on a vector yields the tangential component
Tangent = lambda v, n: v - n*dot(v, n)

# The ALE problem needs to extend boundary deformation to the entire domain
# to define mesh displacement field
vec_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))

n = ufl.FacetNormal(mesh)

scal_el = element("Lagrange", mesh.basix_cell(), 1)
bdm_el = element("BDM", mesh.basix_cell(), 1)
dg_el  = element("DG", mesh.basix_cell(), 0)
dg_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
V = dfx.fem.functionspace(mesh, bdm_el)
Q = dfx.fem.functionspace(mesh, dg_el)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

# Symmetric gradient
Eps = lambda arg: ufl.sym(grad(arg))

mu = 7e-4*1e-2 # Dynamic viscosity [kg/(cm*s)]
BDM_penalty = 50

def stabilization(u: ufl.TrialFunction, v: ufl.TestFunction, consistent: bool=True):
    """ Displacement/Flux Stabilization term from Krauss et al paper. 

    Parameters
    ----------
    u : ufl.TrialFunction
        The finite element trial function.
    
    v : ufl.TestFunction
        The finite element test function.
    
    consistent : bool
        Add symmetric gradient terms to the form if True.

    Returns
    -------
    ufl.Coefficient
        Stabilization term for the bilinear form.
    """

    hA = ufl.avg(ufl.CellDiameter(mesh)) # Facet normal vector and average cell diameter
    dS = ufl.Measure('dS', domain=mesh) # Interior facet integral measure

    if consistent: # Add symmetrization terms
        return (-inner(Avg(2*mu*Eps(u), n), Jump(Tangent(v, n)))*dS
                -inner(Avg(2*mu*Eps(v), n), Jump(Tangent(u, n)))*dS
                + 2*mu*(BDM_penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS)

    # For preconditioning
    return 2*mu*(BDM_penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS

beta = dfx.fem.Constant(mesh, dfx.default_scalar_type(10.0)) # Noslip penalty parameter

p_bc = dfx.fem.Function(Q)
p_bc_expr = PressureBC()

# Tangential traction BC
tau = 0#dfx.fem.Constant(mesh, dfx.default_scalar_type(0.1))
tau_vec   = tau*ufl.as_vector((0, 1, 1)) # Stress vector to be projected tangentially onto the mesh
tangent_traction_dorsal = lambda n: Tangent(tau_vec, n) # Use the tau expression to define the tangent traction vector

# ds_ = ds(noslip_bdry_tags)
# Stokes problem in reference domain accounting for the deformation
a00 = (2*mu*inner(Eps(u), Eps(v))*dx # Viscous dissipation
      + stabilization(u, v) # BDM stabilization
      - mu*inner(dot(grad(u).T, n), v)*(ds(CANAL_OUT) + ds(LATERAL_APERTURES)) # Parallel flow at inlet/outlet
    #   + beta*dot(Tangent(u, n), Tangent(v, n))*J*ds_ # Weakly enforce zero tangential velocity
      )
a01 = inner(p, div(v))*dx
a10 = inner(q, div(u))*dx
a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*dx

L0 = inner(dfx.fem.Function(V), v)*dx \
    + inner(Tangent(v, n), tangent_traction_dorsal(n))*ds(cilia_tags) \
    + inner(p_bc*n, v)*ds(CANAL_OUT)
L1 = inner(dfx.fem.Function(Q), q)*dx

a = dfx.fem.form([[a00, a01], [a10, a11]])
L = dfx.fem.form([L0, L1])

# Set boundary conditions on velocity
v_wall = dfx.fem.Function(V) # Wall deformation BC
v_wall_expr = WallDeformation(derivative=True)
facets_wall = np.concatenate(([ft.find(tag) for tag in wall_displacement_tags]))
v_dofs_wall = dfx.fem.locate_dofs_topological(V, facet_dim, facets_wall)

v_chp = dfx.fem.Function(V) # Choroid plexus BC
v_chp_expr = ChoroidPlexusFlux()
# v_chp.interpolate(v_chp_expr)
facets_chp = np.concatenate(([ft.find(tag) for tag in choroid_plexus_tags]))
v_dofs_chp = dfx.fem.locate_dofs_topological(V, facet_dim, facets_chp)

bcs_stokes = [dfx.fem.dirichletbc(v_wall, v_dofs_wall),
              dfx.fem.dirichletbc(v_chp, v_dofs_chp)]


uh = dfx.fem.Function(dfx.fem.functionspace(mesh, vec_el)); uh.name = 'uh'
ph = dfx.fem.Function(dfx.fem.functionspace(mesh, dg_el)); ph.name = 'ph'

velocity_output = dfx.io.VTKFile(comm, f"./output/stationary-mesh/{mesh_prefix}-BDM/velocity.pvd", "w")
pressure_output = dfx.io.VTKFile(comm, f"./output/stationary-mesh/{mesh_prefix}-BDM/pressure.pvd", "w")

def assemble_system(lhs_form, rhs_form, bcs):
    A = dfx.fem.petsc.assemble_matrix_nest(lhs_form, bcs=bcs)
    A.assemble()

    b = dfx.fem.petsc.assemble_vector_nest(rhs_form)
    dfx.fem.petsc.apply_lifting_nest(b, lhs_form, bcs=bcs)
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    spaces = dfx.fem.extract_function_spaces(rhs_form)
    bcs0 = dfx.fem.bcs_by_block(spaces, bcs)
    dfx.fem.petsc.set_bc_nest(b, bcs0)
    return A, b

def create_block_solver(A):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    return ksp

def solve_stokes():

    A, b = assemble_system(a, L, bcs_stokes)

    ksp = create_block_solver(A)

    u, p = dfx.fem.Function(V), dfx.fem.Function(Q)
    w = PETSc.Vec().createNest([u.x.petsc_vec, p.x.petsc_vec])
    ksp.solve(b, w)
    assert ksp.getConvergedReason() > 0
    u.x.scatter_forward()
    p.x.scatter_forward()

    return u, p

print("Number of dofs Stokes eqs: ", V.dofmap.index_map.size_global+Q.dofmap.index_map.size_global)

tic = time.perf_counter()

for t in np.linspace(0, 1, 12):
    # Update time variable of normal traction BC and interpolate BC function
    p_bc_expr.t = t

    # Update time variable of wall deformation velocity BC
    v_wall_expr.t = t

    # Interpolate BC functions
    p_bc.interpolate(p_bc_expr)
    v_wall.interpolate(v_wall_expr)

    uh_, ph_ = solve_stokes() # Solve the Stokes equations

    # Update output functions
    uh.interpolate(uh_)
    ph.interpolate(ph_)

    # Write output
    velocity_output.write_mesh(mesh, t)
    velocity_output.write_function(uh, t)
    pressure_output.write_mesh(mesh, t)
    pressure_output.write_function(ph, t)

    # Calculate mean pressure
    vol = assemble_scalar(1*dx)
    print("Mean pressure: ", 1/vol*assemble_scalar(ph*dx))

print(f"Solution loop time elapsed: {time.perf_counter()-tic:.4f} sec")

# Close output files
velocity_output.close()
pressure_output.close()