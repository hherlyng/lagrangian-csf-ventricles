import ufl
import time

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from ufl       import inner, dot, grad, sym, div
from scifem    import assemble_scalar
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from utilities.fem import stabilization, tangent, eps
from dolfinx.fem.petsc import LinearProblem

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

cilia_tags = (AQUEDUCT_WALL, FORAMINA_34_WALL, LATERAL_VENTRICLES_WALL, FOURTH_VENTRICLE_WALL,
              CANAL_WALL, THIRD_VENTRICLE_WALL, CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD, CHOROID_PLEXUS_FOURTH)
choroid_plexus_tags = (CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD, CHOROID_PLEXUS_FOURTH)
impermeability_tags = (tag for tag in cilia_tags if tag not in choroid_plexus_tags)

# Read mesh
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
num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts # Total number of facets

ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
dx = ufl.Measure('dx', domain=mesh, subdomain_data=ct)

# Choroid plexus BC
chp_area = assemble_scalar(1*ds(choroid_plexus_tags)) # The area of the choroid plexus boundary
# v_in = 0.31e-6 # Volumetric flux through ventricles from Vinje et al. [m^3/s]
v_in = 5.833e-9 # Corresponds to 504 ml production per day [Czosnyka et al.] # Volumetric flux through ventricles [m^3/s] (1.8e-9 based on aqueduct net flow measurements)
chp_velocity = v_in/chp_area # The choroid plexus influx velocity [m/s]

class ChoroidPlexusFlux:
    def __init__(self, z_positive: bool): 
        self.velocity_value = chp_velocity #/np.sqrt(3) normalize so that velocity vector has unit length?
        self.sign = 1 if z_positive else -1

    def __call__(self, x):
        return self.velocity_value*np.stack((np.ones(x.shape[1]),
                                             np.ones(x.shape[1]),
                                   self.sign*np.ones(x.shape[1])))

n = ufl.FacetNormal(mesh)

vec_el  = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
scal_el = element("Lagrange", mesh.basix_cell(), 1)
bdm_el = element("BDM", mesh.basix_cell(), 1)
dg_el  = element("DG", mesh.basix_cell(), 0)
dg_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
V = dfx.fem.functionspace(mesh, bdm_el)
Q = dfx.fem.functionspace(mesh, dg_el)
DG_vec = dfx.fem.functionspace(mesh, dg_vec_el)
v_zero = dfx.fem.Function(V)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)


mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(7e-4)) #[kg/(m*s)] #*1e-2 # Dynamic viscosity [kg/(cm*s)]
penalty = dfx.fem.Constant(mesh, dfx.default_scalar_type(25.0))

# Tangential traction BC
tau_val = 7.89e-3*1e-2 # Tangential traction force density [Pa]
tau = dfx.fem.Function(V)
tau_input = dfx.fem.Function(DG_vec)
cilia_direction_filename = '../output/checkpoints/cilia-direction-vectors/mesh-coarse/'
a4d.read_function(filename=cilia_direction_filename, u=tau_input)
tau.interpolate(tau_input)

# ds_ = ds(noslip_bdry_tags)
# Stokes problem in reference domain accounting for the deformation
a00 = (2*mu*inner(eps(u), eps(v))*dx # Viscous dissipation
      + stabilization(u, v, mu, penalty) # BDM stabilization
      - mu*inner(dot(grad(u).T, n), v)*(ds(CANAL_OUT)+ds(LATERAL_APERTURES)) # Parallel flow at inlet/outlet
      )
a01 = inner(p, div(v))*dx
a10 = inner(q, div(u))*dx
a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*dx

L0 = inner(v_zero, v)*dx \
   + inner(tau_val*tangent(tau, n), tangent(v, n))*ds(cilia_tags)
L1 = inner(dfx.fem.Function(Q), q)*dx

a = dfx.fem.form([[a00, a01],
                  [a10, a11]])
L = dfx.fem.form([L0, L1])

# Set choroid plexus inflow velocity BC strongly
# Create expressions with positive and negative z-component of the velocity
v_chp_positive_z = dfx.fem.Function(V)
v_chp_expr_positive_z = ChoroidPlexusFlux(z_positive=True) 
v_chp_negative_z = dfx.fem.Function(V)
v_chp_expr_negative_z = ChoroidPlexusFlux(z_positive=False) 
facets_chp_laterals = ft.find(CHOROID_PLEXUS_LATERAL)
v_dofs_chp_laterals = dfx.fem.locate_dofs_topological(V, facet_dim, facets_chp_laterals)
facets_chp_third = ft.find(CHOROID_PLEXUS_THIRD)
v_dofs_chp_third = dfx.fem.locate_dofs_topological(V, facet_dim, facets_chp_third)
facets_chp_fourth = ft.find(CHOROID_PLEXUS_FOURTH)
v_dofs_chp_fourth = dfx.fem.locate_dofs_topological(V, facet_dim, facets_chp_fourth)

bcs_stokes = [dfx.fem.dirichletbc(v_chp_positive_z, v_dofs_chp_laterals),
              dfx.fem.dirichletbc(v_chp_negative_z, v_dofs_chp_third),
              dfx.fem.dirichletbc(v_chp_positive_z, v_dofs_chp_fourth)
]

# Impose deformation velocity on the rest of the boundary
v_defo = dfx.fem.Function(dfx.fem.functionspace(mesh, bdm_el)); # Deformation velocity
facets_defo = np.concatenate(([ft.find(tag) for tag in impermeability_tags]))
v_dofs_defo = dfx.fem.locate_dofs_topological(V, facet_dim, facets_defo)

bcs_stokes.append(dfx.fem.dirichletbc(v_defo, v_dofs_defo))

def assemble_nested_system(lhs_form: dfx.fem.form, rhs_form: dfx.fem.form, bcs: list[dfx.fem.dirichletbc]):
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
def create_solver(A: PETSc.Mat):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    return ksp
def solve_stokes(a: dfx.fem.form, L: dfx.fem.form, bcs: list[dfx.fem.DirichletBC]):

    A, b = assemble_nested_system(a, L, bcs)

    ksp = create_solver(A)

    u, p = dfx.fem.Function(V), dfx.fem.Function(Q)
    w = PETSc.Vec().createNest([u.x.petsc_vec, p.x.petsc_vec])
    ksp.solve(b, w)
    assert ksp.getConvergedReason() > 0

    # MPI communcation
    u.x.scatter_forward()
    p.x.scatter_forward()

    return u, p

print("Number of dofs Stokes eqs: ")
print(f"Total:\t\t{V.dofmap.index_map.size_global+Q.dofmap.index_map.size_global}")
print(f"Velocity:\t{V.dofmap.index_map.size_global}")
print(f"Pressure:\t{Q.dofmap.index_map.size_global}")

tic = time.perf_counter()

if __name__=='__main__':
    from sys import argv
    write_cpoint = True if int(argv[1])==1 else False

    # I/O functions
    uh = dfx.fem.Function(dfx.fem.functionspace(mesh, dg_vec_el)); uh.name = 'uh' # Stokes velocity
    ph = dfx.fem.Function(dfx.fem.functionspace(mesh, dg_el)); ph.name = 'ph' # Pressure

    velocity_output_filename = f"../output/deforming-mesh-{mesh_prefix}/BDM_chp+cilia+defo_velocity.pvd"
    velocity_output = dfx.io.VTKFile(comm, velocity_output_filename, "w")
    pressure_output_filename = f"../output/deforming-mesh-{mesh_prefix}/BDM_chp+cilia+defo_pressure.pvd"
    pressure_output = dfx.io.VTKFile(comm, pressure_output_filename, "w")
    vh_input_filename = f"../output/checkpoints/deforming-mesh-{mesh_prefix}/deformation_velocity/"

    if write_cpoint:
        cpoint_filename = f"../output/checkpoints/deforming-mesh-{mesh_prefix}/BDM_chp_velocity"
        a4d.write_mesh(cpoint_filename, mesh, store_partition_info=True)

    T = 1
    dt = 0.02
    N = int(T/dt)

    for t in np.linspace(0, T, N):

        # Update deformation velocity
        a4d.read_function(filename=vh_input_filename, u=v_defo, time=t)
        
        # Account for deformation in choroid plexus production
        v_chp_positive_z.interpolate(v_chp_expr_positive_z) 
        v_chp_negative_z.interpolate(v_chp_expr_negative_z)
        print(f'Max production velocity: {v_chp_positive_z.x.array.max():.2e}')
        print(f'Min production velocity: {v_chp_positive_z.x.array.min():.2e}')
        v_chp_positive_z.x.array[:] -= v_defo.x.array.copy()
        v_chp_negative_z.x.array[:] -= v_defo.x.array.copy()

        print(f'Max deformation velocity: {v_defo.x.array.max():.2e}')
        print(f'Min deformation velocity: {v_defo.x.array.min():.2e}')


        uh_, ph_ = solve_stokes(a, L, bcs_stokes) # Solve the Stokes equations

        # Interpolate output functions
        uh.interpolate(uh_)
        ph.interpolate(ph_)

        # Write output
        velocity_output.write_mesh(mesh, t)
        velocity_output.write_function(uh, t)
        pressure_output.write_mesh(mesh, t)
        pressure_output.write_function(ph, t)

        if write_cpoint: a4d.write_function(cpoint_filename, uh_, time=t)

        # Calculate mean pressure
        vol = assemble_scalar(1*ufl.dx(mesh))
        print("Mean pressure: ", 1/vol*assemble_scalar(ph*ufl.dx(mesh)))

    print(f"Solution loop time elapsed: {time.perf_counter()-tic:.4f} sec")

    # Close output files
    velocity_output.close()
    pressure_output.close()