import ufl
import time
import basix

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from ufl       import inner, dot, grad, div
from scifem    import assemble_scalar, create_real_functionspace
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from utilities.fem import stabilization, eps, assemble_nested_system, create_normal_contribution_bc
from utilities.normals_and_tangents import facet_normal_approximation

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

impermeability_tags = (AQUEDUCT_WALL, FORAMINA_34_WALL, LATERAL_VENTRICLES_WALL, FOURTH_VENTRICLE_WALL,
                       CANAL_WALL, THIRD_VENTRICLE_WALL)
choroid_plexus_tags = (CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD, CHOROID_PLEXUS_FOURTH)

class ChoroidPlexusFlux:
    """ Choroid plexus boundary condition expression.
    """
    def __init__(self, ds: ufl.Measure, z_positive: bool): 
        # Set the influx velocity
        chp_area = assemble_scalar(1*ds(choroid_plexus_tags)) # The area of the choroid plexus boundary
        # v_in = 0.31e-6 # Volumetric flux through ventricles from Vinje et al. [m^3/s]
        v_in = 5.833e-9 # Corresponds to 504 ml production per day [Czosnyka et al.] # Volumetric flux through ventricles [m^3/s] (1.8e-9 based on aqueduct net flow measurements)
        chp_velocity = v_in/chp_area # The choroid plexus influx velocity [m/s]
        self.velocity_value = chp_velocity #/np.sqrt(3) normalize so that velocity vector has unit length?
        
        # Set the feet-head direction sign
        self.sign = 1 if z_positive else -1

    def __call__(self, x):
        return self.velocity_value*np.stack((np.zeros(x.shape[1]),
                                             np.zeros(x.shape[1]),
                                   self.sign*np.ones (x.shape[1])))

def setup_stokes_problem(mesh: dfx.mesh.Mesh, ft: dfx.mesh.MeshTags):
    facet_dim = mesh.topology.dim-1
    mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity

    ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
    dx = ufl.Measure('dx', domain=mesh)

    n = ufl.FacetNormal(mesh)

    bdm1_el = element("BDM", mesh.basix_cell(), 1)
    dg0_el  = element("DG", mesh.basix_cell(), 0)
    V = dfx.fem.functionspace(mesh, bdm1_el)
    Q = dfx.fem.functionspace(mesh, dg0_el)
    v_zero = dfx.fem.Function(V)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(7e-4)) #[kg/(m*s)] #*1e-2 # Dynamic viscosity [kg/(cm*s)]
    penalty = dfx.fem.Constant(mesh, dfx.default_scalar_type(25.0))

    # Stokes problem in reference domain accounting for the deformation
    a00 = (2*mu*inner(eps(u), eps(v))*dx # Viscous dissipation
        + stabilization(u, v, mu, penalty) # BDM stabilization
        - mu*inner(dot(grad(u).T, n), v)*(ds(CANAL_OUT)+ds(LATERAL_APERTURES)) # Parallel flow at inlet/outlet
        )
    a01 = inner(p, div(v))*dx
    a10 = inner(q, div(u))*dx
    a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*dx

    L0 = inner(v_zero, v)*dx 
    L1 = inner(dfx.fem.Function(Q), q)*dx

    a = dfx.fem.form([[a00, a01],
                      [a10, a11]])
    L = dfx.fem.form([L0, L1])

    # Set choroid plexus inflow velocity BC strongly
    # Create expressions with positive and negative z-component of the velocity,
    # and interpolate the expressions into finite element functions.
    chp_flux = 5.833e-9 # Corresponds to 504 ml production per day [Czosnyka et al.]
    chp_area = assemble_scalar(1*ds(choroid_plexus_tags)) # The area of the choroid plexus boundary
    chp_velocity = chp_flux/chp_area
    facets_chp = np.concatenate(([ft.find(tag) for tag in choroid_plexus_tags]))
    v_chp_expr = create_normal_contribution_bc(V, -chp_velocity*n, facets_chp)

    v_chp = dfx.fem.Function(V)
    v_chp.interpolate(v_chp_expr)

    # Find the dofs of facets tagged with choroid plexus tags
    v_dofs_chp = dfx.fem.locate_dofs_topological(V, mesh.topology.dim-1, facets_chp)
    
    bcs = [dfx.fem.dirichletbc(v_chp, v_dofs_chp)]

    # Impose impermeability condition on the rest of the boundary
    v_zero = dfx.fem.Function(V) # Zero velocity
    facets_imperm = np.concatenate(([ft.find(tag) for tag in impermeability_tags])) # Facets where u.n=0
    v_dofs_zero = dfx.fem.locate_dofs_topological(V, facet_dim, facets_imperm) # Dofs where u.n=0

    bcs.append(dfx.fem.dirichletbc(v_zero, v_dofs_zero)) # Add BC object to list

    return a, L, bcs, V, Q, ds

def create_direct_solver(A: PETSc.Mat, comm: MPI.Comm):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    return ksp

def solve_stokes(a: dfx.fem.form, L: dfx.fem.form, bcs: list[dfx.fem.DirichletBC],
                 uh: dfx.fem.Function, ph: dfx.fem.Function):

    A, b = assemble_nested_system(a, L, bcs)

    ksp = create_direct_solver(A, comm)

    w = PETSc.Vec().createNest([uh.x.petsc_vec, ph.x.petsc_vec])
    ksp.solve(b, w)
    assert ksp.getConvergedReason() > 0

    # MPI communcation
    uh.x.scatter_forward()
    ph.x.scatter_forward()

    return uh, ph

if __name__=='__main__':
    
    from sys import argv

    # Check if velocity checkpoints should be written
    write_cpoint = True if int(argv[1])==1 else False

    comm = MPI.COMM_WORLD # MPI communicator
    mesh_prefix = 'medium' # Mesh version
    
    # Read mesh and facet tags from file
    with dfx.io.XDMFFile(comm,
         f"../geometries/{mesh_prefix}_ventricles_mesh_tagged.xdmf", "r") as xdmf:
         mesh = xdmf.read_mesh() # Read mesh
         mesh.topology.create_entities(mesh.topology.dim-1) # Create facets
         ft = xdmf.read_meshtags(mesh, name="ft") # Read facet tags

    # Setup the Sokes problem
    a, L, bcs, V, Q, ds = setup_stokes_problem(mesh, ft)
    
    # Solution functions
    uh = dfx.fem.Function(V)
    ph = dfx.fem.Function(Q)

    print("Number of dofs Stokes eqs: ")
    print(f"Total:\t\t{V.dofmap.index_map.size_global+Q.dofmap.index_map.size_global}")
    print(f"Velocity:\t{V.dofmap.index_map.size_global}")
    print(f"Pressure:\t{Q.dofmap.index_map.size_global}")

    # I/O function: Stokes velocity in DG1
    dg1_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    uh_out = dfx.fem.Function(dfx.fem.functionspace(mesh, dg1_vec_el))
    uh_out.name = 'uh' 

    velocity_output_filename = f"../output/{mesh_prefix}-mesh/flow/velocity_chp.pvd"
    velocity_output = dfx.io.VTKFile(comm, velocity_output_filename, "w")
    pressure_output_filename = f"../output/{mesh_prefix}-mesh/flow/pressure_chp.pvd"
    pressure_output = dfx.io.VTKFile(comm, pressure_output_filename, "w")

    if write_cpoint:
        cpoint_filename = f"../output/{mesh_prefix}-mesh/flow/checkpoints/velocity_chp/"
        a4d.write_mesh(cpoint_filename, mesh, store_partition_info=True)

    # Timestamp
    tic = time.perf_counter()

    # Solve the steady state Stokes equations
    uh_, ph_ = solve_stokes(a, L, bcs, uh, ph) 
    
    print(f"Solution time elapsed: {time.perf_counter()-tic:.4f} sec")
    
    # Interpolate velocity into DG1 output function
    uh_out.interpolate(uh_)

    # Write output
    velocity_output.write_mesh(mesh)
    velocity_output.write_function(uh_out)
    pressure_output.write_mesh(mesh)
    pressure_output.write_function(ph_)

    # Write checkpoint
    if write_cpoint: a4d.write_function(cpoint_filename, uh_)

    # Calculate mean pressure
    vol = assemble_scalar(1*ufl.dx(mesh))
    print(f"Mean pressure: {1/vol*assemble_scalar(ph_*ufl.dx(mesh)):.2e}")

    # Calculate choroid plexus CSF flux
    ds_chp = ds(choroid_plexus_tags)
    n = ufl.FacetNormal(mesh)
    prod_total = assemble_scalar(dot(uh_, n)*ds_chp)
    prod_laterals = assemble_scalar(dot(uh_, n)*ds(CHOROID_PLEXUS_LATERAL))
    prod_third = assemble_scalar(dot(uh_, n)*ds(CHOROID_PLEXUS_THIRD))
    prod_fourth = assemble_scalar(dot(uh_, n)*ds(CHOROID_PLEXUS_FOURTH))
    print("Choroid plexus production:\n")
    print(f"Total:\t\t{prod_total:.2e}")
    print(f"Laterals:\t{prod_laterals:.2e}")
    print(f"Third:\t\t{prod_third:.2e}")
    print(f"Fourth:\t\t{prod_fourth:.2e}")

    # Close output files
    velocity_output.close()
    pressure_output.close()