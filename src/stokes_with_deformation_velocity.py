import ufl
import time

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from ufl       import inner, dot, grad, div
from scifem    import assemble_scalar
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from utilities.fem import stabilization, tangent, eps, assemble_nested_system

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
impermeability_tags = [tag for tag in cilia_tags if tag not in choroid_plexus_tags]

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

def setup_stokes_problem(mesh: dfx.mesh.Mesh, ft: dfx.mesh.MeshTags, mesh_prefix: str):
    facet_dim = mesh.topology.dim-1
    mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity

    ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
    dx = ufl.Measure('dx', domain=mesh)

    n = ufl.FacetNormal(mesh)

    bdm1_el = element("BDM", mesh.basix_cell(), 1)
    dg0_el  = element("DG", mesh.basix_cell(), 0)
    dg1_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    V = dfx.fem.functionspace(mesh, bdm1_el)
    Q = dfx.fem.functionspace(mesh, dg0_el)
    DG_vec = dfx.fem.functionspace(mesh, dg1_vec_el)
    v_zero = dfx.fem.Function(V)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(7e-4)) #[kg/(m*s)] #*1e-2 # Dynamic viscosity [kg/(cm*s)]
    penalty = dfx.fem.Constant(mesh, dfx.default_scalar_type(25.0))

    # Tangential traction BC
    tau_val = 7.89e-3#*1e-2 # Tangential traction force density [Pa]
    tau = dfx.fem.Function(V)
    tau_input = dfx.fem.Function(DG_vec)
    cilia_direction_filename = f'../output/checkpoints/cilia-direction-vectors/mesh-{mesh_prefix}/'
    a4d.read_function(filename=cilia_direction_filename, u=tau_input)
    tau.interpolate(tau_input)

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
    v_chp_negative_z = dfx.fem.Function(V)
    facets_chp_laterals = ft.find(CHOROID_PLEXUS_LATERAL)
    v_dofs_chp_laterals = dfx.fem.locate_dofs_topological(V, facet_dim, facets_chp_laterals)
    facets_chp_third = ft.find(CHOROID_PLEXUS_THIRD)
    v_dofs_chp_third = dfx.fem.locate_dofs_topological(V, facet_dim, facets_chp_third)
    facets_chp_fourth = ft.find(CHOROID_PLEXUS_FOURTH)
    v_dofs_chp_fourth = dfx.fem.locate_dofs_topological(V, facet_dim, facets_chp_fourth)

    bcs = [dfx.fem.dirichletbc(v_chp_positive_z, v_dofs_chp_laterals),
           dfx.fem.dirichletbc(v_chp_negative_z, v_dofs_chp_third),
           dfx.fem.dirichletbc(v_chp_positive_z, v_dofs_chp_fourth)
    ]

    # Impose deformation velocity on the rest of the boundary
    v_defo = dfx.fem.Function(V) # Deformation velocity
    facets_defo = np.concatenate(([ft.find(tag) for tag in impermeability_tags]))
    v_dofs_defo = dfx.fem.locate_dofs_topological(V, facet_dim, facets_defo)

    bcs.append(dfx.fem.dirichletbc(v_defo, v_dofs_defo))

    return a, L, bcs, V, Q, ds, v_chp_positive_z, v_chp_negative_z, v_defo

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
    write_cpoint = True if int(argv[1])==1 else False

    # Read mesh
    comm = MPI.COMM_WORLD
    mesh_prefix = 'coarse'
    v_defo_input_filename = f"../output/checkpoints/deforming-mesh-{mesh_prefix}/deformation_velocity/"
    mesh = a4d.read_mesh(v_defo_input_filename, comm, read_from_partition=True)
    # mesh.topology.create_entities(mesh.topology.dim-1) # Create facets
    ft   = a4d.read_meshtags(v_defo_input_filename, mesh, meshtag_name='ft')

    # Setup the Sokes problem
    a, L, bcs, V, Q, ds, \
    v_chp_positive_z, v_chp_negative_z, v_defo = setup_stokes_problem(mesh, ft, mesh_prefix)
    
    # Solution functions
    uh = dfx.fem.Function(V)
    ph = dfx.fem.Function(Q)

    # Create BC expressions
    v_chp_expr_positive_z = ChoroidPlexusFlux(ds, z_positive=True) 
    v_chp_expr_negative_z = ChoroidPlexusFlux(ds, z_positive=False) 

    print("Number of dofs Stokes eqs: ")
    print(f"Total:\t\t{V.dofmap.index_map.size_global+Q.dofmap.index_map.size_global}")
    print(f"Velocity:\t{V.dofmap.index_map.size_global}")
    print(f"Pressure:\t{Q.dofmap.index_map.size_global}")

    # I/O function: Stokes velocity in DG1
    dg1_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    uh_out = dfx.fem.Function(dfx.fem.functionspace(mesh, dg1_vec_el))
    uh_out.name = 'uh' 

    cg1_vec_el  = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    uh_cg = dfx.fem.Function(dfx.fem.functionspace(mesh, cg1_vec_el))
    uh_cg.name = 'uh_cg'

    velocity_output_filename = f"../output/deforming-mesh-{mesh_prefix}/BDM_chp+cilia+defo_velocity.pvd"
    velocity_output = dfx.io.VTKFile(comm, velocity_output_filename, "w")
    pressure_output_filename = f"../output/deforming-mesh-{mesh_prefix}/BDM_chp+cilia+defo_pressure.pvd"
    pressure_output = dfx.io.VTKFile(comm, pressure_output_filename, "w")

    velocity_xdmf = dfx.io.XDMFFile(comm, velocity_output_filename.removesuffix('.pvd')+'.xdmf', 'w')
    velocity_xdmf.write_mesh(mesh)

    if write_cpoint:
        cpoint_filename = f"../output/checkpoints/deforming-mesh-{mesh_prefix}/BDM_chp_velocity"
        a4d.write_mesh(cpoint_filename, mesh, store_partition_info=True)

    T = 1
    N = 30

    tic = time.perf_counter()

    for t in np.linspace(0, T, N+1):

        # # Update deformation velocity
        a4d.read_function(filename=v_defo_input_filename, u=v_defo, time=t)
        
        # Account for deformation in choroid plexus production
        v_chp_positive_z.interpolate(v_chp_expr_positive_z) 
        v_chp_negative_z.interpolate(v_chp_expr_negative_z)
        print(f'Max production velocity: {comm.allreduce(v_chp_positive_z.x.array.max(), op=MPI.MAX):.2e}')
        print(f'Min production velocity: {comm.allreduce(v_chp_positive_z.x.array.min(), op=MPI.MIN):.2e}')  
        v_chp_positive_z.x.array[:] -= v_defo.x.array.copy()
        v_chp_negative_z.x.array[:] -= v_defo.x.array.copy()

        print(f'Max deformation velocity: {comm.allreduce(v_defo.x.array.max(), op=MPI.MAX):.2e}')
        print(f'Min deformation velocity: {comm.allreduce(v_defo.x.array.min(), op=MPI.MIN):.2e}')

        # Solve the Stokes equations
        uh_, ph_ = solve_stokes(a, L, bcs, uh, ph) 

        # Interpolate velocity into DG1 output function
        uh_out.interpolate(uh_)
        uh_cg.interpolate(uh_out)

        # Write output
        velocity_output.write_mesh(mesh, t)
        velocity_output.write_function(uh_out, t)
        pressure_output.write_mesh(mesh, t)
        pressure_output.write_function(ph_, t)
        velocity_xdmf.write_function(uh_cg, t=t)

        if write_cpoint: a4d.write_function(cpoint_filename, uh_, time=t)

        # Calculate mean pressure
        vol = assemble_scalar(1*ufl.dx(mesh))
        print("Mean pressure: ", 1/vol*assemble_scalar(ph_*ufl.dx(mesh)))

    print(f"Solution loop time elapsed: {time.perf_counter()-tic:.4f} sec")

    # Close output files
    velocity_output.close()
    pressure_output.close()