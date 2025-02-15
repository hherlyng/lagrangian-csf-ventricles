import ufl
import time

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from ufl       import inner, dot, div, grad
from scifem    import assemble_scalar
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from utilities.fem import stabilization, tangent, eps, assemble_nested_system, create_normal_contribution_bc

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
deformation_tags = [tag for tag in cilia_tags if tag not in choroid_plexus_tags]

def calculate_fluxes(ds: ufl.Measure, tags: tuple[int],
                                  uh: dfx.fem.Function, uh_rel: dfx.fem.Function,
                                  n: ufl.FacetNormal):
    """ Calculate the total amount of CSF produced by the choroid plexi. """

    # Calculate production (minus signs because n=outward unit normal)
    prod_total = assemble_scalar(-dot(uh_rel, n)*ds(tags))
    prod_laterals = assemble_scalar(-dot(uh_rel, n)*ds(tags[0]))
    prod_third = assemble_scalar(-dot(uh_rel, n)*ds(tags[1]))
    prod_fourth = assemble_scalar(-dot(uh_rel, n)*ds(tags[2]))
    
    # Convert to ml/day
    conversion_factor = 86400*1e6 # Convert m^3/second -> ml/day
    values = conversion_factor*np.array([prod_total, prod_laterals, prod_third, prod_fourth])

    # Print the values
    print("Choroid plexus production in ml/day:\n")
    print(f"Total:\t\t{values[0]:.3g}")
    print(f"Laterals:\t{values[1]:.3g}")
    print(f"Third:\t\t{values[2]:.3g}")
    print(f"Fourth:\t\t{values[3]:.3g}")

    # Check that mass is conserved by calculating total boundary flux
    total_boundary_flux = assemble_scalar(-dot(uh, n)*ds)
    assert total_boundary_flux < 1e-10, print("Mass conservation violated.")

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
    v_zero = dfx.fem.Function(V) # Zero velocity for BCs
    v_defo = dfx.fem.Function(V) # Deformation velocity

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(7e-4)) #[kg/(m*s)] #*1e-2 # Dynamic viscosity [kg/(cm*s)]
    penalty = dfx.fem.Constant(mesh, dfx.default_scalar_type(100.0))

    # Tangential traction BC
    tau_val = 0#7.89e-3*1e-1 # Tangential traction force density [Pa]
    tau = dfx.fem.Function(V)
    tau_input = dfx.fem.Function(DG_vec)
    cilia_direction_filename = f'../output/{mesh_prefix}-mesh/flow/checkpoints/cilia-direction-vectors'
    a4d.read_function(filename=cilia_direction_filename, u=tau_input)
    tau.interpolate(tau_input)

    # Stokes problem in reference domain accounting for the deformation
    a00 = (2*mu*inner(eps(u), eps(v))*dx # Viscous dissipation
         + stabilization(u, v, mu, penalty) # BDM stabilization
         - mu*inner(dot(grad(u).T, n), v)*(ds(CANAL_OUT)+ds(LATERAL_APERTURES)) # Parallel flow at inlet/outlet
          )
    a01 = p*div(v)*dx

    a10 = q*div(u)*dx
    a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*dx

    L0 = inner(v_zero, v)*dx \
    #    + inner(tau_val*tangent(tau, n), tangent(v, n))*ds(cilia_tags) \
    L1 = inner(dfx.fem.Function(Q), q)*dx

    a = dfx.fem.form([[a00, a01],
                      [a10, a11]])
    L = dfx.fem.form([L0, L1])

    # Set choroid plexus inflow velocity BC strongly
    # Create expressions with positive and negative z-component of the velocity,
    # and interpolate the expressions into finite element functions.
    chp_prod = 5.833e-9 # Corresponds to 504 ml production per day [Czosnyka et al.]
    chp_area = assemble_scalar(1*ds(choroid_plexus_tags)) # The area of the choroid plexus boundary
    chp_velocity = chp_prod/chp_area
    facets_chp = np.concatenate(([ft.find(tag) for tag in choroid_plexus_tags]))
    v_chp_expr = create_normal_contribution_bc(V, -chp_velocity*n, facets_chp)
    v_chp = dfx.fem.Function(V)

    # Find the dofs of facets tagged with choroid plexus tags
    v_dofs_chp = dfx.fem.locate_dofs_topological(V, facet_dim, facets_chp)
    
    bcs = [dfx.fem.dirichletbc(v_chp, v_dofs_chp)]

    # Impose deformation velocity on the rest of the boundary
    facets_defo = np.concatenate(([ft.find(tag) for tag in deformation_tags]))
    v_dofs_defo = dfx.fem.locate_dofs_topological(V, facet_dim, facets_defo)

    bcs.append(dfx.fem.dirichletbc(v_defo, v_dofs_defo))

    return a, L, bcs, V, Q, ds, v_chp, v_chp_expr, v_defo

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
    comm = uh.function_space.mesh.comm
    
    A, b = assemble_nested_system(a, L, bcs)

    ksp = create_direct_solver(A, comm)

    w = PETSc.Vec().createNest([uh.x.petsc_vec, ph.x.petsc_vec])
    ksp.solve(b, w)
    assert ksp.getConvergedReason() > 0, print(ksp.getConvergedReason())

    # MPI communcation
    uh.x.scatter_forward()
    ph.x.scatter_forward()

    return uh, ph

if __name__=='__main__':
    from sys import argv
    write_cpoint = True if int(argv[1])==1 else False

    # Read mesh
    comm = MPI.COMM_WORLD
    mesh_prefix = 'medium'
    v_defo_input_filename = f"../output/{mesh_prefix}-mesh/deformation/checkpoints/time_dep_deformation_velocity/"
    mesh = a4d.read_mesh(v_defo_input_filename, comm, read_from_partition=False)
    ft   = a4d.read_meshtags(v_defo_input_filename, mesh, meshtag_name='ft')

    # Setup the Sokes problem
    a, L, bcs, V, Q, ds, \
    v_chp, v_chp_expr, v_defo = setup_stokes_problem(mesh, ft, mesh_prefix)
    
    # Solution functions
    uh = dfx.fem.Function(V); uh.name = 'velocity'
    ph = dfx.fem.Function(Q); ph.name = 'pressure'
    uh_rel = dfx.fem.Function(V)

    print("Number of dofs Stokes eqs: ")
    print(f"Total:\t\t{V.dofmap.index_map.size_global+Q.dofmap.index_map.size_global}")
    print(f"Velocity:\t{V.dofmap.index_map.size_global}")
    print(f"Pressure:\t{Q.dofmap.index_map.size_global}\n")

    # I/O function: Stokes velocity in DG1
    dg1_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    uh_out = dfx.fem.Function(dfx.fem.functionspace(mesh, dg1_vec_el))
    uh_out.name = 'velocity' 

    velocity_output_filename = f"../output/{mesh_prefix}-mesh/flow/velocity_chp+cilia+defo.pvd"
    velocity_output = dfx.io.VTKFile(comm, velocity_output_filename, "w")
    pressure_output_filename = f"../output/{mesh_prefix}-mesh/flow/pressure_chp+cilia+defo.pvd"
    pressure_output = dfx.io.VTKFile(comm, pressure_output_filename, "w")

    if write_cpoint:
        cpoint_filename = f"../output/{mesh_prefix}-mesh/flow/checkpoints/chp+cilia+defo"
        a4d.write_mesh(cpoint_filename, mesh, store_partition_info=True)
        a4d.write_meshtags(cpoint_filename, mesh, ft)

    T = 2
    dt = 0.02
    N = int(T / dt)
    times = np.linspace(0, T, N+1)

    tic = time.perf_counter()

    for t in times:

        # Update deformation velocity
        a4d.read_function(filename=v_defo_input_filename, u=v_defo, time=t)
        
        # Account for deformation in choroid plexus flux BC
        v_chp.interpolate(v_chp_expr)
        v_chp.x.array[:] += v_defo.x.array.copy()

        # Solve the Stokes equations
        uh, ph = solve_stokes(a, L, bcs, uh, ph) 

        # Interpolate velocity into DG1 output function
        uh_out.interpolate(uh)

        # Write output
        velocity_output.write_mesh(mesh, t)
        velocity_output.write_function(uh_out, t)
        pressure_output.write_mesh(mesh, t)
        pressure_output.write_function(ph, t)

        if write_cpoint:
            a4d.write_function(cpoint_filename, uh, time=t)
            a4d.write_function(cpoint_filename, ph, time=t)

        # Calculate mean pressure
        vol = assemble_scalar(1*ufl.dx(mesh))
        print("Mean pressure: ", 1/vol*assemble_scalar(ph*ufl.dx(mesh)))

        # Calculate choroid plexus CSF flux
        normal_vec = ufl.FacetNormal(mesh) # Facet normal vector of mesh
        uh_rel.x.array[:] = uh.x.array.copy() - v_defo.x.array.copy() # Relative velocity
        calculate_fluxes(ds=ds, tags=choroid_plexus_tags, uh=uh, uh_rel=uh_rel, n=normal_vec)

    print(f"Solution loop time elapsed: {time.perf_counter()-tic:.4f} sec")

    # Close output files
    velocity_output.close()
    pressure_output.close()