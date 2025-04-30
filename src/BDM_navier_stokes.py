import ufl
import time

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from ufl       import inner, dot, div, grad, nabla_grad, avg, jump
from scifem    import assemble_scalar
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from utilities.fem import stabilization, tangent, eps, create_normal_contribution_bc, calculate_mean, calculate_norm_L2
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, create_matrix_block

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
    # total_boundary_flux = assemble_scalar(-dot(uh, n)*ds)
    # assert total_boundary_flux < 1e-10, print("Mass conservation violated.")

def setup_variational_problem(mesh: dfx.mesh.Mesh,
                              ft: dfx.mesh.MeshTags,
                              k: int):
    facet_dim = mesh.topology.dim-1
    mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity

    n  = ufl.FacetNormal(mesh)
    dx = ufl.Measure('dx', domain=mesh) # Cell integral
    ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft) # Exterior facet integral
    dS = ufl.Measure('dS', mesh) # Interior facet integral

    bdm_el = element("BDM", mesh.basix_cell(), k)
    dg_el  = element("DG", mesh.basix_cell(), k-1)
    V = dfx.fem.functionspace(mesh, bdm_el)
    Q = dfx.fem.functionspace(mesh, dg_el)
    v_defo = dfx.fem.Function(V) # Deformation velocity
    u_ = dfx.fem.Function(V) # Previous timestep velocity

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    rho = dfx.fem.Constant(mesh, dfx.default_scalar_type(1e3)) # Fluid density [kg/m^3]
    mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(7e-4)) # Dynamic viscosity [kg/(m*s)]
    gamma = dfx.fem.Constant(mesh, dfx.default_scalar_type(100.0)) # BDM stabilization penalty parameter
    dt = dfx.fem.Constant(mesh, dfx.default_scalar_type(timestep)) # Timestep

    # Tangential traction BC
    tau_val = 0#7.89e-3*1e-1 # Tangential traction force density [Pa]
    tau = dfx.fem.Function(V)
    # tau_input = dfx.fem.Function(DG_vec)
    # cilia_direction_filename = f'../output/{mesh_prefix}-mesh/flow/checkpoints/cilia-direction-vectors'
    # a4d.read_function(filename=cilia_direction_filename, u=tau_input)
    # tau.interpolate(tau_input)

    # Stokes problem
    a00 = (rho/dt * inner(u, v) * dx # Time derivative
         + 2*mu*inner(eps(u), eps(v))*dx # Viscous dissipation
         + stabilization(u, v, mu, gamma) # BDM stabilization
         - mu*inner(dot(grad(u).T, n), v)*(ds(CANAL_OUT)+ds(LATERAL_APERTURES)) # Parallel flow at inlet/outlet
          )
    a01 = -p*div(v)*dx
    a10 = -q*div(u)*dx
    a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*dx

    L0 = rho/dt * inner(u_, v) * dx # Time derivative
    #L0 += inner(tau_val*tangent(tau, n), tangent(v, n))*ds(cilia_tags) \
    L1 = inner(dfx.fem.Function(Q), q)*dx

    a_stokes = dfx.fem.form([[a00, a01],
                      [a10, a11]])
    L = dfx.fem.form([L0, L1])

    # Navier-Stokes problem
    a00 += rho*inner(dot(u_, nabla_grad(u)), v) * dx # Convective term

    # Add convective term stabilization
    zeta = ufl.conditional(ufl.lt(dot(u_, n), 0), 1, 0) # Upwind velocity operator (equals 1 on inflow boundary, 0 on outflow boundary)
    a00 += (- rho*1/2*dot(jump(u_), n('+')) * avg(dot(u, v))  * dS 
            - rho*dot(avg(u_), n('+')) * dot(jump(u), avg(v)) * dS 
            - zeta*rho*1/2*dot(u_, n) * dot(u, v) * (ds(CANAL_OUT)+ds(LATERAL_APERTURES))
    )

    a = dfx.fem.form([[a00, a01],
                      [a10, a11]])

    # Set choroid plexus inflow velocity BC strongly
    # Create expressions with positive and negative z-component of the velocity,
    # and interpolate the expressions into finite element functions.
    chp_prod = 5.833e-9 # Corresponds to 504 ml production per day [Czosnyka et al.]
    chp_area = assemble_scalar(1*ds(choroid_plexus_tags)) # The area of the choroid plexus boundary
    chp_velocity = chp_prod/chp_area
    v_chp_expr = create_normal_contribution_bc(V, -chp_velocity*n + dot(v_defo, n)*n, facets_chp) # Account for deformation in choroid plexus flux BC
    v_chp = dfx.fem.Function(V)
    v_chp.interpolate(v_chp_expr)
    facets_chp = np.concatenate(([ft.find(tag) for tag in choroid_plexus_tags]))
    v_dofs_chp = dfx.fem.locate_dofs_topological(V, facet_dim, facets_chp) # Find the dofs of facets tagged with choroid plexus tags
    
    bcs = [dfx.fem.dirichletbc(v_chp, v_dofs_chp)]

    # Impose deformation velocity on the rest of the boundary
    facets_defo = np.concatenate(([ft.find(tag) for tag in deformation_tags]))
    v_dofs_defo = dfx.fem.locate_dofs_topological(V, facet_dim, facets_defo)

    bcs.append(dfx.fem.dirichletbc(v_defo, v_dofs_defo))

    return a_stokes, a, L, bcs, V, Q, ds, u_, v_chp, v_defo, n, facets_chp

if __name__=='__main__':
    from sys import argv
    write_cpoint = True if int(argv[1])==1 else False

    # Initialize
    comm = MPI.COMM_WORLD # MPI communicator
    k = 1 # Element degree

    # Temporal parameters
    T = 1
    timestep = 0.02
    N = int(T / timestep)
    times = np.linspace(0, T, N+1)

    # Read mesh and meshtags
    mesh_prefix = 'coarse'
    v_defo_input_filename = f"../output/{mesh_prefix}-mesh/deformation/checkpoints/time_dep_deformation_velocity_dt={timestep:.4g}_T={T:.4g}/"
    mesh = a4d.read_mesh(v_defo_input_filename, comm, read_from_partition=False)
    ft   = a4d.read_meshtags(v_defo_input_filename, mesh, meshtag_name='ft')


    # Setup the Stokes problem
    a_stokes, a, L, bcs, V, Q, ds, \
    u_, v_chp, v_defo, n, facets_chp = setup_variational_problem(mesh, ft, k)
    
    # Solution functions
    uh = dfx.fem.Function(V); uh.name = 'velocity'
    ph = dfx.fem.Function(Q); ph.name = 'pressure'
    uh_rel_ = dfx.fem.Function(V)

    print("Number of dofs Navier-Stokes eqs: ")
    print(f"Total:\t\t{V.dofmap.index_map.size_global+Q.dofmap.index_map.size_global}")
    print(f"Velocity:\t{V.dofmap.index_map.size_global}")
    print(f"Pressure:\t{Q.dofmap.index_map.size_global}\n")

    # I/O function: Stokes velocity in DG1
    dg_vec_el = element("DG", mesh.basix_cell(), k, shape=(mesh.geometry.dim,))
    DG_vec = dfx.fem.functionspace(mesh, dg_vec_el)
    uh_out = dfx.fem.Function(DG_vec); uh_out.name = "velocity"
    uh_rel = dfx.fem.Function(DG_vec); uh_rel.name = "relative_velocity"

    output_dir = f'../output/{mesh_prefix}-mesh/flow/navier-stokes/'
    velocity_output_filename = output_dir + 'velocity_chp+cilia+defo.bp'
    velocity_output = dfx.io.VTXWriter(comm, velocity_output_filename, [uh_out, uh_rel], "BP4")
    pressure_output_filename = output_dir + 'pressure_chp+cilia+defo.bp'
    pressure_output = dfx.io.VTXWriter(comm, pressure_output_filename, [ph], "BP4")

    if write_cpoint:
        cpoint_filename = output_dir + 'checkpoints/chp+cilia+defo'
        a4d.write_mesh(cpoint_filename, mesh, store_partition_info=True)
        a4d.write_meshtags(cpoint_filename, mesh, ft)

    tic = time.perf_counter()

    # Solve the Stokes problem and use it
    # as initial condition for Navier-Stokes
    # Assemble the Stokes problem
    A_stokes = assemble_matrix_block(a_stokes,bcs=bcs)
    A_stokes.assemble()
    b = assemble_vector_block(L, a_stokes, bcs=bcs)

    # Create the Navier-Stokes problem
    A = create_matrix_block(a)
    x = A.createVecRight() # Solution vector

    # Create and configure solver
    ksp = PETSc.KSP().create(comm)  # type: ignore
    ksp.setOperators(A_stokes)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    opts = PETSc.Options()  # type: ignore
    opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
    opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
    opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
    opts["ksp_error_if_not_converged"] = 1 # Throw an error if KSP solver does not converge
    ksp.setFromOptions()

    # Create offsets for the blocked functions
    offset_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    offset_p = offset_u + Q.dofmap.index_map.size_local*Q.dofmap.index_map_bs

    # Update deformation velocity
    a4d.read_function(filename=v_defo_input_filename, u=v_defo, name="defo_velocity", time=times[0])

    # Set choroid plexus inflow velocity BC strongly
    # Create expressions with positive and negative z-component of the velocity,
    # and interpolate the expressions into finite element functions.
    chp_prod = 5.833e-9 # Corresponds to 504 ml production per day [Czosnyka et al.]
    chp_area = assemble_scalar(1*ds(choroid_plexus_tags)) # The area of the choroid plexus boundary
    chp_velocity = chp_prod/chp_area
    v_chp_expr = create_normal_contribution_bc(V, -chp_velocity*n + dot(v_defo, n)*n, facets_chp) # Account for deformation in choroid plexus flux BC
    v_chp.interpolate(v_chp_expr)

    # Solve the Stokes problem 
    ksp.solve(b, x)
    u_.x.array[:offset_u] = x.array_r[:offset_u]
    u_.x.scatter_forward()
    uh_out.interpolate(u_) # Interpolate velocity into visualization function
    ph.x.array[:(offset_p-offset_u)] = x.array_r[offset_u:offset_p]
    ph.x.scatter_forward()

    # Set Navier-Stokes system matrix as KSP operator
    ksp.setOperators(A)

    for t in times:
        print(f"Time = {t:.4g}")

        # Update deformation velocity
        a4d.read_function(filename=v_defo_input_filename, u=v_defo, name="defo_velocity", time=float(f"{t:.4g}"))
        
        v_chp_expr = create_normal_contribution_bc(V, -chp_velocity*n + dot(v_defo, n)*n, facets_chp) # Account for deformation in choroid plexus flux BC
        v_chp.interpolate(v_chp_expr)

        tic_solve = time.perf_counter()
        # Solve the Navier-Stokes equations
        A.zeroEntries()
        assemble_matrix_block(A, a, bcs=bcs)
        A.assemble()

        with b.localForm() as b_loc: b_loc.set(0)
        assemble_vector_block(b, L, a, bcs=bcs)

        # Compute solution
        ksp.solve(b, x)

        print(f"Solve time: {time.perf_counter()-tic_solve:.4f} sec")

        uh.x.array[:offset_u] = x.array[:offset_u]
        uh.x.scatter_forward()
        ph.x.array[:(offset_p-offset_u)] = x.array[offset_u:offset_p]
        ph.x.scatter_forward()
        ph.x.array[:] -= calculate_mean(mesh, ph, dX=ufl.dx) # Subtract mean pressure so that mean=0
        
        # Update u_n
        u_.x.array[:] = uh.x.array

        # Interpolate velocity into DG1 output function
        uh_out.interpolate(uh)
        uh_rel_.x.array[:] = uh.x.array.copy() - v_defo.x.array.copy() # Relative velocity
        uh_rel.interpolate(uh_rel_)
        # Write output
        velocity_output.write(t)
        pressure_output.write(t)

        if write_cpoint:
            a4d.write_function(cpoint_filename, uh_out, time=t)
            a4d.write_function(cpoint_filename, uh_rel, time=t)
            a4d.write_function(cpoint_filename, ph, time=t)

        # Compute divergence L2 norm to check mass conservation
        e_div_u = calculate_norm_L2(comm, div(uh), dX=ufl.dx)
        print(f"e_div_u = {e_div_u}")
        # assert np.isclose(e_div_u, 0.0, atol=float(1.0e5 * np.finfo(dfx.default_real_type).eps))

        # Calculate choroid plexus CSF flux
        calculate_fluxes(ds=ds, tags=choroid_plexus_tags, uh=uh, uh_rel=uh_rel, n=n)

    print(f"Solution loop time elapsed: {time.perf_counter()-tic:.4f} sec")

    # Close output files
    velocity_output.close()
    pressure_output.close()