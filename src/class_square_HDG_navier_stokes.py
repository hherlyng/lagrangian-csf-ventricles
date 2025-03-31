from mpi4py   import MPI
from petsc4py import PETSc

import ufl
import time
import numpy   as np
import dolfinx as dfx

from ufl    import div, dot, grad, inner, outer
from scifem import assemble_scalar
from utilities.fem     import tangent, create_normal_contribution_bc, compute_cell_boundary_integration_entities, calculate_mean, calculate_norm_L2
from utilities.mesh    import create_square_mesh_with_tags
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block

# Jump operator
jump = lambda phi, n: outer(phi('+'), n('+')) + outer(phi('-'), n('-'))

LEFT=1; RIGHT=2; BOT=3; TOP=4

class NavierStokesProblem:

    # Simulation parameters
    t = 0.0
    t_end = 5.0
    num_time_steps = 10
    times = np.linspace(0, t_end, num_time_steps+1)    
    delta_t = t_end/num_time_steps # Timestep size
    polynomial_degree = 1 # Polynomial degree
    penalty_value = 10 * polynomial_degree**2
    nu_value = 1.0 # Kinematic viscosity

    interior_tag = 0
    boundary_tags = [LEFT, RIGHT, BOT, TOP]
    neumann_tags = [RIGHT, LEFT]
    dirichlet_tags = []
    flux_tags = [BOT]
    # neumann_tags = [RIGHT]
    # dirichlet_tags = [LEFT]
    # flux_tags = [BOT, TOP]

    def __init__(self, mesh: dfx.mesh.Mesh, ft: dfx.mesh.MeshTags) -> None:
        """ Constructor.

        Parameters
        ----------
        mesh : dfx.mesh.Mesh
            The computational mesh.
        ft : dfx.mesh.MeshTags
            Facet tags for the mesh.
        """
        self.mesh = mesh 
        self.comm = mesh.comm
        self.ft   = ft 

    def solve(self):
        # Function spaces for velocity, pressure, and output velocity
        V, Q, W = self.V, self.Q, self.W
        
        # Solution functions
        u_h = dfx.fem.Function(V); u_h.name = 'velocity'
        p_h = dfx.fem.Function(Q); p_h.name = 'pressure'

        # I/O function: Stokes velocity in DG1
        u_h_out = dfx.fem.Function(W)
        u_h_out.name = 'velocity' 

        velocity_output = dfx.io.VTKFile(self.comm, velocity_output_filename, "w")
        pressure_output = dfx.io.VTKFile(self.comm, pressure_output_filename, "w")

        tic = time.perf_counter()

        for _ in self.times:

            self.t += self.delta_t

            # Solve the Navier-Stokes equations
            u_h, p_h = self.solve_navier_stokes(u_h, p_h) 

            # Interpolate velocity into output function
            u_h_out.interpolate(u_h)

            # Write output
            velocity_output.write_mesh(mesh, self.t)
            velocity_output.write_function(u_h_out, self.t)
            pressure_output.write_mesh(mesh, self.t)
            pressure_output.write_function(p_h, self.t)

        print(f"Solution loop time elapsed: {time.perf_counter()-tic:.4f} sec")

        # Close output files
        velocity_output.close()
        pressure_output.close()

        # Compute divergence L2 norm to check mass conservation
        e_div_u = calculate_norm_L2(self.comm, div(u_h), dX=self.dx)
        assert np.isclose(e_div_u, 0.0, atol=float(1.0e5 * np.finfo(dfx.default_real_type).eps))

        # Compute flux on boundaries with flux BC
        n = ufl.FacetNormal(self.mesh)
        flux = assemble_scalar(dot(u_h, n)*self.ds(tuple(self.flux_tags)))
        total_flux = assemble_scalar(dot(u_h, n)*self.ds)

        if comm.rank == 0:
            print(f"e_div_u = {e_div_u}")
            print(f"Flux = {flux}")
            print(f"Total flux = {total_flux}")


    def setup_weak_form(self):
        
        # Initial setup
        mesh = self.mesh
        h = ufl.CellDiameter(mesh) 
        n = ufl.FacetNormal(mesh)
        dt = dfx.fem.Constant(mesh, dfx.default_scalar_type(self.delta_t)) # Timestep size
        nu = dfx.fem.Constant(mesh, dfx.default_scalar_type(self.nu_value)) # Kinematic viscosity
        alpha = dfx.fem.Constant(mesh, dfx.default_scalar_type(self.penalty_value*mesh.geometry.dim)) # Interior penalty parameter
        ds_interior = self.ds(self.interior_tag) # Interior cell integrals (on submesh)
        ds_N = self.ds(tuple(self.neumann_tags)) # Neumann BC boundary integral
        ds_D = self.ds(tuple(self.dirichlet_tags)) # Dirichlet BC boundary integral
        ds_flux = self.ds(tuple(self.flux_tags)) # Flux BC boundary integral

        dx = self.dx # Cell integrals (on mesh)

        # Trial and test functions
        u, p, ubar, pbar = ufl.TrialFunctions(self.M)
        v, q, vbar, qbar = ufl.TestFunctions(self.M)

        # Functions at previous timestep
        self.u_ = dfx.fem.Function(self.V)
        self.ubar_ = dfx.fem.Function(self.V_bar)

        # Upwind velocity operator
        lmbda = ufl.conditional(ufl.gt(dot(self.u_, n), 0), 1, 0)

        # Tangential traction
        tangent_vector = ufl.as_vector((0.1, 0.0))

        # Bilinear form
        a = (
            # Time derivative 
            inner(u / dt, v) * dx 

            # Momentum terms, this is a(u_h, v_h) in Joe Dean's thesis
            + nu*inner(grad(u), grad(v)) * dx 
            - nu*inner(u - ubar, dot(grad(v), n)) * ds_interior
            - nu*inner(v - vbar, dot(grad(u), n)) * ds_interior
            + nu*alpha/h * inner(u - ubar, v - vbar) * ds_interior

            # b(v_h, p_h) terms
            - inner(p, div(v)) * dx
            + inner(dot(v, n), pbar) * ds_interior

            # c(u_h, u_h, v_h) terms
            - inner(outer(self.u_, u), grad(v)) * dx
            + inner(outer(u, self.u_) - outer(u - ubar, lmbda*self.u_), outer(v - vbar, n)) * ds_interior

            # Neumann BC terms
            + (1-lmbda) * dot(self.ubar_, n) * dot(ubar, vbar) * ds_N
            - inner(dot(ubar, n), qbar) * ds_N
            - inner(dot(vbar, n), pbar) * ds_N

            # b(u_h, q_h) terms
            - inner(q, div(u)) * dx
            + inner(dot(u, n), qbar) * ds_interior
        )

        # Linear form
        f = dfx.fem.Function(self.V)
        self.u_D = dfx.fem.Function(self.V_bar)
        self.u_flux = dfx.fem.Function(self.V_bar)

        L = inner(f, v) * dx + (
            # Time derivative
            inner(self.u_ / dt, v) * dx
            # Dirichlet BC terms
            + inner(dot(self.u_D, n), qbar) * ds_D
            + inner(tangent_vector, vbar)*self.ds(TOP)
            + inner(dot(self.u_flux, n), qbar) * ds_flux
            - inner(dfx.fem.Constant(mesh, 0.0)*n, vbar) * ds_N # Zero pressure
        )
        # Add zero block to pressure or else PETSc will complain
        L += inner(dfx.fem.Constant(mesh, dfx.default_real_type(0.0)), q) * dx

        # Get the block-structured forms and compile them
        self.a = dfx.fem.form(ufl.extract_blocks(a), entity_maps=self.subentities_map)
        self.L = dfx.fem.form(ufl.extract_blocks(L), entity_maps=self.subentities_map)

        # Boundary conditions
        # Prescribed profile
        if len(self.dirichlet_tags)>0:
            inflow_facets = self.mesh_to_submesh[np.concatenate(([self.ft.find(tag) for tag in self.dirichlet_tags]))]
            inflow_dofs = dfx.fem.locate_dofs_topological(self.V_bar, self.facet_dim, inflow_facets)
            u_parabolic = lambda x: np.vstack((.5*(x[1]-x[1]**2),
                                                np.zeros_like(x[1])))
            self.u_D.interpolate(u_parabolic)
            bc_u = dfx.fem.dirichletbc(self.u_D, inflow_dofs)
            self.bcs = [bc_u]
        else:
            self.bcs = []

        # Flux boundary condition: First
        flux_facets_mesh = np.concatenate(([self.ft.find(tag) for tag in self.flux_tags]))
        flux_facets_submesh = self.mesh_to_submesh[flux_facets_mesh]

        # Create the flux function in a (non-broken) BDM space on the mesh
        flux_expr_mesh = create_normal_contribution_bc(self.V_flux, -0.10*n, flux_facets_mesh)
        u_flux_mesh = dfx.fem.Function(self.V_flux)
        u_flux_mesh.interpolate(flux_expr_mesh)

        # Interpolate the function from the mesh onto the submesh
        interpolation_data = dfx.fem.create_interpolation_data(V_to=self.V_bar,
                                                            V_from=self.V_flux,
                                                            cells=flux_facets_submesh)
        self.u_flux.interpolate_nonmatching(u_flux_mesh,
                                    cells=flux_facets_submesh,
                                    interpolation_data=interpolation_data)

        # Set the BC                               
        flux_dofs = dfx.fem.locate_dofs_topological(self.V_bar, self.facet_dim, flux_facets_submesh)
        self.bcs.append(dfx.fem.dirichletbc(self.u_flux, flux_dofs))   

    def setup_integral_measures(self):
        mesh = self.mesh

        # Cell integral measure
        self.dx = ufl.Measure('dx', domain=mesh)

        # Generate integration entities for the interior facets on the submesh
        cell_facet_map = compute_cell_boundary_integration_entities(mesh)
        interior_cell_boundaries_tag = self.interior_tag
        facet_integral_entities = [(interior_cell_boundaries_tag, cell_facet_map)]

        # Add exterior facet integral entities
        for tag in self.boundary_tags:
            facet_integral_entities += [(int(tag), 
                dfx.fem.compute_integration_domains(
                    dfx.fem.IntegralType.exterior_facet,
                    mesh.topology,
                    self.ft.find(int(tag)),
                    self.facet_dim
                )
            )
        ]

        # Facet integral measure
        self.ds = ufl.Measure('ds', domain=mesh, subdomain_data=facet_integral_entities)

    def setup_function_spaces(self):
        """ Create finite element function spaces for the velocity and for the pressure,
            both on the mesh and the submesh. """
        mesh, submesh = self.mesh, self.submesh
        k = self.polynomial_degree

        # Function spaces for the weak form
        self.V = dfx.fem.functionspace(mesh, ('Discontinuous Brezzi-Douglas-Marini', k))
        self.Q = dfx.fem.functionspace(mesh, ('Discontinuous Lagrange', k-1))
        self.V_bar = dfx.fem.functionspace(submesh, ('Discontinuous Lagrange', k, (mesh.geometry.dim,)))
        self.Q_bar = dfx.fem.functionspace(submesh, ('Discontinuous Lagrange', k))
        self.M = ufl.MixedFunctionSpace(self.V, self.Q, self.V_bar, self.Q_bar)
    
        dofmap_size = (self.V.dofmap.index_map.size_global
            + self.Q.dofmap.index_map.size_global
            + self.V_bar.dofmap.index_map.size_global*self.V_bar.dofmap.index_map_bs # Vector space
            + self.Q_bar.dofmap.index_map.size_global)

        print(f'Setting up finite element spaces.')
        print(f'Size of global dofmap: {dofmap_size}')

        # Function space for visualising the velocity field
        self.W = dfx.fem.functionspace(mesh, ('Discontinuous Lagrange', k, (mesh.geometry.dim,)))

        # Function space for the normal flux boundary condition
        self.V_flux = dfx.fem.functionspace(mesh, ('Brezzi-Douglas-Marini', k))

    def calculate_offsets(self):
        """ Create offsets for accessing the blocked functions
            in the local degree of freedom vectors. """

        V, Q, V_bar = self.V, self.Q, self.V_bar
        offset_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs # Velocity dofs offset
        offset_p = offset_u + Q.dofmap.index_map.size_local*Q.dofmap.index_map_bs # Pressure dofs offset
        offset_ubar = offset_p \
                    + V_bar.dofmap.index_map.size_local*V_bar.dofmap.index_map_bs # Facet velocity dofs offset

        self.offsets = [offset_u, offset_p, offset_ubar] # Store the offsets

    def setup_direct_solver(self):
        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(self.A)
        solver.setType("preonly")
        pc = solver.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()
        opts = PETSc.Options()  # type: ignore
        opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
        opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
        opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
        opts["ksp_error_if_not_converged"] = 1 # Throw an error if KSP solver does not converge
        solver.setFromOptions()

        self.solver = solver

    def setup_navier_stokes_linear_system(self):
        """ Initialize the linear system of the Navier-Stokes problem,
            and create a linear solver for the problem. """

        self.A = assemble_matrix_block(self.a, bcs=self.bcs)
        self.A.assemble()
        self.b = assemble_vector_block(self.L, self.a, bcs=self.bcs) # Right-hand side vector
        self.x = self.A.createVecRight() # Solution vector

        self.calculate_offsets()

    def solve_navier_stokes(self, u_h: dfx.fem.Function, p_h: dfx.fem.Function):
        """ Assemble and solve the Navier-Stokes system of equations. """

        self.A.zeroEntries()
        assemble_matrix_block(self.A, self.a, bcs=self.bcs)
        self.A.assemble()

        with self.b.localForm() as b_loc: b_loc.set(0)
        assemble_vector_block(self.b, self.L, self.a, bcs=self.bcs)

        self.solver.solve(self.b, self.x)
        assert self.solver.getConvergedReason() > 0, print(self.solver.getConvergedReason())

        # Get degree of freedom offsets
        offset_u, offset_p, offset_ubar = self.offsets

        # Update solution functions
        u_h.x.array[:offset_u] = self.x.array_r[:offset_u]
        u_h.x.scatter_forward()
        p_h.x.array[:(offset_p-offset_u)] = self.x.array_r[offset_u:offset_p]
        p_h.x.scatter_forward()
        p_h.x.array[:] -= calculate_mean(self.mesh, p_h, dX=self.dx)

        # Update previous timesteps
        self.u_.x.array[:] = u_h.x.array
        self.ubar_.x.array[:(offset_ubar-offset_p)] = self.x.array_r[offset_p:offset_ubar]
        self.ubar_.x.scatter_forward()

        return u_h, p_h


    def create_submesh_and_subentities_map(self):
        """ Create submesh in the facet dimension of the mesh. """
        facet_dim = self.facet_dim = self.mesh.topology.dim-1
        num_facets = (self.mesh.topology.index_map(facet_dim).size_local  # Total number of facets
                    + self.mesh.topology.index_map(facet_dim).num_ghosts) # local to proc
        facets = np.arange(num_facets, dtype=np.int32)

        # Create the submesh and submesh-to-mesh mapping
        submesh, submesh_to_mesh = dfx.mesh.create_submesh(self.mesh, facet_dim, facets)[:2]
        submesh.topology.create_connectivity(submesh.topology.dim, submesh.topology.dim)

        # Since the bilinear and linear forms are formulated on the mesh,
        # we need a mapping between the facets in the mesh and the cells in the 
        # submesh (these are the same entities). This mapping is simply
        # the inverse mapping of submesh_to_mesh
        mesh_to_submesh = np.zeros(num_facets, dtype=np.int32)
        mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh), dtype=np.int32)
        self.subentities_map = {submesh : mesh_to_submesh}

        # Store submesh and mappings
        self.submesh = submesh
        self.submesh_to_mesh = submesh_to_mesh
        self.mesh_to_submesh = mesh_to_submesh

if __name__=='__main__':
    from sys import argv
    # Read mesh
    comm = MPI.COMM_WORLD
    N = int(argv[1])
    mesh, ft = create_square_mesh_with_tags(N=N, comm=comm)

    output_dir = f'../output/square-mesh/flow/navier-stokes/'
    velocity_output_filename = output_dir + 'velocity.pvd'
    pressure_output_filename = output_dir + 'pressure.pvd'

    problem = NavierStokesProblem(mesh=mesh, ft=ft)
    problem.create_submesh_and_subentities_map()
    problem.setup_function_spaces()
    problem.setup_integral_measures()
    problem.setup_weak_form()
    problem.setup_navier_stokes_linear_system()
    problem.setup_direct_solver()
    problem.solve()
        