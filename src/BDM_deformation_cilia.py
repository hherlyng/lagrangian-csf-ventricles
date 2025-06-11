import ufl
import time
import argparse

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from ufl       import inner, dot, grad, det, inv, jump, avg, nabla_grad
from scifem    import assemble_scalar
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from utilities.fem import create_normal_contribution_bc
from utilities.parsers import CustomParser
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, create_matrix_block, create_vector_block

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
CORPUS_CALLOSUM = 110

# Cell tags
CANAL = 3
THIRD_VENTRICLE = 4
AQUEDUCT = 5
FORAMINA_34 = 6
LATERAL_VENTRICLES = 7
FOURTH_VENTRICLE = 8

# Operators for BDM interior facet stabilization terms
# NOTE: these are the jump operators from Krauss, Zikatonov paper.
# Jump is just a difference and it preserves the rank 
Jump = lambda arg: arg('+') - arg('-')

# Average uses dot with normal and AGAIN MINUS; it reduces the rank
Avg = lambda arg, n: .5*(dot(arg('+'), n('+')) - dot(arg('-'), n('-')))

# Action of (1 - n x n) on a vector yields the tangential component
Tangent = lambda v, n: v - n*dot(v, n)

zero_traction_tags = (CANAL_OUT, LATERAL_APERTURES)
cilia_tags = (AQUEDUCT_WALL, FORAMINA_34_WALL, LATERAL_VENTRICLES_WALL, FOURTH_VENTRICLE_WALL,
              CANAL_WALL, THIRD_VENTRICLE_WALL, CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD,
              CHOROID_PLEXUS_FOURTH, CORPUS_CALLOSUM)
wall_deformation_tags = cilia_tags


class FluidSolverALE:
    # Solve fluid equations of motion in a moving domain
    # by an ALE method. Wall motion is 
    # prescribed in time, given by solutions to the 
    # time-dependent linear elasticity equations.
    comm = MPI.COMM_WORLD # MPI communicator
    quadrature_degree = 8
    read_time  = 0
    write_time = 0
    period = 1
    output_interval = 5

    def __init__(self, T: float,
                       timestep: float,
                       mesh_prefix: str,
                       solver_type: str,
                       write_checkpoint: int,
                       write_output: int):
        """ Constructor. """

        self.T = T
        self.timestep = timestep
        self.N = int(T / timestep)
        self.times = np.linspace(0, T, self.N+1)
        self.final_period_start = 0.0#int(T - self.period)
        self.num_timesteps_per_period = int(self.period / timestep)
        self.mesh_prefix = mesh_prefix
        self.solver_type = solver_type
        self.defo_input_filename = f"../output/{mesh_prefix}-mesh/deformation/checkpoints/displacement_velocity_dt={timestep:.4g}_T={T:.4g}/"
        self.write_cpoint = write_checkpoint
        self.write_output = write_output

        self.mesh = a4d.read_mesh(self.defo_input_filename, self.comm)
        self.out_mesh = a4d.read_mesh(self.defo_input_filename, self.comm)
        self.ft   = a4d.read_meshtags(self.defo_input_filename, self.mesh, meshtag_name='ft')
        self.fdim = self.mesh.topology.dim-1

    def setup(self):
        mesh = self.mesh
        dx = ufl.Measure('dx', domain=mesh, metadata={'quadrature_degree' : self.quadrature_degree}) # Cell integral
        ds = ufl.Measure('ds', domain=mesh, subdomain_data=self.ft, metadata={'quadrature_degree' : self.quadrature_degree}) # Exterior facet integral
        dS = ufl.Measure('dS', domain=mesh, metadata={'quadrature_degree' : self.quadrature_degree}) # Interior facet integral

        # Create finite element function in P2 space for the mesh displacement
        vec_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
        W = dfx.fem.functionspace(mesh, vec_el)
        self.wh = dfx.fem.Function(W)

        # Now that we have that we can define the Stokes problem in the deformed coordinates
        r = ufl.SpatialCoordinate(mesh)
        chi = r + self.wh          
        F = grad(chi) # Deformation gradient
        J = det(F) # Jacobian 
        self.n_hat = n_hat = ufl.FacetNormal(mesh) # Facet normal on the reference mesh
        n = J*inv(F.T)*n_hat # Deformed domain facet normal and surface integral measure
        hA = ufl.avg(ufl.CellDiameter(mesh)) # Average cell diameter of reference mesh

        k = 1
        bdm_el = element("BDM", mesh.basix_cell(), k)
        dg_el  = element("DG", mesh.basix_cell(), k-1)
        dg_vec_el = element("DG", mesh.basix_cell(), k, shape=(mesh.geometry.dim,))
        self.V = V = dfx.fem.functionspace(mesh, bdm_el)
        Q = dfx.fem.functionspace(mesh, dg_el)
        self.u_ = dfx.fem.Function(V) # Velocity at previous timestep
        self.u_defo = dfx.fem.Function(V) # Deformation velocity
        c_vel = self.u_ - self.u_defo # Convection velocity

        # Trial and test functions
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
                        
        # Define differential operators that map from deformed to reference configuration
        Grad = lambda arg: dot(grad(arg), inv(F)) # Gradient
        Nabla_Grad = lambda arg: dot(nabla_grad(arg), inv(F)) # Gradient operator for the convective term
        Div = lambda arg: ufl.tr(dot(grad(arg), inv(F))) # Divergence
        Eps = lambda arg: ufl.sym(Grad(arg)) # Symmetric gradient

        mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(7e-4)) #[kg/(m*s)] #*1e-2 # Dynamic viscosity [kg/(cm*s)]
        rho = dfx.fem.Constant(mesh, dfx.default_scalar_type(1e3))
        gamma = dfx.fem.Constant(mesh, dfx.default_scalar_type(150.0))

        dt = dfx.fem.Constant(mesh, dfx.default_scalar_type(self.timestep))

        # Tangential traction BC
        tau_val = 7.89e-3 # Tangential traction force density [Pa]
        tau_mag = dfx.fem.Constant(mesh, dfx.default_scalar_type(tau_val))
        tau_vec = dfx.fem.Function(V)
        a4d.read_function(filename=f"../output/medium-mesh/flow/navier-stokes/checkpoints/cilia_direction_vectors/",
                          u=tau_vec,
                          name="cilia_direction"
                        )
        tau = tau_mag * tau_vec / ufl.sqrt(inner(tau_vec, tau_vec))

        # Navier-Stokes problem in reference domain accounting for the deformation
        a00  = rho/dt * inner(u, v)*J*dx # Time derivative
        a00 += (2*mu*inner(Eps(u), Eps(v))*J*dx # Viscous dissipation
                - mu*inner(dot(Grad(u).T, n), v)*ds(zero_traction_tags) # Parallel flow at inlet/outlet
                )
        a00 += (-inner(Avg(2*mu*Eps(u), n), Jump(v))*dS # Stabilization term to ensure
                -inner(Avg(2*mu*Eps(v), n), Jump(u))*dS # tangential continuity
                +2*mu*(gamma/hA)*inner(Jump(u), Jump(v))*dS
                )
        a01 = inner(p, Div(v))*J*dx
        a10 = inner(q, Div(u))*J*dx
        a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*J*dx

        self.a_stokes = dfx.fem.form([[a00, a01], [a10, a11]])

        L0  = rho/dt * inner(self.u_, v)*J*dx # Time derivative
        L0 += inner(tau, Tangent(v, n))*ds(cilia_tags) 

        L1 = inner(dfx.fem.Function(Q), q)*J*dx

        self.L = dfx.fem.form([L0, L1])

        if self.solver_type=="navier-stokes":

            # Navier-Stokes problem
            a00 += rho*inner(dot(c_vel, Nabla_Grad(u)), v)*J*dx # Convective term

            # Add convective term stabilization
            zeta = ufl.conditional(ufl.lt(dot(c_vel, n), 0), 1, 0) # Upwind velocity operator (equals 1 on inflow boundary, 0 on outflow boundary)
            a00 += (- rho*1/2*dot(jump(c_vel), n('+')) * avg(dot(u, v))  * dS 
                    - rho*dot(avg(c_vel), n('+')) * dot(jump(u), avg(v)) * dS 
                    - zeta*rho*1/2*dot(c_vel, n) * dot(u, v) * ds(zero_traction_tags)
            )

            self.a = dfx.fem.form([[a00, a01], [a10, a11]])
        else:

            self.a = self.a_stokes


        # Set boundary conditions on velocity
        facets_wall_defo = np.concatenate(([self.ft.find(tag) for tag in wall_deformation_tags]))
        u_dofs_defo = dfx.fem.locate_dofs_topological(V, self.fdim, facets_wall_defo)

        self.bcs = [dfx.fem.dirichletbc(self.u_defo, u_dofs_defo)]

        # Define deforming mesh and reference coordinates (coordinates of mesh at t=0)
        self.x_reference = self.out_mesh.geometry.x.copy()

        # Compute cells for point evaluation of the deformation function wh
        cells = []
        points_on_proc = []
        bb_tree = dfx.geometry.bb_tree(self.out_mesh, mesh.topology.dim)
        cell_candidates = dfx.geometry.compute_collisions_points(bb_tree, self.x_reference)
        colliding_cells = dfx.geometry.compute_colliding_cells(self.out_mesh, cell_candidates, self.x_reference)
        for i, point in enumerate(self.x_reference):
            if len(colliding_cells.links(i)>0):
                cc = colliding_cells.links(i)[0]
                cells.append(cc)
                points_on_proc.append(point)

        # Convert to numpy arrays
        self.cells = np.array(cells)
        self.points_on_proc = np.array(points_on_proc)

        vol = self.comm.allreduce(
                    dfx.fem.assemble_scalar(
                        dfx.fem.form(
                            dfx.fem.Constant(mesh, dfx.default_real_type(1.0)) * ufl.dx(self.out_mesh))
                        ),
                    op=MPI.SUM
                )
        
        # Calculate offsets
        self.offset_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        self.offset_p = self.offset_u + Q.dofmap.index_map.size_local*Q.dofmap.index_map_bs

        # Setup I/O functions and files
        self.uh = dfx.fem.Function(dfx.fem.functionspace(self.out_mesh, dg_vec_el)); self.uh.name = 'velocity'
        self.ph = dfx.fem.Function(dfx.fem.functionspace(self.out_mesh, dg_el)); self.ph.name = 'pressure'
        self.uh_rel = dfx.fem.Function(dfx.fem.functionspace(self.out_mesh, dg_vec_el)); self.uh_rel.name = 'relative_velocity'
        
        self.uh_ = dfx.fem.Function(V); self.uh_.name = "velocity"
        self.ph_ = dfx.fem.Function(Q); self.ph_.name = "pressure"
        self.uh_rel_ = dfx.fem.Function(V); self.uh_rel_.name = "relative_velocity"
        self.u_defo_read = dfx.fem.Function(dfx.fem.functionspace(mesh, element("BDM", mesh.basix_cell(), 1)))

        if self.write_output:
            velocity_output_filename = f"../output/{self.mesh_prefix}-mesh/flow/{self.solver_type}/BDM_deformation+cilia_velocity_T={self.T:.4g}_dt={self.timestep:.4g}.pvd"
            self.velocity_output = dfx.io.VTKFile(self.comm, velocity_output_filename, "w")
            self.uh_vtx = dfx.fem.Function(dfx.fem.functionspace(self.mesh, dg_vec_el)); self.uh_vtx.name = "relative_velocity"
            self.velocity_vtx = dfx.io.VTXWriter(self.comm, velocity_output_filename.removesuffix(".pvd") + ".bp", [self.uh_vtx], "BP4")
            pressure_output_filename = f"../output/{self.mesh_prefix}-mesh/flow/{self.solver_type}/BDM_deformation+cilia_pressure_T={self.T:.4g}_dt={self.timestep:.4g}.pvd"
            self.pressure_output = dfx.io.VTKFile(self.comm, pressure_output_filename, "w")

        if self.write_cpoint:
            self.cpoint_filename = f"../output/{self.mesh_prefix}-mesh/flow/{self.solver_type}/checkpoints/BDM_deformation+cilia_velocity_T={self.T}_dt={self.timestep}"
            a4d.write_mesh(self.cpoint_filename, mesh)
            a4d.write_meshtags(self.cpoint_filename, mesh, self.ft)
        
        # Compile form used to calculate the mean pressure
        self.mean_pressure_form  = dfx.fem.form((1/vol) * self.ph * ufl.dx(self.out_mesh))
        self.mean_pressure_form_ = dfx.fem.form((1/vol) * self.ph_* J*dx)

        # Set up Stokes problem linear system
        self.A = create_matrix_block(self.a)
        self.xh = self.A.createVecRight() # Solution vector
        self.b = create_vector_block(self.L)
        self.create_solver()
        
        print("Global number of dofs: ", V.dofmap.index_map.size_global+Q.dofmap.index_map.size_global)

    def create_solver(self):
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(self.A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        opts = PETSc.Options()  # type: ignore
        opts["mat_mumps_icntl_14"] = 80  # Increase MUMPS working memory
        opts["mat_mumps_icntl_24"] = 1  # Option to support solving a singular matrix (pressure nullspace)
        opts["mat_mumps_icntl_25"] = 0  # Option to support solving a singular matrix (pressure nullspace)
        opts["ksp_error_if_not_converged"] = 1 # Throw an error if KSP solver does not converge
        ksp.setFromOptions()

        # Store solver
        self.ksp = ksp

    def solve_blocked_system(self):
        
        self.A.zeroEntries()
        assemble_matrix_block(self.A, self.a, bcs=self.bcs)
        self.A.assemble()

        with self.b.localForm() as b_loc: b_loc.set(0)
        assemble_vector_block(self.b, self.L, self.a, bcs=self.bcs)
        
        self.ksp.solve(self.b, self.xh)
        assert self.ksp.getConvergedReason() > 0

        # Update and MPI communcation
        self.uh_.x.array[:self.offset_u] = self.xh.array[:self.offset_u]
        self.uh_.x.scatter_forward()
        self.ph_.x.array[:(self.offset_p-self.offset_u)] = self.xh.array[self.offset_u:self.offset_p]
        self.ph_.x.scatter_forward()

    def solve(self):

        if self.solver_type=="navier-stokes":
            # Solve the Stokes problem and use it
            # as initial condition for Navier-Stokes
            print("Solving Stokes eqs for initial condition ...")
            self.solve_initial_condition()
            print("Stokes eqs solved and initial conditon set.\nEntering solution time-loop ...")

        tic = time.perf_counter()

        for i, t in enumerate(self.times[:5]):

            print(f"Time = {t:.5g} sec")

            if (self.read_time-1)==self.num_timesteps_per_period:
                self.read_time = 0

            # Read deformation from file
            a4d.read_function(filename=self.defo_input_filename, u=self.wh, name="defo_displacement", time=self.read_time)
            a4d.read_function(filename=self.defo_input_filename, u=self.u_defo_read, name="defo_velocity", time=self.read_time)
            self.u_defo.interpolate(self.u_defo_read)

            u_chp_updated = create_normal_contribution_bc(
                                    self.V, 
                                    (-self.chp_velocity*self.n_hat
                                        + dot(self.u_defo, self.n_hat)*self.n_hat),
                                    self.facets_chp)
            self.u_chp.interpolate(u_chp_updated)
        
            self.solve_blocked_system() # Solve the fluid equations of motion

            # Update finite element functions
            self.uh.interpolate(self.uh_) # Output velocity (deforming domain)
            self.ph.interpolate(self.ph_) # Output pressure (deforming domain)
            self.u_.x.array[:] = self.uh_.x.array.copy() # Previous timestep velocity (reference configuration)
            self.uh_rel_.x.array[:] = self.uh_.x.array.copy() - self.u_defo.x.array.copy() # Relative velocity (reference configuration)
            self.uh_rel.interpolate(self.uh_rel_) # Relative velocity (deforming domain)

            if len(self.points_on_proc)>0:
                wh_x_reference = self.wh.eval(x=self.x_reference, cells=self.cells) # Evaluate the deformed coordinates at the reference coordinates

                # Update output mesh
                self.out_mesh.geometry.x[:, :self.out_mesh.geometry.dim] = self.x_reference[:, :self.out_mesh.geometry.dim] + wh_x_reference

            # Calculate mean pressures and subtract to make means = 0
            mean_pressure = dfx.fem.assemble_scalar(self.mean_pressure_form)
            self.ph.x.array[:] -= mean_pressure
            mean_pressure_ = dfx.fem.assemble_scalar(self.mean_pressure_form_)
            self.ph_.x.array[:] -= mean_pressure_
            
            if t >= self.final_period_start:
                if self.write_output and (i % self.output_interval == 0):
                    self.velocity_output.write_mesh(self.out_mesh, t)
                    self.velocity_output.write_function(self.uh, t)
                    self.velocity_output.write_function(self.uh_rel, t)
                    self.pressure_output.write_mesh(self.out_mesh, t)
                    self.pressure_output.write_function(self.ph, t)
                    self.uh_vtx.interpolate(self.uh_rel_)
                    self.velocity_vtx.write(t)

                if self.write_cpoint:
                    a4d.write_function(self.cpoint_filename, self.uh_, time=self.write_time)
                    a4d.write_function(self.cpoint_filename, self.uh_rel_, time=self.write_time)
                    a4d.write_function(self.cpoint_filename, self.ph_, time=self.write_time)
                    self.write_time += 1
            
            self.read_time += 1

        print(f"Solution loop time elapsed: {time.perf_counter()-tic:.4f} sec")

        if self.write_output:
            self.velocity_output.close()
            self.pressure_output.close()
            self.velocity_vtx.close()

    def solve_initial_condition(self):

        # Assemble the Stokes problem
        assemble_matrix_block(self.A, self.a_stokes, bcs=self.bcs)
        self.A.assemble()
        assemble_vector_block(self.b, self.L, self.a_stokes, bcs=self.bcs)

        # Read deformation velocity
        a4d.read_function(filename=self.defo_input_filename, u=self.u_defo, name="defo_velocity", time=self.times[0])

        # Solve and update solution functions
        self.ksp.solve(self.b, self.xh)
        self.u_.x.array[:self.offset_u] = self.xh.array[:self.offset_u]
        self.u_.x.scatter_forward()

def main(argv=None):

    parser = argparse.ArgumentParser(formatter_class=CustomParser)

    input_opts = parser.add_argument_group("Input options", "Options for reading input")
    input_opts.add_argument("-m", "--mesh_prefix", type=str, help="Mesh prefix")

    temporal_opts = parser.add_argument_group("Temporal options", "Options for the temporal discretization")
    temporal_opts.add_argument("-T", "--final_time", type=float, help="Final time of simulation")
    temporal_opts.add_argument("-dt", "--timestep", type=float, help="Timestep size")

    solver_opts = parser.add_argument_group("Solver options", "Options for the linear solver")
    solver_opts.add_argument("-d", "--direct", default=1, type=int, help="Use direct solver")
    solver_opts.add_argument("-g", "--governing_equations", type=str, help="Governing equations (Stokes or Navier-Stokes)")

    output_opts = parser.add_argument_group("Output options", "Options for writing output")
    output_opts.add_argument("-c", "--checkpoint", default=0, type=int, help="Write checkpoint")
    output_opts.add_argument("-o", "--output", default=0, type=int, help="Write output")

    args = parser.parse_args(argv)
    if args.mesh_prefix not in ["coarse", "medium", "fine"]:
        raise ValueError(f'Unknown mesh prefix, "coarse", "medium", or "fine".')
    if args.governing_equations not in ["stokes", "navier-stokes"]:
        raise ValueError(f'Unknown governing equations, choose "stokes" or "navier-stokes".')

    solver = FluidSolverALE(args.final_time,
                            args.timestep,
                            args.mesh_prefix,
                            args.governing_equations,
                            args.checkpoint,
                            args.output)
    solver.setup()
    solver.solve()

if __name__=='__main__':
    main()