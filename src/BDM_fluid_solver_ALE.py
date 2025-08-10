import ufl
import time
import argparse

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from ufl       import inner, dot, jump, avg, nabla_grad, grad, det, inv
from scifem    import assemble_scalar
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from utilities.fem import create_normal_contribution_bc
from utilities.mesh import create_cilia_meshtags
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
THIRD_RIGHT = 111
THIRD_LEFT = 112
LATERAL_RIGHT = 113
LATERAL_LEFT = 114
THIRD_ANTERIOR = 115
THIRD_POSTERIOR = 116
THIRD_FLOOR = 117
LATERAL_FLOOR = 118
THIRD_NO_CILIA = 119
ZERO_SOLID_TRACTION = 1000
CHOROID_PLEXUS_LATERAL_ZERO_SOLID_TRACTION = 1001

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

# Mesh marker tags
zero_traction_tags = (CANAL_OUT, LATERAL_APERTURES)
displacement_tags = (AQUEDUCT_WALL, FORAMINA_34_WALL, LATERAL_VENTRICLES_WALL, FOURTH_VENTRICLE_WALL,
                     CANAL_WALL, THIRD_VENTRICLE_WALL, CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD,
                     CHOROID_PLEXUS_FOURTH, CORPUS_CALLOSUM, THIRD_LEFT, THIRD_RIGHT, LATERAL_LEFT,
                     LATERAL_RIGHT, THIRD_ANTERIOR, THIRD_POSTERIOR, THIRD_FLOOR, LATERAL_FLOOR,
                     ZERO_SOLID_TRACTION, CHOROID_PLEXUS_LATERAL_ZERO_SOLID_TRACTION)
cilia_tags = (AQUEDUCT_WALL, FORAMINA_34_WALL, LATERAL_VENTRICLES_WALL, FOURTH_VENTRICLE_WALL,
              CANAL_WALL, THIRD_VENTRICLE_WALL, CORPUS_CALLOSUM, THIRD_LEFT, THIRD_RIGHT,
              LATERAL_LEFT, LATERAL_RIGHT, THIRD_ANTERIOR, THIRD_POSTERIOR, THIRD_FLOOR,
              LATERAL_FLOOR, ZERO_SOLID_TRACTION)
choroid_plexus_tags = (CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD, CHOROID_PLEXUS_FOURTH,
                       CHOROID_PLEXUS_LATERAL_ZERO_SOLID_TRACTION) 
navier_slip_tags = (AQUEDUCT_WALL, FORAMINA_34_WALL, LATERAL_VENTRICLES_WALL, FOURTH_VENTRICLE_WALL,
                     CANAL_WALL, THIRD_VENTRICLE_WALL, CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD,
                     CHOROID_PLEXUS_FOURTH, CORPUS_CALLOSUM, THIRD_LEFT, THIRD_RIGHT, LATERAL_LEFT,
                     LATERAL_RIGHT, THIRD_ANTERIOR, THIRD_POSTERIOR, THIRD_FLOOR, LATERAL_FLOOR,
                     ZERO_SOLID_TRACTION, CHOROID_PLEXUS_LATERAL_ZERO_SOLID_TRACTION)

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
    mean_ICP = 10*133.3 # Mean intracranial pressure of 10 mmHg
    
    wall_deformation_tags = displacement_tags

    models = {1 : "deformation",
              2 : "deformation+cilia",
              3 : "deformation+production",
              4 : "deformation+cilia+production"
    }

    bc_types_dict = {1 : ["deformation"],
                     2 : ["deformation", "cilia"],
                     3 : ["deformation", "production"],
                     4 : ["deformation", "cilia", "production"],
    }

    # Form compilation optimization options
    jit_options = {"cffi_extra_compile_args": ["-O3", "-march=native"]} 

    def __init__(self, 
                    T: float,
                    timestep: float,
                    num_periods: float,
                    mesh_suffix: str,
                    stiffness: int,
                    polynomial_degree: int,
                    element_degree: int,
                    model_version: int,
                    solver_type: str,
                    write_checkpoint: int,
                    write_output: int,
                    calc_cilia_direction_vectors: int
                ):
        """ Constructor. """

        self.timestep = timestep
        self.N = int(num_periods / timestep)
        self.times = np.linspace(0, num_periods, self.N+1)
        self.final_period_start = int(num_periods - self.period)
        self.num_timesteps_per_period = int(self.period / timestep)+1
        self.polynomial_degree = polynomial_degree
        self.element_degree = element_degree
        self.mesh_suffix = mesh_suffix
        self.solver_type = solver_type
        self.model_version = model_version
        self.bc_types = self.bc_types_dict[model_version] if not calc_cilia_direction_vectors else ["production"]
        self.defo_input_filename = f"../output/mesh_{mesh_suffix}/deformation_p={polynomial_degree}_E={stiffness:.0f}_k={element_degree}_T={T:.0f}/checkpoints/displacement_velocity_dt={timestep:.4g}/"
        self.write_cpoint = write_checkpoint
        self.write_output = write_output
        self.output_dir = f"../output/mesh_{mesh_suffix}/flow_p={polynomial_degree}_E={stiffness:.0f}_k={element_degree}_dt={timestep:.4g}_T={T:.0f}/"
        self.calc_cilia_direction_vectors = calc_cilia_direction_vectors
        
        self.mesh = a4d.read_mesh(self.defo_input_filename, self.comm)
        self.out_mesh = a4d.read_mesh(self.defo_input_filename, self.comm)
        self.ft   = a4d.read_meshtags(self.defo_input_filename, self.mesh, meshtag_name='ft')
        self.fdim = self.mesh.topology.dim-1
        
        if "cilia" in self.bc_types:
            # Remove lower 1/3 of third ventricle from cilia tags
            # and add the corresponding tag to the displacement boundary
            self.ft = create_cilia_meshtags(self.mesh, self.ft)
            self.wall_deformation_tags = displacement_tags + (THIRD_NO_CILIA,)
            
        if "production" in self.bc_types:
            # Normal BC will acount for the deformation at choroid plexus
            self.wall_deformation_tags = [tag for tag in self.wall_deformation_tags if tag not in choroid_plexus_tags]

        self.jit_options["cache_dir"] = f"./cache/flow_p={polynomial_degree}_E={stiffness:.0f}_k={element_degree}_dt={timestep:.4g}_T={T:.0f}/"

    def setup(self):

        mesh = self.mesh
        k = self.element_degree

        # Integration measures
        dx = ufl.Measure('dx', domain=mesh, metadata={'quadrature_degree' : self.quadrature_degree}) # Cell integral
        self.ds = ds = ufl.Measure('ds', domain=mesh, subdomain_data=self.ft, metadata={'quadrature_degree' : self.quadrature_degree}) # Exterior facet integral
        dS = ufl.Measure('dS', domain=mesh, metadata={'quadrature_degree' : self.quadrature_degree}) # Interior facet integral

        self.n_hat = n_hat = ufl.FacetNormal(mesh) # Facet normal 
        h =  2*ufl.Circumradius(mesh)
        hA = ufl.avg(ufl.CellDiameter(mesh)) # Average cell diameter

        # Velocity and pressure finite element functions
        bdm_el = element("BDM", mesh.basix_cell(), k)
        dg_el  = element("DG", mesh.basix_cell(), k-1)
        dg_vec_el = element("DG", mesh.basix_cell(), k, shape=(mesh.geometry.dim,))
        self.V = V = dfx.fem.functionspace(mesh, bdm_el)
        Q = dfx.fem.functionspace(mesh, dg_el)
        self.u_ = dfx.fem.Function(V) # Velocity at previous timestep
        self.u_defo = dfx.fem.Function(V) # Deformation velocity

        # Trial and test functions
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

        mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(7e-4)) #[kg/(m*s)] #*1e-2 # Fluid dynamic viscosity [kg/(cm*s)]
        rho = dfx.fem.Constant(mesh, dfx.default_scalar_type(1e3)) # Fluid density
        gamma = dfx.fem.Constant(mesh, dfx.default_scalar_type(150)) # BDM penalty parameter
        alpha = dfx.fem.Constant(mesh, dfx.default_scalar_type(mu.value/(15e-6))) # Navier slip friction coefficient

        dt = dfx.fem.Constant(mesh, dfx.default_scalar_type(self.timestep))

        # Define displacement function
        cg_vec_el = element("Lagrange", mesh.basix_cell(), self.polynomial_degree, shape=(mesh.geometry.dim,))
        W = dfx.fem.functionspace(mesh, cg_vec_el)
        self.wh = wh = dfx.fem.Function(W)

        # Now that we have that we can define the Stokes problem in the deformed coordinates
        r = ufl.SpatialCoordinate(mesh)
        chi = r + wh          
        F = grad(chi) # Deformation gradient
        J = det(F) # Jacobian 
        n_hat = ufl.FacetNormal(mesh)

        # Define ALE operators
        Grad = lambda arg: dot(grad(arg), inv(F))
        Div = lambda arg: ufl.tr(dot(grad(arg), inv(F)))
        Nabla_Grad = lambda arg: dot(nabla_grad(arg), inv(F))

        # Symmetric gradient
        Eps = lambda arg: ufl.sym(Grad(arg))
        self.n = n = J*inv(F.T)*n_hat # Physical domain facet normal vector

        # Navier-Stokes problem in reference domain accounting for the deformation
        a00  = rho/dt * inner(u, v)*J*dx # Time derivative
        a00 += 2*mu*inner(Eps(u), Eps(v))*J*dx # Viscous dissipation
        a00 += (-inner(Avg(2*mu*Eps(u), n), Jump(v))*dS # Stabilization term to ensure
                -inner(Avg(2*mu*Eps(v), n), Jump(u))*dS # tangential continuity
                +2*mu*(gamma/hA)*inner(Jump(u), Jump(v))*dS
                )
        a00 += alpha * inner(Tangent(u, n), Tangent(v, n)) * ds(navier_slip_tags) # Navier slip term
        a01 = inner(p, Div(v))*J*dx
        a10 = inner(q, Div(u))*J*dx
        a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*J*dx # Zero block

        L0  = rho/dt * inner(self.u_, v)*J*dx # Time derivative

        if "cilia" in self.bc_types:
            # Weakly impose tangential traction to represent cilia
            tau_val = 7.89e-3 # Tangential traction force density [Pa]
            tau_mag = dfx.fem.Constant(mesh, dfx.default_scalar_type(tau_val))
            tau_vec = dfx.fem.Function(V)
            a4d.read_function(filename=f"../output/mesh_{self.mesh_suffix}/cilia_direction_vectors_k={k}/",
                                u=tau_vec,
                                name="cilia_direction"
                                )
            tau = tau_mag * tau_vec / ufl.sqrt(inner(tau_vec, tau_vec))

            L0 += inner(tau, Tangent(v, n))*ds(cilia_tags)

        L1 = inner(dfx.fem.Function(Q), q)*J*dx # Zero block

        # Compile Stokes bilinear form and linear form
        self.a_stokes = dfx.fem.form([[a00, a01], [a10, a11]], jit_options=self.jit_options)
        self.L = dfx.fem.form([L0, L1], jit_options=self.jit_options)

        if self.solver_type=="navier-stokes":
            c_vel = self.u_ - self.u_defo # Convection velocity

            # Navier-Stokes problem
            a00 += rho*inner(dot(c_vel, Nabla_Grad(u)), v)*J*dx # Convective term

            # Add convective term stabilization
            zeta = ufl.conditional(ufl.lt(dot(c_vel, n), 0), 1, 0) # Upwind velocity operator (equals 1 on inflow boundary, 0 on outflow boundary)
            a00 += (- rho*1/2*dot(jump(c_vel), n('+')) * avg(dot(u, v))  * dS 
                    - rho*dot(avg(c_vel), n('+')) * dot(jump(u), avg(v)) * dS 
                    - zeta*rho*1/2*dot(c_vel, n) * dot(u, v) * ds(zero_traction_tags)
            )

            self.a = dfx.fem.form([[a00, a01], [a10, a11]], jit_options=self.jit_options)
        else:

            self.a = self.a_stokes

        # Set boundary conditions on velocity
        facets_wall_defo = np.concatenate(([self.ft.find(tag) for tag in self.wall_deformation_tags]))
        u_dofs_defo = dfx.fem.locate_dofs_topological(V, self.fdim, facets_wall_defo)
        self.bcs = [dfx.fem.dirichletbc(self.u_defo, u_dofs_defo)]

        if "production" in self.bc_types:
            # Set choroid plexus inflow velocity BC strongly
            # Create expressions with positive and negative z-component of the velocity,
            # and interpolate the expressions into finite element functions.
            self.chp_prod = 5.833e-9 # Corresponds to 504 ml production per day [Czosnyka et al.]
            chp_area = assemble_scalar(1*ds(choroid_plexus_tags)) # The area of the choroid plexus boundary
            self.Q_tilde = dfx.fem.Constant(mesh, dfx.default_scalar_type(self.chp_prod/chp_area))
            self.chp_facets = np.concatenate(([self.ft.find(tag) for tag in choroid_plexus_tags]))
            chp_flux = create_normal_contribution_bc(V, 
                            (-n_hat*self.Q_tilde
                            + dot(self.u_defo, n_hat)*n_hat),
                            self.chp_facets)
            # Set flux strongly
            self.chp_bc = dfx.fem.Function(V)
            self.chp_bc.interpolate(chp_flux)
            chp_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim-1, self.chp_facets)
            self.bcs.append(dfx.fem.dirichletbc(self.chp_bc, chp_dofs))

        # Calculate offsets
        self.offset_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        self.offset_p = self.offset_u + Q.dofmap.index_map.size_local*Q.dofmap.index_map_bs

        # Setup I/O functions and files      
        self.uh_ = dfx.fem.Function(V); self.uh_.name = "velocity"
        self.ph_ = dfx.fem.Function(Q); self.ph_.name = "pressure"
        self.uh_rel_ = dfx.fem.Function(V); self.uh_rel_.name = "relative_velocity"
        self.u_defo_read = dfx.fem.Function(dfx.fem.functionspace(mesh, element("BDM", mesh.basix_cell(), k)))

        if self.write_output:
            if self.calc_cilia_direction_vectors:
                velocity_output_filename = f"../output/mesh_{self.mesh_suffix}/cilia_direction_vectors_k={k}.bp/"
            else:
                velocity_output_filename = self.output_dir+f"{self.solver_type}/BDM_{self.models[self.model_version]}_velocity.bp"
            self.uh_dg_ = dfx.fem.Function(dfx.fem.functionspace(self.mesh, dg_vec_el)); self.uh_dg_.name = "relative_velocity"
            self.velocity_output = dfx.io.VTXWriter(self.comm, velocity_output_filename, [self.uh_dg_], "BP4")
            pressure_output_filename = self.output_dir+f"{self.solver_type}/BDM_{self.models[self.model_version]}_pressure.bp"
            self.pressure_output = dfx.io.VTXWriter(self.comm, pressure_output_filename, [self.ph_], "BP4")

        if self.write_cpoint:
            if self.calc_cilia_direction_vectors:
                self.cpoint_filename = f"../output/mesh_{self.mesh_suffix}/cilia_direction_vectors_k={k}/"
            else:
                self.cpoint_filename = self.output_dir+f"{self.solver_type}/checkpoints/BDM_{self.models[self.model_version]}_velocity"
            a4d.write_mesh(self.cpoint_filename, mesh)
            a4d.write_meshtags(self.cpoint_filename, mesh, self.ft)
        
        # Compile forms used to calculate volumes and mean pressures
        self.vol = dfx.fem.form(dfx.fem.Constant(mesh, dfx.default_real_type(1.0)) * J*dx, jit_options=self.jit_options)
        self.mean_pressure_form_ = dfx.fem.form(self.ph_ * J*dx, jit_options=self.jit_options)

        # Compile form to calculate H(div) error
        self.e_div_form = dfx.fem.form(inner(Div(self.uh_), Div(self.uh_))*J*dx, jit_options=self.jit_options)

        # Set up Stokes problem linear system
        self.A = create_matrix_block(self.a) # System matrix
        self.xh = self.A.createVecRight() # Solution vector
        self.b = create_vector_block(self.L) # RHS vector
        self.create_solver()
        
        print("Global number of dofs: ", V.dofmap.index_map.size_global+Q.dofmap.index_map.size_global)

    def create_solver(self):
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(self.A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        opts = PETSc.Options() 
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

    def solve_cilia_direction_vectors(self):

        # Set name of the output functions
        self.uh_.name = "cilia_direction"
        
        tic = time.perf_counter()

        for t in self.times:

            print(f"Time = {t:.5g} sec")
            
            self.solve_blocked_system() # Solve the fluid equations of motion

            # Check if steady state is reached
            
            flux = assemble_scalar(dot(self.uh_, self.n_hat)*self.ds(choroid_plexus_tags))
            print("Choroid plexus flux = ", flux)
            relative_error = (np.abs(flux) - (self.chp_prod))/self.chp_prod*100
            print("Relative error [%] = ", relative_error)
            if np.less(relative_error, 0.1): # Less than 0.1 percent error
                break 
                
            self.u_.x.array[:] = self.uh_.x.array.copy()

        # Write to checkpoint file
        a4d.write_function_on_input_mesh(filename=self.cpoint_filename, u=self.uh_)

        if self.write_output:
            self.uh_dg_.interpolate(self.uh_)
            self.velocity_output.write(0.0)
            self.velocity_output.close()

        print(f"Solution loop time elapsed: {time.perf_counter()-tic:.4f} sec")

    def solve_initial_condition(self):

        # Read deformation velocity from file
        a4d.read_function(filename=self.defo_input_filename, u=self.u_defo_read, name="defo_velocity", time=self.times[0])
        self.u_defo.interpolate(self.u_defo_read)
        
        # Assemble the Stokes problem
        assemble_matrix_block(self.A, self.a_stokes, bcs=self.bcs)
        self.A.assemble()
        assemble_vector_block(self.b, self.L, self.a_stokes, bcs=self.bcs)

        # Solve and update solution functions
        self.ksp.solve(self.b, self.xh)
        self.u_.x.array[:self.offset_u] = self.xh.array[:self.offset_u]
        self.u_.x.scatter_forward()

    def solve(self):


        if self.calc_cilia_direction_vectors:
            # Calculate the cilia direction vectors and exit
            self.solve_cilia_direction_vectors()

        else:

            if self.solver_type=="navier-stokes":
                # Solve the Stokes problem and use it
                # as initial condition for Navier-Stokes
                print("Solving Stokes equations for initial condition ...")
                self.solve_initial_condition()
                print("Stokes equations solved and initial conditon set.\nNow solving the Navier-Stokes equations ...")
            
            tic = time.perf_counter()
            
            # Solve the time-dependent fluid equations

            for i, t in enumerate(self.times):

                print(f"Time = {t:.5g} sec")

                if self.read_time==self.num_timesteps_per_period: 
                    self.read_time = 0 # Reset read time

                # Read deformation velocity from file
                a4d.read_function(filename=self.defo_input_filename, u=self.u_defo_read, name="defo_velocity", time=self.read_time)
                self.u_defo.interpolate(self.u_defo_read)

                if "production" in self.bc_types:
                    # Set choroid plexus flux strongly
                    chp_flux = create_normal_contribution_bc(self.V, 
                                    (-self.Q_tilde/dot(self.n_hat, self.n)
                                    + dot(self.u_defo, self.n)/dot(self.n_hat, self.n))*self.n_hat,
                                    self.chp_facets)
                    self.chp_bc.interpolate(chp_flux)
            
                self.solve_blocked_system() # Solve the fluid equations of motion
        
                # Update finite element functions
                self.u_.x.array[:] = self.uh_.x.array.copy() # Previous timestep velocity (reference configuration)
                self.uh_rel_.x.array[:] = self.uh_.x.array.copy() - self.u_defo.x.array.copy() # Relative velocity (reference configuration)

                # Calculate mean pressure
                vol = self.comm.allreduce(dfx.fem.assemble_scalar(self.vol), op=MPI.SUM)
                mean_pressure_ = 1/vol*self.comm.allreduce(dfx.fem.assemble_scalar(self.mean_pressure_form_), op=MPI.SUM)
                self.ph_.x.array[:] += (self.mean_ICP - mean_pressure_) # Correct mean so that mean(p) = mean_ICP

                # Calculate H(div) error, this should be close to machine precision
                e_div_u = dfx.fem.assemble_scalar(self.e_div_form)
                assert np.isclose(e_div_u, 0.0, atol=1e-12), print(f"{e_div_u=}")
                
                if t >= self.final_period_start:
                    if self.write_output and (i % self.output_interval)==0:
                        self.uh_dg_.interpolate(self.uh_rel_)
                        self.velocity_output.write(t)
                        self.pressure_output.write(t)

                    if self.write_cpoint:
                        a4d.write_function(self.cpoint_filename, self.uh_, time=self.write_time)
                        a4d.write_function(self.cpoint_filename, self.uh_rel_, time=self.write_time)
                        a4d.write_function(self.cpoint_filename, self.ph_, time=self.write_time)
                        self.write_time += 1
                
                self.read_time += 1

            if self.write_output:
                self.velocity_output.close()
                self.pressure_output.close()

            print(f"Solution time elapsed: {time.perf_counter()-tic:.4f} sec")
            print(f"Output directory: {self.output_dir}")

def main(argv=None):

    parser = argparse.ArgumentParser(formatter_class=CustomParser)

    input_opts = parser.add_argument_group("Input options", "Options for reading input")
    input_opts.add_argument("-T", "--final_time", type=float, help="Final time of deformation simulation")
    input_opts.add_argument("-m", "--mesh_suffix", type=int, default=0, help="Mesh suffix, number of refinements (0=standard, 1=refined etc.)")
    input_opts.add_argument("-s", "--stiffness", type=float, help="Material stiffness (Young's modulus)")
    input_opts.add_argument("-p", "--polynomial_degree", type=int, help="Polynomial degree of solid model elements")

    discretization_opts = parser.add_argument_group("Temporal options", "Options for the discretization")
    discretization_opts.add_argument("-n", "--num_periods", type=int, help="Number of periods (cardiac cycles) to simulate")
    discretization_opts.add_argument("-dt", "--timestep", type=float, help="Timestep size")
    discretization_opts.add_argument("-k", "--element_degree", type=int, default=1, help="Finite element degree")

    model_opts = parser.add_argument_group("Model options", "Options for the fluid model")
    model_opts.add_argument("-g", "--governing_equations", type=str, help="Governing equations (Stokes or Navier-Stokes)")
    model_opts.add_argument("-v", "--model_version", type=int, help="The model version/which flow mechanisms to consider")
    model_opts.add_argument("-cd", "--cilia_direction", type=int, default=0, help="Solve for the cilia direction vectors (steady-state production velocity)")

    solver_opts = parser.add_argument_group("Solver options", "Options for the linear solver")
    solver_opts.add_argument("-d", "--direct", type=int, default=1, help="Use direct solver")

    output_opts = parser.add_argument_group("Output options", "Options for writing output")
    output_opts.add_argument("-c", "--checkpoint", type=int, default=0, help="Write checkpoint")
    output_opts.add_argument("-o", "--output", type=int, default=0, help="Write output")

    args = parser.parse_args(argv)
    if args.mesh_suffix not in range(3):
        raise ValueError(f'Unknown mesh prefix, "coarse", "medium", or "fine".')
    if args.governing_equations not in ["stokes", "navier-stokes"]:
        raise ValueError(f'Unknown governing equations, choose "stokes" or "navier-stokes".')

    solver = FluidSolverALE(args.final_time,
                            args.timestep,
                            args.num_periods,
                            args.mesh_suffix,
                            args.stiffness,
                            args.polynomial_degree,
                            args.element_degree,
                            args.model_version,
                            args.governing_equations,
                            args.checkpoint,
                            args.output,
                            args.cilia_direction)    
    solver.setup()
    solver.solve()

if __name__=='__main__':
    main()