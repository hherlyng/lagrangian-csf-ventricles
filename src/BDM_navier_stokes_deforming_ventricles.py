import ufl
import time

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from sys       import argv
from ufl       import inner, dot, grad, det, inv, jump, avg, nabla_grad
from scifem    import assemble_scalar
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from utilities.fem import create_normal_contribution_bc
from dolfinx.fem.petsc import LinearProblem, assemble_matrix_block, assemble_vector_block, create_matrix_block, create_vector_block

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
                          CANAL_WALL, THIRD_VENTRICLE_WALL, CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD, CHOROID_PLEXUS_FOURTH)
choroid_plexus_tags = (CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD, CHOROID_PLEXUS_FOURTH)
wall_deformation_tags = [tag for tag in cilia_tags if tag not in choroid_plexus_tags]

# Solve the Navier-Stokes equations in a moving domain
# by an ALE method. Wall motion is 
# prescribed in time, given by solutions to the 
# time-dependent linear elasticity equations.
comm = MPI.COMM_WORLD # MPI communicator
T = float(argv[2]) # Final time
timestep = float(argv[3])
N = int(T / timestep) # Number of timesteps
times = np.linspace(0, T, N+1)
mesh_prefix_input = int(argv[4])
if mesh_prefix_input==1:
    mesh_prefix = 'coarse'
elif mesh_prefix_input==2:
    mesh_prefix = 'medium'
elif mesh_prefix_input==3:
    mesh_prefix = 'fine'
else:
    raise ValueError(f'Unknown mesh prefix, choose 1 (coarse), 2 (medium), or 3 (fine).')
defo_input_filename = f"../output/{mesh_prefix}-mesh/deformation/checkpoints/time_dep_deformation_velocity_dt={timestep:.4g}_T={T:.4g}/"
mesh = a4d.read_mesh(defo_input_filename, comm, read_from_partition=False)
out_mesh = a4d.read_mesh(defo_input_filename, comm, read_from_partition=False)
ft   = a4d.read_meshtags(defo_input_filename, mesh, meshtag_name='ft')
facet_dim = mesh.topology.dim-1

mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity
num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts # Total number of facets

quadrature_degree = 8
dx = ufl.Measure('dx', domain=mesh, metadata={'quadrature_degree' : quadrature_degree}) # Cell integral
ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft, metadata={'quadrature_degree' : quadrature_degree}) # Exterior facet integral
dS = ufl.Measure('dS', domain=mesh, metadata={'quadrature_degree' : quadrature_degree}) # Interior facet integral

# Create finite element function in P2 space for the mesh displacement
vec_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
W = dfx.fem.functionspace(mesh, vec_el)
wh = dfx.fem.Function(W)

# Now that we have that we can define the Stokes problem in the deformed coordinates
r = ufl.SpatialCoordinate(mesh)
chi = r + wh          
F = grad(chi) # Deformation gradient
J = det(F) # Jacobian 
n_hat = ufl.FacetNormal(mesh) # Facet normal on the reference mesh
n = J*inv(F.T)*n_hat # Deformed domain facet normal and surface integral measure
hA = ufl.avg(ufl.CellDiameter(mesh)) # Average cell diameter of reference mesh

scal_el = element("Lagrange", mesh.basix_cell(), 1)
bdm_el = element("BDM", mesh.basix_cell(), 1)
dg_el  = element("DG", mesh.basix_cell(), 0)
dg_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
V = dfx.fem.functionspace(mesh, bdm_el)
Q = dfx.fem.functionspace(mesh, dg_el)
u_zero = dfx.fem.Function(V)
u_ = dfx.fem.Function(V) # Velocity at previous timestep
u_defo = dfx.fem.Function(V) # Deformation velocity

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
BDM_penalty = dfx.fem.Constant(mesh, dfx.default_scalar_type(50.0))
dt = dfx.fem.Constant(mesh, dfx.default_scalar_type(timestep))

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

    if consistent: # Add symmetrization terms
        return (-inner(Avg(2*mu*Eps(u), n), Jump(v))*dS
                -inner(Avg(2*mu*Eps(v), n), Jump(u))*dS
                + 2*mu*(BDM_penalty/hA)*inner(Jump(u), Jump(v))*dS)

    # For preconditioning
    return 2*mu*(BDM_penalty/hA)*inner(Jump(u), Jump(v))*dS

# Tangential traction BC
tau_val = 7.89e-3 # Tangential traction force density [Pa]
tau = dfx.fem.Constant(mesh, dfx.default_scalar_type(tau_val))
tau_vec   = tau*ufl.as_vector((0, 1, 1)) # Stress vector to be projected tangentially onto the mesh
tangent_traction_dorsal = lambda n: Tangent(tau_vec, n) # Use the tau expression to define the tangent traction vector
c_vel = u_ - u_defo # Convection velocity
# Navier-Stokes problem in reference domain accounting for the deformation
a00 = (2*mu*inner(Eps(u), Eps(v))*J*dx # Viscous dissipation
      + stabilization(u, v) # BDM stabilization
      - mu*inner(dot(Grad(u).T, n), v)*ds(zero_traction_tags) # Parallel flow at inlet/outlet
      )
a01 = inner(p, Div(v))*J*dx
a10 = inner(q, Div(u))*J*dx
a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*J*dx

a_stokes = dfx.fem.form([[a00, a01], [a10, a11]])

L0 = inner(u_zero, v)*J*dx
L1 = inner(dfx.fem.Function(Q), q)*J*dx

# Navier-Stokes problem
a00 += rho/dt * inner(u, v)*J*dx # Time derivative
a00 += rho*inner(dot(c_vel, Nabla_Grad(u)), v)*J*dx # Convective term

# Add convective term stabilization
zeta = ufl.conditional(ufl.lt(dot(c_vel, n), 0), 1, 0) # Upwind velocity operator (equals 1 on inflow boundary, 0 on outflow boundary)
a00 += (- rho*1/2*dot(jump(c_vel), n('+')) * avg(dot(u, v))  * dS 
        - rho*dot(avg(c_vel), n('+')) * dot(jump(u), avg(v)) * dS 
        - zeta*rho*1/2*dot(c_vel, n) * dot(u, v) * ds(zero_traction_tags)
)

a = dfx.fem.form([[a00, a01], [a10, a11]])
L = dfx.fem.form([L0, L1])

# Set boundary conditions on velocity
facets_wall_defo = np.concatenate(([ft.find(tag) for tag in wall_deformation_tags]))
u_dofs_defo = dfx.fem.locate_dofs_topological(V, facet_dim, facets_wall_defo)

# Set choroid plexus inflow velocity BC strongly
# Create expressions with positive and negative z-component of the velocity,
# and interpolate the expressions into finite element functions.
chp_prod = 5.833e-9 # Corresponds to 504 ml production per day [Czosnyka et al.]
chp_area = assemble_scalar(1*ds(choroid_plexus_tags)) # The area of the choroid plexus boundary
chp_velocity = chp_prod/chp_area
facets_chp = np.concatenate(([ft.find(tag) for tag in choroid_plexus_tags]))
u_chp = create_normal_contribution_bc(V, (-chp_velocity*n_hat + dot(u_defo, n_hat)*n_hat), facets_chp)
u_dofs_chp_prod = dfx.fem.locate_dofs_topological(V, facet_dim, facets_chp)

bcs_fluid = [dfx.fem.dirichletbc(u_defo, u_dofs_defo),
             dfx.fem.dirichletbc(u_chp, u_dofs_chp_prod)]

# Define deforming mesh and reference coordinates (coordinates of mesh at t=0)
x_reference = out_mesh.geometry.x.copy()

# Compute cells for point evaluation of the deformation function wh
cells = []
points_on_proc = []
bb_tree = dfx.geometry.bb_tree(out_mesh, mesh.topology.dim)
cell_candidates = dfx.geometry.compute_collisions_points(bb_tree, x_reference)
colliding_cells = dfx.geometry.compute_colliding_cells(out_mesh, cell_candidates, x_reference)
for i, point in enumerate(x_reference):
    if len(colliding_cells.links(i)>0):
        cc = colliding_cells.links(i)[0]
        cells.append(cc)
        points_on_proc.append(point)
# Convert to numpy arrays
cells = np.array(cells)
points_on_proc = np.array(points_on_proc)

def create_solver(A):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
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
    return ksp

def solve_navier_stokes_blocked():

    A.zeroEntries()
    assemble_matrix_block(A, a, bcs=bcs_fluid)
    A.assemble()

    with b.localForm() as b_loc: b_loc.set(0)
    assemble_vector_block(b, L, a, bcs=bcs_fluid)
    
    u, p = dfx.fem.Function(V), dfx.fem.Function(Q)
    ksp.solve(b, x_sol)
    assert ksp.getConvergedReason() > 0

    # Update and MPI communcation
    u.x.array[:offset_u] = x_sol.array_r[:offset_u]
    u.x.scatter_forward()
    p.x.array[:(offset_p-offset_u)] = x_sol.array_r[offset_u:offset_p]
    p.x.scatter_forward()

    return u, p

vol = mesh.comm.allreduce(
        dfx.fem.assemble_scalar(
            dfx.fem.form(
                dfx.fem.Constant(mesh, dfx.default_real_type(1.0)) * ufl.dx(out_mesh))
            ),
        op=MPI.SUM
)


if __name__=='__main__':
    write_cpoint = True if int(argv[1])==1 else False

    uh = dfx.fem.Function(dfx.fem.functionspace(out_mesh, dg_vec_el)); uh.name = 'velocity'
    ph = dfx.fem.Function(dfx.fem.functionspace(out_mesh, dg_el)); ph.name = 'pressure'
    uh_rel_ = dfx.fem.Function(V)
    uh_rel = dfx.fem.Function(dfx.fem.functionspace(out_mesh, dg_vec_el)); uh_rel.name = 'relative_velocity'

    dg1_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    u_defo_read = dfx.fem.Function(V)

    velocity_output_filename = f"../output/{mesh_prefix}-mesh/flow/navier-stokes/BDM_deforming_velocity.pvd"
    velocity_output = dfx.io.VTKFile(comm, velocity_output_filename, "w")
    pressure_output_filename = f"../output/{mesh_prefix}-mesh/flow/navier-stokes/BDM_deforming_pressure.pvd"
    pressure_output = dfx.io.VTKFile(comm, pressure_output_filename, "w")

    if write_cpoint:
        cpoint_filename = f"../output/{mesh_prefix}-mesh/flow/navier-stokes/checkpoints/BDM_deforming_velocity"
        a4d.write_mesh(cpoint_filename, mesh, store_partition_info=True)
        a4d.write_meshtags(cpoint_filename, mesh, ft)
    
    # Compile form used to calculate the mean pressure
    mean_pressure_form = dfx.fem.form((1/vol) * ph * ufl.dx(out_mesh))
    
    print("Number of dofs Navier-Stokes eqs: ", V.dofmap.index_map.size_global+Q.dofmap.index_map.size_global)

    tic = time.perf_counter()
    print("Solving Stokes eqs for initial condition ...")
    # Solve the Stokes problem and use it
    # as initial condition for Navier-Stokes
    # Assemble the Stokes problem
    A_stokes = assemble_matrix_block(a_stokes,bcs=bcs_fluid)
    A_stokes.assemble()
    b = assemble_vector_block(L, a_stokes, bcs=bcs_fluid)

    # Create the Navier-Stokes problem
    A = create_matrix_block(a)
    x_sol = A.createVecRight() # Solution vector

    # Create linear solver and offsets for the blocked functions
    ksp = create_solver(A_stokes)
    offset_u = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    offset_p = offset_u + Q.dofmap.index_map.size_local*Q.dofmap.index_map_bs

    # Update deformation velocity
    a4d.read_function(filename=defo_input_filename, u=u_defo, name="defo_velocity", time=times[0])

    # Solve and update solution functions
    ksp.solve(b, x_sol)
    u_.x.array[:offset_u] = x_sol.array_r[:offset_u]
    u_.x.scatter_forward()
    uh.interpolate(u_) # Interpolate velocity into visualization function
    ph.x.array[:(offset_p-offset_u)] = x_sol.array_r[offset_u:offset_p]
    ph.x.scatter_forward()

    print("Stokes eqs solved and initial conditon set.\n Entering solution time-loop ...")

    # Set Navier-Stokes system matrix as KSP operator
    ksp.setOperators(A)
    
    for t in times:

        print(f"Time = {t:.4f} sec")

        # Read deformation from file
        a4d.read_function(filename=defo_input_filename, u=wh, name="defo_displacement", time=float(f"{t:.4g}"))
        a4d.read_function(filename=defo_input_filename, u=u_defo_read, name="defo_velocity", time=float(f"{t:.4g}"))
        u_defo.interpolate(u_defo_read)

        u_chp_updated = create_normal_contribution_bc(V, (-chp_velocity*n_hat + dot(u_defo, n_hat)*n_hat), facets_chp)
        u_chp.interpolate(u_chp_updated)
    
        uh_, ph_ = solve_navier_stokes_blocked() # Solve the Stokes equations

        # Update output functions
        uh.interpolate(uh_)
        ph.interpolate(ph_)

        u_.x.array[:] = uh_.x.array.copy() # Update previous timestep velocity

        # Update relative velocity
        uh_rel_.x.array[:] = uh_.x.array.copy() - u_defo.x.array.copy()
        uh_rel.interpolate(uh_rel_)

        if len(points_on_proc)>0:
            wh_x_reference = wh.eval(x=x_reference, cells=cells) # Evaluate the deformed coordinates at the reference coordinates

            # Update output mesh
            out_mesh.geometry.x[:, :out_mesh.geometry.dim] = x_reference[:, :out_mesh.geometry.dim] + wh_x_reference

        # Calculate mean pressure and subtract to make mean = 0
        mean_pressure = comm.allreduce(dfx.fem.assemble_scalar(mean_pressure_form), op=MPI.SUM)
        ph.x.array[:] -= mean_pressure

        # Write output
        velocity_output.write_mesh(out_mesh, t)
        velocity_output.write_function(uh, t)
        velocity_output.write_function(uh_rel, t)
        pressure_output.write_mesh(out_mesh, t)
        pressure_output.write_function(ph, t)

        if write_cpoint:
            a4d.write_function(cpoint_filename, uh, time=t)
            a4d.write_function_on_input_mesh(cpoint_filename, uh_rel, time=t)
            a4d.write_function(cpoint_filename, ph, time=t)

    print(f"Solution loop time elapsed: {time.perf_counter()-tic:.4f} sec")

    # Close output files
    velocity_output.close()
    pressure_output.close()