import ufl
import time

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from ufl       import inner, dot, grad, det, inv
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

zero_deformation_tags = zero_traction_tags = (CANAL_OUT, LATERAL_APERTURES)
wall_deformation_tags = (AQUEDUCT_WALL, FORAMINA_34_WALL, LATERAL_VENTRICLES_WALL, FOURTH_VENTRICLE_WALL,
                          CANAL_WALL, THIRD_VENTRICLE_WALL, CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD, CHOROID_PLEXUS_FOURTH)
cilia_tags = wall_deformation_tags
choroid_plexus_tags = (CHOROID_PLEXUS_LATERAL, CHOROID_PLEXUS_THIRD, CHOROID_PLEXUS_FOURTH)
impermeability_tags = [tag for tag in cilia_tags if tag not in choroid_plexus_tags]

# Solve stationary Stokes in moving domain by ALE method. Wall motion is 
# prescribed in time, given by solutions to the 
# time-dependent linear elasticity equations.
comm = MPI.COMM_WORLD
mesh_prefix = 'medium'
defo_input_filename = f"../output/{mesh_prefix}-mesh/deformation/checkpoints/time_dep_deformation_velocity/"
mesh = a4d.read_mesh(defo_input_filename, comm, read_from_partition=False)
out_mesh = a4d.read_mesh(defo_input_filename, comm, read_from_partition=False)
ft   = a4d.read_meshtags(defo_input_filename, mesh, meshtag_name='ft')
facet_dim = mesh.topology.dim-1

mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity
num_facets = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts # Total number of facets

dx = ufl.Measure('dx', domain=mesh) # Cell integral
ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft) # Exterior facet integral
dS = ufl.Measure('dS', domain=mesh) # Interior facet integral

# The ALE problem needs to extend boundary deformation to the entire domain
# to define mesh displacement field
vec_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
W = dfx.fem.functionspace(mesh, vec_el)

# Test and trial functions
w, dw = ufl.TrialFunction(W), ufl.TestFunction(W)

# BC functions
w_wall = dfx.fem.Function(W)
w_zero = dfx.fem.Function(W)

a_ale = inner(grad(w), grad(dw))*dx
L_ale = inner(w_zero, dw)*dx

# BCs
facets_wall_defo = np.concatenate(([ft.find(tag) for tag in wall_deformation_tags]))
w_dofs_wall_defo = dfx.fem.locate_dofs_topological(W, facet_dim, facets_wall_defo)

facets_zero_disp = np.concatenate(([ft.find(tag) for tag in zero_deformation_tags]))
w_dofs_zero_disp = dfx.fem.locate_dofs_topological(W, facet_dim, facets_zero_disp)

bcs_ale = [dfx.fem.dirichletbc(w_wall, w_dofs_wall_defo), # Imposed wall displacement
           dfx.fem.dirichletbc(w_zero, w_dofs_zero_disp) # Zero displacement at spinal canal
           ]

# This serves to define the deformation of the mesh
wh = dfx.fem.Function(W)

# Now that we have that we can define the Stokes problem in the deformed coordinates
r = ufl.SpatialCoordinate(mesh)
chi = r + wh          
F = grad(chi) # Deformation gradient
J = det(F) # Jacobian 
n = ufl.FacetNormal(mesh)

scal_el = element("Lagrange", mesh.basix_cell(), 1)
bdm_el = element("BDM", mesh.basix_cell(), 1)
dg_el  = element("DG", mesh.basix_cell(), 0)
dg_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
V = dfx.fem.functionspace(mesh, bdm_el)
Q = dfx.fem.functionspace(mesh, dg_el)
u_zero = dfx.fem.Function(V)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
                  
Grad = lambda arg: dot(grad(arg), inv(F))
Div = lambda arg: inner(grad(arg), inv(F))

# Symmetric gradient
Eps = lambda arg: dot(ufl.sym(grad(arg)), inv(F))

mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(7e-4)) #[kg/(m*s)] #*1e-2 # Dynamic viscosity [kg/(cm*s)]
rho = dfx.fem.Constant(mesh, dfx.default_scalar_type(1e3))
BDM_penalty = dfx.fem.Constant(mesh, dfx.default_scalar_type(50.0))

timestep = 0.2
T = 1
N = int(T / timestep)

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
        return (-inner(Avg(2*mu*Eps(u), n), Jump(Tangent(v, n)))*J('+')*dS
                -inner(Avg(2*mu*Eps(v), n), Jump(Tangent(u, n)))*J('+')*dS
                + 2*mu*(BDM_penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*J('+')*dS)

    # For preconditioning
    return 2*mu*(BDM_penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*J('+')*dS

beta = dfx.fem.Constant(mesh, dfx.default_scalar_type(10.0)) # Noslip penalty parameter

# Tangential traction BC
tau_val = 7.89e-3 # Tangential traction force density [Pa]
tau = dfx.fem.Constant(mesh, dfx.default_scalar_type(tau_val))
tau_vec   = tau*ufl.as_vector((0, 1, 1)) # Stress vector to be projected tangentially onto the mesh
tangent_traction_dorsal = lambda n: Tangent(tau_vec, n) # Use the tau expression to define the tangent traction vector

# ds_ = ds(noslip_bdry_tags)
# Stokes problem in reference domain accounting for the deformation
a00 = (2*mu*inner(Eps(u), Eps(v))*J*dx # Viscous dissipation
      + stabilization(u, v) # BDM stabilization
      - mu*inner(dot(Grad(u).T, n), v)*J*ds(zero_traction_tags) # Parallel flow at inlet/outlet
      )
a01 = inner(p, Div(v))*J*dx
a10 = inner(q, Div(u))*J*dx
a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*J*dx

L0 = inner(u_zero, v)*J*dx \
#    + inner(Tangent(v, n), tangent_traction_dorsal(n))*J*ds(cilia_tags) \
L1 = inner(dfx.fem.Function(Q), q)*J*dx

a = dfx.fem.form([[a00, a01], [a10, a11]])
L = dfx.fem.form([L0, L1])

# Set boundary conditions on velocity
u_dofs_impermeability= dfx.fem.locate_dofs_topological(V, facet_dim, facets_wall_defo)

# Set choroid plexus inflow velocity BC strongly
# Create expressions with positive and negative z-component of the velocity,
# and interpolate the expressions into finite element functions.
chp_prod = 5.833e-9 # Corresponds to 504 ml production per day [Czosnyka et al.]
chp_area = assemble_scalar(1*ds(choroid_plexus_tags)) # The area of the choroid plexus boundary
chp_velocity = chp_prod/chp_area
facets_chp = np.concatenate(([ft.find(tag) for tag in choroid_plexus_tags]))
u_chp = create_normal_contribution_bc(V, -chp_velocity*n, facets_chp)
u_dofs_chp_prod = dfx.fem.locate_dofs_topological(V, facet_dim, facets_chp)

bcs_fluid = [dfx.fem.dirichletbc(u_zero, u_dofs_impermeability),
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

# Define the mesh deformation problem
ale_problem = LinearProblem(a_ale, L_ale, bcs_ale, 
                            wh,
                            petsc_options={"ksp_type" : "preonly",
                                           "pc_type" : "lu",
                                           "pc_factor_mat_solver_type" : "mumps"})
    

A = create_matrix_block(a)
b = create_vector_block(L)
x_sol = A.createVecRight()

# Configure linear solver
ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
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

print("Number of dofs mesh motion problem: ", W.dofmap.index_map.size_global)
print("Number of dofs Stokes eqs: ", V.dofmap.index_map.size_global+Q.dofmap.index_map.size_global)

def solve_navier_stokes_blocked(uh: dfx.fem.Function, ph: dfx.fem.Function):

    A.zeroEntries()
    assemble_matrix_block(A, a, bcs=bcs_fluid)
    A.assemble()

    with b.localForm() as b_loc: b_loc.set(0)
    assemble_vector_block(b, L, a, bcs=bcs_fluid)
    
    ksp.solve(b, x_sol)
    assert ksp.getConvergedReason() > 0

    # Update and MPI communcation
    uh.x.array[:offset_u] = x_sol.array[:offset_u]
    uh.x.scatter_forward()
    ph.x.array[:(offset_p-offset_u)] = x_sol.array[offset_u:offset_p]
    ph.x.scatter_forward()

    return uh, ph


vol = mesh.comm.allreduce(
        dfx.fem.assemble_scalar(dfx.fem.form(
                                dfx.fem.Constant(mesh, dfx.default_real_type(1.0)) * ufl.dx(out_mesh))
                                ), op=MPI.SUM
)

if __name__=='__main__':
    tic = time.perf_counter()
    from sys import argv
    write_cpoint = True if int(argv[1])==1 else False

    uh_ = dfx.fem.Function(V)
    ph_ = dfx.fem.Function(Q)
    uh = dfx.fem.Function(dfx.fem.functionspace(out_mesh, dg_vec_el)); uh.name = 'velocity'
    ph = dfx.fem.Function(dfx.fem.functionspace(out_mesh, dg_el)); ph.name = 'pressure'

    velocity_output_filename = f"../output/{mesh_prefix}-mesh/flow/stokes/BDM_deforming_velocity.pvd"
    velocity_output = dfx.io.VTKFile(comm, velocity_output_filename, "w")
    pressure_output_filename = f"../output/{mesh_prefix}-mesh/flow/stokes/BDM_deforming_pressure.pvd"
    pressure_output = dfx.io.VTKFile(comm, pressure_output_filename, "w")

    if write_cpoint:
        cpoint_filename = f"../output/{mesh_prefix}-mesh/flow/stokes/checkpoints/BDM_deforming_velocity"
        a4d.write_mesh(cpoint_filename, mesh, store_partition_info=True)
        a4d.write_meshtags(cpoint_filename, mesh, ft)
    
    mean_pressure_form = dfx.fem.form((1/vol) * ph * ufl.dx(out_mesh))

    for t in np.linspace(0, T, N+1):

        a4d.read_function(filename=defo_input_filename, u=w_wall, name="defo_displacement", time=float(f"{t:.4g}"))
        wh.interpolate(w_wall)

        tic_solve = time.perf_counter()
        uh_, ph_ = solve_navier_stokes_blocked(uh_, ph_)
        print(f"Solve time: {time.perf_counter()-tic_solve:.4f} sec")

        # Update output functions
        uh.interpolate(uh_)
        ph.interpolate(ph_)
        ph.x.array[:] -= assemble_scalar(mean_pressure_form)

        if len(points_on_proc)>0:
            wh_x_reference = wh.eval(x=x_reference, cells=cells) # Evaluate the deformed coordinates at the reference coordinates

            # Update output mesh
            out_mesh.geometry.x[:, :out_mesh.geometry.dim] = x_reference[:, :out_mesh.geometry.dim] + wh_x_reference

        # Write output
        velocity_output.write_mesh(out_mesh, t)
        velocity_output.write_function(uh, t)
        pressure_output.write_mesh(out_mesh, t)
        pressure_output.write_function(ph, t)

        if write_cpoint:
            a4d.write_function_on_input_mesh(cpoint_filename, uh, time=t)
            a4d.write_function_on_input_mesh(cpoint_filename, ph, time=t)

        # Calculate mean pressure
        vol = assemble_scalar(1*ufl.dx(out_mesh))
        print("Mean pressure: ", 1/vol*assemble_scalar(ph*ufl.dx(out_mesh)))

    print(f"Solution loop time elapsed: {time.perf_counter()-tic:.4f} sec")

    # Close output files
    velocity_output.close()
    pressure_output.close()