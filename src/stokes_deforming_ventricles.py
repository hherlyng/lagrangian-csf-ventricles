import ufl

import numpy   as np
import dolfinx as dfx

from ufl       import inner, dot, grad, det, inv
from scifem    import create_real_functionspace, assemble_scalar
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from dolfinx.fem.petsc import LinearProblem

# Facet tags
CILIA_TAG = 101
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

# Cell tags
CANAL = 3
THIRD_VENTRICLE = 4
AQUEDUCT = 5
FORAMINA_34 = 6
LATERAL_VENTRICLES = 7
FOURTH_VENTRICLE = 8

zero_displacement_tags = [CANAL_OUT, FOURTH_VENTRICLE_OUT]
wall_displacement_tags = [AQUEDUCT_WALL, FORAMINA_34_WALL, LATERAL_VENTRICLES_WALL, FOURTH_VENTRICLE_WALL,
                          CANAL_WALL, THIRD_VENTRICLE_WALL]

# Solve stationary Stokes in moving domain by ALE method. Wall motion is 
# prescribed in time.
comm = MPI.COMM_WORLD
with dfx.io.XDMFFile(comm, "./geometries/mesh_with_cilia.xdmf", "r") as xdmf:
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

# Parameters of the displacement
T_wall = 0.5
A_wall = 0.005

# Displacement expression classes
class WallDeformation:
    def __init__(self):
        self.t = 0
        self.A = A_wall
        self.T = T_wall
        
    def __call__(self, x):
        return np.stack((self.A*4*x[0]*(1-x[0])*np.sin(2*np.pi*self.t/self.T),
                         np.zeros(x.shape[1]),
                         np.zeros(x.shape[1])))

# Velocity expression classes
class WallVelocity:
    def __init__(self):
        self.t = 0
        self.A = A_wall
        self.T = T_wall
        
    def __call__(self, x):
        return np.stack((self.A*4*x[0]*(1-x[0])*np.cos(2*np.pi*self.t/self.T)*2*np.pi/self.T,
                         np.zeros(x.shape[1]),
                         np.zeros(x.shape[1])))

# The ALE problem needs to extend boundary deformation to the entire domain
# to define mesh displacement field
vec_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
W = dfx.fem.functionspace(mesh, vec_el)

# Test and trial functions
w, dw = ufl.TrialFunction(W), ufl.TestFunction(W)

# BC functions
u_wall = dfx.fem.Function(W)
zero = dfx.fem.Function(W)

a_ale = inner(grad(w), grad(dw))*dx
L_ale = inner(zero, dw)*dx

# BCs
facets_wall = np.concatenate(([ft.find(tag) for tag in wall_displacement_tags]))
dofs_wall = dfx.fem.locate_dofs_topological(W, facet_dim, facets_wall)
u_wall_expr = WallDeformation()

facets_zero_disp = np.concatenate(([ft.find(tag) for tag in zero_displacement_tags]))
dofs_zero_disp = dfx.fem.locate_dofs_topological(W, facet_dim, facets_zero_disp)

bcs_ale = [dfx.fem.dirichletbc(u_wall, dofs_wall), # Imposed wall displacement
           dfx.fem.dirichletbc(zero, dofs_zero_disp) # Zero displacement at spinal canal
           ]


# Now we can define the Stokes problem in the deformed coordinates
wh = dfx.fem.Function(W) # Function that defines the deformation of the mesh
r = ufl.SpatialCoordinate(mesh)
chi = r + wh          
F = grad(chi) # Deformation gradient
J = det(F) # Jacobian 
n = ufl.FacetNormal(mesh)

scal_el = element("Lagrange", mesh.basix_cell(), 1)
V = dfx.fem.functionspace(mesh, vec_el)
Q = dfx.fem.functionspace(mesh, scal_el)
R = create_real_functionspace(mesh)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
lm, dlm = ufl.TrialFunction(R), ufl.TestFunction(R)
                  
Grad = lambda arg: dot(grad(arg), inv(F))
Div = lambda arg: inner(grad(arg), inv(F))

p_bc = dfx.fem.Function(Q)
class PressureBC:
    def __init__(self):
        self.t = 0
        self.A = 1
    def __call__(self, x):
        return x[0]*self.A*np.sin(2*np.pi*self.t)

p_bc_expr = PressureBC()
# Stokes problem in reference domain accounting for the deformation
a00 = inner(Grad(u), Grad(v))*J*dx
a01 = inner(p, Div(v))*J*dx
a02 = None 
a10 = inner(q, Div(u))*J*dx
a11 = dfx.fem.Constant(mesh, dfx.default_scalar_type(0.0))*inner(p, q)*J*dx
a12 = inner(lm, q)*J*dx
a20 = None
a21 = inner(p, dlm)*J*dx
a22 = None

L0 = inner(zero, v)*J*dx + dot(p_bc*n, v)*J*ds(CANAL_OUT)
L1 = inner(dfx.fem.Function(Q), q)*J*dx
L2 = inner(dfx.fem.Function(R), dlm)*J*dx

a = dfx.fem.form([[a00, a01], [a10, a11]])
L = dfx.fem.form([L0, L1])
# a = dfx.fem.form([[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]])
# L = dfx.fem.form([L0, L1, L2])

# Set boundary conditions on velocity
v_wall = dfx.fem.Function(V)
v_wall_expr = WallVelocity()
v_dofs_wall = dfx.fem.locate_dofs_topological(V, facet_dim, facets_wall)

v_noslip = dfx.fem.Function(V)

noslip_dofs = dfx.fem.locate_dofs_topological(V, facet_dim, facets_zero_disp)
bcs_stokes = [dfx.fem.dirichletbc(v_wall, v_dofs_wall),
              dfx.fem.dirichletbc(v_noslip, noslip_dofs)]

# Define deforming mesh and reference coordinates (coordinates of mesh at t=0)
x_reference = out_mesh.geometry.x.copy()

# Compute cells for point evaluation of the deformation function wh
cells = []
bb_tree = dfx.geometry.bb_tree(out_mesh, mesh.topology.dim)
cell_candidates = dfx.geometry.compute_collisions_points(bb_tree, x_reference)
colliding_cells = dfx.geometry.compute_colliding_cells(out_mesh, cell_candidates, x_reference)
for i, point in enumerate(x_reference):
    if len(colliding_cells.links(i)>0):
        cc = colliding_cells.links(i)[0]
        cells.append(cc)
cells = np.array(cells)

uh = dfx.fem.Function(dfx.fem.functionspace(out_mesh, vec_el)); uh.name = 'uh'
ph = dfx.fem.Function(dfx.fem.functionspace(out_mesh, scal_el)); ph.name = 'ph'

velocity_output = dfx.io.VTKFile(comm, "./output/CG/velocity.pvd", "w")
pressure_output = dfx.io.VTKFile(comm, "./output/CG/pressure.pvd", "w")

# Define the mesh deformation problem
ale_problem = LinearProblem(a_ale, L_ale, bcs_ale, 
                            wh,
                            petsc_options={"ksp_type" : "preonly",
                                           "pc_type" : "lu",
                                           "pc_factor_mat_solver_type" : "mumps"})

def assemble_system(lhs_form, rhs_form, bcs):
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

def create_block_solver(A):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    return ksp

def solve_stokes(u_element, p_element, domain):
    V = dfx.fem.functionspace(domain, u_element)
    Q = dfx.fem.functionspace(domain, p_element)

    A, b = assemble_system(a, L, bcs_stokes)

    ksp = create_block_solver(A)

    u, p, lm = dfx.fem.Function(V), dfx.fem.Function(Q), dfx.fem.Function(R)
    # w = PETSc.Vec().createNest([u.x.petsc_vec, p.x.petsc_vec, lm.x.petsc_vec])
    w = PETSc.Vec().createNest([u.x.petsc_vec, p.x.petsc_vec])
    ksp.solve(b, w)
    assert ksp.getConvergedReason() > 0
    u.x.scatter_forward()
    p.x.scatter_forward()

    return u, p

for time in np.linspace(0, 0.5, 30):
    # Update time variable of boundary conditions 
    # Wall deformation
    u_wall_expr.t = time

    # Wall velocity
    v_wall_expr.t = time

    # Pressure BC
    p_bc_expr.t = time

    # Interpolate BC expressions into BC functions
    u_wall.interpolate(u_wall_expr)
    v_wall.interpolate(v_wall_expr)
    p_bc.interpolate(p_bc_expr)
    
    # Solve the linear problems
    ale_problem.solve() # Solve the mesh motion problem
    uh_, ph_ = solve_stokes(vec_el, scal_el, mesh) # Solve the Stokes equations
    
    # Update output functions
    uh.x.array[:] = uh_.x.array.copy()
    ph.x.array[:] = ph_.x.array.copy()

    wh_x_reference = wh.eval(x=x_reference, cells=cells) # Evaluate the deformed coordinates at the reference coordinates

    # Update output mesh
    out_mesh.geometry.x[:, :out_mesh.geometry.dim] = x_reference[:, :out_mesh.geometry.dim] + wh_x_reference

    # Write output
    velocity_output.write_mesh(out_mesh, time)
    velocity_output.write_function(uh, time)
    pressure_output.write_mesh(out_mesh, time)
    pressure_output.write_function(ph, time)

    # Calculate mean pressure
    vol = assemble_scalar(1*ufl.dx(out_mesh))
    print("Mean pressure: ", 1/vol*assemble_scalar(ph*ufl.dx(out_mesh)))

# Close output files
velocity_output.close()
pressure_output.close()