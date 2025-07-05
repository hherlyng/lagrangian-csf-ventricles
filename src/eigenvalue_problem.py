import ufl
import sys

import numpy   as np
import dolfinx as dfx

from ufl       import inner, grad, sym
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from dolfinx.fem.petsc    import create_matrix, create_vector
from utilities.deformation_data import (DisplacementCorpusCallosumCephalocaudal, 
                                        DisplacementCaudateNucleusHeadLateral,
                                        DisplacementCanalAndFourthVentricleAnteroposterior,
                                        DisplacementThirdVentricleLateral)

print = PETSc.Sys.Print

# Mesh tags
CANAL_WALL = 13
FOURTH_VENTRICLE_WALL = 18
CANAL_OUT  = 23
LATERAL_APERTURES = 28
CORPUS_CALLOSUM = 110
THIRD_RIGHT = 111
THIRD_LEFT = 112
LATERAL_RIGHT = 113
LATERAL_LEFT = 114
THIRD_ANTERIOR = 115
THIRD_POSTERIOR = 116

# Solve linear elasticity equation on the ventricles. Wall motion is 
# prescribed in time at a single point (close to corpus callosum).
write_output = int(sys.argv[1])
comm = MPI.COMM_WORLD
mesh_suffix = '0'
with dfx.io.XDMFFile(comm, f"../geometries/ventricles_{mesh_suffix}.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    gdim = mesh.geometry.dim
    
    # Generate mesh entities    
    facet_dim = mesh.topology.dim-1
    mesh.topology.create_entities(facet_dim) # Create facets
    mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity
    ft = xdmf.read_meshtags(mesh, "ft")
    ct = xdmf.read_meshtags(mesh, "ct")

ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft) # Boundary integral measure
dx = ufl.Measure('dx', domain=mesh, subdomain_data=ct) # Volume integral measure
eps = lambda arg: sym(grad(arg)) # The symmetric gradient

# Material parameters
E = float(sys.argv[3]) #3156 # Modulus of elasticity [Pa]
nu = 0.479 # Poisson's ratio [-]
eta_value = 2*E / (1+nu) # First Lamé parameter value
lam_value = nu*E / ((1+nu) * (1-2*nu)) # Second Lamé parameter value
eta = dfx.fem.Constant(mesh, eta_value) # First Lamé parameter
lam = dfx.fem.Constant(mesh, lam_value) # Second Lamé parameter
rho = dfx.fem.Constant(mesh, 1000.0) # Ventricular wall density [kg/m^3]
print("Value of Lamé parameters:")
print(f"eta \t= {eta_value:.2f}\nlambda \t= {lam_value:.2f}")

# Define Generalized-alpha method parameters
gamma = dfx.fem.Constant(mesh, dfx.default_scalar_type(1/2))
beta  = dfx.fem.Constant(mesh, dfx.default_scalar_type(1/4*(gamma.value + 1/2)**2))

# Temporal parameters
timestep = 0.001
dt = dfx.fem.Constant(mesh, timestep) 
period = 1
num_periods = int(sys.argv[4])
T = period*num_periods
N = int(T / timestep)
times = np.linspace(0, T, N+1)

# Finite elements
p = int(sys.argv[2])
vec_el = element("Lagrange", mesh.basix_cell(), p, shape=(mesh.geometry.dim,))
W = dfx.fem.functionspace(mesh, vec_el)
W_x = W.sub(0) # x displacement space
W_y = W.sub(1) # y displacement space
W_z = W.sub(2) # z displacement space
wh = dfx.fem.Function(W) # Displacement function
zero = dfx.fem.Function(W)

print(f"\nNumber of degrees of freedom: {W.dofmap.index_map.size_global*W.dofmap.index_map_bs}")

# Test and trial functions
w, dw = ufl.TrialFunction(W), ufl.TestFunction(W)

# Stress tensor
sigma = lambda w: 2.0*eta*eps(w) + lam*ufl.tr(eps(w))*ufl.Identity(mesh.geometry.dim)

# Dirichlet BCs on corpus callosum and canal wall
cc_disp_expr = DisplacementCorpusCallosumCephalocaudal(period=period, timestep=timestep, final_time=T)
cwfv_disp_expr = DisplacementCanalAndFourthVentricleAnteroposterior(period=period, timestep=timestep, final_time=T)
tv_disp_expr = DisplacementThirdVentricleLateral(period=period, timestep=timestep, final_time=T)
lv_disp_expr = DisplacementCaudateNucleusHeadLateral(period=period, timestep=timestep, final_time=T)
cc_disp_func = dfx.fem.Function(W)
cwfv_disp_func = dfx.fem.Function(W)
tv_disp_func_right = dfx.fem.Function(W)
tv_disp_func_left  = dfx.fem.Function(W)
lv_disp_func_right = dfx.fem.Function(W)
lv_disp_func_left  = dfx.fem.Function(W)
bcs = []

# Set BCs
cc_dofs = dfx.fem.locate_dofs_topological((W_z, W), facet_dim, ft.find(CORPUS_CALLOSUM))
bcs.append(dfx.fem.dirichletbc(cc_disp_func, cc_dofs, W_z))

lv_dofs_right = dfx.fem.locate_dofs_topological((W_x, W), facet_dim, ft.find(LATERAL_RIGHT))
bcs.append(dfx.fem.dirichletbc(lv_disp_func_right, lv_dofs_right, W_x))

lv_dofs_left  = dfx.fem.locate_dofs_topological((W_x, W), facet_dim, ft.find(LATERAL_LEFT))
bcs.append(dfx.fem.dirichletbc(lv_disp_func_left, lv_dofs_left, W_x))

tv_dofs_right = dfx.fem.locate_dofs_topological((W_x, W), facet_dim, ft.find(THIRD_RIGHT))
bcs.append(dfx.fem.dirichletbc(tv_disp_func_right, tv_dofs_right, W_x))

tv_dofs_left  = dfx.fem.locate_dofs_topological((W_x, W), facet_dim, ft.find(THIRD_LEFT))
bcs.append(dfx.fem.dirichletbc(tv_disp_func_left, tv_dofs_left, W_x))

# Anchor perimeter of outlet
# Get the facets of the outlet and the canal wall
outlet_facets = ft.indices[ft.values==CANAL_OUT]
wall_facets = ft.indices[ft.values==CANAL_WALL]

# Get the connectivity from facets (dim-1) to vertices (dim=0)
mesh.topology.create_connectivity(facet_dim, 0)
f_to_v = mesh.topology.connectivity(facet_dim, 0)

# Parallell communication
if len(outlet_facets) > 0:
    local_outlet_vertices = np.unique(np.concatenate([f_to_v.links(f) for f in outlet_facets]))
else:
    local_outlet_vertices = np.array([], dtype=np.int32)

all_outlet_vertices_list = comm.allgather(local_outlet_vertices) # Gather all local vertex arrays from all processes
global_outlet_vertices = np.unique(np.concatenate(all_outlet_vertices_list)) # Create a single global array of unique vertices

if len(wall_facets) > 0:
    local_wall_vertices = np.unique(np.concatenate([f_to_v.links(f) for f in wall_facets]))
else:
    local_wall_vertices = np.array([], dtype=np.int32)

# The perimeter vertices are the intersection of the two sets of vertices
# Gather and create global array to compute intersection globally
all_wall_vertices_list = comm.allgather(local_wall_vertices)
global_wall_vertices = np.unique(np.concatenate(all_wall_vertices_list))
outlet_perimeter_vertices = np.intersect1d(global_outlet_vertices, global_wall_vertices)

# Find which outlet_perimeter_vertices are owned locally on the process
owned_vertex_range = mesh.topology.index_map(0).local_range # Get the start and end+1 of the owned vertex range on this process

# Create a boolean mask to filter the vertices
is_owned = (outlet_perimeter_vertices >= owned_vertex_range[0]) & \
           (outlet_perimeter_vertices < owned_vertex_range[1])

# Apply the mask to get only the vertices owned by this process
local_perimeter_vertices = outlet_perimeter_vertices[is_owned]

# Now find the outlet perimeter dofs
mesh.topology.create_connectivity(0, mesh.topology.dim) # Create connectivity from vertices (dim=0) to cells (dim)
outlet_perimeter_dofs_x = dfx.fem.locate_dofs_topological((W_x, W), 0, local_perimeter_vertices)
outlet_perimeter_dofs_y = dfx.fem.locate_dofs_topological((W_y, W), 0, local_perimeter_vertices)

bcs.append(dfx.fem.dirichletbc(zero, outlet_perimeter_dofs_x, W_x))
bcs.append(dfx.fem.dirichletbc(zero, outlet_perimeter_dofs_y, W_y))

##--------- THE EIGENVALUE PROBLEM ---------## 
import slepc4py
from slepc4py import SLEPc
slepc4py.init(sys.argv)
PETSc.Options().setValue("-eps_view", None)
PETSc.Options().setValue("-eps_monitor", None)

m = rho * inner(w, dw) * dx # Mass bilinear form
a = inner(sigma(w), eps(dw)) * dx # Stiffness bilinear form

# Apply boundary conditions to the stiffness matrix assembly
K = dfx.fem.petsc.assemble_matrix(dfx.fem.form(a), bcs=bcs)
K.assemble()

# The mass matrix is not affected by Dirichlet BCs
M = dfx.fem.petsc.assemble_matrix(dfx.fem.form(m))
M.assemble()
print("Assembled")

# Set up and solve the eigenvalue problem with SLEPc
sl_eps = SLEPc.EPS().create(MPI.COMM_WORLD)
sl_eps.setOperators(K, M)
sl_eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
sl_eps.setDimensions(nev=20) # Number of eigenvalues to compute

# --- Explicitly configure the Shift-and-Invert solver ---
st = sl_eps.getST() # Get the Spectral Transform object
st.setType(SLEPc.ST.Type.SINVERT) # Set the transform type to Shift-and-Invert
st.setShift(1.0) # Set the shift value directly, the target: lambda = omega**2
ksp = st.getKSP() # Get the KSP object used by the ST
ksp.setType(PETSc.KSP.Type.PREONLY)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.LU)
pc.setFactorSolverType("mumps")
sl_eps.setFromOptions()

# Solve
sl_eps.solve()

# Extract and print the results
nconv = sl_eps.getConverged()
print(f"Number of converged eigenvalues: {nconv}")

if nconv > 0:
    # Create a dolfinx function to store the eigenvector (mode shape)
    uh = dfx.fem.Function(W, name="Mode Shape")
    # Create vectors for the real and imaginary parts of the eigenvector
    vr, vi = K.getVecs()

    print("\n   Frequency (Hz)")
    print("--------------------")
    for i in range(nconv):
        lambda_i = sl_eps.getEigenpair(i, vr, vi).real
        # Convert eigenvalue ω² to frequency in Hz
        freq_hz = np.sqrt(lambda_i) / (2 * np.pi)
        
        # Check for non-physical "zero-energy" modes close to 0 Hz
        if freq_hz > 1e-2:
            print(f"{i+1:2d}: {freq_hz:8.4f}")

if write_output:
    # Save the first physical mode shape to a file for visualization
    if nconv > 0:
        with dfx.io.VTXWriter(MPI.COMM_WORLD, "mode_shape.bp", [uh], "BP5") as vtx:
            # Find first non-zero frequency mode
            for i in range(nconv):
                lambda_i = sl_eps.getEigenpair(i, vr, vi).real
                if np.sqrt(lambda_i) / (2 * np.pi) > 1e-2:
                    # Copy the eigenvector from the PETSc vector to the dolfinx function
                    uh.x.petsc_vec.setArray(vr.getArray())
                    uh.x.scatter_forward()
                    vtx.write(0.0)
                    print(f"\nSaved mode shape for frequency {np.sqrt(lambda_i)/(2*np.pi):.4f} Hz to mode_shape.vtx")
                    break