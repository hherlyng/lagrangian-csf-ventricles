from mpi4py import MPI
from basix.ufl import element
import numpy as np
import dolfinx as dfx
import adios4dolfinx as a4d

# The MPI communicator
comm = MPI.COMM_WORLD

# The checkpoint filename to read data from
checkpoint_filename = "/path/to/checkpoint_file.bp"  # <-- UPDATE THIS PATH TO YOUR CHECKPOINT FILE

# Read the mesh and define the times
mesh = a4d.read_mesh(checkpoint_filename, comm)
times = np.arange(1001)

# Create a finite element function in a CG function space
W = dfx.fem.functionspace(
            mesh,
            element(
                "Lagrange",
                mesh.basix_cell(),
                degree=4,
                shape=(mesh.geometry.dim,)
            )
        )
wh = dfx.fem.Function(W)

# Extract dof coordinates
dof_coordinates = W.tabulate_dof_coordinates()

# Track global extrema
global_max_val = -np.inf
global_min_val = np.inf
global_max_loc = None
global_min_loc = None
global_max_time = None
global_min_time = None

# Loop over all times and keep tabs on the global extrema
for t in times:
    print(f"Time t = {t}")
    a4d.read_function(checkpoint_filename, wh, time=t, name="defo_displacement")
    
    # Extract nodal displacement values (flattened array)
    u = wh.x.array.reshape((-1, mesh.geometry.dim))

    # Compute magnitude at each DOF
    mag = np.linalg.norm(u, axis=1)

    # Local extrema
    local_max_idx = np.argmax(mag)
    local_min_idx = np.argmin(mag)

    local_max_val = mag[local_max_idx]
    local_min_val = mag[local_min_idx]

    # Update global extrema
    if local_max_val > global_max_val:
        global_max_val = local_max_val
        global_max_loc = dof_coordinates[local_max_idx]
        global_max_time = t

    if local_min_val < global_min_val:
        global_min_val = local_min_val
        global_min_loc = dof_coordinates[local_min_idx]
        global_min_time = t

# Print results
print("=== Global displacement extrema over all time steps ===")
print(f"Max displacement: {global_max_val}")
print(f"  at location: {global_max_loc}")
print(f"  at time: {global_max_time}")

print(f"Min displacement: {global_min_val}")
print(f"  at location: {global_min_loc}")
print(f"  at time: {global_min_time}")
