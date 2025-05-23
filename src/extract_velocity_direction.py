import ufl

import numpy as np
import dolfinx as dfx
import adios4dolfinx as a4d

from mpi4py import MPI
from basix.ufl import element

comm = MPI.COMM_WORLD
mesh_prefix = "medium"
cpoint_input_filename = f"../output/{mesh_prefix}-mesh/flow/navier-stokes/checkpoints/production_velocity/"

# Read mesh and BDM velocity function
mesh = a4d.read_mesh(cpoint_input_filename, comm)
n = ufl.FacetNormal(mesh)
bdm_el = element("BDM", cell=mesh.basix_cell(), degree=1)
u_bdm = dfx.fem.Function(dfx.fem.functionspace(mesh, bdm_el))
a4d.read_function(cpoint_input_filename, u_bdm, name="velocity")

# Reshape vector and normalize
u_reshaped = u_bdm.x.array.reshape((int(u_bdm.x.array.__len__()/mesh.geometry.dim), mesh.geometry.dim))
u_reshaped_norm = np.sqrt(u_reshaped[:, 0]**2 + u_reshaped[:, 1]**2 + u_reshaped[:, 2]**2)
where_norm_is_zero = np.where(u_reshaped_norm < 1e-10)[0]
u_reshaped_norm[where_norm_is_zero] = 1.0
for i in range(mesh.geometry.dim): u_reshaped[:, i] /= u_reshaped_norm
u_bdm.x.array[:] = u_reshaped.ravel()

check_results = True
if check_results:
    with dfx.io.VTXWriter(comm, "../output/cilia-direction-vectors/u_dg_normalized.bp", [u_bdm], "BP4") as vtx:
        vtx.write(t=0)  

# Write to checkpoint file
cpoint_output_filename = f"../output/checkpoints/cilia-direction-vectors/{mesh_prefix}-mesh/"
a4d.write_function_on_input_mesh(filename=cpoint_output_filename, u=u_bdm)