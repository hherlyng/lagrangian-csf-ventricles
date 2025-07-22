from mpi4py import MPI
import numpy   as np
import pandas  as pd
import dolfinx as dfx

mesh_filename = f"../../geometries/ventricles_0"
with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_filename+".xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(ghost_mode=dfx.mesh.GhostMode.none)
    tdim = mesh.topology.dim
    fdim = mesh.topology.dim-1
    ct = xdmf.read_meshtags(mesh, "ct")
    mesh.topology.create_entities(fdim)
    ft = xdmf.read_meshtags(mesh, "ft")

# Copy old facet tags
num_facets = mesh.topology.index_map(fdim).size_local + mesh.topology.index_map(fdim).num_ghosts
new_facets = np.arange(num_facets, dtype=np.int32)
new_ft_values = np.copy(ft.values)
mesh.topology.create_connectivity(fdim, tdim)
mesh.topology.create_connectivity(tdim, fdim)
c_to_f = mesh.topology.connectivity(tdim, fdim)
boundary_facets = dfx.mesh.exterior_facet_indices(mesh.topology)

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
ZERO_SOLID_TRACTION = 1000
CHOROID_PLEXUS_LATERAL_ZERO_SOLID_TRACTION = 1001
third_tags = (14,
              111,
              112,
              115,
              116,
              117,
              1000
              )
no_cilia_tag = 119
z_0 = 0.0009093324794586006
z_1 = 0.01194469730250278
y_0 = 0.02399382568705812
y_1 = -0.02174830491322183
line = lambda y: z_0 + (z_1-z_0)/(y_1-y_0)*(y - y_0)
no_cilia_facets = dfx.mesh.locate_entities(mesh, fdim, lambda x: x[2] < line(x[1]))
third_facets = ft.find(third_tags[0])
for tag in third_tags[1:]:
    third_facets = np.concatenate((third_facets, ft.find(tag)))
facets_to_mark = []
[facets_to_mark.append(facet) for facet in third_facets if facet in no_cilia_facets]

facets_to_mark = np.array(facets_to_mark, dtype=np.int32)

for facet_to_mark in facets_to_mark:
    new_ft_values[facet_to_mark] = no_cilia_tag

new_ft = dfx.mesh.meshtags(mesh, fdim, new_facets, new_ft_values)
new_ft.name = "ft"

# Write the mesh with new facet tags to file
with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_filename+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(new_ft, mesh.geometry)
    xdmf.write_meshtags(ct, mesh.geometry)