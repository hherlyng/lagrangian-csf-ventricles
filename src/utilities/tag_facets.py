from mpi4py import MPI
import numpy   as np
import pandas  as pd
import dolfinx as dfx

mesh_version = "medium"
# mesh_filename = f"../geometries/{mesh_version}_ventricles_mesh"
mesh_filename = f"../../geometries/ventricles_0_tagged"
cell_labels = range(3, 9)  # Extract cells with labels between 3 and 8
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

tags = [101, # Lateral ventricles choroid plexus
        103, # Third ventricle choroid plexus
        104, # Fourth ventricle choroid plexus
        28, # Lateral apertures boundary
        110, # Corpus callosum boundary
        112, # Third ventricle left boundary
        111, # Third ventricle right boundary
        114, # Lateral ventricle left boundary
        113,  # Lateral ventricle right boundary
        115, # Third ventricle anterior boundary
        116, # Third ventricle posterior boundary
        117, # Third ventricle floor boundary
        118, # Lateral ventricle floor boundary
        1000, # Zero traction
        ] 

input_dir = "../../geometries/selected_facets/"
# csv_filenames = [f"selected_facets_lateralChP_{mesh_version}.csv",
#                  f"selected_facets_thirdChP_{mesh_version}.csv",
#                  f"selected_facets_fourthChP_{mesh_version}.csv",
#                  f"selected_facets_lateralApertures_{mesh_version}.csv",
#                  f"selected_facets_laterals_corpus_callosum_{mesh_version}.csv",
#                  f"selected_facets_3V_left_{mesh_version}.csv",
#                  f"selected_facets_3V_right_{mesh_version}.csv",
#                  f"selected_facets_LV_left_{mesh_version}.csv",
#                  f"selected_facets_LV_right_{mesh_version}.csv",
#                  f"selected_facets_3V_anterior_{mesh_version}.csv",
#                  f"selected_facets_3V_posterior_{mesh_version}.csv",
#                  f"selected_facets_3V_floor_{mesh_version}.csv",
#                 ]
csv_filenames = ["corpus_callosum.csv",
                 "3V_floor.csv",
                 "zero_traction_3V.csv",
                 "LV_left.csv",
                 "LV_floor.csv",
                 "zero_traction_LV_1.csv",
                 "zero_traction_LV_2.csv",
]

tags = [110, # Corpus callosum boundary
        117, # Third ventricle floor boundary
        1000, # Zero traction
        114, # Lateral ventricle left boundary
        118, # Lateral ventricle floor boundary
        1000, # Zero traction
        1000, # Zero traction
]
for i, csv_filename in enumerate(csv_filenames):
    df = pd.read_csv(input_dir+csv_filename)
    selected_facets = df['vtkOriginalCellIds'].values
    
    facets_to_mark = []
    [facets_to_mark.append(facet) for facet in selected_facets if facet in boundary_facets]

    facets_to_mark = np.array(facets_to_mark, dtype=np.int32)

    # Loop over the facets and mark them with the cilia tag
    tag = tags[i]

    for facet_to_mark in facets_to_mark:
        new_ft_values[facet_to_mark] = tag

# Find some LV and ChP facets to retag
LATERAL_VENTRICLES_WALL = 17
CHOROID_PLEXUS_LATERAL  = 101
LV_facets = ft.find(LATERAL_VENTRICLES_WALL)
ChP_facets = ft.find(CHOROID_PLEXUS_LATERAL)
mesh.topology.create_connectivity(fdim, 0)
f_to_v = mesh.topology.connectivity(fdim, 0)
for i, facets in enumerate([LV_facets, ChP_facets]):
    for facet in facets:
        for vertex in f_to_v.links(facet):
            v = int(vertex)
            cond = (np.logical_and(
                    np.logical_and(
                    mesh.geometry.x[v, 0] > -0.0239759,
                    mesh.geometry.x[v, 0] < 0.019865),
                mesh.geometry.x[v, 1] > -0.00675))
            if i==1: cond = np.invert(cond)
            if cond:
                if not new_ft_values[facet]==118 and not new_ft_values[facet]==114:
                    new_ft_values[facet] = 1000 if i==0 else 1001 # Zero traction tag

new_ft = dfx.mesh.meshtags(mesh, fdim, new_facets, new_ft_values)
new_ft.name = "ft"

# Write the mesh with new facet tags to file
with dfx.io.XDMFFile(MPI.COMM_WORLD, f"{mesh_filename}_new.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(new_ft, mesh.geometry)
    xdmf.write_meshtags(ct, mesh.geometry)