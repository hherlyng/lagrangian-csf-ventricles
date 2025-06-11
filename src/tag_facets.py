from mpi4py import MPI
import numpy   as np
import pandas  as pd
import dolfinx as dfx

mesh_version = "fine"
mesh_filename = f"../geometries/{mesh_version}_ventricles_mesh"
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
        110] # Corpus callosum boundary
csv_filenames = [f"../geometries/selected_facets_lateralChP_{mesh_version}.csv",
                 f"../geometries/selected_facets_thirdChP_{mesh_version}.csv",
                 f"../geometries/selected_facets_fourthChP_{mesh_version}.csv",
                 f"../geometries/selected_facets_lateralApertures_{mesh_version}.csv",
                 f"../geometries/selected_facets_corpus_callosum_{mesh_version}.csv"]
for i, csv_filename in enumerate(csv_filenames):
    df = pd.read_csv(csv_filename)
    selected_facets = df['vtkOriginalCellIds'].values
    
    facets_to_mark = []
    [facets_to_mark.append(facet) for facet in selected_facets if facet in boundary_facets]
    if i==1:
        facets_rm = dfx.mesh.locate_entities_boundary(mesh, fdim, lambda x: x[1]>0.01119)
        facets_to_mark = [facet for facet in facets_to_mark if facet not in facets_rm]
        # Fix for fine mesh
        if mesh_version=="fine":
            if 303358 in facets_rm: facets_to_mark.append(303358)
            if 300832 in facets_rm: facets_to_mark.append(300832)

    facets_to_mark = np.array(facets_to_mark, dtype=np.int32)

    # Loop over the facets and mark them with the cilia tag
    tag = tags[i]
    for facet_to_mark in facets_to_mark:
        new_ft_values[facet_to_mark] = tag

new_ft = dfx.mesh.meshtags(mesh, fdim, new_facets, new_ft_values)
new_ft.name = "ft"

# Write the mesh with new facet tags to file
with dfx.io.XDMFFile(MPI.COMM_WORLD, f"../geometries/{mesh_filename}_tagged.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(new_ft, mesh.geometry)
    xdmf.write_meshtags(ct, mesh.geometry)