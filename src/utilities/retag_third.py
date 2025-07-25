import numpy   as np
import dolfinx as dfx

def create_cilia_meshtags(mesh: dfx.mesh.Mesh, old_ft: dfx.mesh.MeshTags) -> dfx.mesh.MeshTags:

    tdim = mesh.topology.dim
    fdim = tdim-1

    # Copy old facet tags
    num_facets = mesh.topology.index_map(fdim).size_local + mesh.topology.index_map(fdim).num_ghosts
    new_facets = np.arange(num_facets, dtype=np.int32)
    new_ft_values = np.copy(old_ft.values)
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(tdim, fdim)

    third_tags = (14,
                111,
                112,
                115,
                116,
                117,
                1000
                )
    no_cilia_tag = 119

    # Define a line that separates the caudal 1/3 of the third ventricle
    z_0 = 0.0009093324794586006
    z_1 = 0.01194469730250278
    y_0 = 0.02399382568705812
    y_1 = -0.02174830491322183
    line = lambda y: z_0 + (z_1-z_0)/(y_1-y_0)*(y - y_0)
    no_cilia_facets = dfx.mesh.locate_entities(mesh, fdim, lambda x: x[2] < line(x[1]))
    third_facets = old_ft.find(third_tags[0])
    for tag in third_tags[1:]:
        third_facets = np.concatenate((third_facets, old_ft.find(tag)))
    facets_to_mark = []
    [facets_to_mark.append(facet) for facet in third_facets if facet in no_cilia_facets]

    facets_to_mark = np.array(facets_to_mark, dtype=np.int32)

    for facet_to_mark in facets_to_mark:
        new_ft_values[facet_to_mark] = no_cilia_tag

    new_ft = dfx.mesh.meshtags(mesh, fdim, new_facets, new_ft_values)
    new_ft.name = "ft"

    return new_ft