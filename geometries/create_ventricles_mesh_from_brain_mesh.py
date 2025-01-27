from mpi4py import MPI
import numpy   as np
import dolfinx as dfx

def create_submesh_from_mesh(mesh_filename: str, facet_filename: str, labels_list: list):

    with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_filename, "r") as xdmf:
        mesh = xdmf.read_mesh()
        tdim = mesh.topology.dim
        fdim = mesh.topology.dim-1
        mesh.topology.create_entities(fdim)
        mesh.topology.create_connectivity(tdim, fdim)
        ct = xdmf.read_meshtags(mesh, "mesh")
    
    # Read the facet tags
    with dfx.io.XDMFFile(MPI.COMM_WORLD, facet_filename, "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, "mesh")

    cells = []
    # Get cell indices within the label range
    for cell_index in range(mesh.topology.index_map(tdim).size_global):
        cell_label = ct.values[cell_index]
        if cell_label in labels_list:
            cells.append(cell_index)
    
    # Create submesh
    submesh, entity_map, _, _ = dfx.mesh.create_submesh(mesh, tdim, np.array(cells, dtype=np.int32))

    # Transfer cell tags
    sub_cmap = submesh.topology.index_map(tdim)
    num_sub_cells = sub_cmap.size_local + sub_cmap.size_global
    sub_cells = np.arange(num_sub_cells, dtype=np.int32)
    sub_ct = np.empty(num_sub_cells, dtype=np.int32)
    for child, parent in zip(sub_cells, entity_map):
        sub_ct[child] = ct.values[parent]
    
    sub_ct = dfx.mesh.meshtags(submesh, tdim, sub_cells, sub_ct)
    sub_ct.name = "ct"

    # Transfer facet tags
    submesh.topology.create_entities(fdim)
    submesh.topology.create_connectivity(tdim, fdim)
    submesh.topology.create_connectivity(fdim, tdim)
    subf_map = submesh.topology.index_map(fdim)
    submesh.topology.create_connectivity(tdim, fdim)
    c_to_f = mesh.topology.connectivity(tdim, fdim)
    c_to_f_sub = submesh.topology.connectivity(tdim, fdim)
    num_sub_facets = subf_map.size_local + subf_map.size_global
    sub_facets = np.arange(num_sub_facets, dtype=np.int32)
    sub_ft = np.empty(num_sub_facets, dtype=np.int32)

    for i, entity in enumerate(entity_map):
        parent_facets = c_to_f.links(entity)
        child_facets = c_to_f_sub.links(i)
        for child, parent in zip(child_facets, parent_facets):
            sub_ft[child] = ft.values[parent]

    sub_ft = dfx.mesh.meshtags(submesh, fdim, sub_facets, sub_ft)
    sub_ft.name = "ft"

    return submesh, sub_ft, sub_ct

if __name__=='__main__':
    # Define mesh filenames and the cell tags for the submesh to be extracted
    mesh_file  = "./brain-meshes/midBrainMesh_labels.xdmf"
    facet_file = "./brain-meshes/midBrainMesh_label_boundaries.xdmf"
    output_filename = "medium_ventricles_mesh.xdmf"
    cell_labels = range(3, 9)  # Extract cells with labels between 3 and 8

    # Create the submesh
    new_mesh, new_ft, new_ct = create_submesh_from_mesh(mesh_filename=mesh_file, facet_filename=facet_file, labels_list=cell_labels)
    

    print("Mesh created, writing to file.")
    with dfx.io.XDMFFile(new_mesh.comm, output_filename, "w") as xdmf:
        xdmf.write_mesh(new_mesh)
        xdmf.write_meshtags(new_ft, new_mesh.geometry)
        xdmf.write_meshtags(new_ct, new_mesh.geometry)
    print(f"Success writing the mesh to the file {output_filename}.")