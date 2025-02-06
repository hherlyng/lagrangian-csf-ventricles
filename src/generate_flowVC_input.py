import struct
import numpy         as np
import dolfinx       as dfx
import adios4dolfinx as a4d

from mpi4py          import MPI
from basix.ufl       import element

#--------------------------------------------------------------------#
#  NOTE: this script only works when run in serial, not in parallell.
#--------------------------------------------------------------------#

def generate_data(input_filename: str, output_dir: str, write_xdmf_check: bool=True):
    # Read mesh
    mesh = a4d.read_mesh(filename=input_filename,
                        comm=MPI.COMM_WORLD,
                        engine="BP4",
                        ghost_mode=dfx.mesh.GhostMode.shared_facet)

    # Entity dimensions of cells, facets and vertices for the 3D mesh
    cdim = mesh.topology.dim
    fdim = cdim-1
    vdim = cdim-3 

    # Create entities and connectivities of mesh
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(cdim, fdim) # Cell-to-facet
    mesh.topology.create_connectivity(cdim, vdim) # Cell-to-vertex
    mesh.topology.create_connectivity(fdim, cdim) # Facet-to-cell
    mesh.topology.create_connectivity(fdim, vdim) # Facet-to-vertex
    c_to_f = mesh.topology.connectivity(cdim, fdim) # Cell-to-facet
    c_to_v = mesh.topology.connectivity(cdim, vdim) # Cell-to-vertex
    f_to_c = mesh.topology.connectivity(fdim, cdim) # Facet-to-cell
    f_to_v = mesh.topology.connectivity(fdim, vdim) # Facet-to-vertex
    bdry_facets = dfx.mesh.exterior_facet_indices(mesh.topology) # Facets on the bounary of the mesh

    num_cells = mesh.topology.index_map(cdim).size_global
    cells     = np.arange(num_cells,  dtype=np.int32)

    # Create adjacency list
    cell_neighbors = {}
    for cell in cells:
        cell_neighbors[cell] = np.array([-1]*4, dtype=np.int32)
        nodes = c_to_v.links(cell)
        face1 = [nodes[0], nodes[2], nodes[3]]
        face2 = [nodes[0], nodes[1], nodes[3]]
        face3 = [nodes[0], nodes[1], nodes[2]]
        face4 = [nodes[1], nodes[2], nodes[3]]
        for facet in c_to_f.links(cell):
            if facet in bdry_facets:
                element_index = -1 # Boundary facing facet
            else:
                facet_cells = f_to_c.links(facet)
                for facet_cell in facet_cells:
                    if (cell!=facet_cell):
                        element_index = facet_cell
            facet_nodes = f_to_v.links(facet)
            if len(np.intersect1d(facet_nodes, face1))==len(facet_nodes):
                idx = 0
            elif len(np.intersect1d(facet_nodes, face2))==len(facet_nodes):
                idx = 1
            elif len(np.intersect1d(facet_nodes, face3))==len(facet_nodes):
                idx = 2
            elif len(np.intersect1d(facet_nodes, face4))==len(facet_nodes):
                idx = 3
            else:
                print("Cell number ", cell)
                print("Facet number: ", facet)
                raise RuntimeError("Mismatch between cell facet nodes and searched facet nodes.")
            cell_neighbors[cell][idx] = element_index

    # Create adjacency array by recasting the cell neighbors dictionary into a one-dimensional array 
    adjacency = np.array(list(cell_neighbors.values()), dtype=np.int32).reshape((1, num_cells*(cdim+1)))[0]
    adjacency = np.insert(adjacency, 0, num_cells, axis=0) # Prepend the number of elements

    # Get the connectivity array which is stored in the dofmap of the dolfinx.mesh geometry
    connectivity = mesh.geometry.dofmap.reshape((1, num_cells*(cdim+1)))[0]
    connectivity = np.insert(connectivity, 0, num_cells, axis=0) # Prepend the number of elements

    # Get the coordinates of the nodes which is stored in the dolfinx.mesh geometry object
    num_nodes   = mesh.geometry.x.shape[0] # Get the total number of nodes
    coordinates = np.array(mesh.geometry.x.reshape((1, num_nodes*cdim))[0], dtype="double") 

    # Write the arrays to binary files
    with open(output_dir+"brain_adjacency.bin", "w+b")    as file_1, \
         open(output_dir+"brain_connectivity.bin", "w+b") as file_2, \
         open(output_dir+"brain_coordinates.bin", "w+b")  as file_3:
        adjacency.tofile(file_1)
        connectivity.tofile(file_2)
        file_3.write(struct.pack("i", num_nodes)) # Write integer num_nodes
        coordinates.tofile(file_3) # Write array of coordinates cast as double-precision floats

    #--------------------- Velocity I/O -----------------------#
    # Define finite element function for the input velocity
    cg1_el = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.topology.dim,))
    V = dfx.fem.functionspace(mesh=mesh, element=cg1_el)
    u = dfx.fem.Function(V)

    if write_xdmf_check:
        xdmf_u = dfx.io.XDMFFile(MPI.COMM_WORLD, output_dir+"bin_velocity_check.xdmf", "w")
        xdmf_u.write_mesh(mesh)

    timestamps = np.arange(0, 30)

    # Loop over the timestamps
    for timestamp in timestamps:
        # Read the velocity data
        a4d.read_function(filename=input_filename, u=u, time=timestamp)

        # Cast velocity as numpy.array and prepend timestamp
        velocity = u.x.array
        velocity = np.array(np.insert(velocity, 0, timestamp, axis=0), dtype="double")
        if write_xdmf_check: xdmf_u.write_function(u, timestamp) # Write to xdmf
        
        # Write the velocity data to binary file

        with open(output_dir+f"brain_vel.{timestamp}.bin", "w+b") as file:
            velocity.tofile(file) # Write to binary file

    if write_xdmf_check: xdmf_u.close()

if __name__=='__main__':
    mesh_prefix = "medium"  
    velocity_input_filename = \
        f"../output/checkpoints/projections/deforming-mesh-{mesh_prefix}/BDM_chp+cilia+defo_velocity"
    output_dir = f"/Users/hherlyng/flowVC/bin/deforming-brain-mesh-{mesh_prefix}/BDM_chp+cilia+defo/"
    generate_data(input_filename=velocity_input_filename, output_dir=output_dir)