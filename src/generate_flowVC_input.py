import struct
import argparse
import numpy         as np
import dolfinx       as dfx
import adios4dolfinx as a4d

from mpi4py            import MPI
from basix.ufl         import element
from utilities.parsers import CustomParser

#--------------------------------------------------------------------#
#  NOTE: this script only works when run in serial, not in parallell.
#--------------------------------------------------------------------#

def generate_data(input_filename: str,
                  output_dir: str,
                  output_prefix: str,
                  steady_state: bool,
                  N: int,
                  write_xdmf_check: bool=True):
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
    with open(output_dir+f"{output_prefix}_adjacency.bin", "w+b")    as file_1, \
         open(output_dir+f"{output_prefix}_connectivity.bin", "w+b") as file_2, \
         open(output_dir+f"{output_prefix}_coordinates.bin", "w+b")  as file_3:
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
        xdmf_u = dfx.io.XDMFFile(MPI.COMM_WORLD, output_dir+f"bin_velocity.xdmf", "w")
        xdmf_u.write_mesh(mesh)

    if steady_state:
        timestamp = 0
        # Read the velocity data
        a4d.read_function(filename=input_filename, u=u, time=timestamp)

        # Cast velocity as numpy.array and prepend timestamp
        velocity = u.x.array
        velocity0 = np.array(np.insert(velocity, 0, timestamp, axis=0), dtype="double")
        velocity1 = np.array(np.insert(velocity, 0, timestamp+1, axis=0), dtype="double")
        if write_xdmf_check: xdmf_u.write_function(u, timestamp) # Write to xdmf
        
        # Write the velocity data to binary files
        with open(output_dir+f"{output_prefix}_vel.{timestamp}.bin", "w+b")   as file0, \
             open(output_dir+f"{output_prefix}_vel.{timestamp+1}.bin", "w+b") as file1:
            velocity0.tofile(file0)
            velocity1.tofile(file1)

    else:
        timestamps = np.arange(1, N+1)

        # Loop over the timestamps
        for timestamp in timestamps:
            # Read the velocity data
            a4d.read_function(filename=input_filename, u=u, time=timestamp)

            # Cast velocity as numpy.array and prepend timestamp
            velocity = u.x.array
            velocity = np.array(np.insert(velocity, 0, timestamp, axis=0), dtype="double")
            if write_xdmf_check: xdmf_u.write_function(u, timestamp) # Write to xdmf
            
            # Write the velocity data to binary file
            with open(output_dir+f"{output_prefix}_vel.{timestamp}.bin", "w+b") as file:
                velocity.tofile(file) # Write to binary file

    if write_xdmf_check: xdmf_u.close()

def main(argv=None):

    parser = argparse.ArgumentParser(formatter_class=CustomParser)

    opts = parser.add_argument_group("Options", "Options for velocity data to be generated. ")
    opts.add_argument("-s", "--steady_state", type=int, help="Steady state velocity field or not")
    opts.add_argument("-T", "--final_time", type=float, help="Final time of simulation")
    opts.add_argument("-k", "--element_degree", type=int, help="Velocity finite element degree")
    opts.add_argument("-p", "--polynomial_degree", type=int, help="Displacement finite element degree")
    opts.add_argument("-dt", "--timestep", type=float, help="Timestep size")
    opts.add_argument("-e", "--element_family", type=str, default="BDM", help="Finite element family")
    opts.add_argument("-g", "--governing_equations", type=str, help="Governing equations (Stokes or Navier-Stokes)")
    opts.add_argument("-o", "--output_prefix", type=str, default="brain", help="Prefix for output binaries")
    opts.add_argument("-m", "--mesh_suffix", type=int, help="Mesh suffix, specifies mesh version")
    opts.add_argument("-v", "--model_variation", type=int, help="Model variation, which flow mechanisms are considered")

    args = parser.parse_args(argv)
    if args.mesh_suffix not in range(3):
        raise ValueError("Unknown mesh prefix, choose 0, 1, or 2.")
    if args.governing_equations not in ["stokes", "navier-stokes"]:
        raise ValueError("Unknown governing equations, choose 'stokes' or 'navier-stokes'.")
    if args.model_variation not in range(5):
        raise ValueError("Unknown model variation.")

    # Set model variation name
    models = {1 : "deformation",
              2 : "deformation+cilia",
              3 : "deformation+production",
              4 : "deformation+cilia+production"
    }
    model_variation = models[args.model_variation]
    
    # Set steady state flag
    steady = True if args.steady_state==1 else False
    
    # Set temporal parameters
    T = args.final_time
    dt = args.timestep
    N = int(1 / dt)+1
    print("Number of timestamps: ", N)

    # Define output directory
    mesh_prefixes = {0 : "medium",
                     1 : "fine",
                     2 : "very_fine"
    }
    bin_dir = f"{args.output_prefix}/{mesh_prefixes[args.mesh_suffix]}-mesh"
    
    flowVC_path = "path/to/flowVC"  # <-- CHANGE THIS TO THE ACTUAL PATH TO flowVC
    output_dir = f"{flowVC_path}/bin/{bin_dir}/{args.governing_equations}/" \
        + f"p={args.polynomial_degree}_k={args.element_degree}/{model_variation}/"

    velocity_input_filename = \
        f"../output/mesh_{args.mesh_suffix}/" \
       +f"flow_p={args.polynomial_degree}_E=1500_k={args.element_degree}_dt={args.timestep}_T={T:.0f}/" \
       +f"{args.governing_equations}/checkpoints/{args.element_family}_{model_variation}_velocity_projection"
    
    # Generate binary data
    generate_data(input_filename=velocity_input_filename,
                  output_dir=output_dir,
                  output_prefix=args.output_prefix,
                  N=N,
                  steady_state=steady)

if __name__=='__main__':
    main()