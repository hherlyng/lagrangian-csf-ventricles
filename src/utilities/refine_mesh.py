from mpi4py import MPI
import sys
import numpy as np
import dolfinx as dfx

input_mesh_suffix = int(sys.argv[1])
output_mesh_suffix = input_mesh_suffix+1

with dfx.io.XDMFFile(MPI.COMM_WORLD, f"../../geometries/ventricles_{input_mesh_suffix}.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    tdim = mesh.topology.dim
    fdim = mesh.topology.dim-1
    mesh.topology.create_entities(tdim-2)
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, tdim)
    ft = xdmf.read_meshtags(mesh, "ft")
    ct = xdmf.read_meshtags(mesh, "ct")

mesh_refined, parent_cells, parent_facets = dfx.mesh.refine(mesh, option=dfx.mesh.RefinementOption.parent_cell_and_facet)
mesh_refined.topology.create_connectivity(fdim, tdim)

child_vertices = dfx.mesh.entities_to_geometry(mesh_refined, tdim, np.arange(len(parent_cells), dtype=np.int32))
parent_vertices = dfx.mesh.entities_to_geometry(mesh, tdim, parent_cells)


ft_refined = dfx.mesh.MeshTags(dfx.cpp.refinement.transfer_facet_meshtag(ft._cpp_object, mesh_refined._cpp_object.topology, parent_cells, parent_facets))
ft_refined.name = "ft"
ct_refined = dfx.mesh.MeshTags(dfx.cpp.refinement.transfer_cell_meshtag(ct._cpp_object, mesh_refined._cpp_object.topology, parent_cells))
ct_refined.name = "ct"

with dfx.io.XDMFFile(mesh_refined.comm, f"../../geometries/ventricles_{output_mesh_suffix}.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_refined)
    xdmf.write_meshtags(ft_refined, mesh_refined.geometry)
    xdmf.write_meshtags(ct_refined, mesh_refined.geometry)