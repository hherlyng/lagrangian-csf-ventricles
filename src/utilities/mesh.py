from mpi4py import MPI
import numpy   as np
import dolfinx as dfx
from dolfinx.cpp.mesh import CellType

def create_unit_square_mesh(N: int,
                            comm: MPI.Comm=MPI.COMM_WORLD,
                            cell_type: CellType=CellType.triangle,
                            ghost_mode=dfx.mesh.GhostMode.shared_facet) \
                         -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags)):
        """ Create a unit square mesh with N x N cells, with boundary facet tags:
                Left   = 1 \n
                Right  = 2 \n
                Bottom = 3 \n
                Top    = 4

        Parameters
        ----------
        N : int
            Mesh cells in x and y directions (total # cells will be N x N).
        
        comm:  MPI.Comm
            MPI communicator, by default MPI.COMM_WORLD.
        
        ghost_mode
            Mode for handling ghosting of mesh cells and nodes, by default dfx.mesh.GhostMode.shared_facet.

        diagonal
            Direction of the diagonal of the triangles, by default from left to right.

        Returns
        -------
        mesh : dfx.mesh.Mesh
            The mesh.
            
        ft   : dfx.mesh.Meshtags
            The mesh facet tags.
        """
        mesh = dfx.mesh.create_unit_square(
                                comm,
                                N, N,
                                cell_type=cell_type,
                                ghost_mode=ghost_mode
                                )
        def left(x): return np.isclose(x[0], 0.0)
        def right(x): return np.isclose(x[0], 1.0)
        def bottom(x): return np.isclose(x[1], 0.0)
        def top(x): return np.isclose(x[1], 1.0)
        LEFT=1; RIGHT=2; BOT=3; TOP=4

        # Facet tags
        bc_facet_indices, bc_facet_markers = [], []
        fdim = mesh.topology.dim - 1

        inlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, left)
        bc_facet_indices.append(inlet_BC_facets)
        bc_facet_markers.append(np.full_like(inlet_BC_facets, LEFT))

        outlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, right)
        bc_facet_indices.append(outlet_BC_facets)
        bc_facet_markers.append(np.full_like(outlet_BC_facets, RIGHT))

        bottom_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, bottom)
        bc_facet_indices.append(bottom_BC_facets)
        bc_facet_markers.append(np.full_like(bottom_BC_facets, BOT))

        top_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, top)
        bc_facet_indices.append(top_BC_facets)
        bc_facet_markers.append(np.full_like(top_BC_facets, TOP))

        bc_facet_indices = np.hstack(bc_facet_indices).astype(np.int32)
        bc_facet_markers = np.hstack(bc_facet_markers).astype(np.int32)

        sorted_facets = np.argsort(bc_facet_indices)

        facet_tags = dfx.mesh.meshtags(mesh, fdim, bc_facet_indices[sorted_facets], bc_facet_markers[sorted_facets])

        return mesh, facet_tags

def create_rectangle_mesh(N: int,
                       lower_left: list[float],
                       upper_right: list[float],
                       comm: MPI.Comm=MPI.COMM_WORLD,
                       cell_type: CellType=CellType.triangle,
                       ghost_mode=dfx.mesh.GhostMode.shared_facet) \
                    -> tuple((dfx.mesh.Mesh, dfx.mesh.MeshTags)):
        """ Create a unit square mesh with N x N cells, with boundary facet tags:
                Left   = 1 \n
                Right  = 2 \n
                Bottom = 3 \n
                Top    = 4

        Parameters
        ----------
        N : int
            Mesh cells in x and y directions (total # cells will be N x N).
        
        lower_left : list[float]
            [x, y] coordinates of lower-left corner of the rectangle.
        
        upper_right : list[float]
            [x, y] coordinates of upper-right corner of the rectangle.

        comm:  MPI.Comm
            MPI communicator, by default MPI.COMM_WORLD.
        
        ghost_mode
            Mode for handling ghosting of mesh cells and nodes, by default dfx.mesh.GhostMode.shared_facet.

        diagonal
            Direction of the diagonal of the triangles, by default from left to right.

        Returns
        -------
        mesh : dfx.mesh.Mesh
            The mesh.
            
        ft   : dfx.mesh.Meshtags
            The mesh facet tags.
        """
        mesh = dfx.mesh.create_rectangle(
                                comm,
                                points=[lower_left,
                                        upper_right],
                                n=[N, N],
                                cell_type=cell_type,
                                ghost_mode=ghost_mode
                                )
        def left(x): return np.isclose(x[0], lower_left[0])
        def right(x): return np.isclose(x[0], upper_right[0])
        def bottom(x): return np.isclose(x[1], lower_left[1])
        def top(x): return np.isclose(x[1], upper_right[1])
        LEFT=1; RIGHT=2; BOT=3; TOP=4

        # Facet tags
        bc_facet_indices, bc_facet_markers = [], []
        fdim = mesh.topology.dim - 1

        inlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, left)
        bc_facet_indices.append(inlet_BC_facets)
        bc_facet_markers.append(np.full_like(inlet_BC_facets, LEFT))

        outlet_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, right)
        bc_facet_indices.append(outlet_BC_facets)
        bc_facet_markers.append(np.full_like(outlet_BC_facets, RIGHT))

        bottom_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, bottom)
        bc_facet_indices.append(bottom_BC_facets)
        bc_facet_markers.append(np.full_like(bottom_BC_facets, BOT))

        top_BC_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, top)
        bc_facet_indices.append(top_BC_facets)
        bc_facet_markers.append(np.full_like(top_BC_facets, TOP))

        bc_facet_indices = np.hstack(bc_facet_indices).astype(np.int32)
        bc_facet_markers = np.hstack(bc_facet_markers).astype(np.int32)

        sorted_facets = np.argsort(bc_facet_indices)

        facet_tags = dfx.mesh.meshtags(mesh, fdim, bc_facet_indices[sorted_facets], bc_facet_markers[sorted_facets])

        return mesh, facet_tags