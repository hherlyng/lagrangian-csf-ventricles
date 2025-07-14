import ufl
import gmsh
import numpy as np
import dolfinx as dfx
import adios4dolfinx as a4d

from mpi4py    import MPI
from basix.ufl import element
from dolfinx.fem.petsc import LinearProblem

def projection_problem_CG_to_BDM(vh_cg: dfx.fem.Function,
                                   vh_bdm: dfx.fem.Function,
                                   dx: ufl.Measure) -> LinearProblem:
     V = vh_bdm.function_space
     eta, zeta = ufl.TrialFunction(V), ufl.TestFunction(V)
     a = ufl.inner(eta, zeta)*dx
     L = ufl.inner(vh_cg, zeta)*dx

     return LinearProblem(a, L, bcs=[], u=vh_bdm)

def mesh_rectangle_aqueduct_top(mesh0: dfx.mesh.Mesh, ghost_mode: dfx.mesh.GhostMode=dfx.mesh.GhostMode.shared_facet, num_refinements: int=3):
    """ Mesh a rectangle slice that lies in the xy-plane of at the top of the aqueduct.

    Parameters
    ----------
    mesh0 : dfx.mesh.Mesh
        The brain ventricles mesh.

    Returns
    -------
    dfx.mesh.Mesh
        Rectangle slice mesh.
    """
    comm = mesh0.comm # MPI communicator
    rank = comm.rank  # MPI process rank
    gmsh.initialize()
    
    x_min = -0.0045
    x_max = 0.0008
    x_mid = .5*(x_min+x_max)
    y_min = -0.016
    y_max = -0.010
    y_mid = .5*(y_min+y_max)
    z_min = 0.0025
    z_max = 0.0025
    z = 0.0020

    dx = x_max - x_min
    dy = y_max - y_min
        
    if rank==0:
        gmsh.model.occ.addRectangle(x=x_min, y=y_min, z=z, dx=dx, dy=dy, tag=1)
        gmsh.model.occ.rotate(dimTags=[(2, 1)], x=x_mid, y=y_mid, z=z, ax=1, ay=0, az=0, angle=0)
     #    gmsh.model.occ.rotate(dimTags=[(2, 1)], x=x_mid, y=y_mid, z=z, ax=1, ay=0, az=0, angle=-np.pi/30)
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(dim=2)
        gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], 1)
        gmsh.model.setPhysicalName(surfaces[0][0], 1, "Slice")
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(dim=2)
        gmsh.model.mesh.setOrder(1)
        gmsh.model.mesh.optimize("Netgen")
        for _ in range(num_refinements):
            gmsh.model.mesh.refine()

    partitioner = dfx.mesh.create_cell_partitioner(ghost_mode)
    rectangle_mesh = dfx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3, partitioner=partitioner)[0]

    return rectangle_mesh


# Aq bot (perpendicular to flow?)
x_min = -0.0055
x_max = 0.0015
y_min = -0.025
y_max = -0.0185
z = -0.0165

def nonmatching_mesh_interpolation(mesh0: dfx.mesh.Mesh,
                                   mesh1: dfx.mesh.Mesh, 
                                   u: dfx.fem.Function, 
                                   vector: bool,
                                   family: str):
     """ Interpolate the function u onto a rectangular slice that lies in the yz-plane.

     Parameters
     ----------
     mesh0 : dfx.mesh.Mesh
          Brain ventricles mesh.
     mesh1 : dfx.mesh.Mesh
          Rectangle mesh.
     u : dfx.fem.Function
          Finite element function to be interpolated.
     vector : bool
          Use vector finite elements if True
     family : str
          Finite element family

     Returns
     -------
     dfx.fem.Function
          Function that represents the interpolation of u onto the rectangle slice.
     """

     V0 = u.function_space

     if family=="DG":
          el1 = element("DG", mesh1.basix_cell(), 1, shape=(3,)) if vector else element("DG", mesh1.basix_cell(), 1)
     elif family=="BDM":
          el1 = element("BDM", mesh1.basix_cell(), 1)

     V1 = dfx.fem.functionspace(mesh1, el1)

     # Check that both interfaces of create nonmatching meshes interpolation data returns the same
     mesh1_cell_map = mesh1.topology.index_map(mesh1.topology.dim)
     num_cells_on_proc = mesh1_cell_map.size_local + mesh1_cell_map.num_ghosts
     cells = np.arange(num_cells_on_proc, dtype=np.int32)
     interpolation_data = dfx.fem.create_interpolation_data(V1, V0, cells, padding=1e-14)

     # Interpolate 3D->2D
     u1 = dfx.fem.Function(V1)
     u1.interpolate_nonmatching(u, cells, interpolation_data=interpolation_data)
     u1.x.scatter_forward()

     return u1

if __name__=='__main__':

     test_nr = 1

     comm = MPI.COMM_WORLD
     p = 2
     E = 1500
     T = 3
     dt = 0.001

     if test_nr==0:
          # Test the CG->BDM projection problem
          cpoint_filename = f"../../output/mesh_0/deformation_p={p}_E={E}_T={T}/checkpoints/displacement_velocity_dt={dt:.4g}/"
          mesh = a4d.read_mesh(filename=cpoint_filename, comm=comm)
          cg2_el = element("Lagrange", mesh.basix_cell(), p, shape=(mesh.geometry.dim,))
          vh = dfx.fem.Function(dfx.fem.functionspace(mesh, cg2_el))
          a4d.read_function(cpoint_filename, vh, time=0.20)

          bdm_el = element("BDM", mesh.basix_cell(), 1)
          V = dfx.fem.functionspace(mesh, bdm_el)
          uh = dfx.fem.Function(V)

          problem = projection_problem_CG_to_BDM(vh, uh)
          problem.solve()

          dg_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
          u_dg = dfx.fem.Function(dfx.fem.functionspace(mesh, dg_el))
          u_dg.interpolate(uh)

          cg_el = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
          u_cg = dfx.fem.Function(dfx.fem.functionspace(mesh, cg_el))
          u_cg.interpolate(vh)

          with dfx.io.XDMFFile(comm, "u_cg.xdmf", "w") as xdmf, \
               dfx.io.VTXWriter(comm, "u_dg.bp", [u_dg], "BP4") as vtx:
               xdmf.write_mesh(mesh)
               xdmf.write_function(u_cg)
               vtx.write(0)
     
     elif test_nr==1:
          # Test the slice projection+integration
          cpoint_filename = f"../../output/ex3/mesh_0/checkpoints/BDM_deformation_velocity"
          # cpoint_filename = f"../../output/mesh_0/deformation_p={p}_E={E}_T={T}/checkpoints/displacement_velocity_dt={dt:.4g}/"
          mesh = a4d.read_mesh(filename=cpoint_filename, comm=comm)
          aqueduct_slice = mesh_rectangle_aqueduct_top(mesh0=mesh)

          bdm_el = element("BDM", mesh.basix_cell(), 1)
          V = dfx.fem.functionspace(mesh, bdm_el)
          uh = dfx.fem.Function(V)
          a4d.read_function(filename=cpoint_filename, u=uh, time=750, name="relative_velocity")
          print(uh.x.array.max())
          dg_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
          u_dg = dfx.fem.Function(dfx.fem.functionspace(mesh, dg_el))
          u_dg.interpolate(uh)

          with dfx.io.VTXWriter(comm, "u_dg.bp", [u_dg], "BP4") as vtx:
               vtx.write(0)

          u_slice = nonmatching_mesh_interpolation(mesh0=mesh, mesh1=aqueduct_slice, u=u_dg, vector=True, family="DG")

          dS = ufl.Measure('dS', domain=aqueduct_slice)
          n = ufl.FacetNormal(aqueduct_slice)
          flux = dfx.fem.assemble_scalar(dfx.fem.form(ufl.dot(u_slice('+'), n('+'))*dS))
          
          dg_el_slice = element("DG", aqueduct_slice.basix_cell(), 1, shape=(mesh.geometry.dim,))
          u_dg_slice = dfx.fem.Function(dfx.fem.functionspace(aqueduct_slice, dg_el_slice))
          u_dg_slice.interpolate(u_slice)
          print(f'{flux:.2e}')

          def marker(x,  tol=1e-12):
               lower_bound = lambda x, i, bound: x[i] >= bound - tol
               upper_bound = lambda x, i, bound: x[i] <= bound + tol
               return (
                    lower_bound(x, 0, x_min)
                    & lower_bound(x, 1, y_min)
                    & lower_bound(x, 2, z_min)
                    & upper_bound(x, 0, x_max)
                    & upper_bound(x, 1, y_max)
                    & upper_bound(x, 2, z_max)
               )

          x_min = -0.0045
          x_max = 0.0008
          x_mid = .5*(x_min+x_max)
          y_min = -0.016
          y_max = -0.010
          y_mid = .5*(y_min+y_max)
          delta = 0.00055
          z_min = 0.0025 - delta
          z_max = 0.0025 + delta

          mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
          entities = dfx.mesh.locate_entities(mesh, mesh.topology.dim-1, marker)
          facets = np.arange(mesh.topology.index_map(mesh.topology.dim-1).size_local, dtype=np.int32)
          bdry_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
          entities = [entity for entity in entities if entity not in bdry_facets]
          facet_tags = np.full_like(facets, 1, dtype=np.int32)
          facet_tags[entities] = 10
          ft = dfx.mesh.meshtags(mesh, mesh.topology.dim-1, facets, facet_tags)
          with dfx.io.XDMFFile(comm, "mesh_with_slice_facets.xdmf", "w") as xdmf:
               xdmf.write_mesh(mesh)
               xdmf.write_meshtags(ft, mesh.geometry)

          with dfx.io.VTXWriter(comm, "u_dg_slice.bp", [u_slice], "BP4") as vtx:
               vtx.write(0)

          with dfx.io.XDMFFile(comm, "slice.xdmf", "w") as xdmf:
               xdmf.write_mesh(aqueduct_slice)