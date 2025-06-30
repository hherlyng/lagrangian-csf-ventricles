import ufl
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

if __name__=='__main__':
     # Test the projection problem

     comm = MPI.COMM_WORLD
     vh_cpoint_filename = f"../output/checkpoints/deforming-mesh-coarse/deformation_velocity/"
     mesh = a4d.read_mesh(filename=vh_cpoint_filename, comm=comm)
     cg2_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
     vh = dfx.fem.Function(dfx.fem.functionspace(mesh, cg2_el))
     a4d.read_function(vh_cpoint_filename, vh, time=0.20)

     bdm_el = element("BDM", mesh.basix_cell(), 1)
     V = dfx.fem.functionspace(mesh, bdm_el)
     uh = dfx.fem.Function(V)

     problem = projection_problem_CG2_to_BDM1(vh, uh)
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