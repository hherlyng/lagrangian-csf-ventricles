import ufl
import argparse
import numpy         as np
import dolfinx       as dfx
import adios4dolfinx as a4d

from mpi4py            import MPI
from basix.ufl         import element
from dolfinx.fem.petsc import LinearProblem
from utilities.parsers import CustomParser

def project_velocity_field(k: int,
                           element_family: str):
    # Velocity data
    velocity_input_filename = \
        "/global/D1/homes/hherlyng/lagrangian-csf-ventricles/output/" \
     + f"zfish-mesh/flow/checkpoints/{element_family}_velocity"
    mesh = a4d.read_mesh(filename=velocity_input_filename,
                        comm=MPI.COMM_WORLD,
                        engine="BP4",
                        ghost_mode=dfx.mesh.GhostMode.shared_facet)
    cg1_el = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.topology.dim,))
    cg2_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.topology.dim,))
    CG  = dfx.fem.functionspace(mesh=mesh, element=cg1_el)
    CG2 = dfx.fem.functionspace(mesh=mesh, element=cg2_el)
    projected_space = CG2
    u_cg1 = dfx.fem.Function(CG)

    if element_family=="BDM":
        # Define the input velocity function in a BDM1 space
        bdm1_el = element("BDM", mesh.basix_cell(), k)
        BDM = dfx.fem.functionspace(mesh=mesh, element=bdm1_el)
        u_project = dfx.fem.Function(BDM)
    elif element_family=="DG":
        # Define the input velocity function in a DG1 space
        dg1_el = element("DG", mesh.basix_cell(), k, shape=(mesh.geometry.dim,))
        DG = dfx.fem.functionspace(mesh=mesh, element=dg1_el)
        u_project = dfx.fem.Function(DG)

    # Trial and test functions
    u_projected = dfx.fem.Function(projected_space)
    u, v = ufl.TrialFunction(projected_space), ufl.TestFunction(projected_space)

    # Projection as a variational problem
    a = ufl.inner(u, v) * ufl.dx
    L = ufl.inner(u_project, v) * ufl.dx

    # Problem structure
    problem = LinearProblem(a, L, u=u_projected, petsc_options={"ksp_type" : "preonly",
                                                                "pc_type" : "lu",
                                                                "pc_factor_mat_solver_type" : "mumps"})
 
    output_filename = velocity_input_filename + "_projection"
    a4d.write_mesh(output_filename, mesh)

    # Prepare function evaluation
    bb_tree = dfx.geometry.bb_tree(mesh, mesh.topology.dim)
    cell_candidates = dfx.geometry.compute_collisions_points(bb_tree, mesh.geometry.x)
    colliding_cells = dfx.geometry.compute_colliding_cells(mesh, cell_candidates, mesh.geometry.x)
    eval_cells = []
    for i in range(mesh.geometry.x.shape[0]):
        if len(colliding_cells.links(i))>0:
            eval_cells.append(colliding_cells.links(i)[0])
    eval_cells = np.array(eval_cells, dtype=np.int32)

    a4d.read_function(velocity_input_filename, u_project)
    problem.solve()
    u_cg1.x.array[:] = u_projected.eval(mesh.geometry.x, eval_cells).flatten()
    a4d.write_function(output_filename, u_cg1)

def main(argv=None):

    parser = argparse.ArgumentParser(formatter_class=CustomParser)

    opts = parser.add_argument_group("Options", "Options for velocity data to be projected. ")
    opts.add_argument("-k", "--element_degree", type=int, help="Velocity finite element degree")
    opts.add_argument("-e", "--element_family", type=str, default="BDM", help="Finite element family")
    args = parser.parse_args(argv)
    
    project_velocity_field(
                        args.element_degree,
                        args.element_family,
                        )

if __name__=='__main__':
    main()
