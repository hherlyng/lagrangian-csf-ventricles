import ufl
import argparse
import numpy         as np
import dolfinx       as dfx
import adios4dolfinx as a4d

from mpi4py            import MPI
from basix.ufl         import element
from dolfinx.fem.petsc import LinearProblem
from utilities.parsers import CustomParser

def project_velocity_field(T: float,
                           k: int,
                           p: int,
                           timestep: float,
                           mesh_suffix: str,
                           solver_type: str,
                           model_variation: str,
                           element_family: str,
                           steady: bool):
    # Velocity data
    velocity_input_filename = \
        f"../output/mesh_{mesh_suffix}/" \
       +f"flow_p={p}_E=1500_k={k}_dt={timestep}_T={T:.0f}/{solver_type}/checkpoints/BDM_{model_variation}_velocity"
    mesh = a4d.read_mesh(filename=velocity_input_filename,
                        comm=MPI.COMM_WORLD,
                        engine="BP4")
    num_read_times = int(1/timestep)+1
    timestamps = np.arange(num_read_times)
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
    
    u_project.name = "relative_velocity" # Needs to have the same name as the checkpoint function

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

    if steady:
        a4d.read_function(velocity_input_filename, u_project)
        problem.solve()
        u_cg1.x.array[:] = u_projected.eval(mesh.geometry.x, eval_cells).flatten()
        a4d.write_function(output_filename, u_cg1)
    else:
        for time in timestamps:
            a4d.read_function(velocity_input_filename, u_project, time=time) # Read the input velocity
            problem.solve() # Solve for the CG1 velocity
            u_cg1.x.array[:] = u_projected.eval(mesh.geometry.x, eval_cells).flatten()
            a4d.write_function(output_filename, u_cg1, time=int(time+1))

def main(argv=None):

    parser = argparse.ArgumentParser(formatter_class=CustomParser)

    opts = parser.add_argument_group("Options", "Options for velocity data to be projected. ")
    opts.add_argument("-s", "--steady_state", type=int, help="Steady state velocity field or not")
    opts.add_argument("-T", "--final_time", type=float, help="Final time of simulation")
    opts.add_argument("-k", "--element_degree", type=int, help="Velocity finite element degree")
    opts.add_argument("-p", "--polynomial_degree", type=int, help="Displacement finite element degree")
    opts.add_argument("-dt", "--timestep", type=float, help="Timestep size")
    opts.add_argument("-e", "--element_family", type=str, default="BDM", help="Finite element family")
    opts.add_argument("-g", "--governing_equations", type=str, help="Governing equations (Stokes or Navier-Stokes)")
    opts.add_argument("-m", "--mesh_suffix", type=int, help="Mesh suffix, specifies mesh version")
    opts.add_argument("-v", "--model_variation", type=int, help="Model variation, which flow mechanisms are considered")

    args = parser.parse_args(argv)
    if args.mesh_suffix not in range(3):
        raise ValueError("Unknown mesh prefix, choose 0, 1, or 2.")
    if args.governing_equations not in ["stokes", "navier-stokes"]:
        raise ValueError("Unknown governing equations, choose 'stokes' or 'navier-stokes'.")
    if args.model_variation not in range(5):
        raise ValueError("Unknown model variation.")
    
    steady = True if args.steady_state==1 else False
    print("Number of timestamps: ", int(1 / args.timestep)+1, flush=True)
    models = {1 : "deformation",
              2 : "deformation+cilia",
              3 : "deformation+production",
              4 : "deformation+cilia+production"
    }
    model_variation = models[args.model_variation]
    project_velocity_field(args.final_time,
                           args.element_degree,
                           args.polynomial_degree,
                           args.timestep,
                           args.mesh_suffix,
                           args.governing_equations,
                           model_variation,
                           args.element_family,
                           steady
                        )

if __name__=='__main__':
    main()