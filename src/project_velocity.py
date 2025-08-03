import ufl
import argparse
import numpy         as np
import dolfinx       as dfx
import adios4dolfinx as a4d

from mpi4py            import MPI
from basix.ufl         import element
from dolfinx.fem.petsc import LinearProblem
from utilities.parsers import CustomParser

def project_velocity_field(N: int,
                           T: float,
                           mesh_prefix: str,
                           solver_type: str,
                           model_variation: str,
                           element_family: str,
                           steady: bool):
    # Velocity data
    velocity_input_filename = \
        f"../output/{mesh_prefix}-mesh/flow/{solver_type}/checkpoints/BDM_{model_variation}_velocity"
    mesh = a4d.read_mesh(filename=velocity_input_filename,
                        comm=MPI.COMM_WORLD,
                        engine="BP4",
                        ghost_mode=dfx.mesh.GhostMode.shared_facet)

    timestamps = np.linspace(0, T, N+1)
    cg1_el = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.topology.dim,))

    if element_family=="BDM":
        # Define the input velocity function in a BDM1 space
        bdm1_el = element("BDM", mesh.basix_cell(), 1)
        BDM = dfx.fem.functionspace(mesh=mesh, element=bdm1_el)
        u_project = dfx.fem.Function(BDM)
    elif element_family=="DG":
        # Define the input velocity function in a DG1 space
        dg1_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
        DG = dfx.fem.functionspace(mesh=mesh, element=dg1_el)
        u_project = dfx.fem.Function(DG)
    else:
        # Define the input velocity function in a CG2 space
        cg2_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.topology.dim,))
        CG2 = dfx.fem.functionspace(mesh=mesh, element=cg2_el)
        u_project = dfx.fem.Function(CG2)
    u_project.name = "relative_velocity" # Needs to have the same name as the checkpoint function

    # Define the output velocity function space: CG1
    CG = dfx.fem.functionspace(mesh=mesh, element=cg1_el)
    u, v = ufl.TrialFunction(CG), ufl.TestFunction(CG)

    # Projection as a variational problem
    a = ufl.inner(u, v) * ufl.dx
    L = ufl.inner(u_project, v) * ufl.dx

    # Problem structure
    problem = LinearProblem(a, L, petsc_options={"ksp_type" : "preonly",
                                                "pc_type" : "lu",
                                                "pc_factor_mat_solver_type" : "mumps"})

    output_filename = velocity_input_filename + "_projection"
    a4d.write_mesh(output_filename, mesh)

    if steady:
        a4d.read_function(velocity_input_filename, u_project)
        u_cg = problem.solve()
        a4d.write_function(output_filename, u_cg)
    else:
        for int_time, time in enumerate(timestamps):
            a4d.read_function(velocity_input_filename, u_project, time=time) # Read the input velocity
            u_cg = problem.solve() # Solve for the CG1 velocity
            a4d.write_function(output_filename, u_cg, time=int(int_time+1))

def main(argv=None):

    parser = argparse.ArgumentParser(formatter_class=CustomParser)

    opts = parser.add_argument_group("Options", "Options for velocity data to be projected. ")
    opts.add_argument("-s", "--steady_state", type=int, help="Steady state velocity field or not")
    opts.add_argument("-T", "--final_time", type=float, help="Final time of simulation")
    opts.add_argument("-dt", "--timestep", type=float, help="Timestep size")
    opts.add_argument("-e", "--element_family", type=str, default="BDM", help="Finite element family")
    opts.add_argument("-g", "--governing_equations", type=str, help="Governing equations (Stokes or Navier-Stokes)")
    opts.add_argument("-m", "--mesh_prefix", type=str, help="Mesh prefix")
    opts.add_argument("-v", "--model_variation", type=str, help="Model variation, which flow mechanisms are considered")

    args = parser.parse_args(argv)
    if args.mesh_prefix not in ["coarse", "medium", "fine"]:
        raise ValueError("Unknown mesh prefix, 'coarse', 'medium', or 'fine'.")
    if args.governing_equations not in ["stokes", "navier-stokes"]:
        raise ValueError("Unknown governing equations, choose 'stokes' or 'navier-stokes'.")
    if args.model_variation not in ["deformation+cilia+production", "deformation+cilia", "deformation+production"]:
        raise ValueError("Unknown model variation.")
    
    steady = True if args.steady_state==1 else False
    T = args.final_time
    dt = args.timestep
    N = int(T / dt)
    print("Number of timestamps: ", N)
    project_velocity_field(N,
                           T,
                           args.mesh_prefix,
                           args.governing_equations,
                           args.model_variation,
                           args.element_family,
                           steady
                        )

if __name__=='__main__':
    main()