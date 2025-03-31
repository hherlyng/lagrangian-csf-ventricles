import ufl
import time

import numpy   as np
import dolfinx as dfx
import adios4dolfinx as a4d

from ufl       import inner, div, dot
from scifem    import assemble_scalar
from mpi4py    import MPI
from petsc4py  import PETSc
from basix.ufl import element
from utilities.fem import tangent, eps, assemble_nested_system

print = PETSc.Sys.Print

# Facet tags
ANTERIOR_PRESSURE    = 2 # The pressure BC facets on the anterior ventricle boundary
POSTERIOR_PRESSURE   = 3 # The pressure BC facets on the posterior ventricle boundary
MIDDLE_VENTRAL_CILIA = 5 # The cilia BC facets on the ventral wall of the middle ventricle
MIDDLE_DORSAL_CILIA  = 6 # The cilia BC facets on the dorsal wall of the middle ventricle
ANTERIOR_CILIA1      = 7 # The cilia BC facets on the dorsal, anterior walls of the anterior ventricle
ANTERIOR_CILIA2      = 8 # The cilia BC facets on the dorsal, posterior walls of the anterior ventricle
SLIP                 = 9 # The free-slip facets of the boundary

def setup_stokes_problem(mesh: dfx.mesh.Mesh, ft: dfx.mesh.MeshTags):
    facet_dim = mesh.topology.dim-1 # Facet dimension
    mesh.topology.create_connectivity(facet_dim, facet_dim+1) # Create facet-cell connectivity

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    dx = ufl.Measure("dx", domain=mesh)

    n = ufl.FacetNormal(mesh)

    cg2_vec_el = element("Lagrange", mesh.basix_cell(), degree=2, shape=(mesh.geometry.dim,)) # Second-order Lagrange vector elements
    cg2_el = element("Lagrange", mesh.basix_cell(), degree=2) # Second-order Lagrange elements
    cg1_el  = element("Lagrange", mesh.basix_cell(), degree=1) # First-order Lagrange elements
    V = dfx.fem.functionspace(mesh, cg2_vec_el) # Velocity function space
    Q = dfx.fem.functionspace(mesh, cg1_el) # Pressure function space
    Z = dfx.fem.functionspace(mesh, cg2_el) # Lagrange multiplier function space

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    zeta, eta = ufl.TrialFunction(Z), ufl.TestFunction(Z)

    mu = dfx.fem.Constant(mesh, dfx.default_scalar_type(7e-4)) # Dynamic viscosity [kg/(m*s) = g/(mm*s)]

    #---------------------------------------------------------------#
    # Define the stress vector used in the slip boundary conditions #
    #---------------------------------------------------------------#
    tau = 6.5e-4
    tau_vec   = tau*ufl.as_vector((1, 0, 1)) # Stress vector to be projected tangentially onto the mesh

    # Define coordinates used in the tau expressions
    xx, _, _ = ufl.SpatialCoordinate(mesh)
    x0_dorsal = 0.175
    xe_dorsal = 0.335

    x0_ventral = 0.155
    xe_ventral = 0.310

    # Define the tau expressions
    tau_vec_dorsal = 2.75*tau_vec * (xx - x0_dorsal) / (xe_dorsal - x0_dorsal) # Dorsal cilia lambda expression
    tau_vec_ventral = 0.5*tau_vec * (1 - (xx - x0_ventral) / (xe_ventral - x0_ventral)) # Ventral cilia lambda expression
    tau_vec_anterior1 = 0.4*tau_vec # Anterior cilia lambda expression (anterior part)
    tau_vec_anterior2 = tau_vec # Anterior cilia lambda expression (posterior part)

    # Use the tau expressions to define the tangent traction vectors
    tangent_traction_dorsal    = tangent(tau_vec_dorsal, n)
    tangent_traction_ventral   = tangent(tau_vec_ventral, n)
    tangent_traction_anterior1 = tangent(tau_vec_anterior1, n)
    tangent_traction_anterior2 = tangent(tau_vec_anterior2, n)
    
    # Stokes equations bilinear form in block form
    a00 = 2*mu * inner(eps(u), eps(v)) * dx # Diffusive term
    a01 = - p * div(v) * dx       # Pressure term
    a02 = - zeta * dot(v, n) * ds   # Multiplier trial function term

    a10 = - q * div(u) * dx # Continuity equation
    
    a20 = - eta * dot(u, n) * ds # Multiplier test function term

    # Linear form
    L0 = (inner(tangent_traction_ventral, tangent(v, n))*ds(MIDDLE_VENTRAL_CILIA)
        - inner(tangent_traction_anterior2, tangent(v, n))*ds(ANTERIOR_CILIA2)
        - inner(tangent_traction_anterior1, tangent(v, n))*ds(ANTERIOR_CILIA1)
        - inner(tangent_traction_dorsal, tangent(v, n))*ds(MIDDLE_DORSAL_CILIA)
    )
         
    L1 = inner(dfx.fem.Function(Q), q)*dx # Zero RHS for the pressure
    L2 = inner(dfx.fem.Function(Z), eta) * ds # Zero RHS for multiplier test eq.

    a = dfx.fem.form([[a00, a01, a02],
                      [a10, None, None],
                      [a20, None, None]])
    L = dfx.fem.form([L0, L1, L2])

    bcs = []

    return a, L, bcs, V, Q, Z, ds

def create_direct_solver(A: PETSc.Mat, comm: MPI.Comm):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
    pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
    return ksp

def solve_stokes(a: dfx.fem.form, L: dfx.fem.form, bcs: list[dfx.fem.DirichletBC],
                 uh: dfx.fem.Function, ph: dfx.fem.Function, zh: dfx.fem.Function):

    A, b = assemble_nested_system(a, L, bcs)

    ksp = create_direct_solver(A, comm)

    w = PETSc.Vec().createNest([uh.x.petsc_vec, ph.x.petsc_vec, zh.x.petsc_vec])
    ksp.solve(b, w)
    assert ksp.getConvergedReason() > 0

    # MPI communcation
    uh.x.scatter_forward()
    ph.x.scatter_forward()

    return uh, ph

if __name__=="__main__":
    from sys import argv
    write_cpoint = True if int(argv[1])==1 else False

    # Read mesh
    comm = MPI.COMM_WORLD

    # Read mesh and facet tags from file
    with dfx.io.XDMFFile(comm,
         f"../geometries/zfish/original_ventricles.xdmf", "r") as xdmf:
         mesh = xdmf.read_mesh() # Read mesh
         mesh.topology.create_entities(mesh.topology.dim-1) # Create facets
         ft = xdmf.read_meshtags(mesh, name="ft") # Read facet tags

    # Setup the Sokes problem
    a, L, bcs, V, Q, Z, ds = setup_stokes_problem(mesh, ft)
    
    # Solution functions
    uh = dfx.fem.Function(V)
    ph = dfx.fem.Function(Q)
    zh = dfx.fem.Function(Z)

    print("Number of dofs Stokes eqs: ")
    total_dofs = (V.dofmap.index_map.size_global*V.dofmap.index_map_bs
                 +Q.dofmap.index_map.size_global
                 +Z.dofmap.index_map.size_global)
    print(f"Total:\t\t{total_dofs}")
    print(f"Velocity:\t{V.dofmap.index_map.size_global*V.dofmap.index_map_bs}")
    print(f"Pressure:\t{Q.dofmap.index_map.size_global}")
    print(f"Multiplier:\t{Z.dofmap.index_map.size_global}")

    # I/O function: Stokes velocity in DG1
    dg1_vec_el = element("DG", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    uh_out = dfx.fem.Function(dfx.fem.functionspace(mesh, dg1_vec_el))
    uh_out.name = "uh" 

    velocity_output_filename = f"../output/zfish-mesh/flow/TH_velocity.bp"
    velocity_output = dfx.io.VTXWriter(comm, velocity_output_filename, [uh_out], "BP4")
    pressure_output_filename = f"../output/zfish-mesh/flow/TH_pressure.bp"
    pressure_output = dfx.io.VTXWriter(comm, pressure_output_filename, [ph], "BP4")

    if write_cpoint:
        cpoint_filename = f"../output/zfish-mesh/flow/checkpoints/TH_velocity"
        a4d.write_mesh(cpoint_filename, mesh, store_partition_info=True)

    # Timestamp
    tic = time.perf_counter()

    # Solve the steady state Stokes equations
    uh_, ph_ = solve_stokes(a, L, bcs, uh, ph, zh) 
    
    print(f"Solution time elapsed: {time.perf_counter()-tic:.4f} sec")
    
    # Interpolate velocity into DG1 output function
    uh_out.interpolate(uh_)
    ph.interpolate(ph_)

    # Write output and close files
    velocity_output.write(0)
    pressure_output.write(0)
    velocity_output.close()
    pressure_output.close()

    # Write checkpoint
    if write_cpoint: a4d.write_function(cpoint_filename, uh_)

    # Calculate mean pressure
    vol = assemble_scalar(1*ufl.dx(mesh))
    print("Mean pressure: ", 1/vol*assemble_scalar(ph_*ufl.dx(mesh)))

    # Calculate divergence
    div_u = assemble_scalar(inner(div(uh_out), div(uh_out))*ufl.dx(mesh))
    print("L2 norm of divergence: ", div_u)

    uh_x = uh_out.sub(0).collapse().x.array
    uh_y = uh_out.sub(1).collapse().x.array
    uh_z = uh_out.sub(2).collapse().x.array
    uh_mag = np.sqrt(uh_x**2 + uh_y**2 + uh_z**2)
    print("Max velocity magnitude : ", comm.allreduce(uh_mag.max(), op=MPI.MAX))