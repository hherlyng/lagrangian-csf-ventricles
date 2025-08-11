from mpi4py import MPI
import dolfinx 
import numpy as np
import ufl

import sys

N = int(sys.argv[1])
L = 1.0
H = 1.2
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0,0], [L, H]], [N, N], dolfinx.cpp.mesh.CellType.quadrilateral)

# evaluate an expression at a point

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, )))
u = dolfinx.fem.Function(V)

def f(x):
    return 1/2*x[0]**2 - 2 *x[1]**2, -3/2*x[0]**2 + 1/2*x[1]**2

u.interpolate(f)

from utilities.fem import create_normal_contribution_bc
# prod_func = create_normal_contribution_bc(V, (-Q/(area*area_ref))*n_hat + dot(u_mesh, n_hat)*n_hat, ft.find(RIGHT)) #(-tot_prod + wn_term)/nn_term*n_hat
# normal_bc.interpolate(prod_func)
# Facet expression evaluations

def vertex_to_dof_map_vectorized(V):
    mesh = V.mesh
    num_vertices_per_cell = dolfinx.cpp.mesh.cell_num_entities(
        mesh.topology.cell_type, 0
    )

    dof_layout2 = np.empty((num_vertices_per_cell,), dtype=np.int32)
    for i in range(num_vertices_per_cell):
        var = V.dofmap.dof_layout.entity_dofs(0, i)
        assert len(var) == 1
        dof_layout2[i] = var[0]

    num_vertices = (
        mesh.topology.index_map(0).size_local + mesh.topology.index_map(0).num_ghosts
    )

    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    assert (
        c_to_v.offsets[1:] - c_to_v.offsets[:-1] == c_to_v.offsets[1]
    ).all(), "Single cell type supported"

    vertex_to_dof_map = np.empty(num_vertices, dtype=np.int32)
    vertex_to_dof_map[c_to_v.array] = V.dofmap.list[:, dof_layout2].reshape(-1)
    return vertex_to_dof_map

    


import scifem
import basix.ufl
from dolfinx.fem import Expression, IntegralType, functionspace, Function, compute_integration_domains
def move_to_facet_quadrature(ufl_expr, mesh, sub_facets, scheme="default", degree=6):
    fdim = mesh.topology.dim - 1
    # Create submesh
    bndry_mesh, entity_map, _, _ = dolfinx.mesh.create_submesh(mesh, fdim, sub_facets)
    # Create quadrature space on submesh
    q_el = basix.ufl.quadrature_element(bndry_mesh.basix_cell(), ufl_expr.ufl_shape , scheme, degree)
    Q = functionspace(bndry_mesh, q_el)

    # Compute where to evaluate expression per submesh cell
    integration_entities = compute_integration_domains(IntegralType.exterior_facet, mesh.topology, entity_map, fdim)
    compiled_expr = Expression(ufl_expr, Q.element.interpolation_points())

    # Evaluate expression
    q = Function(Q)
    q.x.array[:] = compiled_expr.eval(mesh, integration_entities).reshape(-1)
    return q

g_h = lambda x: (2*x[0]*(1-x[1]), np.zeros_like(x[0]))

delta_t = 0.1
h_h = g_h / delta_t

u = Function(V, name="Velocity")
u.interpolate(g_h)
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
n = ufl.FacetNormal(mesh)
chi = ufl.SpatialCoordinate(mesh) + u
F = ufl.grad(chi)
J = ufl.det(F)
N = J*ufl.inv(F.T)*n
facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, lambda x: np.isclose(x[0], L))
right_integration_entities = compute_integration_domains(IntegralType.exterior_facet, mesh.topology, facets, 1)

ds_right = ufl.ds(subdomain_data=[(1, right_integration_entities.flatten())], subdomain_id=1)
area = scifem.assemble_scalar(ufl.dot(N, N)*ds_right)
area_ref = scifem.assemble_scalar(dolfinx.fem.Constant(mesh, 1.0)*ds_right)
print(f"{area=}\t{area_ref=}")
flux_value = 0.5
flux = flux_value*N/(area)


flux_integral = scifem.assemble_scalar(ufl.dot(flux, N)*ds_right)
print(f"{flux_integral=}\t{flux_value=}")

q = move_to_facet_quadrature(flux, mesh, facets)
mesh.geometry.x[:,:mesh.geometry.dim] += u.x.array.reshape(-1, mesh.geometry.dim)
q.name = "Q"

q_mesh = q.function_space.mesh
Q  = dolfinx.fem.functionspace(q_mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
q_u = Function(Q)
q_u.interpolate(g_h)
v_to_d = vertex_to_dof_map_vectorized(Q)
num_vertices = q_mesh.topology.index_map(0).size_local

q_mesh.topology.create_connectivity(0, 1)
geometry_indices = dolfinx.mesh.entities_to_geometry(
    q_mesh, 0, np.arange(num_vertices, dtype=np.int32), False)
x = q_mesh.geometry.x
bs = Q.dofmap.bs
for vertex, geom_index in enumerate(geometry_indices):
    dof = v_to_d[vertex]
    for b in range(bs):
        x[geom_index, b] += q_u.x.array[dof*bs+b]


scifem.xdmf.create_pointcloud("flux.xdmf", [q])
with dolfinx.io.VTXWriter(mesh.comm, "flux.bp", [u]) as bp:
    bp.write(0.0)