from mpi4py import MPI

import ufl
import numpy   as np
import dolfinx as dfx

from ufl       import inner, dot, sym, grad, avg
from petsc4py  import PETSc
from basix.ufl import element
from dolfinx.fem.petsc import assemble_matrix_mat, assemble_vector, apply_lifting

# Operators
# NOTE: these are the jump operators from Krauss, Zikatonov paper.
# Jump is just a difference and it preserves the rank 
Jump = lambda arg: arg('+') - arg('-')

# Average uses dot with normal and AGAIN MINUS; it reduces the rank
Avg = lambda arg, n: .5*(dot(arg('+'), n('+')) - dot(arg('-'), n('-')))

# Symmetric gradient
def eps(u: dfx.fem.Function | ufl.Coefficient):
    """ Return the symmetric gradient of u. """
    return sym(grad(u)) 
def tangent(v: dfx.fem.Function | ufl.Coefficient, n: ufl.FacetNormal):
    """ Action of (1 - n x n) on a vector yields the tangential component.

    Parameters
    ----------
    v : dfx.fem.Function | ufl.Coefficient
        The vector to find the tangential component of.
    n : ufl.FacetNormal
        Facet normal vector of mesh.

    Returns
    -------
    Any
        The normal projection of v on n.
    """
    return v - n*dot(v, n)
def stabilization(u: ufl.TrialFunction, v: ufl.TestFunction,
                  mu: dfx.fem.Constant, penalty: dfx.fem.Constant,
                  consistent: bool=True) -> ufl.Form:
    """ Displacement/Flux Stabilization term from Krauss et al paper. 

    Parameters
    ----------
    u : ufl.TrialFunction
        The finite element trial function.
    
    v : ufl.TestFunction
        The finite element test function.
    
    mu : dfx.fem.Constant
        Dynamic viscosity.
    
    penalty : dfx.fem.Constant
        Interior stabilization penalty parameter.
    
    consistent : bool
        Add symmetric gradient terms to the form if True.

    Returns
    -------
    ufl.Coefficient
        Stabilization term for the bilinear form.
    """
    mesh = u._ufl_function_space.mesh
    n, hA = ufl.FacetNormal(mesh), avg(ufl.CellDiameter(mesh)) # Facet normal vector and average cell diameter
    dS = ufl.Measure('dS', domain=mesh) # Interior facet integral measure

    if consistent: # Add symmetrization terms
        return (-inner(Avg(2*mu*eps(u), n), Jump(tangent(v, n)))*dS
                -inner(Avg(2*mu*eps(v), n), Jump(tangent(u, n)))*dS
                + 2*mu*(penalty/hA)*inner(Jump(tangent(u, n)), Jump(tangent(v, n)))*dS)

    # For preconditioning
    return 2*mu*(penalty/hA)*inner(Jump(tangent(u, n)), Jump(tangent(v, n)))*dS

def assemble_system(A: PETSc.Mat,
                    b: PETSc.Vec,
                    lhs_form: dfx.fem.form,
                    rhs_form: dfx.fem.form,
                    bcs: list[dfx.fem.dirichletbc]):
    A.zeroEntries()
    assemble_matrix_mat(A, lhs_form, bcs=bcs)
    A.assemble()
    with b.localForm() as local_b_vec: local_b_vec.set(0.0)
    assemble_vector(b, rhs_form)
    apply_lifting(b, [lhs_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dfx.fem.set_bc(b, bcs)

    return A, b

def assemble_nested_system(lhs_form: dfx.fem.form,
                           rhs_form: dfx.fem.form,
                           bcs: list[dfx.fem.dirichletbc]) -> tuple((PETSc.Mat, PETSc.Vec)):
    A = dfx.fem.petsc.assemble_matrix_nest(lhs_form, bcs=bcs)
    A.assemble()

    b = dfx.fem.petsc.assemble_vector_nest(rhs_form)
    dfx.fem.petsc.apply_lifting_nest(b, lhs_form, bcs=bcs)
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    spaces = dfx.fem.extract_function_spaces(rhs_form)
    bcs0 = dfx.fem.bcs_by_block(spaces, bcs)
    dfx.fem.petsc.set_bc_nest(b, bcs0)

    return A, b

def compute_exterior_facet_entities(mesh: dfx.mesh.Mesh, facets: np.ndarray[np.int32]):
    """ Helper function to compute (cell, local_facet_index) pairs for exterior facets. 
        Copyright (C) 2019-2024 Michal Habera and Jørgen S. Dokken.
    """
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim, tdim - 1)
    c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    integration_entities = np.empty(2 * len(facets), dtype=np.int32)

    for i, facet in enumerate(facets):
        cells = f_to_c.links(facet)
        assert len(cells) == 1
        cell = cells[0]
        local_facets = c_to_f.links(cell)
        local_pos = np.flatnonzero(local_facets == facet)
        integration_entities[2 * i] = cell
        integration_entities[2 * i + 1] = local_pos[0]

    return integration_entities

def create_normal_contribution_bc(Q: dfx.fem.FunctionSpace, expr: ufl.core.expr.Expr, facets: np.typing.NDArray[np.int32]) -> dfx.fem.Function:
    """
    Create function representing normal flux.
    SPDX-License-Identifier:    MIT.
    Author: Jørgen S. Dokken.
    """
    domain = Q.mesh
    Q_el = Q.element

    # Compute integration entities (cell, local_facet index) for all facets
    boundary_entities = compute_exterior_facet_entities(domain, facets)
    interpolation_points = Q_el.basix_element.x
    fdim = domain.topology.dim-1

    c_el = domain.ufl_domain().ufl_coordinate_element()
    ref_top = c_el.reference_topology
    ref_geom = c_el.reference_geometry

    cell_to_facet = {"interval": "vertex",
                     "triangle": "interval", "quadrilateral": "interval",
                     "tetrahedron": "triangle", "hexahedron": "quadrilateral"}
                     
    # Pull back interpolation points from reference coordinate element to facet reference element
    facet_cmap = element(
        "Lagrange", cell_to_facet[domain.topology.cell_name()], c_el.degree, shape=(domain.geometry.dim,), dtype=np.float64)
    facet_cel = dfx.cpp.fem.CoordinateElement_float64(
        facet_cmap.basix_element._e)
    reference_facet_points = None
    
    for i, points in enumerate(interpolation_points[fdim]):
        geom = ref_geom[ref_top[fdim][i]]
        ref_points = facet_cel.pull_back(points, geom)

        # Assert that interpolation points are all equal on all facets
        if reference_facet_points is None:
            reference_facet_points = ref_points
        else:
            assert np.allclose(reference_facet_points, ref_points)

    # Create expression for BC
    normal_expr = dfx.fem.Expression(
        expr, reference_facet_points)

    points_per_entity = [sum(ip.shape[0] for ip in ips)
                         for ips in interpolation_points]
    offsets = np.zeros(domain.topology.dim+2, dtype=np.int32)
    offsets[1:] = np.cumsum(points_per_entity[:domain.topology.dim+1])
    values_per_entity = np.zeros(
        (offsets[-1], domain.geometry.dim), dtype=dfx.default_scalar_type)
    entities = boundary_entities.reshape(-1, 2)
    values = np.zeros(entities.shape[0]*offsets[-1]*domain.geometry.dim)
    for i, entity in enumerate(entities):
        insert_pos = offsets[fdim] + \
            reference_facet_points.shape[0] * entity[1]
        normal_on_facet = normal_expr.eval(domain, entity)
        values_per_entity[insert_pos: insert_pos + reference_facet_points.shape[0]
                          ] = normal_on_facet.reshape(-1, domain.geometry.dim)
        values[i*offsets[-1] *
               domain.geometry.dim: (i+1)*offsets[-1]*domain.geometry.dim] = values_per_entity.reshape(-1)
    qh = dfx.fem.Function(Q)
    qh._cpp_object.interpolate(
        values.reshape(-1, domain.geometry.dim).T.copy(), boundary_entities[::2].copy())
    qh.x.scatter_forward()
    
    return qh

def compute_cell_boundary_int_entities(mesh: dfx.mesh.Mesh):
    """Compute the integration entities for integrals around the
    boundaries of all cells in msh.

    Parameters:
        mesh: The mesh.

    Returns:
        Facets to integrate over, identified by ``(cell, local facet
        index)`` pairs.

    Copyright (C) Joe Dean 2025.
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    n_f = dfx.cpp.mesh.cell_num_entities(mesh.topology.cell_type, fdim)
    n_c = mesh.topology.index_map(tdim).size_local

    return np.vstack((np.repeat(np.arange(n_c), n_f), np.tile(np.arange(n_f), n_c))).T.flatten()

def calculate_norm_L2(comm: MPI.Comm, v: dfx.fem.Function, dX: ufl.Measure):
    """ Compute the L2-norm of v with the integration measure dX. """
    return np.sqrt(
              comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(inner(v, v) * dX)),
              op=MPI.SUM)
              )

def calculate_mean(mesh: dfx.mesh.Mesh, v: dfx.fem.Function, dX: ufl.Measure):
    """ Calculate the average of a function over the domain defined by mesh,
        using the integration measure dX. """
    vol = mesh.comm.allreduce(
        dfx.fem.assemble_scalar(dfx.fem.form(
                                dfx.fem.Constant(mesh, dfx.default_real_type(1.0)) * dX)
                                ), op=MPI.SUM
    )
    return (1/vol) * mesh.comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(v * dX)), op=MPI.SUM)