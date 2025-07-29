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