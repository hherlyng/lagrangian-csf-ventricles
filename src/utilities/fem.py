import ufl
import dolfinx as dfx

from ufl import inner, dot, sym, grad, avg

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
def stabilization(u: ufl.TrialFunction, v: ufl.TestFunction, mu: dfx.fem.Constant, penalty: dfx.fem.Constant, consistent: bool=True):
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