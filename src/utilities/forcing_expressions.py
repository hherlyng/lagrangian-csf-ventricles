import numpy as np
import numpy.typing

class PressureExpression:
    """ A class used to generate expressions for pressure as a function of time.
    """
    def __init__(self, A: float) -> None:
        self.t = 0.0 # Initial time
        self.A = A   # Amplitude

    def __call__(self, x):
        return np.ones(x.shape[1]) * (-self.A * np.sin(2 * np.pi * self.t))

class fExpressionSteady:

    def __init__(self, mu: float, dim: int = 2):
        self.mu = mu
        self.dim = dim
    
    def __call__(self, x):
        if self.dim == 2:
            lapl_term_x = -self.mu * 2*np.pi**2 * np.sin(np.pi*x[0]) * np.sin(np.pi*x[1])
            pres_term_x = 2*np.pi * np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])
            f_x         = -lapl_term_x + pres_term_x

            lapl_term_y = self.mu * 2 * np.pi**2 * np.cos(np.pi*x[0]) * np.sin(np.pi*x[1])
            pres_term_y = 2*np.pi * np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1])
            f_y         = -lapl_term_y + pres_term_y

            return np.stack((f_x, f_y))
        elif self.dim == 3:
            raise NotImplementedError("3D not implemented yet.")

class gExpressionSteady:

    def __init__(self, dim: int = 2):
        self.dim = dim

    def __call__(self, x):
        if self.dim == 2:
            return np.pi * np.cos(np.pi*x[1]) * (np.sin(np.pi*x[0]) - np.cos(np.pi*x[0]))

        elif self.dim == 3:
            raise NotImplementedError("3D not implemented yet.")

# class tauExpressionSteady:


class VelocityExpressionSteady:
    def __init__(self, dim: int = 2):
        self.dim = dim

    def __call__(self, x):
        if self.dim == 2:
            return np.stack((np.sin(np.pi*x[0]) * np.sin(np.pi*x[1]),
                            -np.cos(np.pi*x[0]) * np.sin(np.pi*x[1]),
                            ))
        else:
            raise NotImplementedError("3D not implemented yet.")

class PressureExpressionSteady:
    def __call__(self, x):
        return np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])

class ExactfExpression:
    """ A class used to generate expressions for the function that is the right-hand side
    of the Stokes equation.
    """
    def __init__(self, rho: float, nu: float, dim: int = 3):
        self.t   = 0.0 # Initial time
        self.rho = rho
        self.mu  = rho * nu
        self.dim = dim
    
    def __call__(self, x):
        if self.dim == 2:
            return np.stack(((self.mu*8*np.pi**2 + 2*np.pi - self.rho) * np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * np.exp(-self.t),
                            -(self.mu*8*np.pi**2 - 2*np.pi - self.rho) * np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]) * np.exp(-self.t)
                            ))
        elif self.dim == 3:
            return np.stack(((self.mu*8*np.pi**2 + 2*np.pi - self.rho) * np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * np.exp(-self.t),
                            -(self.mu*8*np.pi**2 - 2*np.pi - self.rho) * np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]) * np.exp(-self.t),
                            np.zeros(x.shape[1])))
        else:
            raise NotImplementedError("Wrong geometry dimensions, choose either 2 or 3.")

class ExactfExpressionSteady:
    """ A class used to generate expressions for the function that is the right-hand side
    of the Stokes equation.
    """
    def __init__(self, rho: float, nu: float, dim: int = 3):
        self.t   = 0.0 # Initial time
        self.rho = rho
        self.mu  = rho * nu
        self.dim = dim
    
    def __call__(self, x):
        if self.dim == 2:
            return np.stack(((self.mu*8*np.pi**2 + 2*np.pi) * np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]),
                            -(self.mu*8*np.pi**2 - 2*np.pi) * np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1])
                            ))
        elif self.dim == 3:
            return np.stack(((self.mu*8*np.pi**2 + 2*np.pi) * np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]),
                            -(self.mu*8*np.pi**2 - 2*np.pi) * np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]),
                            np.zeros(x.shape[1])))
        else:
            raise NotImplementedError("Wrong geometry dimensions, choose either 2 or 3.")

class ExactfExpressionLinearTime:
    def __init__(self, rho: float, nu: float, dim: int = 3):
        self.t = 0.0 # Initial time
        self.rho = rho
        self.mu  = rho * nu
        self.dim = dim
    
    def __call__(self, x):
        if self.dim == 2:
            return np.stack((((self.mu*8*np.pi**2 + 2*np.pi) * (1 + self.t) + self.rho) * np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]),
                            -((self.mu*8*np.pi**2 - 2*np.pi) * (1 + self.t) + self.rho) * np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1])
                            ))
        elif self.dim == 3:
            return np.stack((((self.mu*8*np.pi**2 + 2*np.pi) * (1 + self.t) + self.rho) * np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]),
                            -((self.mu*8*np.pi**2 - 2*np.pi) * (1 + self.t) + self.rho) * np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]),
                            np.zeros(x.shape[1])
                            ))
        else:
            raise NotImplementedError("Wrong geometry dimensions, choose either 2 or 3.")

class ExactfExpressionSpaceIndependent:
    def __init__(self, rho: float, nu: float, dim: int = 3) -> None:
        self.t = 0.0 # Initial time
        self.rho = rho
        self.mu  = rho * nu
        self.dim = dim
    
    def __call__(self, x):
        if self.dim == 2:
            return np.stack((np.ones(x.shape[1]) * self.rho * 2 * self.t,
                             np.ones(x.shape[1]) * self.rho * 2 * self.t,
                            ))
        elif self.dim == 3:
            return np.stack((np.ones(x.shape[1]) * self.rho * 2 * self.t,
                             np.ones(x.shape[1]) * self.rho * 2 * self.t,
                             np.zeros(x.shape[1])
                            ))
        else:
            raise NotImplementedError("Wrong geometry dimensions, choose either 2 or 3.")

class ExactVelocityExpression:
    """ A class used to generate expressions for the exact velocity field of a manufactured solution.
    """
    def __init__(self, dim: int = 3) -> None:
        self.t = 0.0 # Initial time
        self.dim = dim

    def __call__(self, x):
        if self.dim == 2:
            return np.stack((np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * np.exp(-self.t),
                            -np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]) * np.exp(-self.t),
                            ))
        elif self.dim == 3:
            return np.stack((np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * np.exp(-self.t),
                            -np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]) * np.exp(-self.t),
                            np.zeros(x.shape[1])
                            ))
        else:
            raise NotImplementedError("Wrong geometry dimensions, choose either 2 or 3.")

class ExactVelocityExpressionSteady:
    """ A class used to generate expressions for the exact velocity field of a manufactured solution.
    """
    def __init__(self, dim: int = 3) -> None:
        self.t = 0.0 # Initial time
        self.dim = dim

    def __call__(self, x):
        if self.dim == 2:
            return np.stack((np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]),
                            -np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]),
                            ))
        elif self.dim == 3:
            return np.stack((np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]),
                            -np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]),
                            np.zeros(x.shape[1])
                            ))
        else:
            raise NotImplementedError("Wrong geometry dimensions, choose either 2 or 3.")

class ExactVelocityExpressionLinearTime:
    def __init__(self, dim: int = 3) -> None:
        self.t = 0.0 # Initial time
        self.dim = dim

    def __call__(self, x):
        if self.dim == 2:
            return np.stack((np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * (1 + self.t),
                            -np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]) * (1 + self.t),
                            ))
        elif self.dim == 3:
            return np.stack((np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * (1 + self.t),
                            -np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]) * (1 + self.t),
                            np.zeros(x.shape[1])
                            ))
        else:
            raise NotImplementedError("Wrong geometry dimensions, choose either 2 or 3.")

class ExactVelocityExpressionSpaceIndependent:
    def __init__(self, dim: int = 3) -> None:
        self.t = 0.0 # Initial time
        self.dim = dim

    def __call__(self, x):
        if self.dim == 2:
            return np.stack((np.ones(x.shape[1]) * (1 + self.t**2),
                             np.ones(x.shape[1]) * (1 + self.t**2),
                             ))
        elif self.dim == 3:
            return np.stack((np.ones (x.shape[1]) * (1 + self.t**2),
                             np.ones (x.shape[1]) * (1 + self.t**2),
                             np.zeros(x.shape[1])
                            ))
        else:
            raise NotImplementedError("Wrong geometry dimensions, choose either 2 or 3.")

class ExactPressureExpression:
    """ A class used to generate expressions for the exact pressure field of a manufactured solution.
    """
    def __init__(self) -> None:
        self.t = 0.0 # Initial time

    def __call__(self, x):
        return np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * np.exp(-self.t)

class ExactPressureExpressionSteady:
    """ A class used to generate expressions for the exact pressure field of a manufactured solution.
    """
    def __init__(self): pass

    def __call__(self, x):
        return np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1])

class ExactPressureExpressionLinearTime:
    def __init__(self) -> None:
        self.t = 0.0 # Initial time

    def __call__(self, x):
        return np.sin(2*np.pi*x[0]) * np.sin(2*np.pi*x[1]) * (1 + self.t)

class ExactPressureExpressionSpaceIndependent:
    def __init__(self) -> None:
        self.t = 0.0 # Initial time

    def __call__(self, x):
        return np.ones(x.shape[1]) * (1 + self.t)

class CiliaForce:
    """ A class used to generate expressions for the cilia forces as a function of time.
    """
    def __init__(self, A: float) -> None:
        self.A = A # Amplitude (maximum force)
    
    def __call__(self, x: numpy.typing.NDArray[np.float_]) -> numpy.typing.NDArray[np.float_]:
        return np.stack((np.ones (x.shape[1]) * self.A,
                         np.zeros(x.shape[1]),
                         np.zeros(x.shape[1])))

class OscillatoryPressure:
    """ A class used to generate expressions for an oscillatory (sinusoidal) pressure as a function of time.
    """
    def __init__(self, A: float, f: float, c_net: float = 0):
        self.t = 0.0               # Initial time
        self.A = A                 # Amplitude 
        self.omega = 2 * np.pi * f # Angular frequency
        self.c_net = c_net         # Constant reflecting net flow

    def __call__(self, x: numpy.typing.NDArray[np.float_]) -> numpy.typing.NDArray[np.float_]:
        return np.ones(x.shape[1]) * (- self.A * np.sin(self.omega * self.t) + self.c_net)
