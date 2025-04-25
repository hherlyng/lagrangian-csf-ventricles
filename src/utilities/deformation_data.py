import numpy as np
import scipy.interpolate as interp

class WallDeformationCorpusCallosum:
    """ Displacement curve for the corpus callosum in feet-head direction
        (i.e. z axis, positive in cranial direction). (from Kurtcuoglu et al.)"""
    disp = np.array([0, 0.018, 0.014, 0.009, -0.046, -0.071, -0.052, -0.065, -0.047, -0.042, -0.05,
                                -0.047, -0.044, -0.019, 0.0])*1e-3 # Convert from mm to m
    times = np.array([0.0, 0.08, 0.15, 0.21, 0.29, 0.36, 0.42, 0.50, 0.57, 0.63, 0.71, 0.78, 
                                        0.84, 0.90, 1.0])
    disp_interp = interp.CubicSpline(times, disp)
    
    def __init__(self, derivative: bool):
        self.t = 0
        self.disp = self.disp_interp
        self.order = 1 if derivative else 0

    def __call__(self, x):
        amplitude = self.disp(self.t, self.order) # Evaluates function value of spline curve
        return amplitude*np.stack((np.zeros(x.shape[1]),
                                   np.zeros(x.shape[1]),
                                   np.ones (x.shape[1])))

class WallDeformationSpinalCanal:
    """ Displacement curve for the (outlet of) spinal canal in feet-head direction
        (i.e. z axis, positive in cranial direction). (from Enzmann et al.)"""
    # Inserted final as initial position
    disp = np.array([-0.027, 0.0, 0.025, 0.049, 0.058, 0.053, 0.054,
                    0.053, 0.048, 0.024, -0.002, -0.028, -0.051, -0.073,
                    -0.0875, -0.06, -0.027])*1e-3 # Convert from mm to m
    disp -= disp[0] # Reset curve so that inital/final position is 0
    num_datapoints = len(disp)
    times = np.arange(num_datapoints)/(num_datapoints-1)
    disp_interp = interp.CubicSpline(times, disp)

    def __init__(self, derivative: bool):
        self.t = 0
        self.disp = self.disp_interp
        self.order = 1 if derivative else 0

    def __call__(self, x):
        amplitude = self.disp(self.t, self.order) # Evaluates function value of spline curve
        return amplitude*np.stack((np.zeros(x.shape[1]),
                                   np.zeros(x.shape[1]),
                                   np.ones (x.shape[1])))

class WallDeformationCanalWall:
    """ Displacement curve for the (outlet of) spinal canal in feet-head direction
        (i.e. z axis, positive in cranial direction). (from Kurtcuoglu et al.)"""
    # Inserted final as initial position
    disp = np.array([0.0, 0.01, -0.02, -0.081, -0.147, -0.137,
                    -0.142, -0.123, -0.141, -0.117, -0.130, -0.098,
                    -0.068, -0.045, -0.025])*1e-3 # Convert from mm to m
    times = np.array([0.0, 0.08, 0.15, 0.21, 0.29, 0.36, 0.42, 0.50, 0.57, 0.63, 0.71, 0.78, 
                                        0.84, 0.90, 1.0])
    disp_interp = interp.CubicSpline(times, disp)

    def __init__(self, derivative: bool):
        self.t = 0
        self.disp = self.disp_interp
        self.order = 1 if derivative else 0

    def __call__(self, x):
        amplitude = self.disp(self.t, self.order) # Evaluates function value of spline curve
        return amplitude*np.stack((np.zeros(x.shape[1]),
                                   np.zeros(x.shape[1]),
                                   np.ones (x.shape[1])))