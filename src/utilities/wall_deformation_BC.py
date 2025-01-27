import numpy as np
import scipy.interpolate as interp

corpus_callosum_disp = np.array([0, 0.018, 0.014, 0.009, -0.046, -0.071, -0.052, -0.065, -0.047, -0.042, -0.05,
                                -0.047, -0.044, -0.019, 0.0])*1e-3 # Convert from mm to m
disp_time_relative_to_T = np.array([0.0, 0.08, 0.15, 0.21, 0.29, 0.36, 0.42, 0.50, 0.57, 0.63, 0.71, 0.78, 
                                    0.84, 0.90, 1.0])
disp_interp = interp.CubicSpline(disp_time_relative_to_T, corpus_callosum_disp)


class WallDeformation:
    """ Displacement curve for corpus callosum in feet-head direction
        (i.e. z axis, positive in cranial direction). """
    def __init__(self, derivative: bool):
        self.t = 0
        self.disp = disp_interp
        self.order = 1 if derivative else 0

    def __call__(self, x):
        amplitude = self.disp(self.t, self.order) # Evaluates first derivative of spline curve
        return amplitude*np.stack((np.zeros(x.shape[1]),
                                   np.zeros(x.shape[1]),
                                   np.ones (x.shape[1])))