import numpy as np
import scipy.interpolate as interp
from scipy.signal.windows import tukey

class DisplacementCorpusCallosumCephalocaudal:
    """ Displacement curve for the corpus callosum in the cephalocaudal (feet-head) direction
        (z axis, positive in cranial direction (towards head)).
        Adapted from Kurtcuoglu et al. (2007), Soellinger et al. (2007, 2009)."""

    y_max = None
    
    def __init__(self, period: float, timestep: float, final_time: float):
        """ Constructor. Assumes period and final_time are integers. """
        self.t = 0.0
        self.P = period
        self.N = int(period / timestep)+1
        self.T = final_time
        assert final_time > 2*period, print("Ramp up occurs over two periods, ensure final time > 2*period.")
        self.index = 0
        self.set_data()

    def set_data(self):
        x = np.linspace(0.0, 1.0, 11)
        y = [0.0, 0.015, 0.011, -0.050, -0.075, -0.0625, -0.0475, -0.05, -0.0475, -0.03, 0.0]
        disp_data = 2/5*np.array(y)*1e-3 # Convert from mm to m
        times = np.array(x)
        disp_interp = interp.CubicSpline(times, disp_data, bc_type="periodic")
        refined_times = np.linspace(times[0], times[-1], self.N)
        num_final_periods = int(self.T) - int(self.P)
        
        # Construct interpolated deformation data that is windowed 
        # in the initial period
        window = tukey(len(refined_times), alpha=0.25)
        halfway_index = int(self.N/2)
        disp_first_half = window[:halfway_index]*disp_interp(refined_times, 0)[:halfway_index]
        disp_second_half = disp_interp(refined_times, 0)[halfway_index:]
        first_period_windowed = np.concatenate((disp_first_half, disp_second_half))
        final_periods = np.concatenate(([disp_interp(refined_times, 0)[1:] for _ in range(num_final_periods)]))
        self.disp_final = np.concatenate((first_period_windowed, final_periods))
    
    def increment_index(self, t: float):
        self.t = t
        self.index += 1

    def scaling(self):
        if self.t >= (2*self.P):
            return 1.0
        return (6/(2*self.P)**5)*self.t**5 - (15/(2*self.P)**4)*self.t**4 + (10/(2*self.P)**3)*self.t**3

    def __call__(self, x=None):
        """ Evaluates displacement spline function.
        """
        scale_factor = self.scaling()
        self.amplitude = scale_factor*self.disp_final[self.index]
        if x is not None:
            linear_factor = 1/2*(1 + (self.y_max - x[1])/(2*self.y_max))

            return linear_factor * self.amplitude

class DisplacementCaudateNucleusHeadLateral:
    """ Displacement curve for the head of the caudate nucleus
        in the lateral (right-left) direction
        (x axis, positive in right direction).

        If imposed on the left side, the sign of the displacement must
        best switched.

        Adapted from Soellinger et al. (2009), Enzmann & Pelc (1992)."""
    
    def __init__(self, period: float, timestep: float, final_time: float):
        """ Constructor. Assumes period and final_time are integers. """
        self.t = 0.0
        self.P = period
        self.N = int(period / timestep)+1
        self.T = final_time
        assert final_time > 2*period, print("Ramp up occurs over two periods, ensure final time > 2*period.")
        self.index = 0
        self.set_data()

    def set_data(self):
        x = np.linspace(0.0, 1.0, 11)
        y = [0.0, -0.01, -0.03, -0.035, -0.025, -0.02, -0.018, -0.016, -0.009, -0.003, 0.0]
        disp_data = 2/5*1/2*np.array(y)*1e-3 # Convert from mm to m
        times = np.array(x)
        disp_interp = interp.CubicSpline(times, disp_data, bc_type="periodic")
        refined_times = np.linspace(times[0], times[-1], self.N)
        num_final_periods = int(self.T) - int(self.P)
        
        # Construct interpolated deformation data that is windowed 
        # in the initial period
        window = tukey(len(refined_times), alpha=0.25)
        halfway_index = int(self.N/2)
        disp_first_half = window[:halfway_index]*disp_interp(refined_times, 0)[:halfway_index]
        disp_second_half = disp_interp(refined_times, 0)[halfway_index:]
        first_period_windowed = np.concatenate((disp_first_half, disp_second_half))
        final_periods = np.concatenate(([disp_interp(refined_times, 0)[1:] for _ in range(num_final_periods)]))
        self.disp_final = np.concatenate((first_period_windowed, final_periods))
    
    def increment_index(self, t: float):
        self.t = t
        self.index += 1

    def scaling(self):
        if self.t >= (2*self.P):
            return 1.0
        return (6/(2*self.P)**5)*self.t**5 - (15/(2*self.P)**4)*self.t**4 + (10/(2*self.P)**3)*self.t**3

    def __call__(self):
        """ Evaluates displacement spline function.
        """
        scale_factor = self.scaling()
        self.amplitude = scale_factor*self.disp_final[self.index]

class DisplacementThirdVentricleLateral:
    """ Displacement curve for the third ventricle walls in lateral (right-left) direction
        (x axis, positive in right direction).

        If imposed on the left side, the sign of the displacement must
        best switched.

        Adapted from Greitz et al. (1992), Soellinger et al. (2009)."""

    def __init__(self, period: float, timestep: float, final_time: float):
        """ Constructor. Assumes period and final_time are integers. """
        self.t = 0.0
        self.P = period
        self.N = int(period / timestep)+1
        self.T = final_time
        assert final_time > 2*period, print("Ramp up occurs over two periods, ensure final time > 2*period.")
        self.index = 0
        self.set_data()
    
    def set_data(self):
        x = np.linspace(0.0, 1.0, 11)
        y = [0.0, 0.005, -0.02, -0.03, -0.0285, -0.0275, -0.0225, -0.015, -0.0075, -0.005, 0.0]
        disp_data = 2/5*1/2*np.array(y)*1e-3 # Convert from mm to m
        times = np.array(x)
        disp_interp = interp.CubicSpline(times, disp_data, bc_type="periodic")
        refined_times = np.linspace(times[0], times[-1], self.N)
        num_final_periods = int(self.T) - int(self.P)
        
        # Construct interpolated deformation data that is windowed 
        # in the initial period
        window = tukey(len(refined_times), alpha=0.25)
        halfway_index = int(self.N/2)
        disp_first_half = window[:halfway_index]*disp_interp(refined_times, 0)[:halfway_index]
        disp_second_half = disp_interp(refined_times, 0)[halfway_index:]
        first_period_windowed = np.concatenate((disp_first_half, disp_second_half))
        final_periods = np.concatenate(([disp_interp(refined_times, 0)[1:] for _ in range(num_final_periods)]))
        self.disp_final = np.concatenate((first_period_windowed, final_periods))
    
    def increment_index(self, t: float):
        self.t = t
        self.index += 1

    def scaling(self):
        if self.t >= (2*self.P):
            return 1.0
        return (6/(2*self.P)**5)*self.t**5 - (15/(2*self.P)**4)*self.t**4 + (10/(2*self.P)**3)*self.t**3

    def __call__(self):
        """ Evaluates displacement spline function.
        """
        scale_factor = self.scaling()
        self.amplitude = scale_factor*self.disp_final[self.index]

class DisplacementLateralVentricleHorns:
    """ UPDATE """

    def __init__(self, period: float, timestep: float, final_time: float):
        """ Constructor. Assumes period and final_time are integers. """
        self.t = 0.0
        self.P = period
        self.N = int(period / timestep)+1
        self.T = final_time
        assert final_time > 2*period, print("Ramp up occurs over two periods, ensure final time > 2*period.")
        self.index = 0
        self.set_data()
    
    def set_data(self):
        x = np.linspace(0.0, 1.0, 11)
        y = [0.0, -0.01, -0.020, -0.050, -0.075, -0.0625, -0.0475, -0.05, -0.0475, -0.03, 0.0]
        disp_data = 2/5*1/4*np.array(y)*1e-3 # Convert from mm to m
        times = np.array(x)
        disp_interp = interp.CubicSpline(times, disp_data, bc_type="periodic")
        refined_times = np.linspace(times[0], times[-1], self.N)
        num_final_periods = int(self.T) - int(self.P)
        
        # Construct interpolated deformation data that is windowed 
        # in the initial period
        window = tukey(len(refined_times), alpha=0.25)
        halfway_index = int(self.N/2)
        disp_first_half = window[:halfway_index]*disp_interp(refined_times, 0)[:halfway_index]
        disp_second_half = disp_interp(refined_times, 0)[halfway_index:]
        first_period_windowed = np.concatenate((disp_first_half, disp_second_half))
        final_periods = np.concatenate(([disp_interp(refined_times, 0)[1:] for _ in range(num_final_periods)]))
        self.disp_final = np.concatenate((first_period_windowed, final_periods))
    
    def increment_index(self, t: float):
        self.t = t
        self.index += 1

    def scaling(self):
        if self.t >= (2*self.P):
            return 1.0
        return (6/(2*self.P)**5)*self.t**5 - (15/(2*self.P)**4)*self.t**4 + (10/(2*self.P)**3)*self.t**3

    def __call__(self):
        """ Evaluates displacement spline function.
        """
        scale_factor = self.scaling()
        self.amplitude = scale_factor*self.disp_final[self.index]


class DisplacementVentricleFloorCephalocaudal:
    """ UPDATE """

    def __init__(self, period: float, timestep: float, final_time: float):
        """ Constructor. Assumes period and final_time are integers. """
        self.t = 0.0
        self.P = period
        self.N = int(period / timestep)+1
        self.T = final_time
        assert final_time > 2*period, print("Ramp up occurs over two periods, ensure final time > 2*period.")
        self.index = 0
        self.set_data()
    
    def set_data(self):
        x = np.linspace(0.0, 1.0, 11)
        y = [0.0, 0.015, 0.011, -0.050, -0.075, -0.0625, -0.0525, -0.05, -0.0475, -0.03, 0.0]
        disp_data = 2/5*1/2*np.array(y)*1e-3 # Convert from mm to m
        times = np.array(x)
        disp_interp = interp.CubicSpline(times, disp_data, bc_type="periodic")
        refined_times = np.linspace(times[0], times[-1], self.N)
        num_final_periods = int(self.T) - int(self.P)
        
        # Construct interpolated deformation data that is windowed 
        # in the initial period
        window = tukey(len(refined_times), alpha=0.25)
        halfway_index = int(self.N/2)
        disp_first_half = window[:halfway_index]*disp_interp(refined_times, 0)[:halfway_index]
        disp_second_half = disp_interp(refined_times, 0)[halfway_index:]
        first_period_windowed = np.concatenate((disp_first_half, disp_second_half))
        final_periods = np.concatenate(([disp_interp(refined_times, 0)[1:] for _ in range(num_final_periods)]))
        self.disp_final = np.concatenate((first_period_windowed, final_periods))
    
    def increment_index(self, t: float):
        self.t = t
        self.index += 1

    def scaling(self):
        if self.t >= (2*self.P):
            return 1.0
        return (6/(2*self.P)**5)*self.t**5 - (15/(2*self.P)**4)*self.t**4 + (10/(2*self.P)**3)*self.t**3

    def __call__(self):
        """ Evaluates displacement spline function.
        """
        scale_factor = self.scaling()
        self.amplitude = scale_factor*self.disp_final[self.index]