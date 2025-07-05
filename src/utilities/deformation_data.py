import numpy as np
import scipy.interpolate as interp
from scipy.signal.windows import tukey

class DisplacementCorpusCallosumCephalocaudal:
    """ Displacement curve for the corpus callosum in the cephalocaudal (feet-head) direction
        (z axis, positive in cranial direction (towards head)).
        Adapted from Kurtcuoglu et al. (2007), Soellinger et al. (2007, 2009)."""
    
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
        disp_data = np.array(y)*1e-3 # Convert from mm to m
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
        if self.t >= (2*self.T):
            return 1.0
        return (6/(2*self.T)**5)*self.t**5 - (15/(2*self.T)**4)*self.t**4 + (10/(2*self.T)**3)*self.t**3

    def __call__(self):
        """ Evaluates displacement spline function.
        """
        scale_factor = self.scaling()
        self.amplitude = scale_factor*self.disp_final[self.index]

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
        disp_data = np.array(y)*1e-3 # Convert from mm to m
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
        if self.t >= (2*self.T):
            return 1.0
        return (6/(2*self.T)**5)*self.t**5 - (15/(2*self.T)**4)*self.t**4 + (10/(2*self.T)**3)*self.t**3

    def __call__(self):
        """ Evaluates displacement spline function.
        """
        scale_factor = self.scaling()
        self.amplitude = scale_factor*self.disp_final[self.index]

class DisplacementCanalAndFourthVentricleAnteroposterior:
    """ Displacement curve for the region proximal to the canal wall and fourth
        ventricle in the anteroposterior (fore-hind) direction (y axis, positive in fore direction).
        Adapted from Zhong et al. (2009), Greitz et al. (1992)."""

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
        y = [0.0, -0.03, -0.08, -0.108, -0.113, -0.095, -0.06, -0.035, -0.02, -0.012, 0.0]
        disp_data = np.array(y)*1e-3 # Convert from mm to m
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
        disp_data = np.array(y)*1e-3 # Convert from mm to m
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
        if self.t >= (2*self.T):
            return 1.0
        return (6/(2*self.T)**5)*self.t**5 - (15/(2*self.T)**4)*self.t**4 + (10/(2*self.T)**3)*self.t**3

    def __call__(self):
        """ Evaluates displacement spline function.
        """
        scale_factor = self.scaling()
        self.amplitude = scale_factor*self.disp_final[self.index]

class WallDeformationThirdVentricleAnteriorPosterior:
    """ Displacement curve for the third ventricle walls in anterior-posterior direction
        (i.e. y axis, positive in right (?) direction). (Adapted from Soellinger et al., 2009)"""

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
        x = [0.0, 0.09191583610188261, 0.13842746400885936, 0.18715393133997785, 0.2336655592469546, 0.27906976744186046, 0.3289036544850498, 0.41971207087486156, 0.49058693244739754, 0.5592469545957918, 0.6090808416389811, 0.6566998892580288, 0.7021040974529347, 0.7929125138427464, 0.8438538205980066, 0.9335548172757475, 1.0]
        y = [0.0, 0.01298342541436465, 0.019834254143646407, 0.02505524861878453, 0.04845303867403314, 0.06281767955801105, 0.05066298342541436, 0.04955801104972377, 0.03077348066298342, 0.01943093922651934, 0.017773480662983426, 0.016558011049723757, 0.011806629834254143, 0.005298342541436465, 0.002635359116022099, 0.001088397790055248, 0.0]
        disp_data = np.array(y)*1e-3 # Convert from mm to m
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
        if self.t >= (2*self.T):
            return 1.0
        return (6/(2*self.T)**5)*self.t**5 - (15/(2*self.T)**4)*self.t**4 + (10/(2*self.T)**3)*self.t**3

    def __call__(self):
        """ Evaluates displacement spline function.
        """
        scale_factor = self.scaling()
        self.amplitude = scale_factor*self.disp_final[self.index]