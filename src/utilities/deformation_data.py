import numpy as np
import scipy.interpolate as interp
from scipy.signal.windows import tukey

class WallDeformationCorpusCallosum:
    """ Displacement curve for the corpus callosum in feet-head direction
        (i.e. z axis, positive in cranial direction). (Adapted from Kurtcuoglu et al., 2007)"""
    
    def __init__(self, period: float, timestep: float):
        self.t = 0.0
        self.T = period
        self.N = int(period / timestep)+1
        self.index = 0
        self.set_data()

    def set_data(self):
        x = [0.0, 0.07608695652173912, 0.21541501976284583, 0.2875494071146245, 0.341897233201581, 0.43379446640316205, 0.6304347826086956, 0.7687747035573123, 0.858695652173913, 0.942687747035573, 1.0]
        y = [0.0, -0.016636029411764702, -0.028455882352941174, -0.066231617647058826, -0.08240808823529412, -0.079375, -0.07064338235294118, -0.06384191176470588, -0.05400735294117647, -0.0428860294117647, 0.0]
        disp_data = np.array(y)*1e-3 # Convert from mm to m
        times = np.array(x)
        disp_interp = interp.CubicSpline(times, disp_data, bc_type="periodic")
        refined_times = np.linspace(times[0], times[-1], self.N)

        # Construct interpolated deformation data that is windowed 
        # in the initial phase
        window = tukey(len(refined_times), alpha=0.25)
        halfway_index = int(self.N/2)
        disp_first_half = window[:halfway_index]*disp_interp(refined_times, 0)[:halfway_index]
        disp_second_half = disp_interp(refined_times, 0)[halfway_index:]
        disp_two_periods = np.concatenate((disp_interp(refined_times, 0)[1:], disp_interp(refined_times, 0)[1:]))
        self.disp_final = np.concatenate((disp_first_half, disp_second_half, disp_two_periods))
    
    def increment_index(self, t: float):
        self.t = t
        # assert bool(np.logical_and(self.index <= self.N, self.index >= 0))
        # if self.index==(self.N-2):
        #     self.index = 0
        # else:
        self.index += 1

    def scaling(self):
        if self.t >= (2*self.T):
            return 1.0
        return (6/(2*self.T)**5)*self.t**5 - (15/(2*self.T)**4)*self.t**4 + (10/(2*self.T)**3)*self.t**3

    def __call__(self):
        """ Evaluates displacement spline function.
        """
        scale_factor = self.scaling()
        amplitude = scale_factor*self.disp_final[self.index]
        self.applied_bc = amplitude
        return amplitude 

        # return amplitude*np.stack((np.zeros(x.shape[1]),
        #                            np.zeros(x.shape[1]),
        #                            np.ones (x.shape[1])))


class WallDeformationCanalWall:
    """ Displacement curve for the brain stem proximal to the canal wall in feet-head direction
        (i.e. z axis, positive in cranial direction). (Adapted from Kurtcuoglu et al., 2007)"""

    def __init__(self, period: float, timestep: float):
        self.t = 0.0
        self.T = period
        self.N = int(period / timestep)+1
        self.index = 0
        self.set_data()
    
    def set_data(self):
        x = [0.0, 0.07209737827715355, 0.149812734082397, 0.29868913857677903, 0.4803370786516854, 0.6404494382022472, 0.7940074906367042, 0.8913857677902621, 0.950374531835206, 1.0]
        y = [0.0, -0.0150270270270270385, -0.046423423423423406, -0.21018018018018016, -0.20387387387387387, -0.19558558558558558, -0.16801801801801802, -0.130180180180180176, -0.100450450450450445, 0.0]
        disp_data = np.array(y)*1e-3 # Convert from mm to m
        times = np.array(x)
        disp_interp = interp.CubicSpline(times, disp_data, bc_type="periodic")
        refined_times = np.linspace(times[0], times[-1], self.N)
        
        # Construct interpolated deformation data that is windowed 
        # in the initial phase
        window = tukey(len(refined_times), alpha=0.25)
        halfway_index = int(self.N/2)
        disp_first_half = window[:halfway_index]*disp_interp(refined_times, 0)[:halfway_index]
        disp_second_half = disp_interp(refined_times, 0)[halfway_index:]
        disp_two_periods = np.concatenate((disp_interp(refined_times, 0)[1:], disp_interp(refined_times, 0)[1:]))
        self.disp_final = np.concatenate((disp_first_half, disp_second_half, disp_two_periods)) 
    
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
        amplitude = scale_factor*self.disp_final[self.index]
        self.applied_bc = amplitude
        return amplitude
        # return amplitude*np.stack((np.zeros(x.shape[1]),
        #                            np.zeros(x.shape[1]),
        #                            np.ones (x.shape[1])))

class WallDeformationThirdVentricle:
    """ Displacement curve for the third ventricle walls in right-left direction
        (i.e. y axis, positive in right (?) direction). (Adapted from Soellinger et al., 2009)"""

    def __init__(self, period: float, timestep: float):
        self.t = 0.0
        self.T = period
        self.N = int(period / timestep)+1
        self.index = 0
        self.set_data()
    
    def set_data(self):
        x = [0.0, 0.09191583610188261, 0.13842746400885936, 0.18715393133997785, 0.2336655592469546, 0.27906976744186046, 0.3289036544850498, 0.41971207087486156, 0.49058693244739754, 0.5592469545957918, 0.6090808416389811, 0.6566998892580288, 0.7021040974529347, 0.7929125138427464, 0.8438538205980066, 0.9335548172757475, 1.0]
        y = [0.0, 0.036298342541436465, 0.069834254143646407, 0.16005524861878453, 0.24845303867403314, 0.26281767955801105, 0.25066298342541436, 0.24955801104972377, 0.23077348066298342, 0.1943093922651934, 0.17773480662983426, 0.16558011049723757, 0.11806629834254143, 0.036298342541436465, 0.02635359116022099, 0.01088397790055248, 0.0]
        disp_data = np.array(y)*1e-3 # Convert from mm to m
        times = np.array(x)
        disp_interp = interp.CubicSpline(times, disp_data, bc_type="periodic")
        refined_times = np.linspace(times[0], times[-1], self.N)
        
        # Construct interpolated deformation data that is windowed 
        # in the initial phase
        window = tukey(len(refined_times), alpha=0.25)
        halfway_index = int(self.N/2)
        disp_first_half = window[:halfway_index]*disp_interp(refined_times, 0)[:halfway_index]
        disp_second_half = disp_interp(refined_times, 0)[halfway_index:]
        disp_two_periods = np.concatenate((disp_interp(refined_times, 0)[1:], disp_interp(refined_times, 0)[1:]))
        self.disp_final = np.concatenate((disp_first_half, disp_second_half, disp_two_periods)) 
    
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
        amplitude = scale_factor*self.disp_final[self.index]
        self.applied_bc = amplitude
        return amplitude
        # return amplitude*np.stack((np.ones (x.shape[1]),
        #                            np.zeros(x.shape[1]),
        #                            np.zeros(x.shape[1])))