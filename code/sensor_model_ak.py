import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import expon
import pdb

from map_reader import MapReader

class SensorModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):
        """
        Initialize Sensor Model parameters here
        """
        self.map = occupancy_map
        self._max_range = 8183 #1000
        self._sigma_hit = 250 #50 
        self._lambda_short = 0.01 #0.1 #0.01
        self._z_hit = 1000 #1 #2000
        self._z_short = 0.01 #0.1 #0.01
        self._z_max = 0.03 #0.01 #0.01
        self._z_rand = 100000 #700 #15000
        self._min_probability = 1e-7 #0.35

    def p_hit(self, z_kt_star, z_kt):
        if z_kt > self._max_range or z_kt < 0:
            return 0
        else:
            c = 1/math.sqrt(2*math.pi*self._sigma_hit**2)
            p = -0.5*(z_kt - z_kt_star)**2/self._sigma_hit**2                             
            prob = c*math.exp(p)
        return prob

    def p_short(self, z_kt_star, z_kt):
        if z_kt > z_kt_star or z_kt < 0:
            return 0
        else:
            eta = 1/(1 - math.exp(-self._lambda_short*z_kt_star))
            prob = eta * self._lambda_short * math.pow(math.e,-self._lambda_short*z_kt)
        return prob

    def p_max(self, z_kt):
        if z_kt == self._max_range:
            return 1.0
        return 0.0

    def p_rand(self, z_kt):
        if z_kt >= 0 and z_kt < self._max_range:
            return 1/self._max_range
        return 0

 
    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """

        """
        TODO : Add your code here
        """
        # pos_x, pos_y, pos_theta = x_t1
        # print("rob pos: ", min(int(pos_y/10.), 799), min(int(pos_x/10.), 799))
        temp = self.map[min(int(x_t1[1]/10.), 799)][min(int(x_t1[0]/10.), 799)]
        # print("Temp: ", temp)
        if temp > 0.4 or temp == -1:
            return 1e-100
        prob_zt1 = 0.0
        # print("Robot position: ", pos_x,pos_y,pos_theta)

        # laser_x = 25.0 * np.cos(pos_theta)
        # laser_y = 25.0 * np.sin(pos_theta)
        # coord_x = int(round((pos_x + laser_x) / 10.0))
        # coord_y = int(round((pos_y + laser_y) / 10.0))
        x_l = self.shift_to_laser(x_t1)
        coord_x = x_l[0]
        coord_y = x_l[1]
        # print("Laser position: ", coord_x,coord_y)

        z_kt_star = []
        for deg in range (-90,90, 10):
            theta_ray = math.radians(deg) + x_t1[2]
            z_t1_true = self.ray_cast([int(coord_x), int(coord_y)], theta_ray)
            z_kt_star.append(z_t1_true)
            z_t1_k = z_t1_arr[deg+90]
            # pdb.set_trace()from sensor_model_repo import SensorModel
            # if deg == 0:
            #     pdb.set_trace()
            prob = self._z_hit * self.p_hit(z_t1_true,z_t1_k) + self._z_short * self.p_short(z_t1_true,z_t1_k) \
                   + self._z_max * self.p_max(z_t1_k) + self._z_rand * self.p_rand(z_t1_k)
            if prob > 0:
                prob_zt1 += np.log(prob)
            # p1 = self._z_hit * self.p_hit(z_t1_true, z_t1_k)
            # p2 = self._z_short * self.p_short(z_t1_true, z_t1_k)
            # p3 = self._z_max * self.p_max(z_t1_k)
            # p4 = self._z_rand * self.p_rand(z_t1_k)
            # p = p1 + p2 + p3 + p4
            # if p > 0:
            #     q = q + np.log(p)
        return math.exp(prob_zt1)


    def shift_to_laser(self,x_t1): #Add transform to account for position of laser
        x_l = np.zeros(3)
        x_l[0] = int(round((x_t1[0] + 25*math.cos(x_t1[2]))/10)) #resolution
        x_l[1] = int(round((x_t1[1] + 25*math.sin(x_t1[2]))/10))
        x_l[2] = x_t1[2] #- math.pi/2
        return x_l

    def ray_cast(self, x_t, theta):

        z_kt_star = 0

        x0 = int(x_t[0])
        y0 = int(x_t[1])
        x = x0
        y = y0
        # print("Ray casting from: ", x0,y0)
        # pdb.set_trace()
        # print(self.map.shape[1],self.map.shape[0])
        # print("Final angle: ", theta)
        step = 0
        while 0 < x < 800 and 0 < y < 800 and abs(self.map[y][x]) < self._min_probability:
            step += 1
            x0 += 2*np.cos(theta)
            y0 += 2*np.sin(theta)
            x = int(round(x0))
            y = int(round(y0))
            # if step == 49:
            #     print("Final point at step 49: ", x, y)
            #     print("Occupancy map: ", abs(self.map[y, x]))
            #     break
        # print("occupancy_map: ", self.map[y][x])
        # print("steps: ", step)
        # print("Final point: ", x, y)
        # print("Start point: ", x0, y0)
        start_pt = np.array([x_t[0], x_t[1]])
        end_pt = np.array([x, y])
        z_kt_star = np.linalg.norm(end_pt-start_pt) * 10

        return z_kt_star

if __name__=='__main__':
    pass
