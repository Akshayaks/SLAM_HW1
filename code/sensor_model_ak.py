'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
import pdb

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self.map = occupancy_map
        self._z_hit = 1000
        self._z_short = 0.01
        self._z_max = 0.03
        self._z_rand = 100000

        self._sigma_hit = 250
        self._lambda_short = 0.01

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 8183


        # Used in sampling angles in ray casting
        self._subsampling = 1

        self.occupancy_threshold = 1E-7
        self.visualization = True
        self.plot_measurement = True

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        
        prob_zt1 = 0
        robot_x, robot_y, robot_theta = x_t1
        
        temp = self.map[min(int(robot_y/10.), 799)][min(int(robot_x/10.), 799)]
        if temp > 0.4 or temp == -1:
            return 1e-100

        # laser_x = int(round((robot_x + (25 * np.cos(robot_theta))))/10)
        # laser_y = int(round((robot_y + (25 * np.sin(robot_theta))))/10)

        temp_x = 25.0 * np.cos(robot_theta)
        temp_y = 25.0 * np.sin(robot_theta)
        laser_x = int(round((robot_x + temp_x) / 10.0))
        laser_y = int(round((robot_y + temp_y) / 10.0))

        for k in range(-90,90,10): #here K is 180
            
            z_kt = z_t1_arr[k + 90]

            #z_kt_star = self.rayCast(k, robot_theta, laser_x, laser_y)
            z_kt_star = self.ray_cast( laser_x, laser_y, robot_theta, k)
            prob1 = (self._z_hit * self.p_hit(z_kt_star,z_kt))
            prob2 = (self._z_short * self.p_short(z_kt_star,z_kt)) 
            prob3 = (self._z_max * self.p_max(z_kt)) 
            prob4 = (self._z_rand * self.p_rand(z_kt))
            prob = prob1 + prob2 + prob3 + prob4
            if prob > 0:
                prob_zt1 = prob_zt1 + np.log(prob)
        return math.exp(prob_zt1)

    def ray_cast(self, origin_x, origin_y, robot_theta, step):

        z_kt_star = 0

        x_rough = origin_x
        y_rough = origin_y
        x = origin_x
        y = origin_y
        theta = robot_theta + math.radians(step) 
        while 0 < x < self.map.shape[1] and 0 < y < self.map.shape[0] and (abs(self.map[y,x]) < self.occupancy_threshold):
            x_rough += 2*np.cos(theta)
            y_rough += 2*np.sin(theta)
            x = int(round(x_rough))
            y = int(round(y_rough))
        end_p = np.array([x,y])
        start_p = np.array([origin_x,origin_y])
        dist = np.linalg.norm(end_p-start_p) * 10

        return dist


    def rayCast(self, deg, ang, coord_x, coord_y):
        # print("Ray casting from: ", coord_x, coord_y)
        final_angle= ang + math.radians(deg)
        start_x = coord_x
        start_y = coord_y
        final_x = coord_x
        final_y = coord_y
        while 0 < final_x < self.map.shape[1] and 0 < final_y < self.map.shape[0] and abs(self.map[final_y, final_x]) < 0.0000001:
            start_x += 2 * np.cos(final_angle)
            start_y += 2 * np.sin(final_angle)
            final_x = int(round(start_x))
            final_y = int(round(start_y))
        end_p = np.array([final_x,final_y])
        start_p = np.array([coord_x,coord_y])
        dist = np.linalg.norm(end_p-start_p) * 10
        return dist


    def p_hit(self, z_kt_star, z_kt):
        if 0 <= z_kt <= self._max_range:
            prob = (math.exp(-((z_kt - z_kt_star)**2)/(2 * self._sigma_hit**2)))/(math.sqrt(2 * math.pi * self._sigma_hit**2))
            return prob
        return 0

    def p_short(self, z_kt_star, z_kt):
        if 0 <= z_kt <= z_kt_star:
            eta = 1/(1 - math.exp(-self._lambda_short*z_kt_star))
            prob = eta * self._lambda_short * math.exp(-self._lambda_short*z_kt)
            return prob
        return 0.0

    def p_max(self, z_kt):
        if z_kt == self._max_range:
            return 1.0
        return 0.0

    def p_rand(self, z_kt):
        if z_kt >= 0 and z_kt < self._max_range:
            return 1.0/self._max_range
        return 0.0


    

