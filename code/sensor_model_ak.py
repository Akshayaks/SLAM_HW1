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
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self.map = occupancy_map
        self._z_hit = 4 # weight probability for hitting 
        self._z_short = 0.08 # weight probability for robot hitting a shorter distance (corners robots -- decrease to help decrease)
        self._z_max = 0.03 # weight for reaching out of range
        self._z_rand = 500 #weight for random noise (longer time for lesser number of particles; ensure particles are in the correct place)

        self._sigma_hit = 75
        self._lambda_short = 0.25

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000 #constant

        # Used in sampling angles in ray casting
        self._subsampling = 5

        self.probability_min = 0.15 #  keep (if laser goes thru the wall)
        self.visualization = True
        self.plot_measurement = True
        self._thetas_step = np.array(range(-90,90,self._subsampling))
        self._subsamples = self._thetas_step.shape[0]

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """     
        
        
        robot_x_arr = x_t1[:,0]
        robot_y_arr = x_t1[:,1]
        robot_theta_arr = x_t1[:,2]
        
        prob_zt1_array = np.zeros((x_t1.shape[0],1))

        for m in range(x_t1.shape[0]):
            robot_x = robot_x_arr[m]
            robot_y = robot_y_arr[m]
            robot_theta = robot_theta_arr[m]
            laser_x = int(round((robot_x + (25 * np.cos(robot_theta)))/10.0))
            laser_y = int(round((robot_y + (25 * np.sin(robot_theta)))/10.0))


            temp = self.map[min(int(robot_y/10.), 799)][min(int(robot_x/10.), 799)]
            if temp > self.probability_min or temp == -1:
                prob_zt1_array[m] = 1e-100
                continue

            prob_zt1 = 0

            z_kt_array = z_t1_arr[self._thetas_step + 90]
            z_kt_star_array = self.ray_cast_vectors(laser_x, laser_y, np.add(np.radians(self._thetas_step), robot_theta))

            prob = (self._z_hit * self.p_hit(z_kt_star_array,z_kt_array)) + (self._z_short * self.p_short(z_kt_star_array,z_kt_array)) + (self._z_max * self.p_max(z_kt_array)) + (self._z_rand * self.p_rand(z_kt_array))
            
            prob_zt1 = np.sum(np.log(prob[prob!=0]))
            prob_zt1_array[m]= math.exp(prob_zt1)
            
        
        return prob_zt1_array

    def ray_cast_vectors(self, origin_x, origin_y, robot_theta):

        z_kt_star_arr = np.zeros_like(robot_theta)
        for step in range(self._subsamples):
            x_rough = origin_x
            y_rough = origin_y
            x_int = origin_x
            y_int = origin_y
            while 0 < x_int < self.map.shape[1] and 0 < y_int < self.map.shape[0] and (abs(self.map[y_int,x_int]) < self.probability_min):
                x_rough += 2*np.cos(robot_theta[step])
                y_rough += 2*np.sin(robot_theta[step])
                x_int = int(round(x_rough))
                y_int = int(round(y_rough))
            dist = math.sqrt((x_int-origin_x)**2 + (y_int-origin_y)**2) * 10
            if(dist >= self._max_range):
                dist = self._max_range
            z_kt_star_arr[step] = dist
        return z_kt_star_arr

    def ray_cast(self, origin_x, origin_y, robot_theta, step):

        x_rough = origin_x
        y_rough = origin_y
        x_int = origin_x
        y_int = origin_y
        theta = robot_theta + math.radians(step) 
        while 0 < x_int < self.map.shape[1] and 0 < y_int < self.map.shape[0] and (abs(self.map[y_int,x_int]) < self.probability_min):
            x_rough += 2*np.cos(theta)
            y_rough += 2*np.sin(theta)
            x_int = int(round(x_rough))
            y_int = int(round(y_rough))
        dist = math.sqrt((x_int-origin_x)**2 + (y_int-origin_y)**2) * 10
        if(dist >= self._max_range):
            return self._max_range
        return dist
    
    def p_hit(self, z_kt_star, z_kt):
        prob = norm.pdf(z_kt, loc=z_kt_star, scale = self._sigma_hit)
        prob[z_kt < 0] = 0
        prob[z_kt > self._max_range] = 0
        return prob

    def p_short(self, z_kt_star, z_kt):
        prob = expon.pdf(z_kt, loc=0, scale = 1/self._lambda_short)
        prob[z_kt < 0] = 0
        prob[z_kt > z_kt_star] = 0
        return prob

    def p_max(self, z_kt):
        prob = np.zeros(z_kt.shape[0])
        prob[z_kt == self._max_range] = 1
        return prob

    
    def p_rand(self, z_kt):
        prob = np.ones(z_kt.shape[0])/self._max_range
        prob[z_kt < 0] = 0
        prob[z_kt > self._max_range] = 0
        return prob

    def p_hit_scalar(self, z_kt_star, z_kt):
        if 0 <= z_kt <= self._max_range:
            prob = (math.exp(-((z_kt - z_kt_star)**2)/(2 * self._sigma_hit**2)))/(math.sqrt(2 * math.pi * self._sigma_hit**2))
            return prob
        return 0

    def p_short_scalar(self, z_kt_star, z_kt):
        if 0 <= z_kt <= z_kt_star:
            eta = 1/(1 - math.exp(-self._lambda_short*z_kt_star))
            prob = eta * self._lambda_short * math.exp(-self._lambda_short*z_kt)
            return prob
        return 0.0

    def p_max_scalar(self, z_kt):
        if z_kt == self._max_range:
            return 1.0
        return 0.0

    def p_rand_scalar(self, z_kt):
        if z_kt >= 0 and z_kt < self._max_range:
            return 1.0/self._max_range
        return 0.0




