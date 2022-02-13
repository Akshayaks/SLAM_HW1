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
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 1

        self.occupany_threshold = 0.35
        self.visualization = True
        self.plot_measurement = True

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        
        prob_zt1 = 1.0
        x_t = self.shift_to_laser(x_t1)

        if x_t[0] < 0 or x_t[0] >  7000 or x_t[1] < 0 or x_t[1] > 7000:
            return 0

        z_kt_star_array = []
        # theta = x_t[2] - math.pi/2 #-math.pi/2

        for k in range(0,180): #here K is 180
            theta = k * (math.pi/180) + x_t[2]
            z_kt = z_t1_arr[k]
            # print(theta)
            z_kt_star = self.ray_cast(x_t,theta)
            # theta = theta + self._subsampling
            z_kt_star_array.append(z_kt_star)
            if k == 160:
                pdb.set_trace()
            prob = self._z_hit * self.p_hit(z_kt_star,z_kt) + self._z_short * self.p_short(z_kt_star,z_kt) \
                   + self._z_max * self.p_max(z_kt) + self._z_rand * self.p_rand(z_kt)
            prob_zt1 *= prob

        return prob_zt1

    def shift_to_laser(self,x_t1): #Add transform to account for position of laser
        x_l = np.zeros(3)
        x_l[0] = x_t1[0] + (25/10)*math.cos(x_t1[2]) #resolution
        x_l[1] = x_t1[1] + (25/10)*math.sin(x_t1[2])
        x_l[2] = x_t1[2]
        return x_l

    def ray_cast(self, x_t, theta):

        z_kt_star = 0

        x0 = x_t[0]
        y0 = x_t[1]

        for j in range(1,int(self._max_range/10)):

            x = x0 + j*math.cos(theta - math.pi/2)
            y = y0 + j*math.sin(theta - math.pi/2)
            
            print("Occupancy grid: ", self.map[round(y/10)][round(x/10)])
            if x >= 7000 or y >= 7000 or x < 0 or y < 0:
                print("reached boundry")
                # print(round(x),round(y))
                x = x0 + (j-1)*math.cos(theta - math.pi/2)
                y = y0 + (j-1)*math.sin(theta - math.pi/2)
                dist = math.sqrt((y-y0/10)**2 + (x-x0/10)**2)
                print("dist: ", dist)
                z_kt_star = dist
                break

            if self.map[round(x/10)][round(y/10)] < self.occupany_threshold:
                print("reached obstacle: ", j, theta)
                print(x,y, x0,y0)
                dist = math.sqrt((y-y0)**2 + (x-x0)**2)
                print("dist: ", dist)
                z_kt_star = dist
                break

        print("z_kt_star: ", z_kt_star)
        return z_kt_star


    def p_hit(self, z_kt_star, z_kt):
        if z_kt > self._z_max or z_kt < 0:
            return 0
        else:
            c = [1/(math.sqrt(2*math.pi*self._sigma_hit**2))]
            p = -0.5*(z_kt - z_kt_star)**2/self._sigma_hit**2
            eta =  1                                #integral also defined as self._sigma_hit*math.sqrt(2*math.pi)
            prob = eta*c*math.pow(math.e,p)
        return prob

    def p_short(self, z_kt_star, z_kt):
        if z_kt > z_kt_star or z_kt < 0:
            return 0
        else:
            eta = 1/(1 - math.pow(math.e,-self._lambda_short*z_kt_star))
            prob = eta * self._lambda_short * math.pow(math.e,-self._lambda_short*z_kt)
        return prob

    def p_max(self, z_kt):
        if z_kt == self._z_max:
            return 1
        return 0

    def p_rand(self, z_kt):
        if z_kt >= 0 and z_kt < self._z_max:
            return 1/self._z_max
        return 0



    

    def ray_cast_DDA(self,z_kt, x_t1):              #Find zk_t* given zk_t and the occupancy map
        theta = x_t1[2]                        #Laser from 0 -> world frame 90

        m = math.tan(theta)
        step_unit_x = math.sqrt(1 + m**2)           #The distance moved on hypotenuse for unit dist in x and y direction
        step_unit_y = math.sqrt((1/m)**2 + 1)

        x_end = x_t1[0] + z_kt * math.cos(theta) #Calculate the end point of the ray
        y_end = x_t1[1] + z_kt * math.sin(theta)

        x_start = 0   #To move along the ray
        y_start = 0

        x = x_t1[0] 
        y = y_t1[0]

        x_step = 0 #Move in positive or negative direction
        y_step = 0
        len_x = 0 #Length of the true ray
        len_y = 0

        hit_obstacle = False
        dist = 0

        if x_end > x_start:
            x_step = 1
            len_x = (x - x_start) * step_unit_x
        else:
            x_step = -1
            len_x = (x_start - x) * step_unit_x

        if y_end > y_start:
            y_step = 1
            len_y = (y- y_start) * step_unit_y
        else:
            y_step = -1
            len_y = (y_start - y) * step_unit_y

        # dist = math.sqrt(len_x**2 + len_y**2)


        while not hit_obstacle and dist < self._z_max: #as long as it is free 
            if len_x < len_y:
                x += x_step
                dist = len_x
                len_x += step_unit_x


        return z_kt_star

    

