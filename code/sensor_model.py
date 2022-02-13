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
        self._z_hit = 1             #1
        self._z_short = 0.1         #0.1
        self._z_max = 0.1           #0.1
        self._z_rand = 100          #100

        self._sigma_hit = 50        #50
        self._lambda_short = 0.1    #0.1
        
        # self._z_hit = 150           #1
        # self._z_short = 17.5        #0.1
        # self._z_max = 15            #0.1
        # self._z_rand = 100          #100

        # self._sigma_hit = 100       #50
        # self._lambda_short = 15     #0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 5       #2
        self._occupancy_map = occupancy_map
        
        
    def prob(self, z_star, z):
        # hit
        if 0 <= z <= self._max_range:
            p_hit = np.exp(-1/2 * (z - z_star)**2 / (self._sigma_hit**2))
            p_hit = p_hit / (np.sqrt(2 * np.pi * self._sigma_hit**2))
        else:
            p_hit = 0

        # short
        if 0 <= z <= z_star:
            #eta = 1/(1-np.exp(-self._lambda_short*z_star))
            #p_short = eta * self._lambda_short * np.exp(-self._lambda_short * z)
            p_short = self._lambda_short * np.exp(-self._lambda_short * z)
        else:
            p_short = 0


        # max
        if z.astype(int) == self._max_range:
            p_max = 1
        else:
            p_max = 0

        # rand
        if(0<= z < self._max_range):
            p_rand = 1/self._max_range
        else:
            p_rand = 0

        p = self._z_hit * p_hit + self._z_short * p_short + self._z_max * p_max + self._z_rand * p_rand
        # p = p/(self._z_hit + self._z_short + self._z_max + self._z_rand)

        return p

    def getProbability(self, z_star, z_reading):
        # hit
        if 0 <= z_reading <= self._max_range:
            pHit = np.exp(-1 / 2 * (z_reading - z_star) ** 2 / (self._sigma_hit ** 2))
            pHit = pHit / (np.sqrt(2 * np.pi * self._sigma_hit ** 2))

        else:
            pHit = 0

        # short
        if 0 <= z_reading <= z_star:
            # eta = 1.0/(1-np.exp(-lambdaShort*z_star))
            eta = 1
            pShort = eta * self._lambda_short * np.exp(-self._lambda_short * z_reading)

        else:
            pShort = 0

        # max
        if z_reading >= self._max_range:
            pMax = self._max_range
        else:
            pMax = 0

        # rand
        if 0 <= z_reading < self._max_range:
            pRand = 1 / self._max_range
        else:
            pRand = 0

        p = self._z_hit * pHit + self._z_short * pShort + self._z_max * pMax + self._z_rand * pRand
        # p /= (self._z_hit + self._z_short + self._z_max + self._z_rand)
        return p


    def wrap(self, angle):
        angle_wrapped = angle - 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
        return angle_wrapped    

    def raycasting(self,x):
        #position of robot in world frame
        robot_x= x[0]
        robot_y= x[1]

        #laser sensor wrt robot frame in cm
        ds=25

        #transform sensor to world- xs,ys,ts, in occupancy map- divide by resolution
        
        #laser sensor's coordinates x y and theta in world frame and then to occupancy map- divide by resolution
        xs=(robot_x+ds*math.cos(x[2]))/10
        ys=(robot_y+ds*math.sin(x[2]))/10
        ts=x[2]

        #range of laser scan to check: ts-90 to ts+90
        se_angle= np.pi/2

        #store z_star values for the 180 measurements
        # z_star= np.zeros((180,))
        z_star = np.full(180, self._max_range, float)
        #increment for checking in occupancy map
        step=1
        # x_check=xs
        # y_check=ys
        k=0
        beam_angle = np.linspace(self.wrap(ts-se_angle), self.wrap(ts+se_angle), int(180/self._subsampling))
        
        # for i in range(ts-se_angle, ts+se_angle, self._subsampling):
        for i in beam_angle:
            x_check=xs
            y_check=ys
            # x_check += step*math.cos(i)
            # y_check += step*math.sin(i)
            dist=math.sqrt((x_check-xs)**2+(y_check-ys)**2)
            
            while(dist<self._max_range):
                xInt = np.floor(x_check).astype(int)
                yInt = np.floor(y_check).astype(int)
                if(yInt < 0 or xInt < 0):
                    # print("yikes")
                    for h in range(self._subsampling):
                        z_star[k]=self._max_range
                        k+=1
                    break
                # if(xInt < 0):
                #     xInt = 0
                #     print("xikes")
                if(xInt < 800 and yInt < 800 and self._occupancy_map[xInt,yInt] >= self._min_probability):
                    for h in range(self._subsampling):
                        z_star[k]=dist
                        k+=1
                    break
                x_check += step*math.cos(i)
                y_check += step*math.sin(i)
                dist=math.sqrt((x_check-xs)**2+(y_check-ys)**2)
            # if(z_star[k]==0):
                # z_star[k]= self._max_range
            # k+=1
        return z_star

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        prob_zt1 = 1.0        
        K = 180 # z_t : array of 180 range measurements for each laser scan
        z_star= self.raycasting(x_t1) #array of occupancy map based distances for all 180 rays
        for k in range(K):
            # p = self.prob(z_star[k], z_t1_arr[k])
            p = self.getProbability(z_star[k], z_t1_arr[k])
            prob_zt1 *= p
        return prob_zt1
