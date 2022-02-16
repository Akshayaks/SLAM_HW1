'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math
import matplotlib.pyplot as plt

class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.0001
        self._alpha2 = 0.0001
        self._alpha3 = 0.01
        self._alpha4 = 0.01


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        del_rot1 = math.atan2(u_t1[1] - u_t0[1],u_t1[0] - u_t0[0]) - u_t0[2]
        del_trans = math.sqrt((u_t0[0]-u_t1[0])**2 + (u_t0[1]-u_t1[1])**2)
        del_rot2 = u_t1[2] - u_t0[2] - del_rot1

        del_rot1_cap = del_rot1 - self.sample_normal( (self._alpha1*(del_rot1**2)) + (self._alpha2*(del_trans**2)))
        del_trans_cap = del_trans - self.sample_normal( (self._alpha3*(del_trans**2)) + (self._alpha4*(del_rot1**2)) + (self._alpha4*(del_rot2**2)) )
        del_rot2_cap = del_rot2 - self.sample_normal( (self._alpha1*(del_rot2**2)) + (self._alpha2*(del_trans**2)) )

        x_p = np.zeros(3)
        x_p[0] = x_t0[0] + del_trans_cap * np.cos(x_t0[2] + del_rot1_cap)
        x_p[1] = x_t0[1] + del_trans_cap * np.sin(x_t0[2] + del_rot1_cap)
        x_p[2] = x_t0[2] + del_rot1_cap + del_rot2_cap
        return x_p

    def update_vectorized(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        num = x_t0.shape[0]
        ones = np.ones(num)
        del_rot1 = ones * (np.arctan2( np.subtract(u_t1[1], u_t0[1]) ,np.subtract(u_t1[0], u_t0[0])) - u_t0[2])
        del_trans = ones * (np.sqrt((u_t0[0]-u_t1[0])**2 + (u_t0[1]-u_t1[1])**2))
        del_rot2 = ones *(u_t1[2] - u_t0[2] - del_rot1)

        del_rot1_cap = del_rot1 - self.sample_normal( (self._alpha1*(del_rot1**2)) + (self._alpha2*(del_trans**2)), num)
        del_trans_cap = del_trans - self.sample_normal( (self._alpha3*(del_trans**2)) + (self._alpha4*(del_rot1**2)) + (self._alpha4*(del_rot2**2)), num )
        del_rot2_cap = del_rot2 - self.sample_normal( (self._alpha1*(del_rot2**2)) + (self._alpha2*(del_trans**2)) , num)

        x_p = np.zeros_like(x_t0)
        x_p[:,0] = x_t0[:,0] + del_trans_cap * np.cos(x_t0[:,2] + del_rot1_cap)
        x_p[:,1] = x_t0[:,1] + del_trans_cap * np.sin(x_t0[:,2] + del_rot1_cap)
        x_p[:,2] = x_t0[:,2] + del_rot1_cap + del_rot2_cap
        return x_p


    def sample_normal(self,b, num): # sample from a distribution with 0 mean variance = b
        val = np.random.normal(0,np.sqrt(b))
        return val
