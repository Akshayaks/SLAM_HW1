import sys
import numpy as np
import math

class MotionModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self):

        """
        TODO : Initialize Motion Model parameters here
        """

        self.alpha_1 = 0.0001
        self.alpha_2 = 0.0001
        self.alpha_3 = 0.01
        self.alpha_4 = 0.01


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

        b1 = (self.alpha_1*(del_rot1**2)) + (self.alpha_2*(del_trans**2)) 
        b2 = (self.alpha_3*(del_trans**2)) + (self.alpha_4*(del_rot1**2)) + (self.alpha_4*(del_rot2**2))
        b3 = (self.alpha_1*(del_rot2**2)) + (self.alpha_2*(del_trans**2))

        del_rot1_h = del_rot1 - np.random.normal(0,np.sqrt(b1))
        del_trans_h = del_trans - np.random.normal(0,np.sqrt(b2))
        del_rot2_h = del_rot2 - np.random.normal(0,np.sqrt(b3))

        x_t1 = np.zeros(3)
        x_t1[0] = x_t0[0] + del_trans_h*math.cos(x_t0[2] + del_rot1_h)
        x_t1[1] = x_t0[1] + del_trans_h*math.sin(x_t0[2] + del_rot1_h)
        x_t1[2] = x_t0[2] + del_rot1_h + del_rot2_h

        return x_t1

if __name__=="__main__":
    pass



