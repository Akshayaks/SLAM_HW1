'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.stats import norm
import pdb
# from random import seed
# from random import random

# seed(1)
class Resampling:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """
        self.dice_count = 1

    def multinomial_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        weights = X_bar[:, 3]
        total_num = len(weights)
        weights = weights / np.sum(weights)
        resampled_index = np.random.multinomial(self.dice_count*total_num, weights, size=1)[0, :]
        print('resampled_index : ', resampled_index.shape)
        try:
            X_bar_resampled = X_bar[resampled_index, :]
        except:
            ipdb.set_trace()

        return X_bar_resampled

    def low_variance_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        weights = X_bar[:, 3]
        
        num_particles = X_bar.shape[0]
        X_bar_resampled = np.zeros_like(X_bar)
        weights = weights/weights.sum()
        r = np.random.uniform(0, 1/num_particles)
        c = weights[0]
        i = 0

        for m in range(0,num_particles):
            u = r + ((m*1.0)/num_particles)
            while(u > c):
                i = i + 1
                c = c + weights[i]

            X_bar_resampled[m,:] =  X_bar[i,:]

        return X_bar_resampled



    def low_variance_sampler_augmented(self, occupancy_map, obs_threshold, X_bar, wt_f, wt_s, num_particles):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        # weights = X_bar[:, 3]
        
        # num_particles = X_bar.shape[0]
        # X_bar_resampled = np.zeros_like(X_bar)
        # weights = weights/weights.sum()
        # r = np.random.uniform(0, 1/num_particles)
        # c = weights[0]
        # i = 0

        # for m in range(0,num_particles):
        #     u = r + ((m*1.0)/num_particles)
        #     while(u > c):
        #         i = i + 1
        #         c = c + weights[i]

            # X_bar_resampled[m,:] =  X_bar[i,:]


        return X_bar_resampled

    def low_variance_sampler_kidnapped_robot(self, map, obs_threshold, X_bar, X_prev, change, wt_f, wt_s, num_particles):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        # prev_x = np.sum(X_prev[:,0])/num_particles
        # prev_y = np.sum(X_prev[:,1])/num_particles
        # prev_theta = np.sum(X_prev[:,2])/num_particles
        # print("previous position: ", prev_x, prev_y)

        weights = X_bar[:, 3]        
        
        num_particles = X_bar.shape[0]
        
        weights = weights/weights.sum()
        r = np.random.uniform(0, 1/num_particles)
        c = weights[0]

        X_bar_resampled = X_bar.tolist()
        i = 0

        for m in range(num_particles):
            rand_no = np.random.uniform(0,1)
            k = 0
            # r_part = 0
            x_pos = 0
            y_pos = 0 
            theta = 0 
            wt = 0 
            if rand_no < max(0.0,1.0 - wt_f/wt_s):

                while 1:
                    # x_pos = 3000 + np.random.uniform(0,1)*2000
                    # y_pos = np.random.uniform(0,1)*7000
                    x_pos = np.random.normal(prev_x,2*change)
                    y_pos = np.random.normal(prev_y,2*change)
                    theta = np.random.normal(prev_theta,2*chance)
                    # theta = np.random.uniform(-3.14,3.14)
                    wt = 1/num_particles #np.random.uniform(0,1)
                    # print(x_pos,y_pos)
                

                    if map[int(round(y_pos/10))][int(round(x_pos/10))] == 0:
                        # X_bar_resampled.append([x_pos, y_pos, theta, wt])
                        # X_bar.append([x_pos, y_pos, theta, wt])
                        # weights.append(wt)
                        # c += wt
                        # r_part += 1
                        # k += 1
                        continue
                    else:
                        break
                X_bar_resampled.append([x_pos, y_pos, theta, wt])

            
            u = r + ((m*1.0)/num_particles)
            while(u > c and i < num_particles - 1):
                i = i + 1
                c = c + weights[i]

            X_bar_resampled.append(X_bar[i,:])

        return np.array(X_bar_resampled)

    def add_particles(X_bar_resampled):

        mean_weight = np.mean(X_bar_resampled[:,3])

        if(mean_weight < 0.1):
            ## basically all particles are bad
            pass


if __name__ == "__main__":
    pass

