'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os
import random

from map_reader import MapReader
# from motion_model_new import MotionModel
from motion_model import MotionModel
from sensor_model_ak import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time
import math

t = 666
random.seed(t)
np.random.seed(t)

def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    # x_locs = np.random.uniform(0,800,500)
    # y_locs = np.random.uniform(0,800,500)

    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.0001)
    scat.remove()


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()  #Gives probability of each cell being occupied
    print("occupancy_map: ", occupancy_map[400][400])
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    #if args.visualize:
    #    visualize_map(occupancy_map)

    first_time_idx = True

    X = [400, 386, 467, 621]
    Y = [400, 230, 218, 142]
    fig = plt.figure()
    scat = plt.scatter(X, Y, c='r', marker='o')
    colors = ['b', 'g', 'c', 'y']
    robot_theta = [math.pi/2, 0, math.pi/4, math.pi]
    for i in range(0,4):
        for k in range(-90,90,5): #here K is 180
            z_k = sensor_model.plot_ray_cast(X[i], Y[i], robot_theta[i], k)
            x_plot = [z_k[0], X[i]]
            y_plot = [z_k[1], Y[i]]
            plt.plot(x_plot, y_plot, colors[i])
    mng = plt.get_current_fig_manager()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])
    plt.savefig('testing.jpg')
    plt.show()


    
