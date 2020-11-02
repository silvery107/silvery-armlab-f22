#!/usr/bin/python3
"""!
Test the rxarm

TODO: Use this file and modify as you see fit to test the rxarm. You can specify to use the trajectory planner and
which config file to use on the command line use -h from help.
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.realpath(script_path + '/../'))
import sys
import time
from rxarm import RXarm
import numpy as np
import argparse

# Parse cmd line
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trajectory_planner', action='store_true')
parser.add_argument('-c', '--config_file', type=str, default=script_path+'/../config/rxarm_config.csv')
args = parser.parse_args()

rxarm = rxarm(config_file=args.config_file)
rxarm.initialize()
if not rxarm.initialized:
    print('Failed to initialized the rxarm')
    sys.exit(-1)

rxarm.set_velocities_normalized_all(0.5)

tmp_waypoints = [
    [0.0,           0.0,            0.0,            0.0],
    [np.pi * 0.1,   0.0,            np.pi / 2,      0.0],
    [np.pi * 0.25,  np.pi / 2,      -np.pi / 2,     np.pi / 2],
    [np.pi * 0.4,   np.pi / 2,      -np.pi / 2,     0.0],
    [np.pi * 0.55,  0,              0,              0],
    [np.pi * 0.7,   0.0,            np.pi / 2,      0.0],
    [np.pi * 0.85,  np.pi / 2,      -np.pi / 2,     np.pi / 2],
    [np.pi,         np.pi / 2,      -np.pi / 2,     0.0],
    [0.0,           np.pi / 2,      np.pi / 2,      0.0],
    [np.pi / 2,     -np.pi / 2,     np.pi / 2,      0.0]]

waypoints = []
for wp in tmp_waypoints:
    full_wp = [0.0] * rxarm.num_joints
    full_wp[0:len(wp)] = wp
    waypoints.append(full_wp)

if args.trajectory_planner:
    tp = TrajectoryPlanner(rxarm)
    for wp in waypoints:
        tp.set_initial_wp()
        tp.set_final_wp(wp)
        tp.go(max_velocity=2.5)
        time.sleep(1.0)
else:
    for wp in waypoints:
        rxarm.set_positions(wp)
        time.sleep(1)

rxarm.disable_torque()
time.sleep(0.1)
