#!/usr/bin/python
"""!
Test kinematics

TODO: Use this file and modify as you see fit to test kinematics.py
"""
import argparse
import sys
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.realpath(script_path + '/../'))
from kinematics import *
from config_parse import *
from copy import deepcopy

np.set_printoptions(precision=3, suppress=True)
if __name__ == '__main__':
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-c", "--dhconfig", required=True, help="path to DH parameters csv file")

    # args=vars(ap.parse_args())

    passed = True
    vclamp = np.vectorize(clamp)

    dh_params = parse_dh_param_file("../config/rx200_dh.csv")
    M_matrix, S_vectors = parse_pox_param_file("../config/rx200_pox.csv")

    ### Add arm configurations to test here
    fk_angles = np.array([[0.0,           0.0,            0.0,               0.0],
                          [np.pi,           0.0,            0.0,               0.0]])
    
    print('Test FK')
    fk_poses = []
    for joint_angles in fk_angles:
        print('Joint angles:', joint_angles)
        # for i, _ in enumerate(joint_angles):
        #     pose = get_pose_from_T(FK_dh(deepcopy(dh_params), joint_angles, i))
        #     print('Link {} pose: {}'.format(i, pose))
        #     if i == len(joint_angles) - 1:
        #         fk_poses.append(pose)
        pose = FK_pox(joint_angles, M_matrix, S_vectors)
        print("Joint pose:  ", np.asarray(pose))
        print()

    print('Test IK')
    for pose, angles in zip(fk_poses, fk_angles):
        matching_angles = False
        print('Pose: {}'.format(pose))
        options = IK_geometric(deepcopy(dh_params), pose)
        for i, joint_angles in enumerate(options):
            print('Option {}: {}'.format(i, joint_angles))
            compare = vclamp(joint_angles - angles)
            if np.allclose(compare, np.zeros_like(compare), rtol=1e-3, atol=1e-4):
                print('Option {} matches angles used in FK'.format(i))
                matching_angles = True
        if not matching_angles:
            print('No match to the FK angles found!')
            passed = False
        print()
