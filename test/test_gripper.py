#!/usr/bin/python3
"""!
Test the gripper

TODO: Use this file and modify as you see fit to test gripper
"""
import time
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.realpath(script_path + '/../')
from rxarm import RXarm

rexarm = RXArm()

# Test non-blocking versions
rxarm.open_gripper()
time.sleep(1)
rxarm.close_gripper()
time.sleep(1)

rxarm.disable_torque()
time.sleep(1)
