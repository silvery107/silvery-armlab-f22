#!/usr/bin/python
"""!
Test the gripper

TODO: Use this file and modify as you see fit to test gripper
"""
pressure = 0.5 # gripping pressure (0-1)
grip_time = 1.0 # toggle time in s

import time
import os
import sys
import signal
script_path = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.realpath(script_path + '/../'))
print(script_path)
from rxarm import RXArm

rexarm = RXArm()
rexarm.enable_torque()
# Test non-blocking versions
rexarm.set_gripper_pressure(pressure)
time.sleep(1)

def signal_handler(signal, frame):
    rexarm.disable_torque()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    while True:
        rexarm.open_gripper()
        time.sleep(grip_time)
        rexarm.close_gripper()
        time.sleep(grip_time)

main()
