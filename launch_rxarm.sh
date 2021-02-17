#!/bin/bash
gnome-terminal -- roslaunch interbotix_sdk arm_run.launch robot_name:=rx200 use_time_based_profile:=true gripper_operating_mode:=pwm
sleep 5

