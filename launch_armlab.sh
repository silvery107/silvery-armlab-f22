#!/bin/bash
gnome-terminal -- roslaunch realsense2_camera rs_l515.launch align_depth:=true
sleep 3
gnome-terminal -- roslaunch apriltag_ros continuous_detection.launch camera_name:=/camera/color/ image_topic:=image_raw
sleep 3
gnome-terminal -- roslaunch interbotix_sdk arm_run.launch robot_name:=rx200 use_time_based_profile:=true gripper_operating_mode:=pwm
sleep 5

