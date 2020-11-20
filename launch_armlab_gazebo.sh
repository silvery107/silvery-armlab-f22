#!/bin/bash
gnome-terminal -- roslaunch interbotix_gazebo gazebo.launch robot_name:=rx200
sleep 5
gnome-terminal -- python ./gazebo_shim.py
sleep 5
gnome-terminal -- ./gazebo_shim_rs/build/listener
sleep 5
gnome-terminal -- roslaunch apriltag_ros continuous_detection.launch camera_name:=/camera/color/ image_topic:=image_raw
sleep 3
echo "all dependent programs launched. Press play in Gazebo and the launch your control station!"