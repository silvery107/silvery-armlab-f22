#!/bin/bash
gnome-terminal -- roslaunch interbotix_gazebo gazebo.launch robot_name:=rx200
sleep 5
gnome-terminal -- python ./gazebo_shim.py
sleep 5
gnome-terminal -- ./gazebo_shim_rs/build/listener
sleep 5
echo "all dependent programs launched. Press play in Gazebo and the launch your control station!"