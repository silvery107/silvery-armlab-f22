#!/bin/bash
#install interbotix-robot-arm
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source /opt/ros/melodic/setup.bash
mkdir -p ~/interbotix_ws/src
cd ~/interbotix_ws/
catkin_make
echo "source ~/interbotix_ws/devel/setup.bash" >> ~/.bashrc
source ~/interbotix_ws/devel/setup.bash
cd ~/interbotix_ws/src
git clone https://github.com/Interbotix/interbotix_ros_arms.git
cd ~/interbotix_ws/src/interbotix_ros_arms
git checkout melodic
cd ~/interbotix_ws/
rosdep update
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source ~/.bashrc
sudo cp ~/interbotix_ws/src/interbotix_ros_arms/interbotix_sdk/10-interbotix-udev.rules /etc/udev/rules.d
sudo udevadm control --reload-rules && udevadm trigger
