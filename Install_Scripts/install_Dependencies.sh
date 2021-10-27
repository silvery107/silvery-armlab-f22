#!/bin/bash
#
# Install Script for Armlab
#

#initial update
sudo apt-get update
sudo apt-get upgrade

# dev stuff / code tools
sudo apt-get -y install curl wget build-essential cmake dkms \
    git autoconf automake autotools-dev gdb libglib2.0-dev libgtk2.0-dev \
    libusb-dev libusb-1.0-0-dev freeglut3-dev libboost-dev libgsl-dev \
    net-tools doxygen

wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
sudo apt-get update
sudo apt install code

sudo apt-get -y install qtcreator qt4-default qt4-dev-tools

# python 2 stuff
sudo apt-get -y install python-dev python-pip python-cairo \
    python-pygame python-matplotlib python-numpy python-scipy python-pyaudio \
    python-tk ipython pyqt4-dev-tools python-opencv cython

#install some python packages
sudo pip install future
sudo pip install modern_robotics

#Install ROS
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install -y ros-melodic-desktop-full
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool
sudo rosdep init
rosdep update

#install realsense software
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
sudo apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg ros-melodic-realsense2-camera

#install apriltags
sudo apt-get install -y ros-melodic-apriltag-ros
