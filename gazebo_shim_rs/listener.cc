/*
 * Copyright (C) 2012 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/

#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo_client.hh>
#include "ros/ros.h"
#include "std_msgs/Header.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/RegionOfInterest.h"

#include <boost/array.hpp>
#include <signal.h>
#include <iostream>

#define GAZEBO_DEPTH_TOPIC_NAME "~/realsense_cam/rs/stream/depth"
#define GAZEBO_IMAGE_TOPIC_NAME "~/realsense_cam/rs/stream/color"

#define ROS_DEPTH_TOPIC_NAME "/camera/aligned_depth_to_color/image_raw"
#define ROS_IMAGE_TOPIC_NAME "/camera/color/image_raw"
#define ROS_CAMERA_INFO_TOPIC_NAME "/camera/color/camera_info"

// global because I am lazy
ros::Publisher pub_depth;
ros::Publisher pub_color;
ros::Publisher pub_cam_info;


uint32_t seq = 0;
uint32_t depth_received = 0;
uint32_t color_received = 0;

_Float64 _D [5] = {0.0, 0.0, 0.0, 0.0, 0.0};
boost::array<_Float64, 9> _K = {610, 0.0, 320, 0.0, 610, 240, 0.0, 0.0, 1.0};
boost::array<_Float64, 9> _R = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
boost::array<_Float64, 12> _P = {610, 0.0, 320, 0.0, 0.0, 610, 240, 0.0, 0.0, 0.0, 1.0, 0.0};


// adding this to make things exit nicely
void sigintHandler(int sig)
{
  gazebo::client::shutdown();
  ros::shutdown();
}

/////////////////////////////////////////////////
// Function is called everytime a message is received.
void cb_depth(ConstImageStampedPtr &_msg)
{
  std::cout << "\rmsgs received: " << depth_received++ << " depth, " << color_received << " color" << std::flush;

  sensor_msgs::Image to_send;

  std_msgs::Header to_send_header;
  to_send_header.seq = seq++;
  to_send_header.stamp = ros::Time::now();
  to_send_header.frame_id = "camera_color_optical_frame";

  to_send.header = to_send_header;
  to_send.width = (int) _msg->image().width();
  to_send.height = (int) _msg->image().height();
  to_send.encoding = "16UC1";
  to_send.is_bigendian = 0;

  auto data = new unsigned char[_msg->image().data().length() + 1];
  memcpy(data, _msg->image().data().c_str(), _msg->image().data().length());

  to_send.step = _msg->image().width() * 2; //_msg->image().data().length() / _msg->image().height();
  to_send.data = std::vector<unsigned char>(data, data + _msg->image().data().length() + 1);

  pub_depth.publish(to_send);

  delete data;
}

void pub_camInfo(uint32_t seq_in, ros::Time ts_in)
{
  sensor_msgs::CameraInfo to_send;

  std_msgs::Header to_send_header;
  to_send_header.seq = seq_in;
  to_send_header.stamp = ts_in;
  to_send_header.frame_id = "camera_color_optical_frame";

  to_send.header = to_send_header;
  to_send.height = 480;
  to_send.width = 640;
  to_send.distortion_model = "plumb_bob";
  to_send.D = std::vector<_Float64>(_D, _D + 5);
  to_send.K = _K;
  to_send.R = _R;
  to_send.P = _P;
  to_send.binning_x = 0;
  to_send.binning_y = 0;

  sensor_msgs::RegionOfInterest to_send_roi;
  to_send_roi.x_offset = 0;
  to_send_roi.y_offset = 0;
  to_send_roi.height = 0;
  to_send_roi.width = 0;
  to_send_roi.do_rectify = false;

  to_send.roi = to_send_roi;

  pub_cam_info.publish(to_send);
}

void cb_color(ConstImageStampedPtr &_msg)
{
  std::cout << "\rmsgs received: " << depth_received << " depth, " << color_received++ << " color" << std::flush;

  sensor_msgs::Image to_send;

  std_msgs::Header to_send_header;
  to_send_header.seq = seq;
  auto ts = ros::Time::now();
  to_send_header.stamp = ts;
  to_send_header.frame_id = "camera_color_optical_frame";

  to_send.header = to_send_header;
  to_send.width = (int) _msg->image().width();
  to_send.height = (int) _msg->image().height();
  to_send.encoding = "rgb8";
  to_send.is_bigendian = 0;

  auto data = new char[_msg->image().data().length()];
  memcpy(data, _msg->image().data().c_str(), _msg->image().data().length());

  to_send.step = _msg->image().width() * 3;
  to_send.data = std::vector<unsigned char>(data, data + _msg->image().data().length());
  printf("data size: %ld",  _msg->image().data().length());
  pub_color.publish(to_send);

  pub_camInfo(seq++, ts);

  delete data;
}

/////////////////////////////////////////////////
int main(int _argc, char **_argv)
{
  // sigint handler to shutdown both ros and gazebo nodes
  signal(SIGINT, sigintHandler);
  
  // Load gazebo
  gazebo::client::setup(_argc, _argv);

  // Create our nodes for communication in both Gazebo and ROS
  gazebo::transport::NodePtr node(new gazebo::transport::Node());
  node->Init();
  ros::init(_argc, _argv, "rs_shim");
  ros::NodeHandle nh;

  // Listen to Gazebo world_stats topic
  gazebo::transport::SubscriberPtr sub_depth = node->Subscribe(GAZEBO_DEPTH_TOPIC_NAME, cb_depth);
  gazebo::transport::SubscriberPtr sub_color = node->Subscribe(GAZEBO_IMAGE_TOPIC_NAME, cb_color);

  // Create ROS publishers
  pub_depth = nh.advertise<sensor_msgs::Image>(ROS_DEPTH_TOPIC_NAME, 1000);
  pub_color = nh.advertise<sensor_msgs::Image>(ROS_IMAGE_TOPIC_NAME, 1000);
  pub_cam_info = nh.advertise<sensor_msgs::CameraInfo>(ROS_CAMERA_INFO_TOPIC_NAME, 1000);

  // enter the waiting loop
  ros::spin();

  sigintHandler(0);
}
