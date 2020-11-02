#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageListener:
  def __init__(self, topic):
    self.topic = topic
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber(topic,Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
    except CvBridgeError as e:
      print(e)
    (rows,cols,channels) = cv_image.shape
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

class DepthListener:
  def __init__(self, topic):
    self.topic = topic
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber(topic,Image,self.callback)

  def callback(self,data):
    try:
      cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding) << 6
    except CvBridgeError as e:
      print(e)
    (rows,cols) = cv_depth.shape
    cv2.imshow("Depth window", cv_depth)
    cv2.waitKey(3)

def main(args):
  cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
  cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
  image_topic = "/camera/color/image_raw"
  depth_topic = "/camera/aligned_depth_to_color/image_raw"
  image_listener = ImageListener(image_topic)
  depth_listener = DepthListener(depth_topic)
  rospy.init_node('realsense_viewer', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)