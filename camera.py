"""!
Class to represent the camera.
"""

import cv2
import time
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError

import yaml # pip install pyyaml


class Camera():
    """!
    @brief      This class describes a camera.
    """
    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.array([])
        self.extrinsic_matrix = np.array([])
        self.distortion_coefficients = np.array([])
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = list((
            {'id': 'red', 'color': (10, 10, 127)},
            {'id': 'orange', 'color': (30, 75, 150)},
            {'id': 'yellow', 'color': (30, 150, 200)},
            {'id': 'green', 'color': (20, 60, 20)},
            {'id': 'blue', 'color': (100, 50, 0)},
            {'id': 'violet', 'color': (100, 40, 80)})
        )

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file=None):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        if file is not None:
            data = None
            with open(file, "r") as stream:
                data = yaml.safe_load(stream)
            assert (data is not None)
            self.intrinsic_matrix = np.asarray(data["camera_matrix"]["data"], dtype=np.float32).reshape((3, 3))
            self.distortion_coefficients = np.asarray(data["distortion_coefficients"]["data"], dtype=np.float32).reshape((1, 5))
        else:
            self.intrinsic_matrix = np.array([925.27515, 0.0, 653.75928, 
                                            0.0, 938.70001, 367.99236, 
                                            0.0, 0.0, 1.0], dtype=np.float32).reshape((3, 3))
            self.extrinsic_matrix_inv = np.array([1,0,0,-20,
                                                0, -1, 0, 211,
                                                0, 0, -1, 974,
                                                0, 0, 0, 1], dtype=np.float32).reshape((4, 4))
            self.extrinsic_matrix = np.linalg.pinv(self.extrinsic_matrix_inv)

        self.intrinsic_matrix_inv = np.linalg.pinv(self.intrinsic_matrix)

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass

    def detectBlocksInDepthImage(self, _lower=905, _upper=973):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        # !!! Attention, one set of lower and upper only coorespond to one level of blocks
        lower = _lower
        upper = _upper
        cnt_image = self.VideoFrame
        """mask out arm & outside board"""
        mask = np.zeros_like(self.DepthFrameRaw, dtype=np.uint8)
        # !!! Attention to these rectangles's range
        cv2.rectangle(mask, (275,120),(1100,720), 255, cv2.FILLED)
        cv2.rectangle(mask, (575,414),(723,720), 0, cv2.FILLED)
        cv2.rectangle(cnt_image, (275,120),(1100,720), (255, 0, 0), 2)
        cv2.rectangle(cnt_image, (575,414),(723,720), (255, 0, 0), 2)

        img_depth_thr = cv2.bitwise_and(cv2.inRange(self.DepthFrameRaw, lower, upper), mask)
        # depending on your version of OpenCV, the following line could be:
        # contours, _ = cv2.findContours(img_depth_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, _ = cv2.findContours(img_depth_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.block_contours = contours
        # cv2.drawContours(self.VideoFrame, contours, -1, (0,255,255), thickness=1)
        block_detection_single = []
        for contour in contours:
            color = self.retrieve_area_color(contour)
            theta = cv2.minAreaRect(contour)[2]
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.putText(self.VideoFrame, color, (cx-30, cy+40), self.font, 1.0, (0,0,0), thickness=2)
            cv2.putText(self.VideoFrame, str(int(theta)), (cx, cy), self.font, 0.5, (255,255,255), thickness=2)
            # !!! Attention, no block height is taken into consideration for now
            block_detection_single.append([cx, cy, 0])
        
        self.block_detections = np.asarray(block_detection_single)
        print(self.block_detections.shape) # nx3
        print(self.block_detections)

    def retrieve_area_color(self, contour):
        mask = np.zeros(self.VideoFrame.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean = cv2.mean(self.VideoFrame, mask=mask)[:3]
        min_dist = (np.inf, None)
        for label in self.colors:
            d = np.linalg.norm(label["color"] - np.array(mean))
            if d < min_dist[0]:
                min_dist = (d, label["id"])
        return min_dist[1] 


class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image
        

class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data
        # for detection in data.detections:
        #     print(detection.id[0], type(detection.id[0]))
        #     print(detection.pose.pose.pose.position)


class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        #print(self.camera.intrinsic_matrix)


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        while True:
            self.camera.processVideoFrame()
            self.camera.detectBlocksInDepthImage()
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(rgb_frame, depth_frame, tag_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(3)
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
