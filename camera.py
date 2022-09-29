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
import yaml

from utils import DTYPE
from scipy import stats

class BlockDetections():
    def __init__(self):
        self.detected_num = 0
        self.small_num = 0
        self.contours = None
        self.colors = None # range from 0 to 6 w.r. to rainbow color order
        self.thetas = None
        self.sizes = None # 1 for small and 0 for large
        self.uvds = None
        self.xyzs = None
        self.all_contours = None

    def update(self):
        self.contours = np.array(self.contours)
        self.uvds = np.array(self.uvds, dtype=int)
        self.xyzs = np.array(self.xyzs, dtype=DTYPE)
        self.colors = np.array(self.colors, dtype=int)
        self.thetas = np.array(self.thetas, dtype=DTYPE)
        self.sizes = np.array(self.sizes, dtype=int)
        self.detected_num = len(self.contours)
        self.sort()
        # self.colors = 
        # self.thetas = thetas

    def reset(self):
        self.detected_num = 0
        self.small_num = 0
        self.contours = []
        self.colors = []
        self.thetas = []
        self.sizes = []
        self.uvds = []
        self.xyzs = []
        self.all_contours = None

    def _sort_by_idx(self, indices, begin, end):
        self.contours[begin:end] = self.contours[begin:end][indices]
        self.colors[begin:end] = self.colors[begin:end][indices]
        self.thetas[begin:end] = self.thetas[begin:end][indices]
        self.sizes[begin:end] = self.sizes[begin:end][indices]
        self.uvds[begin:end] = self.uvds[begin:end][indices]
        self.xyzs[begin:end] = self.xyzs[begin:end][indices]

    def sort(self):
        """
        Sort blocks by color and size
        size: small to large
        color: rainbow color order
        """
        small_to_large = np.argsort(self.sizes)
        self._sort_by_idx(small_to_large, 0, self.detected_num)
        small_num = np.bincount(self.sizes)[0]
        s_rainbow_order = np.argsort(self.colors[0:small_num])
        self._sort_by_idx(s_rainbow_order, 0, small_num)
        l_rainbow_order = np.argsort(self.colors[small_num:])
        self._sort_by_idx(l_rainbow_order, small_num, self.detected_num)
        self.small_num = small_num
        print(self.colors)


class Camera():
    """!
    @brief      This class describes a camera.
    """
    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280), dtype=np.uint16)
        self.colorReceived = False
        self.depthReceived = False
        self.ProcessVideoFrame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.ProcessVideoFrameLab = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.ProcessDepthFrameRaw = np.zeros((720, 1280), dtype=np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3), dtype=np.uint8)
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
        self.block_detections = BlockDetections()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color_id = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']
        self.color_rgb_mean = np.array([[127, 19, 30],
                                        [164, 66, 5],
                                        [218, 180, 30],
                                        [43, 118, 85],
                                        [0, 65, 117],
                                        [65, 45, 73],
                                        [147, 45, 70]],
                                        dtype=np.uint8)
        self.color_lab_mean = cv2.cvtColor(self.color_rgb_mean[:,None,:], cv2.COLOR_RGB2LAB).squeeze()
        self.size_id = ["large", "small"]

        self.loadCameraCalibration("config/camera_calib.yaml")

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.rectangle(self.ProcessVideoFrame, (275,120),(1100,720), (255, 0, 0), 2)
        cv2.rectangle(self.ProcessVideoFrame, (575,400),(750,720), (255, 0, 0), 2)
        if self.block_detections.detected_num < 1:
            return
        for idx, (color, pixel, point, size, theta) in enumerate(zip(self.block_detections.colors, self.block_detections.uvds, self.block_detections.xyzs, self.block_detections.sizes, self.block_detections.thetas)):
            cx, cy = pixel[:2]
            cv2.putText(self.ProcessVideoFrame, self.color_id[color], (cx-30, cy+30), self.font, 0.5, (0,0,0), thickness=2)
            cv2.putText(self.ProcessVideoFrame, self.size_id[size], (cx-30, cy+45), self.font, 0.5, (0,0,0), thickness=2)
            # cv2.putText(self.ProcessVideoFrame, "+", (cx-12, cy+8), self.font, 1, (0,0,0), thickness=2)
            # cv2.putText(self.ProcessVideoFrame, str(idx), (cx-30, cy+75), self.font, 1, (0,0,0), thickness=2)
            cv2.putText(self.ProcessVideoFrame, str(int(np.rad2deg(theta))), (cx, cy), self.font, 0.5, (255,255,255), thickness=1)
            # cv2.putText(self.ProcessVideoFrame, "%.0f"%(point[2]), (cx-20, cy+55), self.font, 0.5, (0,0,0), thickness=2)

        cv2.drawContours(self.ProcessVideoFrame, self.block_detections.all_contours, -1, (255, 0, 0), 1)
        cv2.drawContours(self.ProcessVideoFrame, self.block_detections.contours, -1, (0, 0, 255), 1)

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
            frame = cv2.resize(self.ProcessVideoFrame, (1280, 720))
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
        pts1 = coord1[0:3].astype(DTYPE)
        pts2 = coord2[0:3].astype(DTYPE)
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
            self.intrinsic_matrix = np.asarray(data["camera_matrix"]["data"], dtype=DTYPE).reshape((3, 3))
            self.distortion_coefficients = np.asarray(data["distortion_coefficients"]["data"], dtype=DTYPE).reshape(-1)
        else:
            self.intrinsic_matrix = np.array([925.27515, 0.0, 653.75928, 
                                            0.0, 938.70001, 367.99236, 
                                            0.0, 0.0, 1.0], dtype=DTYPE).reshape((3, 3))
        self.extrinsic_matrix_inv = np.array([1,0,0,-20,
                                            0, -1, 0, 211,
                                            0, 0, -1, 974,
                                            0, 0, 0, 1], dtype=DTYPE).reshape((4, 4))
        self.extrinsic_matrix = np.linalg.pinv(self.extrinsic_matrix_inv)

        self.intrinsic_matrix_inv = np.linalg.pinv(self.intrinsic_matrix)

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass

    def detectBlocksInDepthImage(self, _lower=700, _upper=960, blind_rect=None):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        self.block_detections.reset()
        lower = _lower
        upper = _upper
        """mask out arm & outside board"""
        # self.ProcessDepthFrameRaw = cv2.GaussianBlur(self.ProcessDepthFrameRaw, (5, 5), 3)
        self.ProcessDepthFrameRaw = cv2.medianBlur(self.ProcessDepthFrameRaw, 3)
        mask = np.zeros_like(self.ProcessDepthFrameRaw, dtype=np.uint8)
        # !!! Attention to these rectangles's range
        cv2.rectangle(mask, (275,120),(1100,720), 255, cv2.FILLED)
        cv2.rectangle(mask, (575,400),(750,720), 0, cv2.FILLED)
        if blind_rect is not None:
            cv2.rectangle(mask, blind_rect[0], blind_rect[1], 0, cv2.FILLED)

        img_depth_thr = cv2.bitwise_and(cv2.inRange(self.ProcessDepthFrameRaw, lower, upper), mask)
        # depending on your version of OpenCV, the following line could be:
        contours, _ = cv2.findContours(img_depth_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        # _, contours, _ = cv2.findContours(img_depth_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.block_detections.all_contours = contours

        cv2.drawContours(self.ProcessVideoFrame, contours, -1, (255, 0, 0), 1)

        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] < 200 or abs(M["m00"]) > 7000:
                # reject false positive detections by area size
                continue
            mask_single = np.zeros_like(self.ProcessDepthFrameRaw, dtype=np.uint8)
            cv2.drawContours(mask_single, [contour], -1, 255, cv2.FILLED)
            depth_single = cv2.bitwise_and(self.ProcessDepthFrameRaw, self.ProcessDepthFrameRaw, mask=mask_single)
            depth_array = depth_single[depth_single>0]
            mode, count = stats.mode(depth_array)
            # !!! Attention to the mode offset, it determines how much of the top surface area will be reserved
            depth_new = cv2.inRange(depth_single, lower, int(mode)+4)
            contours_new, _ = cv2.findContours(depth_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            contours_new_valid = max(contours_new, key=cv2.contourArea) # find the largest contour
            M = cv2.moments(contours_new_valid)
            if abs(M["m00"]) < 200:
                # reject false positive detections by area size
                continue
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cz = self.ProcessDepthFrameRaw[cy, cx]
            # !!! size classification: attention to this moment threshold
            if M["m00"] < 800:
                self.block_detections.sizes.append(1) # 1 for small
            else:
                self.block_detections.sizes.append(0) # 0 for large
            block_ori = - cv2.minAreaRect(contours_new_valid)[2] # turn the range from [-90, 0) to (0, 90]
            self.block_detections.uvds.append([cx, cy, cz])
            self.block_detections.xyzs.append(self.coor_pixel_to_world(cx, cy, cz))
            self.block_detections.contours.append(contours_new_valid)
            self.block_detections.thetas.append(np.deg2rad(block_ori))
            self.block_detections.colors.append(self.retrieve_area_color(self.ProcessVideoFrameLab, contours_new_valid))

        self.block_detections.update()

    def retrieve_area_color(self, frame, contour):
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        mean = np.array(cv2.mean(frame, mask=mask)[:3], dtype=DTYPE)
        # dist = self.color_rgb_mean - mean
        dist = self.color_lab_mean - mean
        dist_norm = np.linalg.norm(dist, axis=1)

        # * Let's directly return color index for easy sorting
        # return self.color_id[np.argmin(dist_norm)]
        return np.argmin(dist_norm)

    def coor_pixel_to_world(self, u, v, z):
        index = np.array([u, v, 1]).reshape((3,1))
        pos_camera = z * np.matmul(self.intrinsic_matrix_inv, index)
        temp_pos = np.array([pos_camera[0][0], pos_camera[1][0], pos_camera[2][0], 1]).reshape((4,1))
        world_pos = np.matmul(self.extrinsic_matrix_inv, temp_pos)
        return world_pos.flatten()


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
        # self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        self.camera.intrinsic_matrix = np.asarray(data.K, dtype=DTYPE).reshape((3,3))
        #print(self.camera.intrinsic_matrix)


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
        self.camera.VideoFrame = cv_image # TODO try .copy() here
        self.camera.colorReceived = True


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
        self.camera.depthReceived = True


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        # camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        # camera_info_listener = CameraInfoListener(camera_info_topic,
        #                                           self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            rospy.sleep(0.5)
        while True:
            if self.camera.colorReceived and self.camera.depthReceived:
                self.camera.ProcessVideoFrame = self.camera.VideoFrame.copy()
                self.camera.ProcessVideoFrameLab = cv2.cvtColor(self.camera.ProcessVideoFrame, cv2.COLOR_RGB2LAB)
                self.camera.ProcessDepthFrameRaw = self.camera.DepthFrameRaw.copy()
                # self.camera.detectBlocksInDepthImage()
                self.camera.processVideoFrame()
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(rgb_frame, depth_frame, tag_frame)

            rospy.sleep(0.04)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.ProcessVideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(3)
                rospy.sleep(0.03)


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
