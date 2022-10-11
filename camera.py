"""!
Class to represent the camera.
"""

from random import random
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
# from cv_bridge import CvBridge, CvBridgeError
import yaml
import csv
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import stats

# TODO
# import torch
# from models.model import BlocksDataset

from utils import DTYPE

class BlockDetections():
    def __init__(self):
        self.detected_num = 0
        self.large_num = 0
        self.contours = None
        self.colors = None # range from 0 to 5 w.r. to rainbow color order
        self.thetas = None
        self.sizes = None # 1 for small and 0 for large
        self.uvds = None
        self.xyzs = None
        self.all_contours = None
        self.has_cluster = False

    def update(self, key="color"):
        self.contours = np.array(self.contours)
        self.uvds = np.array(self.uvds, dtype=int)
        self.xyzs = np.array(self.xyzs, dtype=DTYPE)
        self.colors = np.array(self.colors, dtype=int)
        self.thetas = np.array(self.thetas, dtype=DTYPE)
        self.sizes = np.array(self.sizes, dtype=int)
        self.detected_num = len(self.contours)
        self.sort(key)
        # self.colors = 
        # self.thetas = thetas

    def reset(self):
        self.detected_num = 0
        self.large_num = 0
        self.contours = []
        self.colors = []
        self.thetas = []
        self.sizes = []
        self.uvds = []
        self.xyzs = []
        self.all_contours = None
        self.has_cluster = False

    def _sort_by_idx(self, indices, begin, end):
        self.contours[begin:end] = self.contours[begin:end][indices]
        self.colors[begin:end] = self.colors[begin:end][indices]
        self.thetas[begin:end] = self.thetas[begin:end][indices]
        self.sizes[begin:end] = self.sizes[begin:end][indices]
        self.uvds[begin:end] = self.uvds[begin:end][indices]
        self.xyzs[begin:end] = self.xyzs[begin:end][indices]

    def sort(self, key="color"):
        """
        Sort blocks by color and size
        size: large to small
        color: rainbow color order
        """
        if self.detected_num==0:
            return

        if key == "color":
            large_to_small = np.argsort(self.sizes)
            self._sort_by_idx(large_to_small, 0, self.detected_num)
            large_num = np.bincount(self.sizes)[0]
            self.large_num = large_num
        
            l_rainbow_order = np.argsort(self.colors[0:large_num])
            self._sort_by_idx(l_rainbow_order, 0, large_num)
            s_rainbow_order = np.argsort(self.colors[large_num:])
            self._sort_by_idx(s_rainbow_order, large_num, self.detected_num)

        elif key == "distance":
            dist_xy = np.linalg.norm(self.xyzs[:, :2], axis=1)
            dist_order = np.argsort(dist_xy)
            self._sort_by_idx(dist_order, 0, self.detected_num)
    
        print("[BLOCKS] Find blocks with color {}".format(self.colors))


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
        self.ProcessVideoFrameHSV = np.zeros((720, 1280, 3), dtype=np.uint8)
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
        self.color_id = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        self.color_rgb_mean = np.array([[127, 19, 30],
                                        [164, 66, 5],
                                        [218, 180, 30],
                                        [30, 110, 60],
                                        [5, 60, 110],
                                        [50, 50, 73]],
                                        dtype=np.uint8)
        self.color_lab_mean = cv2.cvtColor(self.color_rgb_mean[:,None,:], cv2.COLOR_RGB2LAB).squeeze()
        self.color_hsv_mean = cv2.cvtColor(self.color_rgb_mean[:,None,:], cv2.COLOR_RGB2HSV).squeeze()
        self.color_mean = np.concatenate((self.color_rgb_mean, self.color_lab_mean, self.color_hsv_mean), axis=1)
        self.size_id = ["large", "small"]

        self.homography = None

        # ML model 
        # TODO
        # self.device = "cpu"
        # self.model = torch.load("models/model_fcn_re101_cpu.pth")
        # self.model.to(self.device)
        # self.model.eval()
        self.model = None
        with open('models/model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        self.loadCameraCalibration("config/camera_calib.yaml")

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        # img_h, img_w = 720, 1280
        # frac = 3.0 / 4.0
        # blind_rectangle = None
        # ignore = 3
        # if ignore==1:
        #     blind_rectangle = [(int(img_w/2), 0), (img_w, int(img_h*frac))]
        # elif ignore==2:
        #     blind_rectangle = [(0, 0), (int(img_w/2), int(img_h/3/2))]
        # elif ignore==3:
        #     blind_rectangle = [(0, int(img_h*frac)), (int(img_w/2), img_h)]
        # elif ignore==4:
        #     blind_rectangle = [(int(img_w/2), int(img_h*frac)), (img_w, img_h)]
        # elif ignore==5: # negative half plane
        #     blind_rectangle = [(0, int(img_h*frac)), (img_w, img_h)]
        # cv2.rectangle(self.ProcessVideoFrame, blind_rectangle[0],blind_rectangle[1], (255, 0, 0), 2)


        cv2.rectangle(self.ProcessVideoFrame, (225, 90),(1090, 700), (255, 0, 0), 2)
        cv2.rectangle(self.ProcessVideoFrame, (575, 400),(750, 700), (255, 0, 0), 2)
        if self.block_detections.detected_num < 1:
            return
        for idx, (color, pixel, point, size, theta) in enumerate(zip(self.block_detections.colors, self.block_detections.uvds, self.block_detections.xyzs, self.block_detections.sizes, self.block_detections.thetas)):
            cx, cy = pixel[:2]
            cv2.putText(self.ProcessVideoFrame, self.color_id[color], (cx-30, cy+30), self.font, 0.5, self.color_rgb_mean[color].tolist(), thickness=2)
            cv2.putText(self.ProcessVideoFrame, self.size_id[size], (cx-30, cy+45), self.font, 0.5, (255,255,255), thickness=2)
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
        self.DepthFrameHSV[..., 0] = self.ProcessDepthFrameRaw >> 1
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

    def detectBlocksInDepthImage(self, _lower=700, _upper=960, blind_rect=None, sort_key="color"):
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
        cv2.rectangle(mask, (225, 90), (1090, 700), 255, cv2.FILLED)
        cv2.rectangle(mask, (575, 400), (750, 700), 0, cv2.FILLED)
        if blind_rect is not None:
            cv2.rectangle(mask, blind_rect[0], blind_rect[1], 0, cv2.FILLED)

        depth_seg = cv2.inRange(self.ProcessDepthFrameRaw, lower, upper)
        img_depth_thr = cv2.bitwise_and(depth_seg, mask)

        contours, _ = cv2.findContours(img_depth_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        self.block_detections.all_contours = contours

        # cv2.drawContours(self.ProcessVideoFrame, contours, -1, (255, 0, 0), 1)

        # depth_stuff = cv2.bitwise_and(self.ProcessDepthFrameRaw, self.ProcessDepthFrameRaw, mask=img_depth_thr)
        # d_array = depth_stuff[depth_stuff>0]
        # d_sort = np.sort(d_array)
        # fig = plt.figure()
        # plt.scatter(np.arange(len(d_sort)), d_sort)
        
        # ax = fig.gca(projection='3d')
        # xspan = np.arange(1280)
        # yspan = np.arange(720)
        # X, Y = np.meshgrid(xspan, yspan)
        # print(np.max(depth_stuff))
        # print(np.min(depth_stuff))
        # surf = ax.plot_surface(X, Y, depth_stuff, cmap=cm.jet,
        #             linewidth=0, antialiased=False)
        # ax.set_zlim(0, 1000)
                    
        # plt.savefig("test.png")

        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] < 200 or abs(M["m00"]) > 7000:
                # reject false positive detections by area size
                continue
            mask_single = np.zeros_like(self.ProcessDepthFrameRaw, dtype=np.uint8)
            cv2.drawContours(mask_single, [contour], -1, 255, cv2.FILLED)
            depth_single = cv2.bitwise_and(self.ProcessDepthFrameRaw, self.ProcessDepthFrameRaw, mask=mask_single)
            depth_array = depth_single[depth_single>=lower]

            # Stats mode range
            mode_real, _ = stats.mode(depth_array)
            print("real mode", mode_real)
            depth_diff = np.abs(depth_array - mode_real)
            depth_array_inliers = depth_array[depth_diff<10]

            # Inter Quartile Range
            # Q1 = np.percentile(depth_array, 25, interpolation = 'midpoint')
            # Q3 = np.percentile(depth_array, 75, interpolation = 'midpoint')
            # IQR = Q3 - Q1
            # mode_lower = Q1 - 1.5 * IQR # outlier lower bound
            # print("IQR lower", mode_lower)
            # depth_array_inliers = depth_array[depth_array>=mode_lower]
            
            mode = np.min(depth_array_inliers)
            print("result min", mode)
            # !!! Attention to the mode offset, it determines how much of the top surface area will be reserved
            depth_new = cv2.inRange(depth_single, lower, int(mode)+3)
            contours_new, _ = cv2.findContours(depth_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            if not contours_new:
                continue
            contours_new_valid = max(contours_new, key=cv2.contourArea) # find the largest contour
            M = cv2.moments(contours_new_valid)
            if abs(M["m00"]) < 200:
                # reject false positive detections by area size
                continue
            elif abs(M["m00"]) > 2000:
                # TODO add seg model
                print("Cluster detected with moment:", M["m00"])
                self.block_detections.reset()
                self.block_detections.all_contours = contours
                self.block_detections.has_cluster = True
                # # generate new mask for new valid contours
                # mask_new_single = np.zeros_like(mask_single, dtype=np.uint8)
                # cv2.drawContours(mask_new_single, [contours_new_valid], -1, 255, cv2.FILLED)
                # # segmente rgb image using new mask
                # rgb_single = cv2.bitwise_and(self.ProcessVideoFrame, self.ProcessVideoFrame, mask=mask_new_single)
                # input_img = BlocksDataset.transform(torch.from_numpy(rgb_single).to(torch.float).permute(2, 0, 1)).unsqueeze(0)
                # # input_img (1, 3, 244, 244)
                # output_pred = self.model(input_img.to(self.device))
                # # output_pred (1, 7, 244, 244)
                # output = torch.argmax(output_pred, 1).squeeze(0).cpu().numpy()
                # # output (244, 244) int64
                # bins = np.bincount(output.flatten())
                # if np.count_nonzero(bins[1:])>1:
                #     output_img = output.astype(np.float32) * 255/6
                #     output_mask = cv2.resize(output_img , (1280,720))
                #     print("Your model really find something??!!")
                #     print("model colors:{}".format(bins[1:]))
                #     cv2.imwrite("data/treasures_%d.png" % (random()*1000), output_mask)
                # pass
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cz = self.ProcessDepthFrameRaw[cy, cx]
            block_ori = cv2.minAreaRect(contours_new_valid)[2] # turn the range from [-90, 0) to (0, 90]

            block_xyz = self.coord_pixel_to_world(cx, cy, cz)

            # !!! size classification: attention to this moment threshold
            if M["m00"] < 800:
                block_xyz[2] = block_xyz[2] - 10
                self.block_detections.sizes.append(1) # 1 for small
            else:
                block_xyz[2] = block_xyz[2] - 19
                self.block_detections.sizes.append(0) # 0 for large

            self.block_detections.uvds.append([cx, cy, cz])
            self.block_detections.xyzs.append(block_xyz)
            self.block_detections.contours.append(contours_new_valid)
            self.block_detections.thetas.append(np.deg2rad(block_ori))
            self.block_detections.colors.append(self.retrieve_area_color(self.ProcessVideoFrame, self.ProcessVideoFrameLab, self.ProcessVideoFrameHSV, contours_new_valid))
            if self.block_detections.has_cluster:
                break

        self.block_detections.update(sort_key)

    def retrieve_area_color(self, frame_rgb, frame_lab, frame_hsv, contour):
        # RGB features
        mask_rgb = np.zeros(frame_rgb.shape[:2], dtype="uint8")
        cv2.drawContours(mask_rgb, [contour], -1, 255, cv2.FILLED)
        mean_rgb = np.array(cv2.mean(frame_rgb, mask=mask_rgb)[:3], dtype=DTYPE)
        # dist_rgb = self.color_rgb_mean - mean_rgb
        # print(dist_rgb.shape)

        # LAB features
        mask_lab = np.zeros(frame_lab.shape[:2], dtype="uint8")
        cv2.drawContours(mask_lab, [contour], -1, 255, cv2.FILLED)
        mean_lab = np.array(cv2.mean(frame_lab, mask=mask_lab)[:3], dtype=DTYPE)
        # dist_lab = self.color_lab_mean - mean_lab
        # print(dist_lab.shape)

        # HSV features
        mask_hsv = np.zeros(frame_hsv.shape[:2], dtype="uint8")
        cv2.drawContours(mask_hsv, [contour], -1, 255, cv2.FILLED)
        mean_hsv = np.array(cv2.mean(frame_hsv, mask=mask_hsv)[:3], dtype=DTYPE)
        # dist_hsv = self.color_hsv_mean - mean_hsv

        # dist = np.concatenate((dist_rgb, dist_lab, dist_hsv), axis=1)

        features = np.concatenate((mean_rgb, mean_lab, mean_hsv))
        # print(features.shape)
        dist = self.color_mean - features
        # print(dist.shape)
        data = features.tolist()

        if self.model is not None:
            pred = self.model.predict(features[None, :])
            data.append(int(pred))
        else:
            dist_norm = np.linalg.norm(dist, axis=1)
            color_idx = np.argmin(dist_norm)
            data.append(color_idx)

        # with open("models/data.csv", 'a') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(data)

        # * Let's directly return color index for easy sorting
        if self.model is not None:
            return int(pred)
        else:
            return color_idx

    def coord_pixel_to_world(self, u, v, z):
        index = np.array([u, v, 1]).reshape((3,1))
        pos_camera = z * np.matmul(self.intrinsic_matrix_inv, index)
        temp_pos = np.array([pos_camera[0][0], pos_camera[1][0], pos_camera[2][0], 1]).reshape((4,1))
        world_pos = np.matmul(self.extrinsic_matrix_inv, temp_pos)
        return world_pos.flatten()


class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        # self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, image_data):
        try:
            # cv_image = self.bridge.imgmsg_to_cv2(image_data, image_data.encoding)
            cv_image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except Exception as e:
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
        # self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, image_data):
        try:
            # cv_image = self.bridge.imgmsg_to_cv2(image_data, image_data.encoding)
            cv_image = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except Exception as e:
            print(e)
        self.camera.VideoFrame = cv_image
        self.camera.colorReceived = True


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        # self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, image_data):
        try:
            # cv_depth = self.bridge.imgmsg_to_cv2(image_data, image_data.encoding)
            cv_depth = np.frombuffer(image_data.data, dtype=np.uint16).reshape(image_data.height, image_data.width)
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except Exception as e:
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
                self.camera.ProcessVideoFrameHSV = cv2.cvtColor(self.camera.ProcessVideoFrame, cv2.COLOR_RGB2HSV)
                if self.camera.homography is not None:
                    self.camera.ProcessDepthFrameRaw = cv2.warpPerspective(self.camera.DepthFrameRaw.copy(), self.camera.homography, (1280, 720))
                else:
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
