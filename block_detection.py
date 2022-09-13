#!/usr/bin/python

import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_simple_detector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    # params.minThreshold = 10
    # params.maxThreshold = 200

    params.filterByColor = True
    params.blobColor = 0

    # Filter by Area.
    # params.filterByArea = True
    # params.minArea = 1500

    # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity =  0.6
    # params.maxCircularity =  0.9

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.9

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.6
    return cv2.SimpleBlobDetector.create(params)

def RGB_split(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([180, 255, 255])
    red_mask = cv2.inRange(img_hsv, low_red, high_red)

    # Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(img_hsv, low_blue, high_blue)

    # Green color
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(img_hsv, low_green, high_green)

    return red_mask, blue_mask, green_mask


# img = cv2.imread("./test/test_blocks.jpg")
img = cv2.imread("./test/test_blocks2.jpeg")
# img = cv2.imread("./test/test_blocks3.jpg")

img = cv2.resize(img, (640, 480))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


red_mask, blue_mask, green_mask = RGB_split(img)

red = cv2.bitwise_and(img, img, mask=red_mask)
blue = cv2.bitwise_and(img, img, mask=blue_mask)
green = cv2.bitwise_and(img, img, mask=green_mask)

img_pair = np.concatenate([red, green, blue], axis=1)

plt.imshow(img_pair[:,:,::-1])
plt.show()

target_frame = img
simple_detector = create_simple_detector()
keypoints = simple_detector.detect(target_frame)
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(im_with_keypoints[:,:,::-1])
plt.show()




