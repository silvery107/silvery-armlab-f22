"""
Generate dataset from data labeled by www.labelstud.io
"""

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--folder", required = True, help = "Path to the label folder")
args = vars(ap.parse_args())

folder_path = args["folder"]
# folder_path = "data/project-1-at-2022-10-01-19-51-ee99dc90"
files = glob.glob(os.path.join(folder_path, "*.png"))

temp = cv2.imread(files[0])
data = []
for idx in range(4):
    fname = "task-" + str(idx + 1)
    mask = np.zeros((720, 1280), dtype=np.float32)
    for file in files:
        if file.find(fname) != -1:
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED) # (720, 1280)
            if file.find("red") != -1:
                mask[img>0] = 1
            elif file.find("orange") != -1:
                mask[img>0] = 2
            elif file.find("yellow") != -1:
                mask[img>0] = 3
            elif file.find("green") != -1:
                mask[img>0] = 4
            elif file.find("blue") != -1:
                mask[img>0] = 5
            elif file.find("purple") != -1:
                mask[img>0] = 6
    data.append(mask)
    # print(mask.shape)
    cv2.imwrite("data/segmentations/seg_" + str(idx + 1) + ".png", mask)

img_pair = np.concatenate([img for img in data], axis=1)

plt.imshow(img_pair*255/6, cmap='gray')
plt.show()