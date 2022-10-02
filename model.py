import numpy as np
import torch
import cv2 as cv
import glob
import os
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101, fcn_resnet50
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import Resize


class BlocksDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, depths_dir, labels_dir):
      self.img_dir = img_dir
      self.depths_dir = depths_dir
      self.labels_dir = labels_dir
      self.label_files = glob.glob(os.path.join(self.labels_dir, "*.png"))
      self.image_files = [fname.replace(self.labels_dir, self.img_dir).replace("seg_", "image_") for fname in self.label_files]
      self.depth_files = [fname.replace(self.labels_dir, self.depths_dir).replace("seg_", "depth_") for fname in self.label_files]
      self.labels = ["red_block", "orange_block", "yellow_block", "green_block", "blue_block", "purple_block"]
      self.resize_fn = Resize((224, 224))
      self.label_resize_fn = Resize((224, 224), interpolation=0)

    def __len__(self):
      return len(self.image_files)

    def __getitem__(self, idx):
      image = cv.imread(self.image_files[idx])
      label = cv.imread(self.label_files[idx], cv.IMREAD_UNCHANGED)#[:,:,0] 
      depth = cv.imread(self.depth_files[idx], cv.IMREAD_UNCHANGED)
      to_return_dict = {\
          "rgb": torch.from_numpy(image).to(torch.float).permute(2,0,1),\
          "segmentation": torch.from_numpy(label).to(torch.long).unsqueeze(0),\
          "depth": torch.from_numpy(depth.astype(np.float16)).unsqueeze(0),\
      }
      # permute(2,0,1): HWC (720, 1280, 3) --> CHW (3, 720, 1280)
      # unsqueeze(0): HW (720, 1280) --> CHW (1, 720, 1280)
      
      to_return_dict["rgb_sized"] = self.resize_fn(to_return_dict["rgb"])
      to_return_dict["segmentation_sized"] = self.label_resize_fn(to_return_dict["segmentation"])
      to_return_dict["depth_sized"] = self.resize_fn(to_return_dict["depth"])

      return to_return_dict

class RoboBlockNet(torch.nn.Module):
  def __init__(self):
    super(RoboBlockNet, self).__init__()
    self.backbone = fcn_resnet101(pretrained=True)
    # self.backbone = fcn_resnet50(pretrained=True)
    # self.backbone = deeplabv3_resnet101(pretrained=True)
    # self.backbone = maskrcnn_resnet50_fpn(pretrained=True)
    self.output = torch.nn.Sequential(
                  torch.nn.Conv2d(21, 7, 1), 
                  torch.nn.Softmax()
                  )
    
  def forward(self, x):
    x = self.backbone(x)['out']
    x = self.output(x)
    return x