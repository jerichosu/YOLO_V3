# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 06:34:45 2022

@author: Jericho
"""
import config
import numpy as np
# import os
# import pandas as pd
import torch
import albumentations as A
# from tqdm import tqdm
import cv2
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from albumentations.pytorch import ToTensorV2

from PIL import ImageFile
# from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    # iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)
# from dataset import YOLODataset
from model import YOLOv3

ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------------------------------------------------------
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
# -----------------------------------------------------------------------------        

WEIGHTS = 'yolov3_pascal_78.1map.pth.tar'

model = YOLOv3(num_classes=20)
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
load_checkpoint(WEIGHTS, model, optimizer, config.LEARNING_RATE)

model.to(config.DEVICE)
model.eval()

# %%

anchors = config.ANCHORS
# transform = config.test_transforms

# some testing images
# path = 'dogs01.jpg'
# path = 'dogs02.jpg'
path = 'people.png'
# path = 'people01.jpg'

img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


IMAGE_SIZE = 416
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
)

inputs = test_transforms(image = img)
print(inputs['image'].shape) # torch.Size([3, 416, 416])
inputs = inputs['image']
print(inputs.shape)



# increase the dimension
inputs = torch.unsqueeze(inputs, 0) # torch.Size([1,3,H,W])
print(inputs.shape)
        
input_tensor = inputs.to(config.DEVICE)

y = model(input_tensor)
print(y[0].shape)
print(y[1].shape)
print(y[2].shape)


# %%
# def plot_image(image, boxes):
#     """Plots predicted bounding boxes on the image"""
#     im = np.array(image)
#     height, width, _ = im.shape

#     # Create figure and axes
#     fig, ax = plt.subplots(1)
#     # Display the image
#     ax.imshow(im)

#     # box[0] is x midpoint, box[2] is width
#     # box[1] is y midpoint, box[3] is height

#     # Create a Rectangle potch
#     for box in boxes:
#         box = box[2:]
#         assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
#         upper_left_x = box[0] - box[2] / 2
#         upper_left_y = box[1] - box[3] / 2
#         rect = patches.Rectangle(
#             (upper_left_x * width, upper_left_y * height),
#             box[2] * width,
#             box[3] * height,
#             linewidth=1,
#             edgecolor="r",
#             facecolor="none",
#         )
#         # Add the patch to the Axes
#         ax.add_patch(rect)

#     plt.show()
# %%

S = [13, 26, 52]
scaled_anchors = torch.tensor(anchors) / (
    1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
)

#######
# for x, y in loader: # x (image): [B, 3, 416, 416], y (label): zeros [[B, 3, 13, 13, 6], [B, 3, 26, 26, 6], [B, 3, 52, 52, 6]]
boxes = []

for i in range(y[0].shape[1]): #y[0]: [B, 3, 13, 13, 6],  y[0].shape[1]: 3
    anchor = scaled_anchors[i] # scaled_anchors: torch.Size([3, 3, 2])
    print(anchor.shape) # torch.Size([3, 2])
    print(y[i].shape) # [B, 3, 13/26/52, 13/26/52, 6]
    boxes += cells_to_bboxes(
        y[i].to("cpu"), is_preds=True, S=y[i].shape[2], anchors=anchor #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    )[0] # scale and augment all 3 predictions from each output into one list which contains all (10647) predictions 

# specified as [class_pred, prob_score, x1, y1, x2, y2]
# iou_threshold (float): threshold where predicted bboxes is correct
# threshold (float): threshold to remove predicted bboxes (independent of IoU)
boxes = nms(boxes, iou_threshold=0.5, threshold=0.2, box_format="midpoint")

# boxes = boxes[0:len(boxes)//3]
print(boxes) # [class, obj, x, y, h, w] (repeated for 3 times)
# plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes) #x: torch.Size([1, 3, 416, 416]), x[0]: torch.Size([3, 416, 416])
plot_image(img, boxes) #x: torch.Size([1, 3, 416, 416]), x[0]: torch.Size([3, 416, 416])



