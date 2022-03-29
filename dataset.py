"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] #[torch.Size([3, 13, 13, 6]), torch.Size([3, 26, 26, 6]), torch.Size([3, 52, 52, 6])], all zeros
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors) # compute iou with each ancher boxes
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode='trunc')
                # scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx] # determing which scale should we put this box in (13, 26 or 52?)
                i, j = int(S * y), int(S * x)  # which cell in (13x13?)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction??????????

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    # transform = config.train_transforms
    transform = config.test_transforms


    dataset = YOLODataset(
        "PASCAL_VOC/train.csv",
        "PASCAL_VOC/images",
        "PASCAL_VOC/labels",
        
        # "COCO/train.csv",
        # "COCO/images/images/",
        # "COCO/labels/labels_new/",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    
    #######
    for x, y in loader: # x (image): [B, 3, 416, 416], y (label): zeros [[B, 3, 13, 13, 6], [B, 3, 26, 26, 6], [B, 3, 52, 52, 6]]
        boxes = []

        for i in range(y[0].shape[1]): #y[0]: [B, 3, 13, 13, 6],  y[0].shape[1]: 3
            anchor = scaled_anchors[i] # scaled_anchors: torch.Size([3, 3, 2])
            # print(anchor.shape) # torch.Size([3, 2])
            # print(y[i].shape) # [B, 3, 13/26/52, 13/26/52, 6]
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0] # scale and augment all 3 predictions from each output into one list which contains all (10647) predictions 
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        # print(boxes) # [class, obj, x, y, h, w] (repeated for 3 times)
        boxes = boxes[0:len(boxes)//3]
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes) #x: torch.Size([1, 3, 416, 416]), x[0]: torch.Size([3, 416, 416])
        break
    
    
    # # this works
    # loop = tqdm(loader, leave=True)
    # for batch_idx, (x, y) in enumerate(loop):
    #     print(batch_idx)
    #     print(x.shape)
    #     print(y[0].shape)
    #     break
    


if __name__ == "__main__":
    test()
