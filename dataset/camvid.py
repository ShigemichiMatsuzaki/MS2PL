import os
import sys
import torch
import torch.utils.data as data
from PIL import Image
import cv2
from collections import OrderedDict
import numpy as np
from torchvision.transforms import functional as F
import random
import glob
from dataset.base_dataset import BaseDataset

CAMVID_CLASS_LIST = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
    "Road_marking",
    "Unlabelled",
]

color_encoding = OrderedDict(
    [
        ("Sky", (128, 128, 128)),
        ("Building", (128, 0, 0)),
        ("Pole", (192, 192, 128)),
        ("Road", (128, 64, 128)),
        ("Pavement", (0, 0, 192)),
        ("Tree", (128, 128, 0)),
        ("SignSymbol", (192, 128, 128)),
        ("Fence", (64, 64, 128)),
        ("Car", (64, 0, 128)),
        ("Pedestrian", (64, 64, 0)),
        ("Bicyclist", (0, 128, 192)),
        ("Unlabelled", (0, 0, 0)),
    ]
)


class CamVidSegmentation(BaseDataset):
    def __init__(
        self,
        root,
        mode="train",
        ignore_idx=255,
        scale=(0.5, 2.0),
        height=512,
        width=1024,
        transform=None,
        label_conversion_to="",
        max_iter=None,
    ):
        super().__init__(
            root,
            mode=mode,
            ignore_idx=ignore_idx,
            scale=scale,
            height=height,
            width=width,
            transform=transform,
            label_conversion_to=label_conversion_to,
            max_iter=max_iter
        )

        if self.mode not in ["train", "val", "test"]:
            print("Invalid dataset mode : {}".format(self.mode))
            raise ValueError

        data_train_image_dir = os.path.join(self.root, self.mode)
        data_train_label_dir = os.path.join(self.root, self.mode + "annot")
        self.images += sorted(glob.glob(os.path.join(data_train_image_dir, "*.png")))
        self.labels += sorted(glob.glob(os.path.join(data_train_label_dir, "*.png")))

        self.num_classes = 13
        if self.label_conversion_to == "greenhouse" or self.label_conversion_to == "imo":
            from .tools.label_conversions import id_camvid_to_greenhouse as label_conversion
            self.num_classes = 3
        elif self.label_conversion_to == "sakaki":
            from .tools.label_conversions import id_camvid_to_sakaki as label_conversion
            self.num_classes = 5

        self.label_conversion_map = label_conversion

        self.size = (height, width)

        if self.max_iter is not None and self.max_iter > len(self.images):
            self.images *= self.max_iter // len(self.images)
            self.labels *= self.max_iter // len(self.labels)

    def __len__(self):
        return len(self.images)
