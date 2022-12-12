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


id_camvid_to_greenhouse = np.array(
    [
        3,  # Sky
        1,  # Building
        1,  # Pole
        2,  # Road
        2,  # Pavement
        0,  # Tree
        1,  # SignSymbol
        1,  # Fence
        1,  # Car
        3,  # Pedestrian
        3,  # Bicyclist
        1,  # Road_marking(?)
        3,  # Unlabeled
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
        label_conversion=False,
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
            label_conversion=label_conversion,
            max_iter=max_iter
        )

        if self.mode not in ["train", "val", "test"]:
            print("Invalid dataset mode : {}".format(self.mode))
            raise ValueError

        data_train_image_dir = os.path.join(self.root, self.mode)
        data_train_label_dir = os.path.join(self.root, self.mode + "annot")
        self.images += sorted(glob.glob(os.path.join(data_train_image_dir, "*.png")))
        self.labels += sorted(glob.glob(os.path.join(data_train_label_dir, "*.png")))

        self.label_conversion_map = id_camvid_to_greenhouse
        self.size = (height, width)

        if self.max_iter is not None:
            self.images *= self.max_iter // len(self.images)
            self.labels *= self.max_iter // len(self.labels)
    #    def __init__(
    #        self,
    #        root,
    #        mode="train",
    #        scale=(0.5, 2.0),
    #        size=(360, 480),
    #        ignore_idx=255,
    #        normalize=True,
    #        label_conversion=False,
    #    ):
    #
    #        if mode not in ["train", "val", "test"]:
    #            print("Invalid dataset mode : {}".format(mode))
    #            raise ValueError
    #
    #        self.mode = mode
    #        self.normalize = normalize
    #        self.images = []
    #        self.labels = []
    #        self.label_conversion = label_conversion
    #        self.ignore_idx = 255
    #
    #
    #        if isinstance(size, tuple):
    #            self.size = size
    #        else:
    #            self.size = (size, size)
    #
    #        if isinstance(scale, tuple):
    #            self.scale = scale
    #        else:
    #            self.scale = (scale, scale)

    def initialize(self):
        if self.mode not in ["train", "val", "test"]:
            print("Invalid dataset mode : {}".format(self.mode))
            raise ValueError

        data_train_image_dir = os.path.join(self.root, self.mode)
        data_train_label_dir = os.path.join(self.root, self.mode + "annot")
        self.images += sorted(glob.glob(os.path.join(data_train_image_dir, "*.png")))
        self.labels += sorted(glob.glob(os.path.join(data_train_label_dir, "*.png")))

        self.label_conversion_map = id_camvid_to_greenhouse
        self.size = (256, 480)

    def __len__(self):
        return len(self.images)

    # def __getitem__(self, index):
    #     rgb_img = Image.open(self.images[index]).convert("RGB")
    #     label_img = Image.open(self.labels[index])

    #     # Resize
    #     rgb_img = F.resize(rgb_img, self.size)
    #     label_img = F.resize(label_img, self.size)

    #     # Convert images to tensors
    #     rgb_img = F.to_tensor(rgb_img)
    #     label_img = np.array(label_img)
    #     if self.label_conversion:
    #         label_img = id_camvid_to_greenhouse[label_img]

    #     label_img = torch.LongTensor(label_img.astype(np.int64))

    #     # Normalize the pixel values
    #     rgb_img = F.normalize(rgb_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    #     return {"image": rgb_img, "label": label_img}
