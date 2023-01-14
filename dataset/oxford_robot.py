import os
import torch
from PIL import Image
from PIL import ImageDraw

# from transforms.segmentation.data_transforms import (
#     RandomFlip,
#     Normalize,
#     Resize,
#     Compose,
# )
import albumentations as A
from torchvision.transforms import functional as F
from collections import OrderedDict
import numpy as np
import copy
import glob

from dataset.base_dataset import BaseTargetDataset

OXFORD_ROBOT_CLASS_LIST = [
    "background",
    "sky",
    "person",
    "two-wheel vehicle",
    "automobile",
    "traffic sign",
    "traffic light",
    "building",
    "grass",
    "parking lot",
    "sidewalk",
    "road",
    "driveway",
    "crosswalk",
    "other road marking",
    "rail ",
    "other obstacle ",
    "lane boundary",
    "other lane marking ",
]

color_encoding = OrderedDict(
    [
        ("background", (0, 0, 0)),
        ("sky", (70, 130, 180)),
        ("person", (220, 20, 60)),
        ("motorcycle", (0, 0, 230)),
        ("car", (0, 0, 142)),
        ("traffic sign", (220, 220, 0)),
        ("traffic light", (250, 170, 30)),
        ("building", (70, 70, 70)),
        ("vegetation", (107, 142, 35)),
        ("wall", (102, 102, 156)),
        ("road", (128, 64, 128)),
        ("terrain", (152, 251, 152)),
        ("sidewalk", (244, 35, 232)),
        ("fence", (190, 153, 153)),
        ("pole", (153, 153, 153)),
        ("rider", (255, 0, 0)),
        ("truck", (0, 0, 70)),
        ("bus", (0, 60, 100)),
        ("train", (0, 80, 100)),
        ("bicycle", (119, 11, 32)),
    ]
)

color_palette = []
# OrderedDict to color palette
for key in color_encoding:
    color_palette.append(color_encoding[key][0])
    color_palette.append(color_encoding[key][1])
    color_palette.append(color_encoding[key][2])

class OxfordRobot(BaseTargetDataset):
    def __init__(
        self,
        dataset_root,
        label_root="",
        mode="train",
        size=(256, 480),
        is_hard_label=False,
        load_labels=True,
    ):
        """Initialize a dataset

        Each line of a data list must be formatted as follows:
            rgb_image_path, depth_image_path, label_path, trav_mask_path, start_point_x, start_point_y, end_point_x, end_point_y

        Parameters
        ----------
        list_name: `str`
            File name of the data list
        root: `str`
            Name of the directory where the data list is stored
        train: `bool`
            True if the dataset is for training
        size: `list`
            Image size to which the image is resized
        raw: `bool`
            True if the original pixel coordinates are used for training of the path line.
          Otherwise, they are scaled in [0, 1]
        is_soft_label: `bool`
            `True` if soft labels are used
        load_labels: `bool`
            Whether to load label data.

        """
        super().__init__(
            label_root=label_root,
            mode=mode,
            size=size,
            is_hard_label=is_hard_label,
            load_labels=load_labels,
        )

        self.dataset_root = dataset_root

        if self.mode == "test":
            self.mode = "val"
        elif self.mode not in ["train", "test", "val", "pseudo"]:
            print("Invalid dataset mode : {}".format(self.mode))
            raise ValueError

        data_dir = os.path.join(self.dataset_root, "OxfordRobot/Oxford_Robot_ICCV19")

        if self.mode == "train" or self.mode == "pseudo":
            self.images = sorted(glob.glob(os.path.join(data_dir, "train","*/*.png")))
            self.labels = [""] * len(self.images)
        else: 
            self.images = sorted(glob.glob(os.path.join(data_dir, "val", "*.png")))
            self.labels = sorted(
                glob.glob(os.path.join(data_dir, "anno", "*.png")))

            print(len(self.images), len(self.labels))

    def label_preprocess(self, label_img):
        """Pre-processing of the label
        
        """
        #
        # Segmentation label
        #
        label_np = np.array(label_img)
        label_np = label_np[:,:,0]
        label_img = Image.fromarray(label_np)
        return label_img
