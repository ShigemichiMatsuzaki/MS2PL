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

FOREST_CLASS_LIST = ["road", "grass", "tree", "sky", "obstacle"]

color_encoding = OrderedDict(
    [
        ("road", (170, 170, 170)),
        ("grass", (0, 255, 0)),
        ("tree", (102, 102, 51)),
        ("sky", (0, 120, 255)),
        ("obstacle", (0, 0, 0)),
    ]
)

color_to_id = {
    (170, 170, 170): 0,
    (0, 255, 0): 1,
    (102, 102, 51): 2,
    (0, 60, 0): 2,
    (0, 120, 255): 3,
    (0, 0, 0): 4,
    (255, 255, 255): 255,
}

id_forest_to_greenhouse = np.array(
    [2, 0, 0, 1, 1], dtype=np.uint8  # road  # grass  # tree  # sky  # obstacle
)

color_palette = [170, 170, 170, 0, 255, 0, 102, 102, 51, 0, 120, 255, 0, 0, 0]


class FreiburgForestDataset(BaseDataset):
    def initialize(self):
        if self.mode == "val":
            self.mode = "test"
        elif self.mode not in ["train", "test"]:
            print("Invalid dataset mode : {}".format(self.mode))
            raise ValueError

        data_dir = os.path.join(self.root, self.mode)

        self.images = sorted(glob.glob(os.path.join(data_dir, "rgb/*.jpg")))
        self.labels = sorted(glob.glob(os.path.join(data_dir, "GT_color/*.png")))
        self.size = (256, 480)
        self.label_conversion_map = id_forest_to_greenhouse

    # def __len__(self):
    #     return len(self.images)

    def label_preprocess(self, label) -> Image:
        """Convert color label to ids

        Parameters
        ----------
        label_pil : `PIL.Image`or numpy.ndarray
            3-channel color label image

        Returns
        -------
        label_img : `PIL.Image`or numpy.ndarray
            1-channel label image
        """
        if isinstance(label, np.ndarray):
            label_img_color_np = label
        else:
            label_img_color_np = np.array(label, np.uint8)

        label_img_id_np = np.zeros((label_img_color_np.shape[:2]), dtype=np.uint8)
        for color in color_to_id:
            label_img_id_np[(label_img_color_np == color).all(axis=2)] = color_to_id[
                color
            ]

        # Convert the label values
        # label_img_id_np -= 1
        # label_img_id_np[label_img_id_np > 4] = self.ignore_idx  # void

        if isinstance(label, np.ndarray):
            label_img = label_img_id_np
        else:
            label_img = Image.fromarray(label_img_id_np)

        return label_img
