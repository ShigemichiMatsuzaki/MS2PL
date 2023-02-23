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

from dataset.base_dataset import BaseTargetDataset

GREENHOUSE_CLASS_LIST = ["plant", "artificial", "ground", "other"]

color_encoding = OrderedDict(
    [
        ("plant", (0, 255, 255)),
        ("artificial_objects", (255, 0, 0)),
        ("ground", (255, 255, 0)),
        ("background", (0, 0, 0)),
    ]
)

color_palette = [0, 255, 255, 255, 0, 0, 255, 255, 0, 0, 0, 0]


class GreenhouseRGBD(BaseTargetDataset):
    def __init__(
        self,
        list_name,
        label_root="",
        mode="train",
        size=(256, 480),
        rough_plant=True,
        is_hard_label=False,
        load_labels=True,
        is_old_label=False,
        max_iter=None,
    ):
        """Initialize a dataset

        Each line of a data list must be formatted as follows:
            rgb_image_path, depth_image_path, label_path, trav_mask_path, start_point_x, start_point_y, end_point_x, end_point_y

        Parameters
        ----------
        list_name: `str`
            File name of the data list
        label_root: `str`
            Name of the directory where the data list is stored
        mode: `bool`
            True if the dataset is for training
        size: `list`
            Image size to which the image is resized
        rough_plant: `bool`
            Use rough annotation
        is_hard_label: `bool`
            `True` if hard labels are used
        load_labels: `bool`
            Whether to load label data.
        is_old_label: `bool`
            If `True`, consider the labels as from an old label set
            i.e., ['traversable plant', 'other plant', 'artificial object', 'ground'].
            Otherwise, ['plant', 'artificial object', 'ground']

        """
        super().__init__(
            label_root=label_root,
            mode=mode,
            size=size,
            is_hard_label=is_hard_label,
            load_labels=load_labels,
            max_iter=max_iter,
        )

        self.data_file = list_name
        self.rough_plant = rough_plant
        self.is_old_label = is_old_label

        # Initialize the lists
        with open(self.data_file, "r") as lines:
            for line in lines:
                # Split a line
                line_split = line.split(",")

                #
                # RGB
                #
                rgb_img_loc = line_split[0].rstrip()
                # Chieck the first character. If it's '%', the line is not read
                if rgb_img_loc == "" or rgb_img_loc[0] == "%":
                    continue
                # Verify the existence of the file
                if rgb_img_loc != "" and not os.path.isfile(rgb_img_loc):
                    print("Not found : " + rgb_img_loc)
                    assert os.path.isfile(rgb_img_loc)
                self.images.append(rgb_img_loc)

                #
                # Segmentation label
                #
                if self.load_labels:
                    if self.label_root:
                        label_img_loc = os.path.join(
                            self.label_root, line_split[1].rstrip())
                    else:
                        label_img_loc = line_split[1].rstrip()

                    if not self.is_hard_label:
                        label_img_loc = label_img_loc.replace("png", "pt")
                else:
                    label_img_loc = ""

                # Verify the existence of the file
                if (
                    self.load_labels
                    and label_img_loc != ""
                    and not os.path.isfile(label_img_loc)
                ):
                    print("Not found : " + label_img_loc)
                    assert os.path.isfile(label_img_loc)
                self.labels.append(label_img_loc)

        if self.max_iter is not None and self.max_iter > len(self.images):
            self.images *= (self.max_iter // len(self.images))
            self.labels *= (self.max_iter // len(self.labels))

    def label_preprocess(self, label_img):
        """Pre-processing of the label

        """
        #
        # Segmentation label
        #
        if not self.rough_plant and not self.is_old_label:
            return label_img

        label_np = np.array(label_img, np.uint8)
        if self.rough_plant:
            # 5: rough plant -> 1: other plant
            label_np[label_np == 5] = 1 if self.is_old_label else 0

        if self.is_old_label:
            label_np[label_np == 0] = 1
            label_np -= 1

        label_img = Image.fromarray(label_np)

        return label_img


class GreenhouseTraversability(BaseTargetDataset):
    def __init__(
        self,
        list_name,
        label_root="",
        mode="train",
        size=(256, 480),
    ):
        """Initialize a dataset

        Each line of a data list must be formatted as follows:
            rgb_image_path, depth_image_path, label_path, trav_mask_path, start_point_x, start_point_y, end_point_x, end_point_y

        Parameters
        ----------
        list_name: `str`
            File name of the data list
        label_root: `str`
            Name of the directory where the data list is stored
        mode: `bool`
            True if the dataset is for training
        size: `list`
            Image size to which the image is resized
        load_labels: `bool`
            Whether to load label data.

        """
        super().__init__(
            label_root=label_root,
            mode=mode,
            size=size,
            is_hard_label=True,
            load_labels=True,
        )

        self.data_file = list_name

        # Initialize the lists
        with open(self.data_file, "r") as lines:
            for line in lines:
                # Split a line
                line_split = line.split(",")

                #
                # RGB
                #
                rgb_img_loc = line_split[0].rstrip()
                # Chieck the first character. If it's '%', the line is not read
                if rgb_img_loc == "" or rgb_img_loc[0] == "%":
                    continue
                # Verify the existence of the file
                if rgb_img_loc != "" and not os.path.isfile(rgb_img_loc):
                    print("Not found : " + rgb_img_loc)
                    assert os.path.isfile(rgb_img_loc)
                self.images.append(rgb_img_loc)

                #
                # Traversability label
                #
                label_img_loc = line_split[1].rstrip()

                # Verify the existence of the file
                if (
                    self.load_labels
                    and label_img_loc != ""
                    and not os.path.isfile(label_img_loc)
                ):
                    print("Not found : " + label_img_loc)
                    assert os.path.isfile(label_img_loc)
                self.labels.append(label_img_loc)

    def label_preprocess(self, label_img: Image):
        """Pre-processing of the label

        Parameters
        ----------
        label_img: `PIL.Image`
            Label image

        Returns
        -------
        label_img: `PIL.Image`
            Label image after conversion

        """
        label_img = label_img.convert("L")
        label_np = np.array(label_img, np.uint8)
        label_np //= 255  # 255 (white) to 1 (positive label)
        label_img = Image.fromarray(label_np)

        return label_img
