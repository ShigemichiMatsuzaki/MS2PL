# ===========================================# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================
import torch
import torch.utils.data as data
import os
from PIL import Image
from collections import OrderedDict
from torchvision.transforms import functional as F
import numpy as np
import glob
from dataset.base_dataset import BaseDataset

CITYSCAPE_CLASS_LIST = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "background",
]

color_encoding = OrderedDict(
    [
        ("road", (128, 64, 128)),
        ("sidewalk", (244, 35, 232)),
        ("building", (70, 70, 70)),
        ("wall", (102, 102, 156)),
        ("fence", (190, 153, 153)),
        ("pole", (153, 153, 153)),
        ("traffic light", (250, 170, 30)),
        ("traffic sign", (220, 220, 0)),
        ("vegetation", (107, 142, 35)),
        ("terrain", (152, 251, 152)),
        ("sky", (70, 130, 180)),
        ("person", (220, 20, 60)),
        ("rider", (255, 0, 0)),
        ("car", (0, 0, 142)),
        ("truck", (0, 0, 70)),
        ("bus", (0, 60, 100)),
        ("train", (0, 80, 100)),
        ("motorcycle", (0, 0, 230)),
        ("bicycle", (119, 11, 32)),
        ("background", (0, 0, 0)),
    ]
)


class CityscapesSegmentation(BaseDataset):
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
        coarse=False,
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
            max_iter=max_iter,
        )

        self.annot_type = "gtCoarse" if coarse else "gtFine"

        image_dir = os.path.join(self.root, "leftImg8bit")
        # label_dir = os.path.join(self.root, self.annot_type)
        label_dir = os.path.join(self.root, self.annot_type)
        data_train_image_dir = os.path.join(image_dir, self.mode)
        data_train_label_dir = os.path.join(label_dir, self.mode)
        self.images += sorted(glob.glob(os.path.join(data_train_image_dir, "*/*.png")))
        self.labels += sorted(
            glob.glob(os.path.join(
                data_train_label_dir, "*/*labelTrainIds.png"))
        )

        if self.mode == "train" and self.annot_type == "gtCoarse":
            data_train_image_dir = os.path.join(image_dir, "train_extra")
            data_train_label_dir = os.path.join(label_dir, "train_extra")
            self.images += sorted(
                glob.glob(os.path.join(data_train_image_dir, "*/*.png"))
            )
            self.labels += sorted(
                glob.glob(os.path.join(
                    data_train_label_dir, "*/*labelTrainIds.png"))
            )

        self.num_classes = 19
        if self.label_conversion_to == "greenhouse":
            from .tools.label_conversions import id_cityscapes_to_greenhouse as label_conversion
            self.num_classes = 3
        elif self.label_conversion_to == "sakaki" or self.label_conversion_to == "imo":
            from .tools.label_conversions import id_cityscapes_to_sakaki as label_conversion
            self.num_classes = 5
        else:
            label_conversion = None

        self.label_conversion_map = label_conversion

        self.size = (height, width)

        if self.max_iter is not None and self.max_iter > len(self.images):
            self.images *= self.max_iter // len(self.images)
            self.labels *= self.max_iter // len(self.labels)

    def label_preprocess(self, label):
        """Convert color label to ids

        Parameters
        ----------
        label : `PIL.Image`or numpy.ndarray
            3-channel color label image

        Returns
        -------
        label_img : `PIL.Image`or `numpy.ndarray`
            1-channel label image
        """
        if self.label_conversion_to:
            if isinstance(label, np.ndarray):
                label_np = label
            else:
                label_np = np.array(label, np.uint8)

            label_np[label_np == self.ignore_idx] = 19

            if isinstance(label, np.ndarray):
                label_img = label_np
            else:
                label_img = Image.fromarray(label_np)
        else:
            label_img = label

        return label_img
