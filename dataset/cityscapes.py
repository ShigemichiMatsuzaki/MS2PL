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
id_cityscapes_to_greenhouse = np.array(
    [
        2,  # Road
        2,  # Sidewalk
        1,  # Building
        1,  # Wall
        1,  # Fence
        1,  # Pole
        1,  # Traffic light
        1,  # Traffic sign
        0,  # Vegetation
        2,  # Terrain
        3,  # Sky
        3,  # Person
        3,  # Rider
        1,  # Car
        1,  # Truck
        1,  # Bus
        1,  # Train
        1,  # Motorcycle
        1,  # Bicycle
        3,  # Background
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
        label_conversion=False,
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
            label_conversion=label_conversion,
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
            glob.glob(os.path.join(data_train_label_dir, "*/*labelTrainIds.png"))
        )

        if self.mode == "train" and self.annot_type == "gtCoarse":
            data_train_image_dir = os.path.join(image_dir, "train_extra")
            data_train_label_dir = os.path.join(label_dir, "train_extra")
            self.images += sorted(
                glob.glob(os.path.join(data_train_image_dir, "*/*.png"))
            )
            self.labels += sorted(
                glob.glob(os.path.join(data_train_label_dir, "*/*labelTrainIds.png"))
            )

        self.label_conversion_map = id_cityscapes_to_greenhouse
        self.size = (height, width)

        if self.max_iter is not None:
            self.images *= self.max_iter // len(self.images)
            self.labels *= self.max_iter // len(self.labels)

    def initialize(self):
        self.annot_type = "gtFine"
        image_dir = os.path.join(self.root, "leftImg8bit")
        # label_dir = os.path.join(self.root, self.annot_type)
        label_dir = os.path.join(self.root, self.annot_type)
        data_train_image_dir = os.path.join(image_dir, self.mode)
        data_train_label_dir = os.path.join(label_dir, self.mode)
        self.images += sorted(glob.glob(os.path.join(data_train_image_dir, "*/*.png")))
        self.labels += sorted(
            glob.glob(os.path.join(data_train_label_dir, "*/*labelTrainIds.png"))
        )

        if self.mode == "train" and self.annot_type == "gtCoarse":
            data_train_image_dir = os.path.join(image_dir, "train_extra")
            data_train_label_dir = os.path.join(label_dir, "train_extra")
            self.images += sorted(
                glob.glob(os.path.join(data_train_image_dir, "*/*.png"))
            )
            self.labels += sorted(
                glob.glob(os.path.join(data_train_label_dir, "*/*labelTrainIds.png"))
            )

        self.label_conversion_map = id_cityscapes_to_greenhouse
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
    #         label_img = id_cityscapes_to_greenhouse[label_img]

    #     label_img = torch.LongTensor(label_img.astype(np.int64))

    #     label_img[label_img >= 19] = 255

    #     # Normalize the pixel values
    #     rgb_img = F.normalize(rgb_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    #     return {"image": rgb_img, "label": label_img}
