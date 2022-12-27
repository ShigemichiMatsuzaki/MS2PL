import os
import sys
import torch
import torch.utils.data as data
from PIL import Image
import cv2
import numpy as np
from torchvision.transforms import functional as F
import albumentations as A


class BaseDataset(data.Dataset):
    def __init__(
        self,
        root,
        mode="train",
        ignore_idx=255,
        scale=(0.5, 2.0),
        height=256,
        width=480,
        transform=None,
        label_conversion=False,
        max_iter=None,
    ):
        """Base class of dataset

        Parameter
        ---------
        root : str
            Root directory of datasets
        mode : str
            Mode of the dataset ['train', 'val', 'test']
        ignore_idx : int
            Label index for the pixels ignored in training
        transform : albumentations

        """
        self.root = root
        self.mode = mode
        self.ignore_idx = ignore_idx
        self.label_conversion = label_conversion
        self.label_conversion_map = None
        # self.transform = transform

        # Declare an augmentation pipeline
        if transform is not None:
            self.transform = transform
        else:
            self.transform = (
                A.Compose(
                    [
                        # A.Resize(width=480, height=256),
                        # A.RandomCrop(width=480, height=256),
                        A.RandomResizedCrop(
                            width=width, height=height, scale=scale),
                        A.HorizontalFlip(p=0.5),
                        A.GaussNoise(p=0.2),
                        A.GaussianBlur(p=0.2),
                        A.RGBShift(p=0.5),
                        A.RandomBrightnessContrast(p=0.2),
                        A.ChannelShuffle(p=0.1),
                    ]
                )
                if self.mode == "train"
                else A.Compose(
                    [
                        A.Resize(width=width, height=height),
                    ]
                )
            )

        self.max_iter = max_iter
        # self.label_preprocess = None

        self.images = []
        self.labels = []

        # self.initialize()

    def initialize(self):
        raise NotImplementedError

    # def label_preprocess(self, label_pil: Image):
    def label_preprocess(self, label_pil):
        return label_pil

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_img = Image.open(self.images[index]).convert("RGB")
        label_img = Image.open(self.labels[index])

        rgb_np = np.array(rgb_img)
        label_np = np.array(label_img)

        label_np = self.label_preprocess(label_np)
        # Resize
        # rgb_img = F.resize(rgb_img, self.size)
        # label_img = F.resize(label_img, self.size)

        # Convert images to tensors
        # label_img = np.array(label_img)
        if self.label_conversion and self.label_conversion_map is not None:
            label_np = self.label_conversion_map[label_np]

        transformed = self.transform(image=rgb_np, mask=label_np)

        rgb_img_orig = F.to_tensor(transformed["image"])
        label_img = torch.LongTensor(transformed["mask"].astype(np.int64))

        # rgb_img_orig = F.to_tensor(rgb_np)
        # label_img = torch.LongTensor(label_np.astype(np.int64))

        # Normalize the pixel values
        rgb_img = F.normalize(
            rgb_img_orig, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )

        return {"image": rgb_img, "image_orig": rgb_img_orig, "label": label_img}
