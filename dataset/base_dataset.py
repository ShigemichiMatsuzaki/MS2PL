import os
import sys
import torch
import torch.utils.data as data
from PIL import Image
import cv2
import numpy as np
import torchvision
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import albumentations as A
import copy


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
        label_conversion_to="",
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
        self.label_conversion_to = label_conversion_to
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
        if self.label_conversion_to and self.label_conversion_map is not None:
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


class BaseTargetDataset(data.Dataset):
    def __init__(
        self,
        label_root="",
        mode="train",
        size=(256, 480),
        scale=(0.70, 1.0),
        is_hard_label=False,
        load_labels=True,
        transform=None,
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
        self.mode = mode
        self.is_hard_label = is_hard_label
        self.load_labels = load_labels
        self.label_root = label_root

        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        self.scale = scale

        # Declare an augmentation pipeline
        if transform is not None:
            self.transform = transform
        else:
            self.transform = (
                A.Compose(
                    [
                        # A.Resize(width=480, height=256),
                        # A.RandomCrop(width=480, height=256),
                        A.Resize(width=self.size[1], height=self.size[0]),
                        A.HorizontalFlip(p=0.5),
                        A.GaussNoise(p=0.1),
                        A.GaussianBlur(p=0.1),
                        A.RGBShift(p=0.1),
                        A.RandomBrightnessContrast(p=0.3),
                        A.ChannelShuffle(p=0.05),
                    ]
                )
                if self.mode == "train"
                else A.Compose(
                    [
                        A.Resize(width=self.size[1], height=self.size[0]),
                    ]
                )
            )

        self.images = []
        self.labels = []

    def initialize(self):
        raise NotImplementedError

    def label_preprocess(self, label):
        """Pre-processing of the label

        """
        raise NotImplementedError

    def set_label_list(self, label_list):
        """Set label list

        Args:
            real_noise (torch.Tensor): A tensor of original noisy images
            real_denoise (torch.Tensor): A tensor of original not noisy images
            tri_denoise (torch.Tensor): A tensor of images converted back to the denoised domain

        Returns:
            Dictionary: A dictionary that stores the losses
        """
        self.labels = copy.deepcopy(label_list)

    def __len__(self):
        """Get length

        Returns:
            Int: The length of the dataset
        """
        return len(self.images)

    def __getitem__(self, index):
        """Initialize

        Args:
            index (Int): An index

        Returns:
            Dictionary: A dictionary that stores data
        """

        rgb_img = Image.open(self.images[index]).convert("RGB")
        rgb_cv_img = np.array(rgb_img)
        label_img = None

        #
        # Segmentation label
        #
        if self.labels and self.labels[index]:
            label_path = self.labels[index]
            if self.is_hard_label:
                label_img = Image.open(self.labels[index])

                label_img = self.label_preprocess(label_img)

                label_img = label_img.resize(rgb_img.size, Image.NEAREST)
                label_cv_img = np.array(label_img)
            else:
                label = torch.load(self.labels[index]).to(torch.float)
                label_cv_img = label.numpy().transpose(1, 2, 0)
        else:
            label_img = Image.new("L", rgb_img.size)
            label_cv_img = np.array(label_img)
            in_batch_mask_label = 0

        # print(rgb_cv_img.shape, label_cv_img.shape)
        transformed = self.transform(image=rgb_cv_img, mask=label_cv_img)

        rgb_img_orig = F.to_tensor(transformed["image"])
        if self.is_hard_label:
            label_img = torch.LongTensor(transformed["mask"].astype(np.int64))
        else:
            label_img = F.to_tensor(transformed["mask"])

        # Normalize the pixel values
        rgb_img = F.normalize(
            rgb_img_orig, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )

        # Spacial transform
        if self.mode == "train":
            i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(
                rgb_img,
                scale=self.scale,
                ratio=(0.75, 1.3333333333333333),
            )

            rgb_img = F.crop(rgb_img, i, j, h, w)
            rgb_img = F.resize(
                rgb_img, self.size,
                interpolation=InterpolationMode.BILINEAR)

            rgb_img_orig = F.crop(rgb_img_orig, i, j, h, w)
            rgb_img_orig = F.resize(
                rgb_img_orig, self.size,
                interpolation=InterpolationMode.BILINEAR)

            label_img = torch.unsqueeze(label_img, dim=0)
            label_img = F.crop(label_img, i, j, h, w)
            label_img = F.resize(
                label_img, self.size,
                interpolation=InterpolationMode.NEAREST)
            label_img = torch.squeeze(label_img)

        return {
            "image": rgb_img,
            "image_orig": rgb_img_orig,
            "image_path": self.images[index],
            "label": label_img,
            "label_path": self.labels[index],
            "name": self.images[index].rsplit("/", 1)[1],
        }
