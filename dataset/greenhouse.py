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


class GreenhouseRGBD(torch.utils.data.Dataset):
    def __init__(
        self,
        list_name,
        root=None,
        train=True,
        size=(256, 480),
        rough_plant=True,
    ):
        """Initialize a dataset

        Each line of a data list must be formatted as follows:
            rgb_image_path, depth_image_path, label_path, trav_mask_path, start_point_x, start_point_y, end_point_x, end_point_y

        Args:
            list_name (string): File name of the data list
            root (string): Name of the directory where the data list is stored
            train (Bool): True if the dataset is for training
            size (list): Image size to which the image is resized
            raw (Bool): True if the original pixel coordinates are used for training of the path line.
              Otherwise, they are scaled in [0, 1]
            rough_plant (Bool):
        """
        self.train = train
        self.rough_plant = rough_plant
        if root is not None:
            data_file = os.path.join(root, list_name)
        else:
            data_file = list_name

        # Initialize the lists
        self.images = []
        self.labels = []
        self.soft_pseudo_labels = []  # Not loaded here
        with open(data_file, "r") as lines:
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
                label_img_loc = line_split[1].rstrip()
                # Verify the existence of the file
                if label_img_loc != "" and not os.path.isfile(label_img_loc):
                    print("Not found : " + label_img_loc)
                    assert os.path.isfile(label_img_loc)
                self.labels.append(label_img_loc)

        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        (self.train_transforms, self.val_transforms) = self.transforms()

    def transforms(self):
        """Get transforms

        Returns:
            Tuple: set of transforms for train data and test data
        """
        train_transforms = A.Compose(
            [
                A.RandomResizedCrop(width=self.size[1], height=self.size[0]),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.5),
                A.GaussianBlur(p=0.5),
                A.RGBShift(p=0.5),
                A.RandomBrightnessContrast(
                    p=1.0, brightness_limit=0.3, contrast_limit=0.3
                ),
            ]
        )
        val_transforms = A.Compose(
            [
                A.Resize(width=480, height=256),
            ]
        )

        return train_transforms, val_transforms

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

    def set_soft_pseudo_label_list(self, soft_pseudo_label_list):
        """Set label list

        Args:
            real_noise (torch.Tensor): A tensor of original noisy images
            real_denoise (torch.Tensor): A tensor of original not noisy images
            tri_denoise (torch.Tensor): A tensor of images converted back to the denoised domain

        Returns:
            Dictionary: A dictionary that stores the losses
        """
        self.soft_pseudo_labels = copy.deepcopy(soft_pseudo_label_list)

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
        orig_img = None
        label_img = None

        #
        # Segmentation label
        #
        if self.labels[index] != "":
            label_img = Image.open(self.labels[index])
            in_batch_mask_label = 1

            if self.rough_plant:
                label_np = np.array(label_img, np.uint8)
                # 5: rough plant -> 1: other plant
                label_np[label_np == 5] = 1
                label_img = Image.fromarray(label_np)

            label_np = np.array(label_img, np.uint8)
            label_np[label_np == 0] = 1
            label_np -= 1
            label_img = Image.fromarray(label_np)
        else:
            label_img = Image.new("L", rgb_img.size)
            in_batch_mask_label = 0

        #
        # Soft pseudo-label
        #
        if self.soft_pseudo_labels:
            soft_pseudo_label = torch.load(self.soft_pseudo_labels[index]).to(
                torch.float
            )
            soft_pseudo_path = self.soft_pseudo_labels[index]
        else:
            soft_pseudo_label = torch.zeros((1, self.size[0], self.size[1]))
            soft_pseudo_path = ""

        # Pre-processing of the data
        rgb_cv_img = np.array(rgb_img)
        label_cv_img = np.array(label_img)
        if self.train:
            transformed = self.train_transforms(image=rgb_cv_img, mask=label_cv_img)
        else:
            transformed = self.val_transforms(image=rgb_cv_img, mask=label_cv_img)

        rgb_img_orig = F.to_tensor(transformed["image"])
        label_img = torch.LongTensor(transformed["mask"].astype(np.int64))

        # Normalize the pixel values
        rgb_img = F.normalize(
            rgb_img_orig, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )

        return {
            "image": rgb_img,
            "image_orig": rgb_img_orig,
            "image_path": self.images[index],
            "label": label_img,
            "label_path": self.labels[index],
            "name": self.images[index].rsplit("/", 1)[1],
        }
