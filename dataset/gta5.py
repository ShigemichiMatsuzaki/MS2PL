import os
from collections import OrderedDict
from dataset.base_dataset import BaseDataset
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import torch
from .cityscapes_labels import labels

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


class GTA5(BaseDataset):
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
            max_iter=max_iter,
        )

        # Image directories
        data_train_image_dir = os.path.join(self.root, "images")
        data_train_label_dir = os.path.join(self.root, "labels")

        if self.mode in ['train', 'val', 'test']:
            data_file = os.path.join(self.root, self.mode + '.lst')
        else:
            print("Mode '{}' is not supported".format(self.mode))
            raise ValueError

        with open(data_file, "r") as lines:
            for line in lines:
                file_name = '{:05d}'.format(int(line.rstrip()))+'.png'
                #
                # RGB
                #
                rgb_img_loc = os.path.join(
                    data_train_image_dir, file_name)

                if not os.path.isfile(rgb_img_loc):
                    print("Not found : " + rgb_img_loc)
                    assert os.path.isfile(rgb_img_loc)
                self.images.append(rgb_img_loc)
                #
                # Segmentation label
                #
                label_img_loc = os.path.join(
                    data_train_label_dir, file_name)

                if not os.path.isfile(label_img_loc):
                    print("Not found : " + label_img_loc)
                    assert os.path.isfile(label_img_loc)

                self.labels.append(label_img_loc)

        self.label_conversion_map = id_cityscapes_to_greenhouse
        self.size = (height, width)

        if self.max_iter is not None and self.max_iter > len(self.images):
            self.images *= self.max_iter // len(self.images)
            self.labels *= self.max_iter // len(self.labels)

        self.id2trainId = np.array([label.trainId for label in labels])
        self.id2trainId[self.id2trainId < 0] = 255
        print(self.id2trainId)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_img = Image.open(self.images[index]).convert("RGB")
        label_img = Image.open(self.labels[index])

        rgb_np = np.array(rgb_img)
        label_np = np.array(label_img).astype(np.uint8)
        label_np = self.id2trainId[label_np]

        # Convert images to tensors
        # label_img = np.array(label_img)
        if self.label_conversion and self.label_conversion_map is not None:
            label_img = self.label_conversion_map[label_img]

        transformed = self.transform(image=rgb_np, mask=label_np)

        rgb_img_orig = F.to_tensor(transformed["image"])
        label_img = torch.LongTensor(transformed["mask"].astype(np.int64))

        # Normalize the pixel values
        rgb_img = F.normalize(
            rgb_img_orig, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )

        return {"image": rgb_img, "image_orig": rgb_img_orig, "label": label_img}
