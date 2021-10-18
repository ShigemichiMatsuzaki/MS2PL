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

FOREST_CLASS_LIST = ['road', 'grass', 'tree', 'sky', 'obstacle']

color_encoding = OrderedDict([
    ('road', (170, 170, 170)),
    ('grass', (0, 255, 0)),
    ('tree', (102, 102, 51)),
    ('sky', (0, 120, 255)),
    ('obstacle', (0, 0, 0))
])

color_to_id = {
    (255, 255, 255) : 0,
    (170, 170, 170) : 1,
    (0, 255, 0)     : 2,
    (102, 102, 51)  : 3,
    (0, 60, 0)      : 3,
    (0, 120, 255)   : 4,
    (0, 0, 0)       : 5
}

color_palette = [
    170, 170, 170,
    0, 255, 0,
    102, 102, 51,
    0, 120, 255,
    0, 0, 0
]

class FreiburgForestDataset(data.Dataset):

    def __init__(self, root='/tmp/dataset/freiburg_forest_annotated/', train=True,
        scale=(0.5, 2.0), size=(480, 256), normalize=True):

        self.root = root
        self.train = train
        self.normalize = normalize

        if self.train:
            data_dir = os.path.join(root, 'train')
        else:
            data_dir = os.path.join(root, 'test')

        self.images = sorted(glob.glob(os.path.join(data_dir, 'rgb/*.jpg')))
        self.masks = sorted(glob.glob(os.path.join(data_dir, 'GT_color/*.png')))

        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rgb_img = Image.open(os.path.join(self.root, self.images[index])).convert('RGB')
        label_img = Image.open(os.path.join(self.root, self.masks[index]))

        # Convert color label to label id
        label_img_color_np = np.array(label_img, np.uint8)
        label_img_id_np = np.zeros((label_img_color_np.shape[:2]), dtype=np.uint8)
        for color in color_to_id:
            label_img_id_np[(label_img_color_np == color).all(axis=2)] = color_to_id[color]
#        label_img = Image.fromarray(label_img_id_np)

        # Convert the label values
        label_img_id_np -= 1
        label_img_id_np[label_img_id_np < 0] = 255 # void 
        label_img = Image.fromarray(label_img_id_np)

        # Resize
        rgb_img   = F.resize(rgb_img, self.size)
        label_img = F.resize(label_img, self.size)

        # Convert images to tensors
        rgb_img = F.to_tensor(rgb_img)
        label_img = torch.LongTensor(np.array(label_img).astype(np.int64))

        # Normalize the pixel values
        if self.normalize:
            rgb_img = F.normalize(rgb_img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        return {"image": rgb_img, "label": label_img}
