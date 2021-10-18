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

CAMVID_CLASS_LIST = [
    'Sky', 
    'Building', 
    'Pole', 
    'Road', 
    'Pavement', 
    'Tree', 
    'SignSymbol', 
    'Fence', 
    'Car', 
    'Pedestrian', 
    'Bicyclist', 
    'Road_marking', 
    'Unlabelled']

color_encoding = OrderedDict([
    ('Sky', (128,128,128)),
    ('Building', (128,0,0)),
    ('Pole', (192,192,128)),
    ('Road', (128,64,128)),
    ('Pavement', (60,40,222)),
    ('Road_marking', (255,69,0)),
    ('Unlabelled', (0,0,0))
])

id_camvid_to_greenhouse = np.array([
    4, # Sky
    2, # Building
    2, # Pole
    3, # Road
    3, # Pavement
    1, # Tree
    2, # SignSymbol
    2, # Fence
    2, # Car
    4, # Pedestrian
    4, # Bicyclist
    2, # Road_marking(?)
    4  # Unlabeled
])

class CamVidSegmentation(data.Dataset):

    def __init__(self, root, mode='train', scale=(0.5, 2.0), size=(360, 480), normalize=True):
        self.mode = mode
        self.normalize = normalize
        self.images = []
        self.labels = []

        if self.mode == 'val':
            data_train_image_dir = os.path.join(root, 'val')
            data_train_label_dir = os.path.join(root, 'valannot')
            self.images += sorted(glob.glob(os.path.join(data_train_image_dir, '*.png')))
            self.labels += sorted(glob.glob(os.path.join(data_train_label_dir, '*.png')))
        elif self.mode == 'test':
            data_train_image_dir = os.path.join(root, 'test')
            data_train_label_dir = os.path.join(root, 'testannot')
            self.images += sorted(glob.glob(os.path.join(data_train_image_dir, '*.png')))
            self.labels += sorted(glob.glob(os.path.join(data_train_label_dir, '*.png')))
        else:
            data_train_image_dir = os.path.join(root, 'train')
            data_train_label_dir = os.path.join(root, 'trainannot')
            self.images += sorted(glob.glob(os.path.join(data_train_image_dir, '*.png')))
            self.labels += sorted(glob.glob(os.path.join(data_train_label_dir, '*.png')))

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
        rgb_img = Image.open(self.images[index]).convert('RGB')
        label_img = Image.open(self.labels[index])

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

