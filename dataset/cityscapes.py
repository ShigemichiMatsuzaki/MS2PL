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

CITYSCAPE_CLASS_LIST = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic light',
    'traffic sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
    'background']

color_encoding = OrderedDict([
    ('road',(128, 64,128)),
    ('sidewalk',(244, 35,232)),
    ('building',(70, 70, 70)),
    ('wall',(102,102,156)),
    ('fence',(190,153,153)),
    ('pole',(153,153,153)),
    ('traffic light',(250,170, 30)),
    ('traffic sign',(220,220,  0)),
    ('vegetation',(107,142, 35)),
    ('terrain',(152,251,152)),
    ('sky',(70,130,180)),
    ('person',(220, 20, 60)),
    ('rider',(255,  0,  0)),
    ('car',(0,  0,142)),
    ('truck',(0,  0, 70)),
    ('bus',(0, 60,100)),
    ('train',(0, 80,100)),
    ('motorcycle',(0,  0,230)),
    ('bicycle',(119, 11, 32)),
    ('background',(0,  0,  0)),
])

class CityscapesSegmentation(data.Dataset):

    def __init__(self, root, mode='train', scale=(0.5, 2.0), size=(1024, 512), ignore_idx=255, coarse=True, normalize=True):

        self.mode = mode
        self.normalize = normalize
        self.annot_type = 'gtFine' if coarse else 'gtCoarse'
        image_dir = os.path.join(root, 'leftImg8bit')
        label_dir = os.path.join(root, self.annot_type)
        self.images = []
        self.labels = []
        if self.mode == 'train':
            data_train_image_dir = os.path.join(image_dir, 'train')
            data_train_label_dir = os.path.join(label_dir, 'train')
            self.images += sorted(glob.glob(os.path.join(data_train_image_dir, '*/*.png')))
            self.labels += sorted(glob.glob(os.path.join(data_train_label_dir, '*/*labelIds.png')))

            if self.annot_type == 'gtCoarse':
                data_train_image_dir = os.path.join(image_dir, 'train_extra')
                data_train_label_dir = os.path.join(label_dir, 'train_extra')
                self.images += sorted(glob.glob(os.path.join(data_train_image_dir, '*/*.png')))
                self.labels += sorted(glob.glob(os.path.join(data_train_label_dir, '*/*labelIds.png')))

        elif self.mode == 'val':
            data_train_image_dir = os.path.join(image_dir, 'val')
            data_train_label_dir = os.path.join(label_dir, 'val')
            self.images += sorted(glob.glob(os.path.join(data_train_image_dir, '*/*.png')))
            self.labels += sorted(glob.glob(os.path.join(data_train_label_dir, '*/*labelIds.png')))
        elif self.mode == 'test':
            label_dir = os.path.join(root, 'gtFine')
            data_train_image_dir = os.path.join(image_dir, 'test')
            data_train_label_dir = os.path.join(label_dir, 'test')
            self.images += sorted(glob.glob(os.path.join(data_train_image_dir, '*/*.png')))
            self.labels += sorted(glob.glob(os.path.join(data_train_label_dir, '*/*labelIds.png')))

        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        if isinstance(scale, tuple):
            self.scale = scale
        else:
            self.scale = (scale, scale)

        self.ignore_idx = ignore_idx

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

