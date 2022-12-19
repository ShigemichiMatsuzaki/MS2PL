# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Original code:
#   https://github.com/microsoft/ProDA/blob/main/calc_prototype.py

import os
import logging
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

def calc_prototype(model, dataset, num_classes, device='cpu'):
    ''' Calculate the prototype feature of each class

    Parameters
    ----------
    model: torch.Module
        A pre-trained model to produce output features
    dataset: torch.utils.data.Dataset
        A dataset of images
    num_classes: 
        The number of classes in the dataset
    
    Returns
    --------
    class_features: ClassFeatures
        An object to store the prototypes

    '''
    #
    # Load datasets (source and target)
    #
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    #
    # Define a class feature object
    #
    class_features = ClassFeatures(num_classes=num_classes, device=device)

    model.eval()
#    model.to('cuda')
    # begin training
    for epoch in range(1):
        for batch in tqdm(data_loader):
            images = batch["image"].to('cuda')
            labels = batch["label"].to(device)

            with torch.no_grad():
                # Get outputs
                out = model(images)
                prob = (out["out"] + out["aux"] * 0.5).to(device) # segmentation
                feat = out["feat"].to(device) # feature (32-dimensional in espdnetue)

                batch, w, h = labels.size()
                newlabels = labels.reshape([batch, 1, w, h]).float()
                newlabels = F.interpolate(newlabels, size=feat.size()[2:], mode='nearest')

                # Calculate the mean of the features for each class in this batch
                vectors, ids = class_features.calculate_mean_vector(feat, prob, newlabels)

                # Update the overall mean of the features
                for t in range(len(ids)):
                    class_features.update_objective_SingleVector(ids[t], vectors[t].detach().to(device), 'mean')

    return class_features

class ClassFeatures(object):
    """ Manage the prototype feature of each class

    Attributes
    ----------
    num_classes : int
        The number of classes
    device :
        Device name used for the computation
    

    Methods
    -------
    calculate_mean_vector(feat_cls, outputs, labels_val=None, thresh=None)

    update_objective_SingleVector(id, vector, objective_vectors, objective_vectors_num, 
                                  name='moving_average', start_mean=True, proto_momentum=0.9999)
        Update the prototype of the specified class with the given vectors

    """
    def __init__(self, num_classes=19, device='cpu'):
        self.device = device
        self.num_classes = num_classes
        self.class_features = [[] for i in range(self.num_classes)]
        self.objective_vectors = torch.zeros([self.num_classes, 32]).to(self.device)
        self.objective_vectors_num = torch.zeros([self.num_classes]).to(self.device)

    def calculate_mean_vector(self, feat_cls, outputs, labels_val=None, thresh=-1):
        ''' Calculate the prototype feature of each class

        Parameters
        ----------
        feat_cls:
            A tensor consisting of K dimensional feature vector (B, K, H, W)
        outputs:
            C dimensional output tensor (B, C, H, W), C: the number of classes
        labels_val:
            Ground truth label tensor (B, 1, H, W)
        thresh:
            Threshold of the confidence of pixels used for calculating the mean vector

        Returns
        --------
        vectors: List
        
        ids: List

        '''
        # Class robability
        outputs_softmax = F.softmax(outputs, dim=1)

        # Mask the features used for the calculation of the prototypes
        #  by the confidence threshold
        outputs_conf = outputs_softmax.max(dim=1, keepdim=True)[0]
        mask = outputs_conf.ge(thresh)

        # Class ID
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)

        # One-hot
        outputs_onehot = self.process_label(outputs_argmax.float())

        # If a ground truth label is not given, use the prediction.
        #  otherwise use the ground truth
        if labels_val is None:
            outputs_pred = outputs_onehot
        else:
            labels_onehot = self.process_label(labels_val)

            # Take AND of the ground truth labels and the predicted labels?
            outputs_pred = labels_onehot * outputs_onehot

        # Average of the one-hot vectors = probability 
        scale_factor = F.adaptive_avg_pool2d(outputs_pred * mask, 1)
        vectors = []
        ids = []
        # For each image in the batch
        for n in range(feat_cls.size()[0]):
            # For each class
            for t in range(self.num_classes):
                # If scale factor of class t (probability over the image) is 0
                #  (= there is no pixel with class t in the image),
                #  skip the following process
                if scale_factor[n][t].item()==0:
                    continue

                # If the number of pixels with class t is less than 10 in the image,
                #  skip the following process
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue

                # feat_cls[n]: (K, H, W)
                # outputs_pred[n][t]: (H, W)
                # extract features with class t
                s = feat_cls[n] * outputs_pred[n][t] * mask[n]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]

                vectors.append(s)
                ids.append(t)

        return vectors, ids


    def update_objective_SingleVector(self, id, vector, name='moving_average', start_mean=True, proto_momentum=0.9999):
        ''' Calculate the prototype feature of each class

        Parameters
        ----------
        id: int
            Class ID
        vector: 
            New vectors
        name:
        start_mean:

        Returns
        --------

        '''
        vector = vector.to(self.device)
        if vector.sum().item() == 0:
            return
        if start_mean and self.objective_vectors_num[id].item() < 100:
            name = 'mean'
        if name == 'moving_average':
            self.objective_vectors[id] = self.objective_vectors[id] * (1 - proto_momentum) + proto_momentum * vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
        elif name == 'mean':
            self.objective_vectors[id] = self.objective_vectors[id] * self.objective_vectors_num[id] + vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors[id] = self.objective_vectors[id] / self.objective_vectors_num[id]
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
        else:
            raise NotImplementedError('no such updating way of objective vectors {}'.format(name))


    def process_label(self, label):
        ''' Convert the label tensor to one-hot representation

        Parameters
        ----------
        label: torch.Tensor
            A tensor of label values (B, 1, H, W)

        Returns
        --------
        pred1: torch.Tensor
            A tensor of one-hot representation of the label (B, C, H, W)

        '''

        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, self.num_classes + 1, w, h).to(self.device)
        id = torch.where(label < self.num_classes, label, torch.Tensor([self.num_classes]).to(self.device))
        pred1 = pred1.scatter_(1, id.long(), 1)

        return pred1


    def full2weak(self, feat, target_weak_params):
        ''' Convert the label tensor to one-hot representation

        Parameters
        ----------
        label: torch.Tensor
            A tensor of label values (B, 1, H, W)

        Returns
        --------
        pred1: torch.Tensor
            A tensor of one-hot representation of the label (B, C, H, W)

        '''#
        for i in range(feat.shape[0]):
            h, w = target_weak_params['RandomSized'][0][i], target_weak_params['RandomSized'][1][i]
            feat_ = F.interpolate(feat[i:i+1], size=[int(h/4), int(w/4)], mode='bilinear', align_corners=True)
            y1, y2, x1, x2 = target_weak_params['RandomCrop'][0][i], target_weak_params['RandomCrop'][1][i], target_weak_params['RandomCrop'][2][i], target_weak_params['RandomCrop'][3][i]
            y1, th, x1, tw = int(y1/4), int((y2-y1)/4), int(x1/4), int((x2-x1)/4)
            feat_ = feat_[:, :, y1:y1+th, x1:x1+tw]
            if target_weak_params['RandomHorizontallyFlip'][i]:
                inv_idx = torch.arange(feat_.size(3)-1,-1,-1).long().to(self.device)
                feat_ = feat_.index_select(3,inv_idx)
            tmp.append(feat_)

        feat = torch.cat(tmp, 0)
        return feat


    def feat_prototype_distance(self, feat):
        N, C, H, W = feat.shape
        feat_proto_distance = -torch.ones((N, self.num_classes, H, W)).to(self.device)
        for i in range(self.num_classes):
            #feat_proto_distance[:, i, :, :] = torch.norm(torch.Tensor(self.objective_vectors[i]).reshape(-1,1,1).expand(-1, H, W).to(feat.device) - feat, 2, dim=1,)
            feat_proto_distance[:, i, :, :] = torch.norm(self.objective_vectors[i].reshape(-1,1,1).expand(-1, H, W) - feat, 2, dim=1,)
        return feat_proto_distance


    def get_prototype_weight(self, feat, label=None, target_weak_params=None, proto_temperature=1.0):
        feat = feat.to(self.device)
        if target_weak_params is not None:
            feat = self.full2weak(feat, target_weak_params)

        feat_proto_distance = self.feat_prototype_distance(feat)
        feat_nearest_proto_distance, feat_nearest_proto = feat_proto_distance.min(dim=1, keepdim=True)

        feat_proto_distance = feat_proto_distance - feat_nearest_proto_distance
        weight = F.softmax(-feat_proto_distance * proto_temperature, dim=1)
        return weight


def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    file_path = os.path.join(logdir, 'run.log')
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--source', action='store_true', help='calc source prototype')
#    parser = parser_(parser)
    opt = parser.parse_args()

#    opt = relative_path_to_absolute_path(opt)
    opt.logdir = opt.logdir.replace(opt.name, 'debug')
    opt.noaug = True
    opt.noshuffle = True
    opt.epochs = 4
    #opt.num_workers = 0

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    calc_prototype(opt, logger)
