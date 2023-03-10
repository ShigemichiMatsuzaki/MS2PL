# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================
import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MIOU(object):
    def __init__(self, num_classes=21):
        self.num_classes = num_classes
        self.epsilon = 1e-6

    def get_iou(self, output, target):
        if isinstance(output, tuple):
            output = output[0]

        if len(output.size()) == 4:  # Case of raw outputs
            _, pred = torch.max(output, 1)
        else:  # Case of argmax
            pred = output

        # histc in torch is implemented only for cpu tensors, so move your tensors to CPU
        if pred.device == torch.device("cuda"):
            pred = pred.cpu()
        if target.device == torch.device("cuda"):
            target = target.cpu()

        pred = pred.type(torch.ByteTensor)
        target = target.type(torch.ByteTensor)

        # shift by 1 so that 255 is 0
        pred += 1
        target += 1

        pred = pred * (target > 0)
        # pred = pred * (target < self.num_classes)
        inter = pred * (pred == target)
        # inter = pred * (target < self.num_classes)
        area_inter = torch.histc(
            inter.float(), bins=self.num_classes, min=1, max=self.num_classes
        )
        area_pred = torch.histc(
            pred.float(), bins=self.num_classes, min=1, max=self.num_classes
        )
        area_mask = torch.histc(
            target.float(), bins=self.num_classes, min=1, max=self.num_classes
        )
        area_union = area_pred + area_mask - area_inter + self.epsilon

        return area_inter.numpy(), area_union.numpy()
