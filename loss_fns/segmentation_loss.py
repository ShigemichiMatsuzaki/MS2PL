# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================
from multiprocessing.sharedctypes import Value
import torch
from torch import nn
from torch.nn import functional as F
import math


class SoftArgMax(nn.Module):
    def __init__(self):
        super(SoftArgMax, self).__init__()

    def soft_arg_max(self, A, beta=500, dim=1, epsilon=1e-12):
        """
        applay softargmax on A and consider mask, return \sum_i ( i * (exp(A_i * beta) / \sum_i(exp(A_i * beta))))
        according to https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
        :param A:
        :param mask:
        :param dim:
        :param epsilon:
        :return:
        """
        # According to https://discuss.pytorch.org/t/apply-mask-softmax/14212/7
        A_max = torch.max(A, dim=dim, keepdim=True)[0]
        A_exp = torch.exp((A - A_max) * beta)
        A_softmax = A_exp / (torch.sum(A_exp, dim=dim, keepdim=True) + epsilon)
        indices = (
            torch.arange(start=0, end=A.size()[dim])
            .float()
            .reshape(1, A.size()[dim], 1, 1)
        )

        return F.conv2d(A_softmax.to("cuda"), indices.to("cuda"))

    def forward(self, x):
        return self.soft_arg_max(x)


class UncertaintyWeightedSegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        class_weights=None,
        ignore_idx=None,
        ignore_index=None,
        device="cuda",
        temperature=1.0,
        reduction="mean",
    ):
        super(UncertaintyWeightedSegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = (
            class_weights
            if class_weights is not None
            else torch.ones(self.num_classes).to(device)
        )
        self.ignore_index = ignore_index if ignore_index is not None else ignore_idx

        self.T = temperature

        # Check the reduction type
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError

        self.reduction = reduction

    def forward(self, pred, target, u_weight, epsilon=1e-12):
        torch.autograd.set_detect_anomaly(True)
        # Calculate softmax probability
        batch_size = pred.size()[0]
        H = pred.size()[2]
        W = pred.size()[3]

        # Standard cross entropy
        ce = F.cross_entropy(
            pred / self.T,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # Rectify the loss with the uncertainty weights
        rect_ce = ce * torch.exp(-u_weight)

        if self.reduction == "mean":
            return rect_ce[target != self.ignore_index].mean()
        elif self.reduction == "sum":
            return rect_ce[target != self.ignore_index].sum()
        elif self.reduction == "none":  # if reduction=='none' or any other
            return rect_ce
        else:
            raise ValueError


class DistillationSegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        class_weights=None,
        ignore_idx=None,
        ignore_index=None,
        device="cuda",
        temperature=1.0,
        reduction="mean",
    ):
        super(DistillationSegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = (
            class_weights
            if class_weights is not None
            else torch.ones(self.num_classes).to(device)
        )
        self.ignore_index = ignore_index if ignore_index is not None else ignore_idx

        self.T = temperature

        # Check the reduction type
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError

        self.reduction = reduction

    def forward(self, pred, target, epsilon=1e-12):
        torch.autograd.set_detect_anomaly(True)
        # Calculate softmax probability
        batch_size = pred.size()[0]
        H = pred.size()[2]
        W = pred.size()[3]

        pred = pred / self.T
        target = target / self.T

        # From this answer
        # https://stackoverflow.com/questions/68907809/soft-cross-entropy-in-pytorch
        p = F.log_softmax(pred, 1)
        class_weights = self.class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        w_labels = class_weights * target

        loss = -(w_labels * p).sum() / (w_labels).sum()

        return loss


class PixelwiseKLD(nn.Module):
    def __init__(self):
        super(PixelwiseKLD, self).__init__()

    def forward(self, dist1, dist2):
        # Calculate probability and log probability
        p1 = F.softmax(dist1, dim=1)
        logp1 = F.log_softmax(dist1, dim=1)
        logp2 = F.log_softmax(dist2, dim=1)

        kld_i = p1 * logp1 - p1 * logp2

        return torch.sum(kld_i, dim=1)
