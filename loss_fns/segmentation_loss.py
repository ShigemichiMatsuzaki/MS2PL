# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================
from multiprocessing.sharedctypes import Value
import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Optional


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
        num_classes: int,
        class_wts: Optional[torch.Tensor] = None,
        ignore_index: int = None,
        device: str = "cuda",
        temperature: float = 1.0,
        reduction: str = "mean",
    ):
        """Cross entropy loss with weights

        Parameters
        ----------
        num_classes: `int`
            Number of classes
        class_wts: ``
        """

        super(UncertaintyWeightedSegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.class_wts = (
            class_wts if class_wts is not None else torch.ones(self.num_classes)
        )

        self.class_wts.to(device)
        self.ignore_index = ignore_index

        self.T = temperature

        # Check the reduction type
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError

        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        u_weight: torch.Tensor,
        epsilon: float = 1e-12,
        is_hard: bool = True,
    ) -> torch.Tensor:
        """Forward calculation

        Parameters
        ----------

        Returns
        -------

        """
        torch.autograd.set_detect_anomaly(True)

        if is_hard:
            seg_loss = F.cross_entropy(
                pred / self.T,
                target,
                weight=self.class_wts,
                ignore_index=self.ignore_index,
                reduction="none",
            )
        else:
            # Standard cross entropy
            pred_prob = F.log_softmax(pred / self.T, dim=1)
            seg_loss = F.kl_div(
                pred_prob, torch.log(target), log_target=True, reduction="none"
            ).sum(dim=1)

        # Rectify the loss with the uncertainty weights
        rect_ce = seg_loss * torch.exp(-u_weight)

        if is_hard and not self.reduction == "none":
            rect_ce = rect_ce[target != self.ignore_index]

        if self.reduction == "mean":
            return rect_ce.mean()
        elif self.reduction == "sum":
            return rect_ce.sum()
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
