# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================
import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
from typing import Optional, Union


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
        is_hard: bool = True,
        is_kld: bool = False,
    ):
        """Cross entropy loss with weights

        Parameters
        ----------
        num_classes: `int`
            Number of classes
        class_wts: `Optional[torch.Tensor]`
            Weights on loss of each class
        ignore_index: `int`
            Label index to ignore in training
        device: `str`
            Device on which the loss is computed
        temperature: `float`
            Temperature parameter of softmax
        reduction: `str`
            Reduction type of the loss
        is_hard: `bool`
            `True` to use hard label
        is_kld: `bool`
            `True` to use KLD loss for soft label. Valid only when `is_hard`==`False`
        
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
        self.is_hard = is_hard
        self.is_kld = is_kld

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        u_weight: Union[torch.Tensor, list],
        epsilon: float = 1e-12,
    ) -> torch.Tensor:
        """Forward calculation

        Parameters
        ----------

        Returns
        -------
        pred: `torch.Tensor`
            Prediction value in a shape (B, C, H, W)
        target: `torch.Tensor`
            Label value (B, H, W)
        u_weight: `Union[torch.Tensor, list]`
            Pixel-wise weight for loss function.
            If it's given as a list, weight is calculated as a product of the elements of the list.
            The weight(s) must be given with the same spatial shape (B, H, W) as `pred`.
        epsilon: `float` 
            CURRENTLY NOT USED.

        """
        torch.autograd.set_detect_anomaly(True)

        if isinstance(u_weight, list) and len(u_weight) > 0:
            u_weight_tmp = torch.ones(u_weight[0].size())
            for l in u_weight:
                u_weight_tmp = u_weight_tmp * l

        if self.is_hard:
            # Standard cross entropy
            seg_loss = F.cross_entropy(
                pred / self.T,
                target,
                weight=self.class_wts,
                ignore_index=self.ignore_index,
                reduction="none",
            )
        elif self.is_kld:
            # KLD between the predicted probability and the soft label
            pred_prob = F.log_softmax(pred / self.T, dim=1)
            seg_loss = F.kl_div(
                pred_prob, target, reduction="none"
            ).sum(dim=1)
        else:
            # Label is the argmax of the soft label
            target = target.argmax(dim=1)
            seg_loss = F.cross_entropy(
                pred / self.T,
                target,
                weight=self.class_wts,
                ignore_index=self.ignore_index,
                reduction="none",
            )

        # Rectify the loss with the uncertainty weights
        rect_ce = seg_loss * u_weight

        if self.is_hard and not self.reduction == "none":
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

class Entropy(nn.Module):
    def __init__(self, reduction='none', log_target=False, num_classes: int=None):
        super(Entropy, self).__init__()

        self.reduction = reduction
        self.log_target = log_target
        self.num_classes = num_classes
    
    def forward(self, p: torch.Tensor):
        """Calculate entropy of the probability distribution in (B, C, H, W)

        Parameters
        ----------
        p: `torch.Tensor`
            Probability distribution
        
        """
        if not self.log_target:
            ent = torch.sum(-p * torch.log(p), dim=1)
        else:
            ent = torch.sum(-torch.exp(p) * p, dim=1)
        
        if self.num_classes is not None:
            ent = ent / np.log(self.num_classes)

        if self.reduction == 'mean':
            ent = torch.mean(ent)
        elif self.reduction == 'sum':
            ent = torch.sum(ent)

        return ent