import torch
from typing import Optional


def get_loss_function(
    loss_name: str,
    class_weights: Optional[torch.Tensor] = None,
    reduction: Optional[str] = "mean",
    ignore_index: Optional[int] = 255,
):
    """Get loss function for semantic segmentation

    Parameters
    ----------
    loss_name: `str`
        Name of the loss function
    class_weights

    Returns
    -------
    loss: `nn.Module`

    """
    if loss_name == "ce":
        loss = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            reduction=reduction,
            ignore_index=ignore_index,
        )
    elif loss_name == "iou":
        from loss_fns.seg_loss.losses_pytorch.dice_loss import IoULoss
        loss = IoULoss()
    elif loss_name == "focal":
        from loss_fns.seg_loss.losses_pytorch.focal_loss import FocalLoss
        loss = FocalLoss()
    elif loss_name == "dice":
        from loss_fns.seg_loss.losses_pytorch.dice_loss import GDiceLossV2
        loss = GDiceLossV2()
    else:
        raise ValueError

    return loss
