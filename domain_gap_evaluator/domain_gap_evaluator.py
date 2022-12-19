"""This implements the method for evaluating the domain gaps of multiple datasets to a target dataset
proposed by Liu et al. in "Who is closer: A computational method for domain gap evaluation" (Pattern Recognition, 2021).
https://www.sciencedirect.com/science/article/pii/S0031320321004738?via%3Dihub
"""
from collections import OrderedDict
import torch
import numpy as np
from tqdm import tqdm
from typing import Union, Optional


def get_output(
    model: torch.Tensor,
    image: torch.Tensor,
    aux_weight: float = 0.5,
    device: str = "cuda",
) -> torch.Tensor:
    """Get output for the given image.
    The output of the model is processed according to its type.

    Parameters
    ----------
    model: torch.Tensor
        Model
    image: torch.Tensor
        Input image
    aux_weight: float
        Weight on the auxiliary output (if exists)
    device: str
        Device on which the computation is carried out

    Returns
    -------
    pred: torch.Tensor
        Predicted output from the model

    """
    output = model(image.to(device))

    if isinstance(output, OrderedDict) or isinstance(output, dict):
        if "aux" in output.keys():
            pred = output["out"] + aux_weight * output["aux"]
        else:
            pred = output["out"]
    elif isinstance(output, tuple):
        pred = output[0] + aux_weight * output[1]
    elif isinstance(output, torch.Tensor):
        pred = output
    else:
        print("Type {} is not supported.".format(type(output)))
        raise ValueError

    return pred


def calc_entropy(pred: torch.Tensor, is_prob: bool = False, reduction: str = "mean"):
    """Calculate entropy using the given class-wise scores

    Parameters
    ----------
    pred: `torch.Tensor`
        Predicted class-wise scores / probability
    is_prob: `bool`
        True if `pred` is normalized probability values
    reduction: `str`
        Condition of the returned entropy values. ["mean", "none"]

    Returns
    ------
    output: `float` or `torch.Tensor`
        Image-wise or pixel-wise entropy values depending on the option `is_prob`

    """
    if is_prob:
        p = pred
    else:
        softmax2d = torch.nn.Softmax2d()
        p = softmax2d(pred)

    logp = torch.log(p)
    pixel_ent = -(p * logp).sum(dim=1, keepdim=True)  # per pixel

    if reduction == "none":
        return pixel_ent
    else:
        return pixel_ent.mean()  # per image


def calc_norm_ent(
    target: Union[torch.utils.data.DataLoader, torch.Tensor],
    model: Optional[torch.Tensor]=None,
    device: str = "cuda",
    reduction: str = "mean",
) -> dict:
    """Calculate domain gap for one source

    Parameters
    ----------
    model: `torch.Tensor`
        Model
    target: `torch.utils.data.DataLoader` or `torch.Tensor`
        Target dataset loader or a tensor to evaluate
    device: `str`
        Device on which the computation is done

    Returns
    -------
    dict: `dict`
        Dictionary that stores entries as follows:
            "ent": `float`
                Normalized entropy value
            "p": `torch.Tensor` or `None`
                For input of `torch.Tensor`,
                return the tensor of predicted probability values
    """

    if model is not None:
        model.eval()

    softmax2d = torch.nn.Softmax2d()
    sum_ent = 0.0

    if isinstance(target, torch.utils.data.DataLoader):
        if model is None:
            print("model must be given if the target is a data loader")
            raise ValueError
        target_loader = target

        with torch.no_grad():
            with tqdm(total=len(target_loader)) as pbar:
                for i, batch in enumerate(tqdm(target_loader)):
                    image = batch["image"].to(device)
                    # output = model(image)

                    # # pred = model(image)["out"]
                    # pred = output["out"] + 0.5 * output["aux"]
                    pred = get_output(model, image, aux_weight=0.5, device=device)

                    # Entropy
                    # p = softmax2d(pred)
                    # logp = torch.log(p)
                    # pixel_ent = -(p * logp).sum(dim=1)  # per pixel
                    # image_ent_sum = pixel_ent.mean()  # per image
                    image_ent_sum = calc_entropy(pred, reduction=reduction)

                    sum_ent += image_ent_sum.item()

            pbar.close()

        # Number of classes
        C = pred.size(1)

        return {"ent": sum_ent / (len(target_loader) * np.log(C)), "out": None}

    elif isinstance(target, torch.Tensor):
        if model is not None:
            pred = get_output(model, target, aux_weight=0.5, device=device)
        else:
            pred = target

        # Entropy
        p = softmax2d(pred)
        image_ent_sum = calc_entropy(pred, is_prob=True, reduction=reduction)

        # Number of classes
        C = pred.size(1)

        return {"ent": image_ent_sum / np.log(C), "out": p}

    else:
        print("Type {} is not supported.".format(type(target)))
        raise ValueError


def calculate_domain_gap(
    model_list: list,
    target: Union[torch.utils.data.DataLoader, torch.Tensor],
    device: str = "cuda",
) -> dict:
    """Evaluate domain gap values for the given models and the target dataset / image

    Parameters
    ----------
    model_list: `list`
        List of models to evaluate
    target: `torch.utils.data.DataLoader` or `torch.Tensor`
        Target dataset loader or a tensor to evaluate
    device: `str`
        Device on which the computation is done

    Returns
    -------
    dict: `dict`
        Dictionary that stores entries as follows
            "domain_gap_list": `list`
                List of the estimated domain gap values
            "output": `list`
                List of the predicted probability values
                (only valid for per-image estimation)
    """
    domain_gap_value_list = []
    output_list = []

    for model in model_list:
        gap = calc_norm_ent(target, model=model, device=device)

        domain_gap_value_list.append(gap["ent"])
        output_list.append(gap["out"])

    return {"domain_gap_list": domain_gap_value_list, "output_list": output_list}
