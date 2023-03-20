# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import os
from typing import Optional
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from PIL import Image
import skimage.data
import skimage.color
import skimage.filters
import skimage.util
import skimage.segmentation

import torch
import torch.nn.functional as F

from domain_gap_evaluator.domain_gap_evaluator import calculate_domain_gap, calc_norm_ent
from dataset.tools.label_conversions import id_camvid_to_greenhouse
from dataset.tools.label_conversions import id_cityscapes_to_greenhouse
from dataset.tools.label_conversions import id_forest_to_greenhouse
from dataset.tools.label_conversions import id_camvid_to_sakaki
from dataset.tools.label_conversions import id_cityscapes_to_sakaki
from dataset.tools.label_conversions import id_forest_to_sakaki
from dataset.tools.label_conversions import id_camvid_to_oxford
from dataset.tools.label_conversions import id_cityscapes_to_oxford
from dataset.tools.label_conversions import id_forest_to_oxford
from utils.calc_prototype import ClassFeatures
from loss_fns.segmentation_loss import Entropy


def propagate_max_label_in_sp(sp, num_classes, min_portion=0.5, ignore_index=4):
    """
    Parameters
    ----------
    sp : `numpy.ndarray`
        Pixel label values in a superpixel
    num_classes : `int`
        The number of classes
    min_portion : `float`
        Minumum necessary proportion of the label in the superpixel to be propagated
    ignore_index : `int`
        ID to return when the maximum proportion is below `min_propotion`

    Returns
    -------
    output : `int`
        A label value that should be assigned to the superpixel
    """
    label_hist = np.bincount(sp, minlength=num_classes)

    valid_label_num = label_hist[0:num_classes].sum()
    argmax = np.argmax(label_hist[0:num_classes])
    #    print(valid_label_num[valid_label_num==argmax].sum() / float(sp.size))
    #    print("portion : {}".format(label_hist[argmax] / float(sp.size)))
    if valid_label_num and (label_hist[argmax] / float(sp.size) > min_portion):
        return argmax
    else:
        return ignore_index


def get_label_from_superpixel(
    rgb_img_np,
    label_img_np,
    sp_type="watershed",
    min_portion=0.5,
    ignore_index=3,
    num_classes=3,
):
    rgb_img_np = skimage.util.img_as_float(rgb_img_np)
    #    print(rgb_img_np.shape)

    # Superpixel segmentation
    if sp_type == "watershed":
        superpixels = skimage.segmentation.watershed(
            skimage.filters.sobel(skimage.color.rgb2gray(rgb_img_np)),
            markers=250,
            compactness=0.001,
        )
    elif sp_type == "quickshift":
        superpixels = skimage.segmentation.quickshift(
            rgb_img_np, kernel_size=3, max_dist=6, ratio=0.5
        )
    elif sp_type == "felzenszwalb":
        superpixels = skimage.segmentation.felzenszwalb(
            rgb_img_np, scale=100, sigma=0.5, min_size=50
        )
    elif sp_type == "slic":
        superpixels = skimage.segmentation.slic(
            rgb_img_np, n_segments=250, compactness=10, sigma=1
        )

    # Define a variable for a new label image
    new_label_img = np.zeros(label_img_np.shape)
    for i in range(0, superpixels.max()):
        # Get indeces of pixels in i+1 th superpixel
        index = superpixels == (i + 1)

        # Get labels within the superpixel
        labels_in_sp = label_img_np[index]

        # Get a label id that should be propagated within the superpixel
        max_label = propagate_max_label_in_sp(
            sp=labels_in_sp,
            num_classes=num_classes,
            min_portion=min_portion,
            ignore_index=ignore_index,
        )

        # Substitute the label in all pixels in the superpixel
        if max_label != ignore_index:
            new_label_img[index] = max_label
        else:
            new_label_img[index] = labels_in_sp

    return new_label_img


def get_output(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: str = "cuda",
):
    """Get an output on the given input image from the given model

    Parameters
    ----------
    model: `torch.nn.Module`
        Model
    image: `torch.Tensor`
        Input tensor
    device: `str`
        Device for computation

    Returns
    --------
    output : `dict`
        `out`: `torch.Tensor`
            A tensor of output as class probabilities
        `feat`: `torch.Tensor`
            A tensor of intermediate features
        `kld` : `torch.Tensor`
            A tensor of pixel-wise KLD (Currently not used, and `None` is set)
    """

    softmax2d = torch.nn.Softmax2d()
    # Forward the data
    output2 = model(image.to(device))

    pred_aux = None
    feat = None
    # Calculate the output from the two classification layers
    if isinstance(output2, OrderedDict) or isinstance(output2, dict):
        pred = output2["out"]
        if "aux" in output2.keys():
            pred_aux = output2["aux"]
        if "feat" in output2.keys():
            feat = output2["feat"]

    if pred_aux is not None:
        output2 = pred + 0.5 * pred_aux
    else:
        output2 = pred

    output = softmax2d(output2)  # softmax2d(output2).cpu().data[0].numpy()

    return {"out": output, "feat": feat, "kld": None}


def merge_outputs(amax_output_list, ignore_index=4):
    """Get an output on the given input image from the given model

    Parameters
    ----------
    amax_output_list : `list`
        List of predicted labels from the source models calculated by argmax
    ignore_index : `int`
        Label ID to be ignored during traing

    Returns
    --------
    pseudo_label : Generated pseudo-label

    """
    pseudo_label = amax_output_list[0]
    ignore_index_tensor = torch.ones_like(pseudo_label) * ignore_index
    for amax_output in amax_output_list[1:]:
        pseudo_label = torch.where(
            pseudo_label == amax_output, pseudo_label, ignore_index_tensor
        )

    return pseudo_label


def generate_pseudo_label(
    model: torch.nn.Module,
    testloader: torch.utils.data.DataLoader,
    num_classes: int,
    ignore_index: int,
    save_path: str,
    device: str = "cuda",
    prototypes: Optional[ClassFeatures] = None,
    proto_rect_thresh: float = 0.9,
    min_portion: float = -1.0,
    label_conversion: Optional[np.ndarray] = None,
):
    """Generate pseudo-labels using a pre-trained model

    Parameters
    ----------
    model: `torch.nn.Module`
        Current model
    testloader: `torch.utils.data.DataLoader`
        Dataloader
    num_classes: `int`
        The number of classes
    ignore_index: `int`
        Label index to ignore in training
    save_path: `str`
        Directory name to save the labels
    prototypes: `ClassFeatures`
        Representative features of the classes for label rectification
    proto_rect_thresh: `float`
        Confidence threshold for pseudo-label generation
    min_portion: `float`
        Minimum portion of the same label in a superpixel to be propagated
    label_conversion: `numpy.ndarray`
        Label conversion

    Returns
    --------
    class_wts: `torch.Tensor`
        Calculated class weights
    label_path_list: `list`
        List of the generated pseudo-labels
    """
    # model for evaluation
    model.eval()
    model.to(device)

    print("Ignore idx = {}".format(ignore_index))
    # evaluation process
    label_path_list = []
    class_array = np.zeros(num_classes)
    with torch.no_grad():
        with tqdm(total=len(testloader)) as pbar:
            for index, batch in enumerate(tqdm(testloader)):
                image = batch["image"].to(device)
                name = batch["name"]

                # Output: tensor, KLD: tensor, feature: tensor
                output = get_output(model, image)
                output_prob = output["out"]
                feature = output["feat"]
                max_output, argmax_output = torch.max(output_prob, dim=1)

                #
                # If the prototypes are given, apply filtering of the labels
                # (Zhang et al., 2021)
                #
                if prototypes is not None:
                    # Class-wise weights based on the distance to the prototype of each class
                    weights = prototypes.get_prototype_weight(
                        feature).to(device)

                    # Rectified output probability
                    rectified_prob = weights * output_prob
                    # Normalize the rectified values as probabilities
                    rectified_prob = rectified_prob / \
                        rectified_prob.sum(1, keepdim=True)
                    # Predicted label map after rectification
                    max_output, argmax_output = rectified_prob.max(
                        1, keepdim=True)

                # Filter out the pixels with a confidence below the threshold
                argmax_output[max_output < proto_rect_thresh] = ignore_index

                # Convert the label space from the source to the target
                if label_conversion is not None:
                    label_conversion = torch.tensor(
                        label_conversion).to(device)
                    argmax_output = label_conversion[argmax_output]

                for i in range(argmax_output.size(0)):
                    amax_output = argmax_output[i].squeeze().cpu().numpy()

                    if min_portion >= 0:
                        # Convert PIL image to Numpy
                        rgb_img_np = (
                            image[i].to("cpu").detach(
                            ).numpy().transpose(1, 2, 0)
                        )

                        amax_output = get_label_from_superpixel(
                            rgb_img_np=rgb_img_np,
                            label_img_np=amax_output,
                            sp_type="watershed",
                            min_portion=min_portion,
                        )

                    for j in range(0, num_classes):
                        class_array[j] += (amax_output == j).sum()

                    file_name = name[i].split("/")[-1]
                    image_name = file_name.rsplit(".", 1)[0]

                    # prob
                    # trainIDs/vis seg maps
                    amax_output = Image.fromarray(amax_output.astype(np.uint8))
                    # Save the predicted images (+ colorred images for visualization)
                    path = "{}/{}.png".format(save_path, image_name)
                    amax_output.save(path)

                    label_path_list.append(path)

    class_array /= class_array.sum()  # normalized
    class_wts = 1 / (class_array + 1e-10)

    print("class_wts : {}".format(class_wts))
    class_wts = torch.from_numpy(class_wts).float().to(device)
    class_wts = torch.clamp(class_wts, min=0.0, max=1e2)

    return class_wts, label_path_list


def generate_pseudo_label_multi_model(
    model_list: list,
    source_dataset_name_list: list,
    target_dataset_name: str,
    data_loader: torch.utils.data.DataLoader,
    num_classes: int,
    save_path: str,
    device: str = "cuda",
    min_portion: float = 0.5,
    ignore_index: int = 4,
    class_weighting: str = "normal",
):
    """Create the model and start the evaluation process."""

    for m in model_list:
        m.eval()
        m.to(device)

    if target_dataset_name == "greenhouse":
        from dataset.greenhouse import color_palette
    elif target_dataset_name == "imo":
        from dataset.imo import color_palette
    elif target_dataset_name == "sakaki":
        from dataset.sakaki import color_palette
    elif target_dataset_name == "oxfordrobot":
        from dataset.oxford_robot import color_palette
    else:
        print("Target {} is not supported.".format(target_dataset_name))
        raise ValueError

    # evaluation process
    class_array = np.zeros(num_classes)
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for index, batch in enumerate(tqdm(data_loader)):
                image = batch["image"].to(device)
                name = batch["name"]

                output_list = []
                for m, os_data in zip(model_list, source_dataset_name_list):
                    # Output: Numpy, KLD: Numpy
                    output = get_output(m, image)["out"]
                    amax_output = output.argmax(dim=1)

                    # Visualize pseudo labels
                    if target_dataset_name == "greenhouse":
                        # save visualized seg maps & predication prob map
                        if os_data == "camvid":
                            label_conversion = id_camvid_to_greenhouse
                        elif os_data == "cityscapes":
                            label_conversion = id_cityscapes_to_greenhouse
                        elif os_data == "forest":
                            label_conversion = id_forest_to_greenhouse
                    elif target_dataset_name == "sakaki" or target_dataset_name == "imo":
                        if os_data == "camvid":
                            label_conversion = id_camvid_to_sakaki
                        elif os_data == "cityscapes":
                            label_conversion = id_cityscapes_to_sakaki
                        elif os_data == "forest":
                            label_conversion = id_forest_to_sakaki

                    elif target_dataset_name == "oxfordrobot":
                        # save visualized seg maps & predication prob map
                        if os_data == "camvid":
                            label_conversion = id_camvid_to_oxford
                        elif os_data == "cityscapes":
                            label_conversion = id_cityscapes_to_oxford
                        elif os_data == "forest":
                            label_conversion = id_forest_to_oxford
                    else:
                        raise ValueError

                    label_conversion_t = torch.tensor(
                        label_conversion).to(device)

                    amax_output = label_conversion_t[
                        amax_output
                    ]  # Torch.cuda or numpy

                    output_list.append(amax_output)

                amax_output = merge_outputs(
                    output_list,
                    ignore_index=ignore_index,
                )

                # Use superpixel to propagate the pseudo-labels to nearby pixels with a similar color
                for i in range(amax_output.size(0)):
                    label = amax_output[i].to("cpu").detach().numpy()
                    if min_portion >= 0:
                        # Convert PIL image to Numpy
                        rgb_img_np = (
                            image[i].to("cpu").detach(
                            ).numpy().transpose(1, 2, 0)
                        )

                        label = get_label_from_superpixel(
                            rgb_img_np=rgb_img_np,
                            label_img_np=label,
                            sp_type="watershed",
                            min_portion=min_portion,
                            ignore_index=ignore_index,
                            num_classes=num_classes,
                        )

                    # Count the number of each class
                    for j in range(0, num_classes):
                        class_array[j] += (label == j).sum()

                    # File name ('xxx.png')
                    filename = name[i].split("/")[-1]
                    label = Image.fromarray(
                        label.astype(np.uint8)).convert("P")
                    label.putpalette(color_palette)
                    # Save the predicted images (+ colorred images for visualization)
                    # label.save("%s/%s.png" % (save_pred_path, image_name))
                    label.save(os.path.join(
                        save_path, filename.replace('.jpg', '.png')))

    #    update_image_list(tgt_train_lst, image_path_list, label_path_list, depth_path_list)

    if class_weighting == "normal":
        class_array /= class_array.sum()  # normalized
        class_wts = 1 / (class_array + 1e-10)
    #        if args.dataset == 'greenhouse' and not args.use_traversable:
    #            class_wts[0] = 0.0
    else:
        class_wts = np.ones(num_classes) / num_classes

    print("class_wts : {}".format(class_wts))
    class_wts = torch.from_numpy(class_wts).float().to(device)
    class_wts = torch.clamp(class_wts, min=0.0, max=1e2)

    # return the dictionary containing all the class-wise confidence vectors,
    # and the class_wts for loss weighting
    return class_wts


def generate_pseudo_label_multi_model_domain_gap(
    model_list: list,
    dg_model_list: list,
    source_dataset_name_list: list,
    target_dataset_name: str,
    data_loader: torch.utils.data.DataLoader,
    num_classes: int,
    save_path: str,
    device: str = "cuda",
    domain_gap_type: str = "none",
    #    use_domain_gap: bool = True,
    #    is_per_pixel: bool = False,
    #    is_per_sample: bool = False,
    ignore_index: int = 4,
    label_normalize: str = "softmax",
    class_weighting: str = "normal",
) -> torch.Tensor:
    """Generate multi-source pseudo-labels with domain gaps

    Generate pseudo-labels from the outputs of multiple source models
    considering relative domain gaps between each source and the target datasets
    so that one having less domain gap contributes more on the resulting pseudo-labels.

    Parameters
    ----------
    model_list: `list`
        List of source models to generate predicted labels
    dg_model_list: `list`
        List of source models to evaluate domain gaps
    source_dataset_name_list: `list`
        List of source dataset names
    target_dataset_name: `str`
        Name of the target dataset
    data_loader: `torch.utils.data.DataLoader`
        Target DataLoader
    num_classes: `int`
        Number of target classes
    device: `str`
        Device on which the computation is done
    domain_gap_type: `str`
        "none": Domain gap is not used
        "per_dataset": A weight is calculated for each dataset
        "per_sample": A weight is calculated for each sample (image)
        "per_pixel": A weight is calculated for each image pixel
    ignore_index: `int`,
        Label to be ignored in the target classes
    softmax_normalize: `bool`
        When calculating the importance weights based on the domain gap,
        if `True`, normalize the gap values by softmax,
        which emphasizes the difference of the values.
        Otherwise, normalize them by the sum.
    label_normalize: `str`
        Normalization method of the soft labels.
        [`L1`, `softmax`]
        Default: L1 (normalize by the sum)
    class_weighting: `str`
        Type of the class weights.
        "normal": More weight on less frequent class
        "inverse": Opposite as "normal"

    Returns
    -------
    class_wts: `torch.Tensor`
        Class weights of the resulting pseudo-labels
    """

    for m in model_list:
        m.eval()
        m.to(device)

    if target_dataset_name == "greenhouse":
        from dataset.greenhouse import color_palette
    elif target_dataset_name == "imo":
        from dataset.imo import color_palette
    elif target_dataset_name == "sakaki":
        from dataset.sakaki import color_palette
    elif target_dataset_name == "oxfordrobot":
        from dataset.oxford_robot import color_palette
    else:
        print("Target {} is not supported.".format(target_dataset_name))
        raise ValueError

    # evaluation process
    class_array = np.zeros(num_classes)

    if domain_gap_type not in ["none", "per_dataset", "per_sample", "per_pixel"]:
        raise ValueError(
            "Domain gap type '{}' is not supported".format(domain_gap_type))
    #
    # Calculate weights based on the domain gaps
    #
    # if not is_per_sample and use_domain_gap:
    if domain_gap_type == "per_dataset":
        # Domain gap. Less value means closer -> More importance.
        domain_gap_list = calculate_domain_gap(dg_model_list, data_loader, device)[
            "domain_gap_list"
        ]

        del dg_model_list

        domain_gap_weight = torch.Tensor(domain_gap_list).to(device)
        # if softmax_normalize:
        #     domain_gap_weight = torch.nn.Softmax2d()(domain_gap_weight)
        # else:
        domain_gap_weight /= domain_gap_weight.sum()

        # Calculate the inverse values of the domain gap
        domain_gap_weight = 1 / (domain_gap_weight + 1e-10)
        domain_gap_weight.to(device)

        print("Weight: {}".format(domain_gap_weight))
    elif domain_gap_type == "none":
        domain_gap_weight = torch.ones(num_classes)

        domain_gap_weight.to(device)

    entropy_layer = Entropy(num_classes=num_classes,)
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for index, batch in enumerate(tqdm(data_loader)):
                image = batch["image"]
                name = batch["name"]

                output_list = []
                gap_total = 0.0
                gap_list = []
                output_total = 0
                ds_index = 0
                for m, os_data in zip(model_list, source_dataset_name_list):
                    # Output: Numpy, KLD: Numpy
                    output = get_output(m, image)["out"]

                    # Extract the maximum value for each target class
                    output_target = torch.zeros(
                        (output.size(0), num_classes,
                         output.size(2), output.size(3))
                    ).to(device)

                    if target_dataset_name == "greenhouse":
                        if os_data == "camvid":
                            label_conversion = id_camvid_to_greenhouse
                        elif os_data == "cityscapes":
                            label_conversion = id_cityscapes_to_greenhouse
                        elif os_data == "forest":
                            label_conversion = id_forest_to_greenhouse
                        else:
                            raise ValueError
                    elif target_dataset_name == "sakaki" or target_dataset_name == "imo":
                        if os_data == "camvid":
                            label_conversion = id_camvid_to_sakaki
                        elif os_data == "cityscapes":
                            label_conversion = id_cityscapes_to_sakaki
                        elif os_data == "forest":
                            label_conversion = id_forest_to_sakaki

                    elif target_dataset_name == "oxfordrobot":
                        if os_data == "camvid":
                            label_conversion = id_camvid_to_oxford
                        elif os_data == "cityscapes":
                            label_conversion = id_cityscapes_to_oxford
                        elif os_data == "forest":
                            label_conversion = id_forest_to_oxford
                        else:
                            raise ValueError

                    label_conversion = torch.Tensor(
                        label_conversion).to(device)

                    for i in range(num_classes):
                        indices = torch.where(label_conversion == i)[0]
                        if indices.size(0):
                            output_target[:, i] = output[:, indices].max(dim=1)[
                                0]

                    # output_target = F.normalize(output_target, p=1)
                    output_target = F.softmax(output_target, dim=1)

                    output_list.append(output_target)

                    if domain_gap_type == "none":  # No domain gap
                        output_total += output_target
                    elif domain_gap_type == "per_sample":
                        domain_gap_w = calc_norm_ent(
                            output_target,
                            reduction="per_sample"
                        )["ent"]
                        output_total += output_target / domain_gap_w
                        gap_total += domain_gap_w
                    elif domain_gap_type == "per_pixel":
                        domain_gap_w = calc_norm_ent(
                            output_target,
                            reduction="none"
                        )["ent"]
                        output_total += output_target / domain_gap_w
                        gap_total += domain_gap_w
                    else:
                        output_total += output_target * \
                            domain_gap_weight[ds_index]
                        ds_index += 1

                if domain_gap_type == "per_sample":
                    output_total *= gap_total

                if label_normalize == "L1":
                    output_total = F.normalize(output_total, p=1)
                elif label_normalize == "softmax":
                    output_total = F.softmax(output_total, dim=1)
                else:
                    print(
                        "Label normalization type {} is not supported.".format(
                            label_normalize
                        )
                    )
                    raise ValueError

                entropy = entropy_layer(output_total)
                label = output_total.argmax(dim=1)
                for j in range(0, num_classes):
                    class_array[j] += (label == j).sum()

                # Save soft pseudo-labels
                for i in range(output_total.size(0)):
                    # Save the probabilities as float16 for saving the storage
                    output_prob_i = output_total[i].squeeze().to(
                        "cpu").type(torch.half)

                    file_name = name[i].split("/")[-1]
                    image_name = file_name.rsplit(".", 1)[0]

                    # Save the predicted images (+ colorred images for visualization)
                    torch.save(
                        output_prob_i,
                        os.path.join(save_path, "{}.pt".format(image_name)),
                    )

                    # Visualize entropy
                    ent = torch.exp(-entropy[i].detach() * 5.0)
                    ent = ent / ent.max()
                    ent = (ent * 255).cpu().byte().numpy()
                    ent = Image.fromarray(ent)
                    ent.save(os.path.join(
                        save_path, "{}_entropy.png".format(image_name)))

                    label = output_total[i].argmax(dim=0).cpu().byte().numpy()
                    label = Image.fromarray(
                        label.astype(np.uint8)).convert("P")
                    label.putpalette(color_palette)
                    # Save the predicted images (+ colorred images for visualization)
                    # label.save("%s/%s.png" % (save_pred_path, image_name))
                    label.save(os.path.join(
                        save_path, "{}_argmax.png".format(image_name)))

    if class_weighting == "normal":
        class_array /= class_array.sum()  # normalized
        class_wts = 1 / (class_array + 1e-10)
    else:
        class_wts = np.ones(num_classes) / num_classes

    print("class_wts : {}".format(class_wts))
    class_wts = torch.from_numpy(class_wts).float().to(device)
    class_wts = torch.clamp(class_wts, min=0.0, max=1e2)

    return class_wts


def class_balanced_pseudo_label_selection(
    P: torch.Tensor,
    num_classes: int,
    ignore_idx: int,
    alpha: float,
    tau: float,
) -> torch.Tensor:
    """Select labels per class

    Parameters
    ----------
    P : `torch.Tensor`
        Predicted probability
    num_classes : `int`
        The number of classes
    ignore_idx : `int`
        Label index to ignore in training
    alpha : `float`
        Proportion of labels to be selected per class
    tau : `float`
        Minimum confidence threshold

    Returns
    -------
    `torch.Tensor`
        Resulting pseudo-labels
    """
    Y = torch.argmax(P, dim=1)
    # Select the labels
    for c in range(num_classes):
        # P_c = sorted(P[:, c, :, :], reverse=True)[0]
        P_c = torch.clone(P[:, c, :, :])
        P_c = torch.reshape(P_c, (-1,))
        P_c = torch.sort(P_c, descending=True)[0]

        # Number of pixels predicted as class c
        n_c = (Y == c).sum()

        # Confidence threshold for pseudo-label c
        th = min(P_c[int(n_c * alpha)], tau)

        # Replace with 'ignore_idx' with the labels
        # with confidence below the threshold
        mask1 = (Y == c)
        mask2 = (P[:, c, :, :] <= th)

        Y[mask1 & mask2] = ignore_idx

    return Y
