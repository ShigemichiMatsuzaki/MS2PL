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

from domain_gap_evaluator.domain_gap_evaluator import calculate_domain_gap
from dataset.camvid import id_camvid_to_greenhouse
from dataset.cityscapes import id_cityscapes_to_greenhouse
from dataset.forest import id_forest_to_greenhouse


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
    model,
    image,
    device="cuda",
    use_depth=False,
):
    """Get an output on the given input image from the given model

    Parameters
    ----------
    model  :

    image  :

    model_name :

    device :

    use_depth :

    Returns
    --------
    output : torch.Tensor
        A tensor of output as class probabilities
    kld : torch.Tensor

    """

    softmax2d = torch.nn.Softmax2d()
    """
    Get outputs from the input images
    """
    # Forward the data
    if not use_depth:  # or outsource == 'camvid':
        output2 = model(image.to(device))

    pred_aux = None
    # Calculate the output from the two classification layers
    if isinstance(output2, OrderedDict) or isinstance(output2, dict):
        pred = output2["out"]
        if "aux" in output2.keys():
            pred_aux = output2["aux"]

    if pred_aux is not None:
        output2 = pred + 0.5 * pred_aux
    else:
        output2 = pred

    output = softmax2d(output2)  # softmax2d(output2).cpu().data[0].numpy()

    return output


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
    args,
    model: torch.nn.Module,
    testloader: torch.utils.data.DataLoader,
    save_path: str,
    proto_rect_thresh: float = 0.9,
    min_portion: float = -1.0,
    label_conversion: Optional[np.ndarray] = None,
):
    """Generate pseudo-labels using a pre-trained model

    Parameters
    ----------
    args:
        Args
    model: `torch.nn.Module`
        Current model
    testloader: `torch.utils.data.DataLoader`
        Dataloader
    save_path: `str`
        Directory name to save the labels
    proto_rect_thresh: `float`
        Confidence threshold for pseudo-label generation
    min_portion: `float`
        Minimum portion of the same lable in a superpixel to be propagated
    label_conversion: `numpy.ndarray`
        Label conversion

    Returns
    --------
    class_weights: `torch.Tensor`
        Calculated class weights
    label_path_list: `list`
        List of the generated pseudo-labels
    """
    # model for evaluation
    model.eval()
    model.to(args.device)

    # evaluation process
    label_path_list = []
    class_array = np.zeros(args.num_classes)
    with torch.no_grad():
        with tqdm(total=len(testloader)) as pbar:
            for index, batch in enumerate(tqdm(testloader)):
                image = batch["image"].to(args.device)
                name = batch["name"]

                # Output: tensor, KLD: tensor, feature: tensor
                output_prob = get_output(model, image)
                max_output, argmax_output = torch.max(output_prob, dim=1)

                # Filter out the pixels with a confidence below the threshold
                argmax_output[max_output <
                              proto_rect_thresh] = args.ignore_index

                # Convert the label space from the source to the target
                if label_conversion is not None:
                    label_conversion = torch.tensor(
                        label_conversion).to(args.device)
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

                    for j in range(0, args.num_classes):
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

    print("class_weights : {}".format(class_wts))
    class_wts = torch.from_numpy(class_wts).float().to(args.device)

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
                    output = get_output(m, image)
                    amax_output = output.argmax(dim=1)

                    # Visualize pseudo labels
                    if target_dataset_name == "greenhouse":
                        # save visualized seg maps & predication prob map
                        if os_data == "camvid":
                            from dataset.camvid import (
                                id_camvid_to_greenhouse as label_conversion,
                            )
                        elif os_data == "cityscapes":
                            from dataset.cityscapes import (
                                id_cityscapes_to_greenhouse as label_conversion,
                            )
                        elif os_data == "forest":
                            from dataset.forest import (
                                id_forest_to_greenhouse as label_conversion,
                            )

                        label_conversion = torch.tensor(
                            label_conversion).to(device)

                        amax_output = label_conversion[
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
                    label.save(os.path.join(save_path, filename))

    #    update_image_list(tgt_train_lst, image_path_list, label_path_list, depth_path_list)

    if class_weighting == "normal":
        class_array /= class_array.sum()  # normalized
        class_weights = 1 / (class_array + 1e-10)
    #        if args.dataset == 'greenhouse' and not args.use_traversable:
    #            class_weights[0] = 0.0
    else:
        class_weights = np.ones(num_classes) / num_classes

    print("class_weights : {}".format(class_weights))
    class_weights = torch.from_numpy(class_weights).float().to(device)

    # return the dictionary containing all the class-wise confidence vectors,
    # and the class_weights for loss weighting
    return class_weights


def generate_pseudo_label_multi_model_domain_gap(
    model_list: list,
    dg_model_list: list,
    source_dataset_name_list: list,
    target_dataset_name: str,
    data_loader: torch.utils.data.DataLoader,
    num_classes: int,
    save_path: str,
    device: str = "cuda",
    is_per_pixel: bool = False,
    ignore_index: int = 4,
    softmax_normalize: bool = False,
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
    is_per_pixel: `bool`
        True if the domain gap values are computed and
        considered per pixel.
        Otherwise, per image.
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
    else:
        print("Target {} is not supported.".format(target_dataset_name))
        raise ValueError

    # evaluation process
    class_array = np.zeros(num_classes)

    # Domain gap. Less value means closer -> More importance.
    domain_gap_list = calculate_domain_gap(dg_model_list, data_loader, device)[
        "domain_gap_list"
    ]

    del dg_model_list

    # Calculate weights based on the domain gaps
    domain_gap_weight = torch.Tensor(domain_gap_list).to(device)
    if softmax_normalize:
        domain_gap_weight = torch.nn.Softmax2d()(domain_gap_weight)
    else:
        domain_gap_weight /= domain_gap_weight.sum()

    # Calculate the inverse values of the domain gap
    domain_gap_weight = 1 / (domain_gap_weight + 1e-10)
    domain_gap_weight.to(device)

    print(domain_gap_weight)

    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for index, batch in enumerate(tqdm(data_loader)):
                image = batch["image"]
                name = batch["name"]

                output_list = []
                output_total = 0
                ds_index = 0
                for m, os_data in zip(model_list, source_dataset_name_list):
                    # Output: Numpy, KLD: Numpy
                    output = get_output(m, image)

                    # Extract the maximum value for each target class
                    output_target = torch.zeros(
                        (output.size(0), num_classes,
                         output.size(2), output.size(3))
                    ).to(device)

                    if os_data == "camvid":
                        label_conversion = id_camvid_to_greenhouse
                    elif os_data == "cityscapes":
                        label_conversion = id_cityscapes_to_greenhouse
                    elif os_data == "forest":
                        label_conversion = id_forest_to_greenhouse
                    else:
                        raise ValueError

                    label_conversion = torch.Tensor(
                        label_conversion).to(device)

                    for i in range(num_classes):
                        indices = torch.where(label_conversion == i)[0]
                        output_target[:, i] = output[:, indices].max(dim=1)[0]

                    output_target = F.normalize(output_target, p=1)

                    output_list.append(output_target)

                    output_total += output_target * domain_gap_weight[ds_index]
                    ds_index += 1

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
                    

    if class_weighting == "normal":
        class_array /= class_array.sum()  # normalized
        class_wts = 1 / (class_array + 1e-10)
    else:
        class_wts = np.ones(num_classes) / num_classes

    print("class_weights : {}".format(class_wts))
    class_wts = torch.from_numpy(class_wts).float().to(device)

    return class_wts
