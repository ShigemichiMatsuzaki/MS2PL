# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import os
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


def update_image_list(
    tgt_train_lst, image_path_list, label_path_list, depth_path_list=None
):
    """Update the list of input data. NOT USED ANYMORE SOON DELETED

    Parameters
    ----------
    tgt_train_lst  :
    image_path_list  :
    label_path_list :
    depth_path_list:

    """

    with open(tgt_train_lst, "w") as f:
        for idx in range(len(image_path_list)):
            if depth_path_list:
                f.write(
                    "%s,%s,%s\n"
                    % (image_path_list[idx], label_path_list[idx], depth_path_list[idx])
                )
            else:
                f.write("%s,%s\n" % (image_path_list[idx], label_path_list[idx]))

    return


def get_output(
    model,
    image,
    model_name="espdnetue",
    device="cuda",
    use_depth=False,
    is_numpy=True,
    merge_feature=True,
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
    if isinstance(output2, OrderedDict):
        pred = output2["out"]
        if "aux" in output2.keys():
            pred_aux = output2["aux"]
    # elif isinstance(output2, dict):
    #     pred = output2["main"]
    #     pred_aux = output2["aux"]
    #     if merge_feature:
    #         feat = output2["feat"]
    #     else:
    #         feat = output2["main_feat"]
    # elif model_name == "espdnetue":
    #     pred = output2[0]
    #     pred_aux = output2[1]
    #     if merge_feature:
    #         feat = output2[2]
    #     else:
    #         feat = output2[3]

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
    model,
    testloader,
    device,
    save_path,
    round_idx,
    args,
    logger,
    ignore_index=4,
    use_trav_mask=False,
    prototypes=None,
    proto_rect_thresh=0.9,
    min_portion=-1.0,
    label_conversion=None,
    use_gpu=False,
    model_ema=None,
):
    """Generate pseudo-labels using a pre-trained model

    Parameters
    ----------
    model :

    testloader :

    device :

    save_path :

    round_idx :

    args :

    logger :

    ignore_index : int

    use_trav_mask :

    prototypes :

    proto_rect_thresh : float

    label_conversion : List

    model_ema:


    Returns
    --------
    tgt_train_lst :

    class_weights :

    label_path_list :

    """
    ## model for evaluation
    model.eval()
    #
    model.to(device)

    save_pred_path = os.path.join(save_path, "pred")
    tgt_train_lst = os.path.join(save_path, "tgt_train.lst")

    ## evaluation process
    logger.info(
        "###### Start evaluating target domain train set in round {}! ######".format(
            round_idx
        )
    )
    image_path_list = []
    label_path_list = []
    depth_path_list = []
    class_array = np.zeros(args.classes)
    with torch.no_grad():
        with tqdm(total=len(testloader)) as pbar:
            for index, batch in enumerate(tqdm(testloader)):
                image = batch["rgb"].to(device)
                name = batch["name"]

                # Output: tensor, KLD: tensor, feature: tensor
                output_prob = get_output(model, image, is_numpy=False)
                max_output, argmax_output = torch.max(output_prob, dim=1)

                #
                # If the prototypes are given, apply filtering of the labels
                #
                if prototypes is not None:
                    if model_ema is not None:
                        model_ema.eval()
                        model_ema.to()
                        _, _, feature = model_ema(image)

                    # Class-wise weights based on the distance to the prototype of each class
                    weights = prototypes.get_prototype_weight(feature).to(device)

                    # Rectified output probability
                    rectified_prob = weights * output_prob
                    # Normalize the rectified values as probabilities
                    rectified_prob = rectified_prob / rectified_prob.sum(
                        1, keepdim=True
                    )
                    # Predicted label map after rectification
                    max_output, argmax_output = rectified_prob.max(1, keepdim=True)

                    print("prob after")
                    print(rectified_prob[0])

                # Filter out the pixels with a confidence below the threshold
                argmax_output[max_output < proto_rect_thresh] = ignore_index - 1

                # Convert the label space from the source to the target
                if label_conversion is not None:
                    label_conversion = torch.tensor(label_conversion).to(device)
                    argmax_output = label_conversion[argmax_output]

                #                output = output.transpose(1,2,0)
                #                amax_output = argmax_output[0].cpu().numpy().transpose(1, 2, 0)
                if use_trav_mask:
                    # If the class is plant (1) and the pixel is traversable (1),
                    #  change the class to 'traversable plant' (0)
                    argmax_output[(argmax_output == 1) & (mask == 255)] = 0

                for i in range(argmax_output.size(0)):
                    amax_output = argmax_output[i].squeeze().cpu().numpy()

                    if min_portion >= 0:
                        # Convert PIL image to Numpy
                        rgb_img_np = (
                            image[i].to("cpu").detach().numpy().transpose(1, 2, 0)
                        )

                        amax_output = get_label_from_superpixel(
                            rgb_img_np=rgb_img_np,
                            label_img_np=amax_output,
                            sp_type="watershed",
                            min_portion=min_portion,
                        )

                    for j in range(0, args.classes):
                        class_array[j] += (amax_output == j).sum()

                    amax_output += 1
                    path_name = name[i]
                    file_name = name[i].split("/")[-1]
                    image_name = file_name.rsplit(".", 1)[0]

                    # prob
                    # trainIDs/vis seg maps
                    amax_output = Image.fromarray(amax_output.astype(np.uint8))
                    # Save the predicted images (+ colorred images for visualization)
                    amax_output.save("%s/%s.png" % (save_pred_path, image_name))

                    image_path_list.append(path_name)
                    label_path_list.append("%s/%s.png" % (save_pred_path, image_name))
                    if args.use_depth:
                        depth_path_list.append(path_name.replace("color", "depth"))

        pbar.close()

    #    update_image_list(tgt_train_lst, image_path_list, label_path_list, depth_path_list)

    if args.class_weighting == "normal":
        class_array /= class_array.sum()  # normalized
        class_weights = 1 / (class_array + 1e-10)
    #        if args.dataset == 'greenhouse' and not args.use_traversable:
    #            class_weights[0] = 0.0
    else:
        class_weights = np.ones(args.classes)

    print("class_weights : {}".format(class_weights))
    class_weights = torch.from_numpy(class_weights).float().to(device)

    return tgt_train_lst, class_weights, label_path_list


def generate_pseudo_label_multi_model(
    model_list,
    source_dataset_name_list,
    target_dataset_name,
    data_loader,
    num_classes,
    device,
    save_path,
    min_portion=0.5,
    ignore_index=4,
    class_weighting="normal",
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

    ## evaluation process
    class_array = np.zeros(num_classes)
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for index, batch in enumerate(tqdm(data_loader)):
                image = batch["rgb"]
                name = batch["name"]

                output_list = []
                for m, os_data in zip(model_list, source_dataset_name_list):
                    # Output: Numpy, KLD: Numpy
                    output = get_output(m, image, is_numpy=False)
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

                        label_conversion = torch.tensor(label_conversion).to(device)

                        amax_output = label_conversion[
                            amax_output
                        ]  # Torch.cuda or numpy

                    elif target_dataset_name == "forest":
                        from dataset.camvid import (
                            id_camvid_to_forest,
                        )
                        from data_loader.segmentation.freiburg_forest import (
                            id_cityscapes_to_forest,
                        )

                        # save visualized seg maps & predication prob map
                        if os_data == "camvid":
                            from data_loader.segmentation.camvid import (
                                color_encoding as class_encoding_tmp,
                            )

                        elif os_data == "cityscapes":
                            from data_loader.segmentation.cityscapes import (
                                color_encoding as class_encoding_tmp,
                            )

                        if os_data == "camvid":
                            amax_output = id_camvid_to_forest[amax_output]
                        elif os_data == "cityscapes":
                            amax_output = id_cityscapes_to_forest[amax_output]

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
                            image[i].to("cpu").detach().numpy().transpose(1, 2, 0)
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
                    label = Image.fromarray(label.astype(np.uint8)).convert("P")
                    label.putpalette(color_palette)
                    # Save the predicted images (+ colorred images for visualization)
                    # label.save("%s/%s.png" % (save_pred_path, image_name))
                    label.save(os.path.join(save_path, filename))

        pbar.close()

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


def generate_soft_pseudo_label(
    model, testloader, device, save_path, round_idx, args, logger
):
    """Generate soft pseudo-labels (probability values)

    Parameters
    ----------
    model :

    testloader :

    device :

    save_path :

    round_idx :

    args :

    logger :


    Returns
    --------
    label_path_list :

    """
    ## model for evaluation
    model.eval()
    #
    model.to(device)

    ## evaluation process
    logger.info(
        "###### Start evaluating target domain train set in round {}! ######".format(
            round_idx
        )
    )
    save_pred_path = os.path.join(save_path, "soft_pseudo_label")
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)

    label_path_list = []
    with torch.no_grad():
        with tqdm(total=len(testloader)) as pbar:
            for batch in tqdm(testloader):
                if args.use_depth:
                    image = batch["rgb"]
                    depth = batch["depth"]
                    name = batch["name"]
                else:
                    image = batch["rgb"]
                    name = batch["name"]

                # Output: tensor, KLD: tensor, feature: tensor
                output_prob = get_output(model, image, is_numpy=False)

                for i in range(output_prob.size(0)):
                    # Save the probabilities as float16 for saving the storage
                    output_prob_i = output_prob[i].squeeze().to("cpu").type(torch.half)

                    file_name = name[i].split("/")[-1]
                    image_name = file_name.rsplit(".", 1)[0]

                    # Save the predicted images (+ colorred images for visualization)
                    torch.save(output_prob_i, "%s/%s.pt" % (save_pred_path, image_name))

                    label_path_list.append("%s/%s.pt" % (save_pred_path, image_name))

        pbar.close()

    return label_path_list


if __name__ == "__main__":
    pass
