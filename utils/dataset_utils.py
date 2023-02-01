import sys
import traceback
import torch
from tqdm import tqdm
from typing import Optional
import albumentations as A

# Supported source datasets
DATASET_LIST = ["camvid", "cityscapes", "forest", "gta5"]


def calculate_class_weights(
    dataset: torch.utils.data.Dataset,
    num_classes: int,
    batch_size: int = 64,
    is_inverse: bool = False,
    max_iter: Optional[int] = -1,
):
    """Calculate class weights for loss function

    Parameters
    ----------
    dataset: `torch.utils.data.Dataset`
        Dataset to calculate the weights
    num_classes: `int`
        The number of the classes in the dataset
    batch_size: `int`
        Batch size to be used in the calculation
    is_inverse: `bool`
        `True` to make the weights proportional to the class frequency
        (usually, the weight is less for a more frequent class)
    max_iter: `int`
        The maximum number of iteration for 

    """
    # Calculate class weight
    print("Calculate class weights")
    class_wts = torch.zeros(num_classes).to("cuda")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    with tqdm(total=len(loader)) as pbar:
        for i, b in enumerate(tqdm(loader)):
            if i == max_iter:
                break

            label = b["label"].to("cuda")
            for j in range(0, num_classes):
                class_wts[j] += (label == j).sum()

    class_wts /= class_wts.sum()  # normalized

    if is_inverse:
        class_wts = 1 / (1 - class_wts + 1e-10)
    else:
        class_wts = 1 / (class_wts + 1e-10)

    return class_wts


def import_dataset(
    dataset_name: str,
    mode: str = "train",
    calc_class_wts: bool = False,
    is_class_wts_inverse: bool = False,
    height: int = 256,
    width: int = 480,
    scale: list = (0.5, 2.0),
    transform: Optional[A.Compose] = None,
    max_iter: Optional[int] = None,
    label_conversion: bool = False,
):
    """Import a designated dataset

    Parameters
    ----------
    dataset_name: `str`
        Name of the dataset to import
    mode: `str`
        Mode of the dataset. ['train', 'val', 'test']
    calc_class_wts: `bool`
        True to calculate class weights based on the frequency
    is_class_wts_inverse: `bool`
        True to calculate class weights propotional to the frequency
        (usually, less frequent classes have more weight)
    height: `int`
        Height of the image size after resizing
    width: `int`
        Width of the image size after resizing
    scale: `list`
        Scales to be used in data augumentation
    transform: `Optional[albumentations.Compose]`
        Transforms to be used in data loading
    max_iter: `Optional[int]`
        Maximum iteration. If the number of data in a dataset is below max_iter, 
        the data are multiplied by max_iter // len(dataset) to (approximately) match 
        the number of the data with this value.
    label_conversion: `bool`
        `True` to convert the labels to the Greenhouse label map. Default: `False`

    Returns
    -------
    dataset: `torch.utils.data.Dataset`
        Imported dataset
    num_classes: `int`
        The number of classes in the imported dataset
    color_encoding: `collection.OrderedDict`
        Label color encoding of the imported dataset
    class_wts: `torch.Tensor`
        Class weights
    """
    # max_iter = 3000
    if dataset_name == DATASET_LIST[0]:
        from dataset.camvid import CamVidSegmentation, color_encoding

        num_classes = 13 if not label_conversion else 3

        dataset = CamVidSegmentation(
            root="/tmp/dataset/CamVid",
            mode=mode,
            max_iter=max_iter,
            height=height,
            width=width,
            scale=scale,
            transform=transform,
            label_conversion=label_conversion,
        )
        dataset_label = CamVidSegmentation(
            root="/tmp/dataset/CamVid",
            mode=mode,
            height=height,
            width=width,
            scale=scale,
            label_conversion=label_conversion,
        )
    elif dataset_name == DATASET_LIST[1]:
        from dataset.cityscapes import CityscapesSegmentation, color_encoding

        dataset = CityscapesSegmentation(
            root="/tmp/dataset/cityscapes",
            mode=mode,
            max_iter=max_iter,
            height=height,
            width=width,
            scale=scale,
            transform=transform,
            label_conversion=label_conversion,
        )

        num_classes = 19 if not label_conversion else 3

    elif dataset_name == DATASET_LIST[2]:
        from dataset.forest import FreiburgForestDataset, color_encoding

        dataset = FreiburgForestDataset(
            root="/tmp/dataset/freiburg_forest_annotated/",
            mode=mode,
            max_iter=max_iter,
            height=height,
            width=width,
            scale=scale,
            transform=transform,
            label_conversion=label_conversion,
        )
        dataset_label = FreiburgForestDataset(
            root="/tmp/dataset/freiburg_forest_annotated/",
            mode=mode,
            height=height,
            width=width,
            scale=scale,
            label_conversion=label_conversion,
        )
        num_classes = 5 if not label_conversion else 3

    elif dataset_name == DATASET_LIST[3]:
        from dataset.gta5 import GTA5, color_encoding

        dataset = GTA5(
            root="/tmp/dataset/gta5/",
            mode=mode,
            max_iter=max_iter,
            height=height,
            width=width,
            scale=scale,
            transform=transform,
            label_conversion=label_conversion,
        )
        dataset_label = GTA5(
            root="/tmp/dataset/gta5/",
            mode=mode,
            height=height,
            width=width,
            scale=scale,
            label_conversion=label_conversion,
        )
        num_classes = 19 if not label_conversion else 3

    else:
        raise Exception

    if label_conversion:
        from dataset.greenhouse import color_encoding

    if calc_class_wts:
        if dataset_name != DATASET_LIST[1]:
            class_wts = calculate_class_weights(
                dataset_label,
                num_classes=num_classes,
                batch_size=64,
                is_inverse=False,
                max_iter=10,
            )
        else:
            class_wts = torch.ones(19)
            class_wts[0] = 2.8149201869965
            class_wts[1] = 6.9850029945374
            class_wts[2] = 3.7890393733978
            class_wts[3] = 9.9428062438965
            class_wts[4] = 9.7702074050903
            class_wts[5] = 9.5110931396484
            class_wts[6] = 10.311357498169
            class_wts[7] = 10.026463508606
            class_wts[8] = 4.6323022842407
            class_wts[9] = 9.5608062744141
            class_wts[10] = 7.8698215484619
            class_wts[11] = 9.5168733596802
            class_wts[12] = 10.373730659485
            class_wts[13] = 6.6616044044495
            class_wts[14] = 10.260489463806
            class_wts[15] = 10.287888526917
            class_wts[16] = 10.289801597595
            class_wts[17] = 10.405355453491
            class_wts[18] = 10.138095855713

        if is_class_wts_inverse:
            class_wts = 1 / (class_wts + 1e-12)
    else:
        class_wts = torch.ones(num_classes) / num_classes

    return dataset, num_classes, color_encoding, class_wts


def import_target_dataset(
    dataset_name: str,
    mode: str = "train",
    data_list_path: str = "",
    pseudo_label_dir: str = "",
    is_hard: bool = False,
    is_old_label: bool = False,
):
    """Import a designated dataset

    Parameters
    ----------
    dataset_name: `str`
        Name of the dataset to import
    mode: `str`
        Mode of the dataset. ['train', 'val', 'test']
    data_list_path: `str`
        Path to data list (for Greenhouse and Imo). Default: `""`
    pseudo_label_dir: `str`
        Path to pseudo labels. Default: `""`
    is_hard: `bool`
        `True` to load hard (one-hot) label. Default: `False`
    is_old_label: `bool`
        `True` to use old label style. Valid only for Greenhouse.
        Default: `False`

    Returns
    -------
    dataset: `torch.utils.data.Dataset`
        Imported dataset
    num_classes: `int`
        The number of classes in the imported dataset
    color_encoding: `collection.OrderedDict`
        Label color encoding of the imported dataset
    """
    # max_iter = 3000
    if dataset_name == "greenhouse":
        from dataset.greenhouse import GreenhouseRGBD, color_encoding, color_palette
        from dataset.greenhouse import GREENHOUSE_CLASS_LIST as class_list
        num_classes = 3
        try:
            if mode == "train":
                dataset_ret = GreenhouseRGBD(
                    list_name=data_list_path,
                    label_root=pseudo_label_dir,
                    mode="train",
                    is_hard_label=is_hard,
                    is_old_label=is_old_label,
                )
            elif mode == "pseudo":
                dataset_ret = GreenhouseRGBD(
                    list_name=data_list_path,
                    label_root=pseudo_label_dir,
                    mode="pseudo",
                    is_hard_label=True,
                    load_labels=False,
                )
            elif mode == "val":
                dataset_ret = GreenhouseRGBD(
                    list_name=data_list_path,
                    mode="val",
                    is_hard_label=True,
                    is_old_label=True,
                )
            elif mode == "test":
                dataset_ret = GreenhouseRGBD(
                    list_name=data_list_path,
                    mode="test",
                    is_hard_label=True,
                    is_old_label=True,
                )

        except Exception as e:
            t, v, tb = sys.exc_info()
            print(traceback.format_exception(t, v, tb))
            print(traceback.format_tb(e.__traceback__))
            print("Failed to load dataset '{}'.".format(dataset_name))
            sys.exit(1)
    elif dataset_name == "imo":
        from dataset.imo import Imo, color_encoding, color_palette
        from dataset.imo import IMO_CLASS_LIST as class_list
        num_classes = 3
        try:
            if mode == "train":
                dataset_ret = Imo(
                    list_name=data_list_path,
                    label_root=pseudo_label_dir,
                    mode="train",
                    is_hard_label=is_hard,
                )
            elif mode == "pseudo":
                dataset_ret = Imo(
                    list_name=data_list_path,
                    label_root=pseudo_label_dir,
                    mode="pseudo",
                    is_hard_label=True,
                    load_labels=False,
                )
            elif mode == "val":
                dataset_ret = Imo(
                    list_name=data_list_path,
                    mode="val",
                    is_hard_label=True,
                )
            elif mode == "test":
                dataset_ret = Imo(
                    list_name=data_list_path,
                    mode="test",
                    is_hard_label=True,
                )

            else:
                raise ValueError
        except Exception as e:
            t, v, tb = sys.exc_info()
            print(traceback.format_exception(t, v, tb))
            print(traceback.format_tb(e.__traceback__))
            print("Failed to load dataset '{}'.".format(dataset_name))
            sys.exit(1)

    elif dataset_name == "sakaki":
        from dataset.sakaki import SakakiDataset, color_encoding, color_palette
        from dataset.sakaki import SAKAKI_CLASS_LIST as class_list
        num_classes = 5
        try:
            if mode == "train":
                dataset_ret = SakakiDataset(
                    list_name=data_list_path,
                    label_root=pseudo_label_dir,
                    mode="train",
                    is_hard_label=is_hard,
                )
            elif mode == "pseudo":
                dataset_ret = SakakiDataset(
                    list_name=data_list_path,
                    label_root=pseudo_label_dir,
                    mode="pseudo",
                    is_hard_label=True,
                    load_labels=False,
                )
            elif mode == "val":
                dataset_ret = SakakiDataset(
                    list_name=data_list_path,
                    mode="val",
                    is_hard_label=True,
                )
            elif mode == "test":
                dataset_ret = SakakiDataset(
                    list_name=data_list_path,
                    mode="test",
                    is_hard_label=True,
                )

            else:
                raise ValueError

        except Exception as e:
            t, v, tb = sys.exc_info()
            print(traceback.format_exception(t, v, tb))
            print(traceback.format_tb(e.__traceback__))
            print("Failed to load dataset '{}'.".format(dataset_name))
            sys.exit(1)
    else:
        print("Target dataset '{}' is not supported.".format(dataset_name))
        raise ValueError

    return dataset_ret, num_classes, color_encoding, color_palette, class_list
