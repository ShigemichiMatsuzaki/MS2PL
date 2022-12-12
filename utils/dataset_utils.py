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
    transform=None,
    max_iter: Optional[int] = None,
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

        num_classes = 13

        dataset = CamVidSegmentation(
            root="/tmp/dataset/CamVid",
            mode=mode,
            max_iter=max_iter,
            height=height,
            width=width,
            scale=scale,
            transform=transform,
        )
        dataset_label = CamVidSegmentation(
            root="/tmp/dataset/CamVid",
            mode=mode,
            height=height,
            width=width,
            scale=scale,
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
        )

        num_classes = 19

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
        )
        dataset_label = FreiburgForestDataset(
            root="/tmp/dataset/freiburg_forest_annotated/",
            mode=mode,
            height=height,
            width=width,
            scale=scale,
        )
        num_classes = 5
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
        )
        dataset_label = GTA5(
            root="/tmp/dataset/gta5/",
            mode=mode,
            height=height,
            width=width,
            scale=scale,
        )
        num_classes = 19
    else:
        raise Exception

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
