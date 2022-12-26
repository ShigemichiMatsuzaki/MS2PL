import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

import numpy as np
import torch.nn.functional as F
import umap
import matplotlib.pyplot as plt
from dataset.greenhouse import GREENHOUSE_CLASS_LIST

from typing import Optional

UMAP_LABEL_TYPES = ['object', 'traversability', 'misclassification']


class LongTensorToRGBPIL(object):
    """Converts a ``torch.LongTensor`` to a ``PIL image``.

    The input is a ``torch.LongTensor`` where each pixel's value
    identifies the class.

    Keyword arguments:
    - rgb_encoding (``OrderedDict``): An ``OrderedDict`` that relates pixel
    values, class names, and class colors.

    """

    def __init__(self, rgb_encoding):
        self.rgb_encoding = rgb_encoding

    def __call__(self, tensor):
        """Performs the conversion from ``torch.LongTensor`` to a ``PIL image``

        Keyword arguments:
        - tensor (``torch.LongTensor``): the tensor to convert

        Returns:
        A ``PIL.Image``.

        """
        # Check if label_tensor is a LongTensor
        if not isinstance(tensor, torch.LongTensor):
            raise TypeError(
                "label_tensor should be torch.LongTensor. Got {}".format(type(tensor))
            )
        # Check if encoding is a ordered dictionary
        if not isinstance(self.rgb_encoding, OrderedDict):
            raise TypeError(
                "encoding should be an OrderedDict. Got {}".format(
                    type(self.rgb_encoding)
                )
            )

        # label_tensor might be an image without a channel dimension, in this
        # case unsqueeze it
        if len(tensor.size()) == 2:
            tensor.unsqueeze_(0)

        # Initialize
        color_tensor = torch.ByteTensor(3, tensor.size(1), tensor.size(2))

        for index, (class_name, color) in enumerate(self.rgb_encoding.items()):
            # Get a mask of elements equal to index
            mask = torch.eq(tensor, index).squeeze_()
            # Fill color_tensor with corresponding colors
            for channel, color_value in enumerate(color):
                color_tensor[channel].masked_fill_(mask, color_value)

        #        return ToPILImage()(color_tensor)
        return color_tensor


def batch_transform(tensor: torch.Tensor, transform: callable):
    """Applies a transform to a batch of samples.

    Parameters
    ----------
    batch : torch.Tensor
        a batch os samples
    transform : callable
        A function/transform to apply to ``batch``

    Returns
    -------
    out : torch.Tensor
        Output tensor

    """

    # Convert the single channel label to RGB in tensor form
    # 1. torch.unbind removes the 0-dimension of "labels" and returns a tuple of
    # all slices along that dimension
    # 2. the transform is applied to each slice
    transf_slices = [transform(tensor) for tensor in torch.unbind(tensor)]

    return torch.stack(transf_slices)


def add_images_to_tensorboard(
    writer: SummaryWriter,
    tensor: torch.Tensor,
    i_iter: int,
    data_tag: str,
    is_label: bool = False,
    color_encoding: dict = None,
) -> None:
    """Write the given image batch to ``SummaryWriter`` of Tensorboard

    Parameters
    ----------
    writer : ``torch.utils.tensorboard.SummaryWriter``
        SummaryWriter
    tensor : ``torch.Tensor``
        Image tensor
    i_iter: ``int``
        Iteration number
    data_tag: ``str``
        Tag of data
    is_label : ``bool``
        ``True`` if the given tensor is segmentation label maps
    color_encoding : ``dict``
        Definition of mapping from object labels to colors. Must be given if ``is_label`` is True

    Returns
    -------
    None

    """

    if not is_label:
        images_grid = torchvision.utils.make_grid(tensor.data.cpu()).numpy()
    else:
        tensor[tensor >= len(color_encoding)] = len(color_encoding) - 1
        # label_to_rgb : Sequence of processes
        #  1. LongTensorToRGBPIL(tensor) -> PIL Image : Convert label tensor to color map
        #  2. transforms.ToTensor() -> Tensor : Convert PIL Image to a tensor
        label_to_rgb = torchvision.transforms.Compose(
            [LongTensorToRGBPIL(color_encoding)]  # ,
        )

        # Do transformation of label tensor and prediction tensor
        images_grid = batch_transform(tensor.data.cpu(), label_to_rgb)
        images_grid = torchvision.utils.make_grid(images_grid).numpy()

    writer.add_image(data_tag, images_grid, i_iter)


def color_encoding_dict_to_palette(color_encoding: OrderedDict) -> list:
    """Convert color_encoding to color palette

    Parameters
    ----------
    color_encoding: `OrderedDict`
        Color encoding in `OrderedDict` type

    Returns
    -------
    color_palette: `list`
        Color palette in `list` type
    """
    color_palette = []
    for v in color_encoding.values():
        for i in v:
            color_palette.append(i)

    return color_palette


def visualize_features(features, labels, label_type='', fig_name='foo'):
    """UMAP -> Visualization by Matplotlib

    Parameters
    ----------
    features: List
        List of features
    labels: List
        Label of lists
    label_type: string
        Type of label to assign to the features.
        'object': Object class
        'traversability': Traversability mask on plant data
        'misclassification': Correctness of the prediction on plant data
    fig_name: string
        Name of the file to save
    """

    # Clear the plot
    plt.clf()
    if not features.size or not labels.size:
        return

    # 次元削減する
    print("UMap")
    mapper = umap.UMAP(random_state=0, verbose=True)
    embedding = mapper.fit_transform(features)

    config_dict = {
        'end_of_plant': ('yellowgreen', '.', 5, 0.1),
        'other_part_of_plant': ('cyan', '.', 5, 0.1),
        'artificial_objects': ('red', '.', 5, 0.1),
        'ground': ('y', '.', 5, 0.1),
        'background': ('k', '.', 5, 0.1),
        '0': ('greenyellow', '*', 20, 1),
        '1': ('darkturquoise', '*', 20, 1),
        '2': ('lightcoral', '*', 20, 1),
        '3': ('gold', '*', 20, 1),
        '4': ('k', '*', 20, 1)}
    
    # 結果を二次元でプロットする
    embedding_x = embedding[:, 0]
    embedding_y = embedding[:, 1]
    print("Label type: ", label_type)
    for n in np.unique(labels):
        plt.scatter(embedding_x[labels == n],
                    embedding_y[labels == n],
                    label=n,
                    c=config_dict[n][0] if label_type == 'object' else None,
                    marker=config_dict[n][1],
                    s=config_dict[n][2],
                    zorder=config_dict[n][3])

    # グラフを表示する
    plt.grid()
    plt.legend()
    plt.savefig(fig_name+'.pdf')
    plt.savefig(fig_name+'.png')


def assign_label_on_features(
    feature: torch.Tensor, 
    labels: torch.Tensor, 
    masks: Optional[torch.Tensor]=None,
    pred_label: Optional[torch.Tensor]=None, 
    label_type: str='', 
    scale_factor: int=4,
    ignore_index: int=-1,
):
    """Create lists of features and the corresponding labels with a specified label type

    Parameters
    ----------
    feature : `torch.Tensor`
        Feature map
    labels:  `torch.Tensor`
        Ground-truth label map
    masks: `torch.Tensor`
        Traversability mask
    pred_label: `torch.Tensor`
        Predicted label
    label_type: `str`
        Type of label to assign to the features.
        'object': Object class
        'traversability': Traversability mask on plant data
        'misclassification': Correctness of the prediction on plant data
    scale_factor: `int`
        Factor to downsample the feature map (1/scale_factor)
    ignore_index: `int`
        Label features with which are ignored in embedding.

    Returns
    -------
    feature_list: `list`
        List of features
    label_list: `list`
        List of labels for UMAP visualization
    
    """
    # Argument check
    #  Is the given label type valid?
    assert label_type in UMAP_LABEL_TYPES #['object', 'traversability', 'misclassification']
    #  Is 'pred_label' given when the label type is 'misclassification'?
    assert label_type != 'misclassification' or pred_label is not None
    assert label_type != 'traversability' or masks is not None

    feature = F.interpolate(feature, scale_factor=1/scale_factor, mode='nearest')
#    feature = feature.squeeze()

    # interpolate is not implemented for integers
    ih = torch.linspace(0, labels.size(1)-1, labels.size(1)//scale_factor).long()
    iw = torch.linspace(0, labels.size(2)-1, labels.size(2)//scale_factor).long()
    labels = labels[:, ih[:, None], iw]
    if masks is not None:
        masks = masks[:, ih[:, None], iw]

    if pred_label is not None:
        pred_label = pred_label[:, ih[:, None], iw]

    feature_list, label_list = [], []
    for n in range(feature.size(0)):
        for i in range(feature.size(2)):
            for j in range(feature.size(3)):
                if label_type == 'object':
                    l = labels[n,i,j].detach().cpu().numpy()
                    if l != ignore_index:
                        feature_list.append(feature[n,:,i,j].detach().cpu().numpy())
                        label_list.append(GREENHOUSE_CLASS_LIST[l])
                elif label_type == 'traversability':
                    # Consider only plant regions
                    if labels[n,i,j].detach().cpu().numpy() in [0, 1]:
                        feature_list.append(feature[n,:,i,j].detach().cpu().numpy())
                        l = masks[n,i,j].detach().cpu().numpy() // 255
                        label_list.append('traversable' if l == 1 else 'non-traversable')
                elif label_type == 'misclassification':
                    # Consider only plant regions
                    gt_label = labels[n,i,j].detach().cpu().numpy()
                    if gt_label in [0, 1]:
                        pred = pred_label[n,i,j].detach().cpu().numpy()

                        if gt_label == 4:
                            continue

                        feature_list.append(feature[n,:,i,j].detach().cpu().numpy())
                        is_correct = gt_label == pred
                        if is_correct:
                            l = 'correct_plant' if is_correct else 0
                        elif pred == 2:
                            l = 'fn_artificial'
                        elif pred == 3:
                            l = 'fn_ground'
                        elif gt_label == 2:
                            l = 'fp_artificial'
                        elif gt_label == 3:
                            l = 'fp_ground'
                        else:
                            l = 'other'

                        label_list.append(l)

    return feature_list, label_list
