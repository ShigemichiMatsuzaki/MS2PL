import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict


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
