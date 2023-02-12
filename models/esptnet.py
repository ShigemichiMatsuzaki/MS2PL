import torch
from .edgenets.model.segmentation.espnetv2 import ESPNetv2Segmentation

class _LabelProbEstimator(torch.nn.Module):
    """This class defines a simple architecture for estimating binary label probability of 1-d features"""

    def __init__(self, in_channels: int=16, use_sigmoid: bool=False, spatial: bool=True):
        """_summary_

        Parameters
        ----------
        in_channels : `int`, optional
            Number of input channels. Default: `16`
        use_sigmoid : `bool`, optional
            `True` to use sigmoid at last. Default: `False`
        spatial : `bool`, optional
            `True` to use 3x3 conv for classification. 
            Else, use 1x1 (point-wise) conv. Default: `True`
        """
        super().__init__()

        if spatial:
            self.conv1x1 = nn.Conv2d(
                in_channels=in_channels, out_channels=1, kernel_size=3, padding=1)
        else:
            self.conv1x1 = nn.Conv2d(
                in_channels=in_channels, out_channels=1, kernel_size=1)

        self.relu = nn.ReLU()

        self.use_sigmoid = use_sigmoid

        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        output = self.conv1x1(x)

        if self.use_sigmoid:
            output = self.sigmoid(output)

        return output


class ESPTNet(ESPNetv2Segmentation):
    # Segmentation + traversability estimation
    def __init__(
        self,
        args,
        classes=21,
        dataset="pascal",
        aux_layer=2,
    ):
        super(ESPNetv2Segmentation, self).__init__(
            args, classes, dataset, aux_layer,
        )

    def forward(self, x: torch.Tensor):
        return 