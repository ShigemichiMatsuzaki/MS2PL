from .base_options import BaseOptions
from distutils.util import strtobool


class PseudoLabelOptions(BaseOptions):
    def __init__(self):
        super(PseudoLabelOptions, self).__init__()
        print("PseudoLabel option")

        # Source
        self.parser.add_argument(
            "--source-dataset-names",
            type=str,
            help="Source datasets to use. Either 'camvid', 'cityscapes', and 'forest', and must be separated by ',', i.e., 'camvid,forest'.",
        )
        self.parser.add_argument(
            "--source-weight-names",
            type=str,
            help="Paths to weight files separated by ','",
        )
        self.parser.add_argument(
            "--source-model-names",
            help="Paths to weight files separated by ','",
        )

        self.parser.add_argument(
            "--target",
            type=str,
            default="greenhouse",
            help="Target dataset",
        )
        self.parser.add_argument(
            "--target-data-list",
            type=str,
            help="List of the target data",
        )

        # Pseudo-label parameters
        self.parser.add_argument(
            "--is-hard",
            default=False,
            type=strtobool,
            help="If True, generate hard pseudo-labels.",
        )
        self.parser.add_argument(
            "--use-domain-gap",
            type=strtobool,
            default=True,
            help="If True, domain gap-based weights are used for soft pseudo-label generation",
        )
        self.parser.add_argument(
            "--is-softmax-normalize",
            type=strtobool,
            default=False,
            help="If set, normalize the domain gaps using softmax. Otherwise by the sum",
        )
        self.parser.add_argument(
            "--is-per-sample",
            type=strtobool,
            default=False,
            help="If set, consider the domain gap per sample. Otherwise, per batch",
        )
        self.parser.add_argument(
            "--is-per-pixel",
            type=strtobool,
            default=False,
            help="If set, consider the domain gap per pixel. Otherwise, per image",
        )

        self.parser.add_argument(
            "--sp-label-min-portion",
            type=float,
            default=0.5,
            help="Minimum proportion of the majority label in a superpixel to propagate",
        )
