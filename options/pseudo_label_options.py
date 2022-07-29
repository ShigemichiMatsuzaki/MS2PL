from .base_options import BaseOptions


class PseudoLabelOptions(BaseOptions):
    def initialize(self):
        super().initialize()

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
            action="store_true",
            help="If set, generate hard pseudo-labels.",
        )
        self.parser.add_argument(
            "--is-softmax-normalize",
            action="store_true",
            help="If set, normalize the domain gaps using softmax. Otherwise by the sum",
        )
        self.parser.add_argument(
            "--is-per-pixel",
            action="store_true",
            help="If set, consider the domain gap per pixel. Otherwise, per image",
        )

        self.parser.add_argument(
            "--superpixel-pseudo-min-portion",
            type=float,
            default=0.5,
            help="Minimum proportion of the majority label in a superpixel to propagate",
        )
