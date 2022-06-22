from .base_options import BaseOptions


class PseudoLabelOptions(BaseOptions):
    def initialize(self):
        super().initialize()

        # Source
        self.parser.add_argument(
            "--source-dataset-names",
            help="Source datasets to use. Either 'camvid', 'cityscapes', and 'forest', and must be separated by ',', i.e., 'camvid,forest'.",
        )
        self.parser.add_argument(
            "--source-weight-names",
            help="Paths to weight files separated by ','",
        )
        self.parser.add_argument(
            "--source-model-names",
            help="Paths to weight files separated by ','",
        )

        self.parser.add_argument(
            "--target",
            default="greenhouse",
            help="Target dataset",
        )
        self.parser.add_argument(
            "--target-data-list",
            help="List of the target data",
        )

        self.parser.add_argument(
            "--superpixel-pseudo-min-portion",
            type=float,
            default=0.5,
            help="Minimum proportion of the majority label in a superpixel to propagate",
        )
