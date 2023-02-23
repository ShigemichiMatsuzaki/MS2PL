from .base_options import BaseOptions
from .pseudo_label_options import PseudoLabelOptions
from distutils.util import strtobool


class TestBaseOptions(BaseOptions):
    def __init__(self):
        # super().__init__()
        super(TestBaseOptions, self).__init__()

        print("TrainBase option")

        # Dataset
        self.parser.add_argument(
            "--target",
            type=str,
            help="Target dataset",
        )
        self.parser.add_argument(
            "--test-data-list-path",
            type=str,
            help="Target test dataset",
        )
        self.parser.add_argument(
            "--test-save-path",
            type=str,
            help="Target test dataset",
        )

class TestSingleModelOptions(TestBaseOptions):
    def __init__(self):
        super(TestSingleModelOptions, self).__init__()

class TestEnsembleOptions(TestBaseOptions):
    def __init__(self):
        super(TestEnsembleOptions, self).__init__()

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
            type=str,
            help="Paths to weight files separated by ','",
        )
        self.parser.add_argument(
            "--domain-gap-type",
            type=str,
            default="per_sample",
            help="Type of domain gap calculation",
        )