from .base_options import BaseOptions
from distutils.util import strtobool


class TrainBaseOptions(BaseOptions):
    def initialize(self):
        super().initialize()

        self.parser.add_argument(
            "--epochs", type=int, default=300, help="The number of training epochs"
        )
        self.parser.add_argument(
            "--resume-from",
            type=str,
            default="",
            help="Weights to resume the training from",
        )
        self.parser.add_argument(
            "--resume-epoch",
            type=int,
            default=0,
            help="Epoch number to resume the training from",
        )
        self.parser.add_argument("--train-image-size-h", type=int, default=256)
        self.parser.add_argument("--train-image-size-w", type=int, default=480)
        self.parser.add_argument("--val-image-size-h", type=int, default=256)
        self.parser.add_argument("--val-image-size-w", type=int, default=480)

        # Optimizer and scheduler
        self.parser.add_argument(
            "--optim", type=str, default="SGD", help="Optimizer. Default: SGD"
        )
        self.parser.add_argument(
            "--lr", type=float, default=1e-4, help="Initial learning rate"
        )
        self.parser.add_argument(
            "--momentum", type=float, default=0.9, help="Momentum for SGD"
        )
        self.parser.add_argument(
            "--weight-decay",
            type=float,
            default=5e-4,
            help="Weight decay by L2 normalization",
        )

        self.parser.add_argument(
            "--scheduler",
            type=str,
            default="polynomial",
            help="Type of the learning rate cheduler. Default: cyclic",
        )
        self.parser.add_argument(
            "--lr-gamma",
            type=strtobool,
            default=False,
            help="Whether to use learning rate warmup",
        )
        self.parser.add_argument(
            "--use-lr-warmup",
            type=strtobool,
            default=False,
            help="Whether to use learning rate warmup",
        )

        self.parser.add_argument(
            "--class-wts-type",
            type=str,
            default="normal",
            choices=["normal", "inverse", "uniform"],
            help="True to set more weight to more frequent classes",
        )


class PreTrainOptions(TrainBaseOptions):
    def initialize(self):
        super().initialize()

        # Dataset
        self.parser.add_argument(
            "--s1-name",
            # choices=DATASET_LIST,
            default="camvid",
            help="The dataset used as S1, the main source",
        )

        # Loss
        self.parser.add_argument(
            "--weight-loss-ent",
            type=float,
            default=0.2,
            help="Weight on the entropy loss",
        )


class TrainOptions(TrainBaseOptions):
    def initialize(self):
        super().initialize()

        self.parser.add_argument(
            "--target",
            type=str,
            default="greenhouse",
            help="Target dataset",
        )

        self.parser.add_argument(
            "--is-hard",
            type=strtobool,
            default=False,
            help="If True, generate hard pseudo-labels.",
        )

        # Loss
        self.parser.add_argument(
            "--weight-loss-ent",
            type=float,
            default=0.2,
            help="Weight on the entropy loss",
        )
        self.parser.add_argument(
            "--use-label-ent-weight",
            type=strtobool,
            default=True,
            help="True to use inverse label entropy as loss weights",
        )
        self.parser.add_argument(
            "--label-weight-temperature",
            type=float,
            default=1.0,
            help="True to use inverse label entropy as loss weights",
        )
        self.parser.add_argument(
            "--label-weight-threshold",
            type=float,
            default=0.5,
            help="Threshold of label weight value. Below this value is set to 0",
        )



        # Pseudo-label update
        self.parser.add_argument(
            "--label-update-epoch",
            type=int,
            default=15,
            help="Epoch at which the pseudo-labels are updated",
        )
        self.parser.add_argument(
            "--conf-thresh",
            type=float,
            default=0.9,
            help="Threshold of confidence to be selected as a pseudo-label",
        )
        self.parser.add_argument(
            "--sp-label-min-portion",
            type=float,
            default=0.7,
            help="Threshold of portion of labels of a certain class that dominates a superpixel to be propagated within the entire superpixel",
        )
        self.parser.add_argument(
            "--pseudo-label-dir",
            type=str,
            default="",
            help="Path to the directory where the pre-trained class weight file is",
        )
        self.parser.add_argument(
            "--use-prototype-denoising",
            type=strtobool,
            default=False,
            help="Whether to use prototype-based denoising",
        )
