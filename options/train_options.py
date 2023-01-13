from .base_options import BaseOptions
from distutils.util import strtobool


class TrainBaseOptions(BaseOptions):
    def initialize(self):
        super().initialize()

        self.parser.add_argument(
            "--epochs", type=int, default=300, help="The number of training epochs"
        )

        self.parser.add_argument(
            "--val-every-epochs", type=int, default=1, help="Validate every this number of epochs"
        )
        self.parser.add_argument(
            "--vis-every-vals", type=int, default=5, help="Visualize every this number of validation processes"
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
            help="Type of the learning rate cheduler. Default: polynomial",
        )
        self.parser.add_argument(
            "--lr-gamma",
            type=float,
            default=0.1,
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

        # Parameters related to cosine-based softmax
        self.parser.add_argument(
            '--use-cosine', 
            default=False, 
            type=strtobool,
            help='True to use cosine-based loss (ArcFace). Valid only when "model"=="espnetv2"'
        )
        self.parser.add_argument(
            '--cos-margin', 
            default=0.1, 
            type=float,
            help='Angle margin'
        )
        self.parser.add_argument(
            '--cos-logit-scale', 
            default=30.0, 
            type=float,
            help='Scale factor for the final logits'
        )
        self.parser.add_argument(
            '--is-easy-margin', 
            default=False, 
            type=strtobool,
            help='Whether to use an easy margin'
        )

        # Hyperparameter tuning
        self.parser.add_argument(
            '--use-optuna', 
            default=False, 
            type=strtobool,
            help='Whether to use automatic hyperparameter tuning by Optuna'
        )
        self.parser.add_argument(
            '--optuna-resume-from', 
            default='', 
            type=str,
            help='Name of existing study'
        )




class PreTrainOptions(TrainBaseOptions):
    def initialize(self):
        super().initialize()

        # Dataset
        self.parser.add_argument(
            "--s1-name",
            required=True,
            nargs="*",
            type=str,
            help="The dataset used as S1, the main source",
        )
        self.parser.add_argument(
            "--use-other-datasets",
            type=strtobool,
            default=False,
            help="True to use other datasets than S1 for domain gap evaluation.",
        )
        self.parser.add_argument(
            "--use-label-conversion",
            type=strtobool,
            default=False,
            help="True to use label conversion",
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

        # Dataset
        self.parser.add_argument(
            "--target",
            type=str,
            default="greenhouse",
            help="Target dataset",
        )
        self.parser.add_argument(
            "--train-data-list-path",
            type=str,
            default="dataset/data_list/train_greenhouse_a.lst",
            help="Target training dataset",
        )
        self.parser.add_argument(
            "--val-data-list-path",
            type=str,
            default="dataset/data_list/val_greenhouse_a.lst",
            help="Target validation dataset",
        )



        # Label type
        self.parser.add_argument(
            "--is-hard",
            type=strtobool,
            default=False,
            help="If True, generate hard pseudo-labels.",
        )

        # Loss
        self.parser.add_argument(
            "--kld-loss-weight",
            type=float,
            default=0.2,
            help="Weight on the KLD loss between main and aux",
        )
        self.parser.add_argument(
            "--entropy-loss-weight",
            type=float,
            default=0.2,
            help="Weight on the entropy loss",
        )

        self.parser.add_argument(
            "--use-kld-class-loss",
            type=strtobool,
            default=False,
            help="True to use KLD loss to train classifier",
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
            required=True,
            nargs="*",
            type=int,
            help="Epoch at which the pseudo-labels are updated",
        )
        self.parser.add_argument(
            "--conf-thresh",
            required=True,
            nargs="*",
            type=float,
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
