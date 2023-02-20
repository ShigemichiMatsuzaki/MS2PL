from .base_options import BaseOptions
from .pseudo_label_options import PseudoLabelOptions
from distutils.util import strtobool


class TrainBaseOptions(BaseOptions):
    def __init__(self):
        # super().__init__()
        super(TrainBaseOptions, self).__init__()

        print("TrainBase option")

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

        # Dataset
        self.parser.add_argument(
            "--target",
            type=str,
            default="",
            help="Target dataset",
        )


class PreTrainOptions(TrainBaseOptions):
    def __init__(self):
        # super().__init__()
        super(PreTrainOptions, self).__init__()

        print("PreTrainBase option")
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
    def __init__(self):
        # super().__init__()
        super(TrainOptions, self).__init__()
        print("Train option")

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
        self.parser.add_argument(
            "--test-data-list-path",
            type=str,
            default="dataset/data_list/test_greenhouse_a.lst",
            help="Target test dataset",
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
        self.parser.add_argument(
            "--is-sce-loss",
            type=strtobool,
            default=False,
            help="Use symmetric cross entropy as a classification loss",
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


class PseudoLabelAndTrainOptions(TrainOptions):
    def __init__(self):
        super(PseudoLabelAndTrainOptions, self).__init__()

        self.parser.add_argument(
            "--generate-pseudo-labels",
            type=strtobool,
            default=True,
            help="True to generate pseudo-labels before training",
        )

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

        # Pseudo-label parameters
        self.parser.add_argument(
            "--is-hard",
            default=False,
            type=strtobool,
            help="If True, generate hard pseudo-labels.",
        )
        # self.parser.add_argument(
        #     "--use-domain-gap",
        #     type=strtobool,
        #     default=True,
        #     help="If True, domain gap-based weights are used for soft pseudo-label generation",
        # )
        self.parser.add_argument(
            "--is-softmax-normalize",
            type=strtobool,
            default=False,
            help="If set, normalize the domain gaps using softmax. Otherwise by the sum",
        )
        self.parser.add_argument(
            "--domain-gap-type",
            type=str,
            help="If True, domain gap-based weights are used for soft pseudo-label generation",
        )
        # self.parser.add_argument(
        #     "--is-per-sample",
        #     type=strtobool,
        #     default=False,
        #     help="If set, consider the domain gap per sample. Otherwise, per batch",
        # )
        # self.parser.add_argument(
        #     "--is-per-pixel",
        #     type=strtobool,
        #     default=False,
        #     help="If set, consider the domain gap per pixel. Otherwise, per image",
        # )

        self.parser.add_argument(
            "--sp-label-min-portion",
            type=float,
            default=0.5,
            help="Minimum proportion of the majority label in a superpixel to propagate",
        )
        self.parser.add_argument(
            "--pseudo-label-batch-size",
            type=int,
            default=1,
            help="Minimum proportion of the majority label in a superpixel to propagate",
        )
        self.parser.add_argument(
            "--initial-pseudo-label-path",
            type=str,
            default="",
            help="Directory where initial pseudo-labels are saved",
        )


class TraversabilityTrainOptions(TrainBaseOptions):
    def __init__(self):
        # super().__init__()
        super(TraversabilityTrainOptions, self).__init__()

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
            default="dataset/data_list/trav_train_greenhouse_b.lst",
            help="Target training dataset",
        )
        self.parser.add_argument(
            "--val-data-list-path",
            type=str,
            default="dataset/data_list/trav_val_greenhouse_b.lst",
            help="Target validation dataset",
        )
        self.parser.add_argument(
            "--test-data-list-path",
            type=str,
            default="dataset/data_list/test_val_greenhouse_a.lst",
            help="Target test dataset",
        )


class MSDACLTrainOptions(TrainBaseOptions):
    def __init__(self):
        # super().__init__()
        super(MSDACLTrainOptions, self).__init__()

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
        self.parser.add_argument(
            "--test-data-list-path",
            type=str,
            default="dataset/data_list/test_greenhouse_a.lst",
            help="Target test dataset",
        )

        self.parser.add_argument(
            "--pseudo-label-dir",
            type=str,
            default="",
            help="Path to the directory where the pre-trained class weight file is",
        )
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
