from .base_options import BaseOptions


class TrainBaseOptions(BaseOptions):
    def initialize(self):
        self.parser.add_argument(
            "--epochs", type=int, default=500, help="The number of training epochs"
        )
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
            default="cyclic",
            help="Type of the learning rate cheduler. Default: cyclic",
        )
        self.parser.add_argument(
            "--lr-gamma",
            action="store_true",
            help="Whether to use learning rate warmup",
        )
        self.parser.add_argument(
            "--use-lr-warmup",
            action="store_true",
            help="Whether to use learning rate warmup",
        )


class PreTrainOptions(TrainBaseOptions):
    def initialize(self):
        TrainBaseOptions.initialize(self)

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
        TrainBaseOptions.initialize(self)
