import argparse
from distutils.util import strtobool


class BaseOptions(object):
    """General options used in training, test, and pseudo-label generation in general."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            conflict_handler='resolve',
        )

        super(BaseOptions, self).__init__()
#        self.initialized = False
        print("Base option")

#    def initialize(self):
        # Directory
        self.parser.add_argument(
            "--save-path", help="Save path (tensorboard etc.)")

        # GPU/CPU
        self.parser.add_argument(
            "--device",
            type=str,
            choices=["cuda", "cpu"],
            default="cuda",
            help="Batch size in training",
        )
        self.parser.add_argument(
            "--model",
            type=str,
            choices=[
                "deeplabv3_mobilenet_v3_large",
                "deeplabv3_resnet50",
                "deeplabv3_resnet101",
                "espnetv2",
                "esptnet",
                "unet",
            ],
            default="deeplabv3_resnet50",
            help="Model",
        )

        # DataLoader
        self.parser.add_argument(
            "--pin-memory",
            action="store_true",
            help="Whether to use 'pin_memory' in DataLoader",
        )
        self.parser.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="Number of workers used in DataLoader. Should be 4 times the number of GPUs?",
        )

        # Training conditions
        self.parser.add_argument(
            "--batch-size", type=int, default=16, help="Batch size in training"
        )

        # Training conditions
        self.parser.add_argument(
            "--ignore-index",
            type=int,
            default=3,
            help="Label to ignore in training",
        )

        # Seed
        self.parser.add_argument(
            "--rand-seed",
            type=int,
            default=0,
            help="Seed of the random value generators",
        )

        self.parser.add_argument(
            "--is-old-label",
            type=strtobool,
            default=False,
        )

        self.parser.add_argument(
            '--use-cosine',
            default=False,
            type=strtobool,
            help='True to use cosine-based loss (ArcFace). Valid only when "model"=="espnetv2"'
        )
        self.parser.add_argument(
            "--resume-from",
            type=str,
            default="",
            help="Weights to resume the training from",
        )

        self.initialized = True

    def parse(self):
        # if not self.initialized:
        #     self.initialize()

        return self.parser.parse_args()
