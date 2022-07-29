import argparse


class BaseOptions:
    """General options used in training, test, and pseudo-label generation in general."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialized = False

    def initialize(self):
        # Directory
        self.parser.add_argument("--save-path", help="Save path (tensorboard etc.)")

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
            choices=["deeplabv3_mobilenet_v3_large", "deeplabv3_resnet50"],
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

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        return self.parser.parse_args()
