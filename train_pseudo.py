# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import os
import numpy as np

import torch
from trainer.vanilla_trainer_with_optuna import PseudoTrainer
from options.train_options import TrainOptions, PseudoLabelAndTrainOptions
from utils.model_io import import_model
from utils.pseudo_label_generator import generate_pseudo_label_multi_model, generate_pseudo_label_multi_model_domain_gap


def main():
    # Get arguments
    # args = parse_arguments()
    # args = TrainOptions().parse()
    args = PseudoLabelAndTrainOptions().parse()
    print(args)

    #
    # Train
    #
    torch.autograd.set_detect_anomaly(True)

    trainer = PseudoTrainer(args)

    if args.use_optuna:
        trainer.optuna_optimize(n_trials=500)
    else:
        if trainer.args.generate_pseudo_labels:
            print("Generate pseudo-labels")
            trainer.import_datasets(pseudo_only=True)
            trainer.generate_pseudo_labels()

        trainer.import_datasets(pseudo_only=False)
        trainer.init_training()

        trainer.fit()


if __name__ == "__main__":
    main()
