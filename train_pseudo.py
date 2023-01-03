# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import random
import numpy as np

import torch
from trainer.vanilla_trainer_with_optuna import PseudoTrainer
from options.train_options import TrainOptions


def main():
    # Get arguments
    # args = parse_arguments()
    args = TrainOptions().parse()
    print(args)

    # Manually set the seeds of random values
    # https://qiita.com/north_redwing/items/1e153139125d37829d2d
    torch.manual_seed(args.rand_seed)
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)

    torch.autograd.set_detect_anomaly(True)

    trainer = PseudoTrainer(args)

    if args.use_optuna:
        trainer.optuna_optimize()
    else:
        trainer.fit()

if __name__ == "__main__":
    main()
