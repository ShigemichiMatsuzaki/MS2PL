# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import os
import shutil
import sys
import traceback
import datetime
import collections
from typing import Optional
import random
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from trainer.vanilla_trainer_with_optuna import PseudoTrainer
from warmup_scheduler import GradualWarmupScheduler

from options.train_options import TrainOptions
from utils.metrics import AverageMeter, MIOU
from utils.visualization import add_images_to_tensorboard, assign_label_on_features
from utils.optim_opt import get_optimizer, get_scheduler
from utils.model_io import import_model
from utils.calc_prototype import calc_prototype

from utils.dataset_utils import import_dataset, DATASET_LIST

from loss_fns.segmentation_loss import UncertaintyWeightedSegmentationLoss, Entropy
from utils.pseudo_label_generator import generate_pseudo_label

from utils.logger import log_training_conditions, log_metrics


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
