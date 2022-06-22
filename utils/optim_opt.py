from multiprocessing.sharedctypes import Value
from xml.dom.minidom import Attr
import torch


def get_optimizer(args, model: torch.Tensor) -> torch.optim.Optimizer:
    """Get optimizer

    Parameters
    ----------
    args :
        Arguments acquired by ``argparse''
    model : ``torch.Tensor''
        A model to optimize

    Return
    ------
    optimizer : ``torch.optim.Optimizer''
        An optimizer

    """
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        print("Invalid optimizer name {}".format(args.optim))
        raise ValueError

    return optimizer


def get_scheduler(
    args, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler

    Parameters
    ----------
    args :
        Arguments acquired by ``argparse''
    optimizer : ``torch.optim.Optimizer''
        Optimizer

    Returns
    -------
    scheduler : ``torch.optim.lr_scheduler._LRScheduler''
        Scheduler

    """
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args.epochs // 4, gamma=0.1
        )
    elif args.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[args.epochs // 6, args.epochs // 3, args.epochs // 2],
            gamma=0.1,
        )
    elif args.scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_gamma
        )
        pass
    elif args.scheduler == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr,
            max_lr=args.lr * 10,
            mode="triangular",
            step_size_up=5,
            step_size_down=10,
        )
    # elif args.scheduler == "linear":
    #     schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=4)
    else:
        print(
            "Learning rate scheduling policy {} is not supported.".format(
                args.scheduler
            )
        )
        raise ValueError

    return scheduler
