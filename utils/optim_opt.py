from multiprocessing.sharedctypes import Value
from xml.dom.minidom import Attr
import torch


class ConstantLR:
    def __init__(self):
        pass

    def step(self):
        pass


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
            # model.parameters(),
            [
                {
                    "params": get_decoder_weights(model, args.model),
                    "lr": args.lr,
                },
                {
                    "params": get_encoder_weights(model, args.model),
                    "lr": args.lr / 10.0,
                },
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(
            # model.parameters(),
            [
                {
                    "params": get_decoder_weights(model, args.model),
                    "lr": args.lr,
                },
                {
                    "params": get_encoder_weights(model, args.model),
                    "lr": args.lr / 10.0,
                },
            ],
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
    elif args.scheduler == "polynomial":
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=args.epochs, power=0.9, last_epoch=-1, verbose=False
        )
    elif args.scheduler == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr,
            max_lr=args.lr * 10,
            mode="triangular",
            step_size_up=5,
            step_size_down=10,
            cycle_momentum=(args.optim == "SGD"),
        )
    elif args.scheduler == "constant":
        scheduler = ConstantLR()
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


def get_encoder_weights(model: torch.nn.Module, model_name: str):
    """ """
    b = []
    if "deeplab" in model_name:
        b.append(model.backbone)
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k
    elif model_name == "espnetv2":
        for w in model.get_basenet_params():
            yield w
    else:
        raise ValueError


def get_decoder_weights(model: torch.nn.Module, model_name: str):
    """ """
    if "deeplab" in model_name:
        b = []
        b.append(model.classifier)
        if model.aux_classifier is not None:
            b.append(model.aux_classifier)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k
    elif model_name == "espnetv2":
        for w in model.get_segment_params():
            yield w
    else:
        raise ValueError
