import torch

SUPPORTED_OPTIMIZERS = ['SGD', 'Adam']
SUPPORTED_SCHEDULERS = ['step', 'multistep', 'exponential', 'polynomial', 'cyclic', 'constant']

class ConstantLR:
    def __init__(self):
        pass

    def step(self):
        pass


def get_optimizer(
    optim_name: str, 
    model_name: str, 
    model: torch.nn.Module, 
    lr: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    """Get optimizer

    Parameters
    ----------
    optim_name: `str`
        Name of optimizer
    model_name: `str`
        Name of model
    model : `torch.nn.Module`
        A model to optimize
    lr: `float`
        Base learning rate
    weight_decay: `float`
        Weight decay
    momentum: `float`
        Momentum value for SGD

    Return
    ------
    optimizer : `torch.optim.Optimizer`
        An optimizer

    """
    if optim_name == "Adam":
        optimizer = torch.optim.Adam(
            # model.parameters(),
            [
                {
                    "params": get_decoder_weights(model, model_name),
                    "lr": lr,
                },
                {
                    "params": get_encoder_weights(model, model_name),
                    "lr": lr / 10.0,
                },
            ],
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optim_name == "SGD":
        optimizer = torch.optim.SGD(
            # model.parameters(),
            [
                {
                    "params": get_decoder_weights(model, model_name),
                    "lr": lr,
                },
                {
                    "params": get_encoder_weights(model, model_name),
                    "lr": lr / 10.0,
                },
            ],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        print("Invalid optimizer name {}".format(optim_name))
        raise ValueError

    return optimizer


def get_scheduler(
    scheduler_name: str, 
    optim_name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    lr: float,
    lr_gamma: float,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler

    Parameters
    ----------
    scheduler_name: `str`
        Name of scheduler
    optimizer : `torch.optim.Optimizer`
        Optimizer
    epochs: `int`
        The total number of training epochs

    Returns
    -------
    scheduler : `torch.optim.lr_scheduler._LRScheduler`
        Scheduler

    """
    if scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, epochs // 4, gamma=0.1
        )
    elif scheduler_name == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[epochs // 6, epochs // 3, epochs // 2],
            gamma=lr_gamma,
        )
    elif scheduler_name == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_gamma
        )
    elif scheduler_name == "polynomial":
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=epochs, power=0.9, last_epoch=-1, verbose=False
        )
    elif scheduler_name == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr,
            max_lr=lr * 10,
            mode="triangular",
            step_size_up=5,
            step_size_down=10,
            cycle_momentum=(optim_name == "SGD"),
        )
    elif scheduler_name == "constant":
        scheduler = ConstantLR()
    # elif scheduler_name == "linear":
    #     schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=4)
    else:
        print(
            "Learning rate scheduling policy {} is not supported.".format(
                scheduler_name
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
