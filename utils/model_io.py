import torch
import torchvision
from typing import Optional


class Arguments(object):
    pass


def make_argument_for_espnetv2(
    num_classes: int, channels: int = 3, s: float = 2.0
) -> Arguments:
    """Return Namespace that stores parameters for ESPNetv2

    Parameters
    ----------

    Returns
    -------
    args: `Namespace`

    """
    args = Arguments()

    args.num_classes = num_classes
    args.channels = channels
    args.s = s

    return args


def import_espnetv2(num_classes: int) -> torch.nn.Module:
    """Wrapper for code to import ESPNetv2

    Parameters
    ----------

    Returns
    -------
    model: `torch.nn.Module`

    """
    from models.edgenets.model.segmentation.espnetv2 import ESPNetv2Segmentation

    args = make_argument_for_espnetv2(num_classes)
    model = ESPNetv2Segmentation(
        args,
        classes=num_classes,
    )

    return model


def import_model(
    model_name: str,
    num_classes: int,
    weights: Optional[str] = None,
    pretrained: bool = False,
    aux_loss: bool = True,
    device: Optional[str] = "cuda",
) -> torch.nn.Module:
    # Import model
    if model_name == "deeplabv3_resnet50":
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=pretrained,
            aux_loss=aux_loss,
            num_classes=num_classes,
        )
    elif model_name == "deeplabv3_mobilenet_v3_large":
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            pretrained=pretrained,
            aux_loss=aux_loss,
            num_classes=num_classes,
        )
    elif model_name == "espnetv2":
        model = import_espnetv2(num_classes=num_classes)
    elif model_name == "unet":
        from models.unet.unet import UNet

        model = UNet(num_classes=num_classes)
    else:
        print("Model {} is not supported.".format(model_name))
        raise ValueError

    if weights is not None:
        print("Weight file : {}".format(weights))
        state_dict = torch.load(weights)
        overlap_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(overlap_dict)

    model.to(device)

    return model
