import os
import torch
import torchvision
from typing import Optional


class Arguments(object):
    pass


def make_argument_for_espnetv2(
    num_classes: int,
    channels: int = 3,
    s: float = 2.0,
    use_cosine: bool = False,
    cos_margin: float = 0.1,
    cos_logit_scale: float = 30.0,
    is_easy_margin: bool = False,
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

    # ArcFace-related parameters
    args.use_cosine = use_cosine
    args.cos_logit_scale = cos_logit_scale
    args.cos_margin = cos_margin
    args.is_easy_margin = is_easy_margin

    return args


def import_espnetv2(
    num_classes: int,
    use_cosine: bool = False,
    cos_margin: float = 0.1,
    cos_logit_scale: float = 30.0,
    is_easy_margin: bool = False,
    use_traversability: bool = False,
) -> torch.nn.Module:
    """Wrapper for code to import ESPNetv2

    Parameters
    ----------
    num_classes: `int`
        Number of classes
    use_cosine: `bool`
        `True` to use cosine-based loss (ArcFace). Default: `True`

    Returns
    -------
    model: `torch.nn.Module`
        Model
    """

    args = make_argument_for_espnetv2(
        num_classes,
        use_cosine=use_cosine,
        cos_margin=cos_margin,
        cos_logit_scale=cos_logit_scale,
        is_easy_margin=is_easy_margin,
    )
    if use_traversability:
        from models.esptnet import ESPTNet
        model = ESPTNet(
            args,
            classes=num_classes,
            spatial=False,
        )
    else:
        from models.edgenets.model.segmentation.espnetv2 import ESPNetv2Segmentation
        model = ESPNetv2Segmentation(
            args,
            classes=num_classes,
        )

    # Load ImageNet pretrained encoder
    weights = "/root/training/models/edgenets/model/classification/model_zoo/espnetv2/espnetv2_s_2.0_imagenet_224x224.pth"

    if os.path.isfile(weights):
        num_gpus = torch.cuda.device_count()
        device = "cuda" if num_gpus >= 1 else "cpu"
        pretrained_dict = torch.load(
            weights, map_location=torch.device(device))
    else:
        print("Weight file {} is not found".format(weights))
        raise FileNotFoundError

    basenet_dict = model.base_net.state_dict()
    overlap_dict = {k: v for k, v in pretrained_dict.items()
                    if k in basenet_dict}

    if len(overlap_dict) == 0:
        print("No overlapping elements")
        raise ValueError

    basenet_dict.update(overlap_dict)
    model.base_net.load_state_dict(basenet_dict)

    return model


def import_model(
    model_name: str,
    num_classes: int,
    weights: Optional[str] = None,
    pretrained: bool = False,
    aux_loss: bool = True,
    device: Optional[str] = "cuda",
    use_cosine: bool = False,
) -> torch.nn.Module:
    """Import model

    Parameters
    ----------
    model_name: `str`
        Name of the model. 
        ['deeplabv3_resnet101', 'deeplabv3_resnet50', 'deeplabv3_mobilenet_v3_large', 'espnetv2', 'unet']
    num_classes: `int`
        Number of classes
    weights: `str`
        Name of weight file. Default: `None`
    pretrained: `bool`
        `True` to use the pretrained weights provided by TorchHub
    aux_loss: `bool`
        `True` to use auxiliary branch. Default: `True`
    device: `str`
        Device. Default: `cuda`
    use_cosine: `bool`
        `True` to use normalized classifier. Valid only for 'espnetv2'.
        Default: `False`

    Returns
    -------
    model: `torch.nn.Module`
        Imported model

    """
    # Import model
    if model_name == "deeplabv3_resnet101":
        model = torchvision.models.segmentation.deeplabv3_resnet101(
            pretrained=pretrained,
            aux_loss=aux_loss,
            num_classes=num_classes,
        )

    elif model_name == "deeplabv3_resnet50":
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
        model = import_espnetv2(
            num_classes=num_classes,
            use_cosine=use_cosine,
        )
    elif model_name == "esptnet":
        model = import_espnetv2(
            num_classes=num_classes,
            use_cosine=use_cosine,
            use_traversability=True,
        )

    elif model_name == "unet":
        from models.unet.unet import UNet

        model = UNet(num_classes=num_classes)
    else:
        print("Model {} is not supported.".format(model_name))
        raise ValueError

    if weights is not None:
        print("Weight file : {}".format(weights))
        state_dict = torch.load(weights)
        model_dict = model.state_dict()
        # print(state_dict.keys())
        # print(model_dict.keys())
        overlap_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
            if k.replace("module.", "") in model_dict
            and model_dict[k.replace("module.", "")].size() == v.size()
        }

        # Just for debugging
        non_overlap_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
            if not (
                k.replace("module.", "") in model_dict
                and model_dict[k.replace("module.", "")].size() == v.size()
            )
            and "depth" not in k
        }

        # for k, v in state_dict.items():
        #     if k.replace("module.", "") in model_dict and model_dict[k.replace("module.", "")].size() != v.size():
        #         print(v.size(), model_dict[k.replace("module.", "")].size())
        #     elif k.replace("module.", "") not in model_dict:
        #         print("{} not in model_dict".format(k.replace("module.", "")))

        # print(non_overlap_dict.keys())
        # print(state_dict["module.bu_dec_l4.merge_layer.3.weight"].size())
        # print(model_dict["bu_dec_l4.merge_layer.3.weight"].size())

        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict)

        print("{} % of paremteres are loaded.".format(
            len(overlap_dict)/len(model_dict) * 100))

    model.to(device)

    return model
