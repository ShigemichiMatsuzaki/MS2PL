import torch
import torchvision


def import_model(
    model_name,
    num_classes,
    weights=None,
    pretrained=False,
    aux_loss=True,
    device="cuda",
):
    # Import model

    if model_name == "deeplabv3_resnet50":
        print("Weight file : {}".format(weights))
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=pretrained,
            aux_loss=aux_loss,
            num_classes=num_classes,
        )
    elif model_name == "deeplabv3_mobilenet_v3_large":
        print("Weight file : {}".format(weights))
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            pretrained=pretrained,
            aux_loss=aux_loss,
            num_classes=num_classes,
        )
    else:
        print("Model {} is not supported.".format(model_name))
        raise ValueError

    if weights is not None:
        state_dict = torch.load(weights)
        overlap_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(overlap_dict)

    model.to(device)

    return model
