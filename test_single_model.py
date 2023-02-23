# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import os
import numpy as np
from PIL import Image

import torch
from utils.logger import log_metrics
from tqdm import tqdm

from utils.metrics import AverageMeter, MIOU
from utils.model_io import import_model
from options.test_options import TestSingleModelOptions

from utils.dataset_utils import import_target_dataset


def test(
    model: torch.Tensor,
    num_classes: int,
    test_loader: torch.utils.data.DataLoader,
    test_save_path: str,
    device: str = "cuda",
    color_palette: list = None,
    class_list: list = None,
) -> None:
    """Test

    Parameters
    ----------
    args: `argparse.Arguments`
        Args
    model: `torch.Tensor`
        Model to train
    test_loader: `torch.utils.data.DataLoader`
        Dataloader for the dataset to train classification
    test_save_path: `str`
        Name of the directory to save the test results
    device: `str`
        Device on which the optimization is carried out. Default: `cuda`
    color_palette: `list`
        List of color values for visualization of the label images
    class_list: `list`
        List of the object classes

    """
    # Set the model to 'eval' mode
    model.eval()

    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    miou_class = MIOU(num_classes=num_classes)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            # Get input image and label batch
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            name = batch["name"]

            # Get output
            output = model(image)

            main_output = output["out"]
            aux_output = output["aux"]

            amax_total = (main_output + 0.5 * aux_output).argmax(dim=1)
            inter, union = miou_class.get_iou(amax_total.cpu(), label.cpu())

            inter_meter.update(inter)
            union_meter.update(union)

            for j in range(amax_total.shape[0]):
                amax_total_np = amax_total[j].cpu().numpy().astype(np.uint8)
                # File name ('xxx.png')
                filename = name[j].split(
                    "/")[-1].replace(".png", "").replace(".jpg", "")
                label = Image.fromarray(amax_total_np).convert("P")
                if color_palette is not None:
                    label.putpalette(color_palette)

                label.save(
                    os.path.join(test_save_path, filename + ".png",)
                )


    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    avg_iou = iou.mean()

    # Logging
    metrics = {class_list[i]: iou[i] for i in range(iou.shape[0])}
    metrics["miou"] = avg_iou
    log_metrics(
        metrics=metrics,
        epoch=0,
        save_dir=test_save_path,
        write_header=True
    )

def main():
    # Get arguments
    # args = parse_arguments()
    args = TestSingleModelOptions().parse()
    print(args)

    torch.autograd.set_detect_anomaly(True)

    #
    # Import datasets (source S1, and the rest A1)
    #
    dataset_test, num_classes, _, color_palette, class_list = import_target_dataset(
        dataset_name=args.target,
        mode="test",
        data_list_path=args.test_data_list_path,
    )

    #
    # Dataloader
    #
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )

    #
    # Define a model
    #
    print("=== Import model ===")
    model = import_model(
        model_name=args.model,
        num_classes=num_classes,
        weights=args.resume_from,
        aux_loss=True,
        device=args.device,
        use_cosine=args.use_cosine,
    )

    model.to(args.device)

    save_path = os.path.join(
        args.test_save_path, 
        args.target, 
        "single_supervised"
    )

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Test
    test(
        model=model,
        num_classes=num_classes,
        test_loader=test_loader,
        test_save_path=save_path,
        device=args.device,
        color_palette=color_palette,
        class_list=class_list,
    )

if __name__ == "__main__":
    main()
