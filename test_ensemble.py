# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from utils.logger import log_metrics
from tqdm import tqdm
from options.test_options import TestEnsembleOptions
from utils.metrics import AverageMeter, MIOU
from utils.model_io import import_model
from utils.dataset_utils import import_target_dataset
from utils.pseudo_label_generator import get_output
from domain_gap_evaluator.domain_gap_evaluator import calculate_domain_gap, calc_norm_ent
from dataset.tools.label_conversions import id_camvid_to_greenhouse
from dataset.tools.label_conversions import id_cityscapes_to_greenhouse
from dataset.tools.label_conversions import id_forest_to_greenhouse
from dataset.tools.label_conversions import id_camvid_to_sakaki
from dataset.tools.label_conversions import id_cityscapes_to_sakaki
from dataset.tools.label_conversions import id_forest_to_sakaki
from dataset.tools.label_conversions import id_camvid_to_oxford
from dataset.tools.label_conversions import id_cityscapes_to_oxford
from dataset.tools.label_conversions import id_forest_to_oxford

def test(
    model_list: list,
    dg_model_list: list,
    source_dataset_name_list: list,
    target_dataset_name: str,
    num_classes: int,
    test_loader: torch.utils.data.DataLoader,
    test_save_path: str,
    domain_gap_type: str = "per_sample",
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
    for m in model_list:
        m.eval()
        m.to(device)

    if domain_gap_type not in ["none", "per_dataset", "per_sample", "per_pixel"]:
        raise ValueError("Domain gap type '{}' is not supported".format(domain_gap_type))
    #
    # Calculate weights based on the domain gaps
    #
    # if not is_per_sample and use_domain_gap:
    if domain_gap_type == "per_dataset":
        # Domain gap. Less value means closer -> More importance.
        domain_gap_list = calculate_domain_gap(dg_model_list, test_loader, device)[
            "domain_gap_list"
        ]

        del dg_model_list

        domain_gap_weight = torch.Tensor(domain_gap_list).to(device)
        # if softmax_normalize:
        #     domain_gap_weight = torch.nn.Softmax2d()(domain_gap_weight)
        # else:
        domain_gap_weight /= domain_gap_weight.sum()

        # Calculate the inverse values of the domain gap
        domain_gap_weight = 1 / (domain_gap_weight + 1e-10)
        domain_gap_weight.to(device)

        print("Weight: {}".format(domain_gap_weight))
    elif domain_gap_type == "none":
        domain_gap_weight = torch.ones(num_classes)

        domain_gap_weight.to(device)

    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    miou_class = MIOU(num_classes=num_classes)

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for index, batch in enumerate(tqdm(test_loader)):
                image = batch["image"]
                label = batch["label"]
                name = batch["name"]

                output_list = []
                gap_total = 0.0
                output_total = 0
                ds_index = 0
                for m, os_data in zip(model_list, source_dataset_name_list):
                    # Output: Numpy, KLD: Numpy
                    output = get_output(m, image)["out"]

                    # Extract the maximum value for each target class
                    output_target = torch.zeros(
                        (output.size(0), num_classes,
                         output.size(2), output.size(3))
                    ).to(device)

                    if target_dataset_name == "greenhouse" or target_dataset_name == "imo":
                        if os_data == "camvid":
                            label_conversion = id_camvid_to_greenhouse
                        elif os_data == "cityscapes":
                            label_conversion = id_cityscapes_to_greenhouse
                        elif os_data == "forest":
                            label_conversion = id_forest_to_greenhouse
                        else:
                            raise ValueError
                    elif target_dataset_name == "sakaki":
                        if os_data == "camvid":
                            label_conversion = id_camvid_to_sakaki
                        elif os_data == "cityscapes":
                            label_conversion = id_cityscapes_to_sakaki
                        elif os_data == "forest":
                            label_conversion = id_forest_to_sakaki

                    elif target_dataset_name == "oxfordrobot":
                        if os_data == "camvid":
                            label_conversion = id_camvid_to_oxford
                        elif os_data == "cityscapes":
                            label_conversion = id_cityscapes_to_oxford
                        elif os_data == "forest":
                            label_conversion = id_forest_to_oxford
                        else:
                            raise ValueError

                    label_conversion = torch.Tensor(
                        label_conversion).to(device)

                    for i in range(num_classes):
                        indices = torch.where(label_conversion == i)[0]
                        if indices.size(0):
                            output_target[:, i] = output[:, indices].max(dim=1)[0]

                    # output_target = F.normalize(output_target, p=1)
                    output_target = F.softmax(output_target, dim=1)

                    output_list.append(output_target)

                    if domain_gap_type == "none": # No domain gap
                        output_total += output_target
                    elif domain_gap_type == "per_sample":
                        domain_gap_w = calc_norm_ent(
                            output_target,
                            reduction="per_sample"
                        )["ent"]
                        output_total += output_target / domain_gap_w
                        gap_total += domain_gap_w
                    elif domain_gap_type == "per_pixel":
                        domain_gap_w = calc_norm_ent(
                            output_target,
                            reduction="none"
                        )["ent"]
                        output_total += output_target / domain_gap_w
                        gap_total += domain_gap_w
                    else:
                        output_total += output_target * \
                            domain_gap_weight[ds_index]
                        ds_index += 1

                if domain_gap_type == "per_sample":
                    output_total *= gap_total

                output_total = F.softmax(output_total, dim=1)
                amax_total = torch.argmax(output_total, dim=1)
                inter, union = miou_class.get_iou(amax_total.cpu(), label.cpu())

                inter_meter.update(inter)
                union_meter.update(union)

                # Save prediction
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
    args = TestEnsembleOptions().parse()
    print(args)

    torch.autograd.set_detect_anomaly(True)

    #
    # Import test dataset
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
    #
    # Load pre-trained models
    #
    source_model_name_list = args.source_model_names.split(",")
    source_weight_name_list = args.source_weight_names.split(",")
    source_dataset_name_list = args.source_dataset_names.split(",")

    assert (len(source_model_name_list) == len(source_weight_name_list)) and (
        len(source_weight_name_list) == len(source_dataset_name_list)
    )

    source_model_list = []
    dg_model_list = []
    for os_m, os_w, os_d in zip(
        source_model_name_list, source_weight_name_list, source_dataset_name_list
    ):
        if os_d == "camvid":
            os_seg_classes = 13
        elif os_d == "cityscapes":
            # os_seg_classes = 19
            os_seg_classes = 20
        elif os_d == "forest" or os_d == "greenhouse":
            os_seg_classes = 5
        else:
            print("{} is not supported.".format(os_d))
            raise ValueError

        os_model = import_model(
            model_name=os_m,
            num_classes=os_seg_classes,
            weights=os_w,
            aux_loss=True,
            device=args.device,
        )
        os_model_dg = import_model(
            model_name=os_m,
            num_classes=os_seg_classes,
            weights=os_w.replace("best_iou", "best_ent_loss"),
            aux_loss=True,
            device=args.device,
        )

        # Model to evaluate domain gap
        source_model_list.append(os_model)
        dg_model_list.append(os_model_dg)

    # Save
    save_path = os.path.join(
        args.test_save_path, 
        args.target, 
        "single_ensemble",
    )

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Test
    test(
        model_list=source_model_list,
        dg_model_list=dg_model_list,
        source_dataset_name_list=source_dataset_name_list,
        target_dataset_name=args.target,
        num_classes=num_classes,
        test_loader=test_loader,
        test_save_path=save_path,
        domain_gap_type=args.domain_gap_type,
        device=args.device,
        color_palette=color_palette,
        class_list=class_list,
    )

if __name__ == "__main__":
    main()
