# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import os
import sys
import traceback
import datetime
import collections
from typing import Optional
import random
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

from options.train_options import TrainOptions
from utils.metrics import AverageMeter, MIOU
from utils.visualization import add_images_to_tensorboard
from utils.optim_opt import get_optimizer, get_scheduler
from utils.model_io import import_model

from utils.dataset_utils import import_dataset, DATASET_LIST

from loss_fns.segmentation_loss import UncertaintyWeightedSegmentationLoss
from utils.pseudo_label_generator import generate_pseudo_label


def train_pseudo(
    args,
    model: torch.Tensor,
    s1_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    seg_loss: torch.nn.Module,
    kld_loss: Optional[torch.nn.Module] = None,
    class_weights: Optional[torch.Tensor] = None,
    weight_kld_loss: float = 0.1,
    writer: Optional[torch.utils.tensorboard.SummaryWriter] = None,
    color_encoding: Optional[collections.OrderedDict] = None,
    epoch: int = -1,
    device: str = "cuda",
) -> None:
    """Main training process

    Parameters
    ----
    args: `argparse.Namespace`
        Arguments given in Argparse format
    model: `torch.Tensor`
        Model to train
    s1_loader: `torch.utils.data.DataLoader`
        Dataloader for the dataset to train classification
    optimizer: `torch.optim.Optimizer`
        Optimizer
    seg_loss: `torch.nn.Module`
        Segmentation loss
    kld_loss: `torch.nn.Module`
        KLD loss for weighting (Optional)
    class_weights: `torch.Tensor`
        Loss weights per class for classification
    weight_kld_loss: `float`
        Weight on the KLD loss
    writer: `torch.utils.tensorboard.SummaryWriter`
        SummaryWriter for TensorBoard
    color_encoding: `OrderedDict`
        Mapping from class labels to a corresponding color
    epoch: `int`
        Current epoch number
    device: `str`
        Device on which the optimization is carried out

    Returns
    -------
    `None`

    """
    # Set the model to 'train' mode
    model.train()

    # Loss function
    class_weights = (
        class_weights.to(device)
        if class_weights is not None
        else torch.ones(args.num_classes).to(device)
    )

    optimizer.zero_grad()

    #
    # Training loop
    #
    loss_cls_acc_val = 0.0
    loss_ent_acc_val = 0.0

    logsoftmax = torch.nn.LogSoftmax(dim=1)
    # Classification for S1
    for i, batch in enumerate(s1_loader):
        # Get input image and label batch
        image = batch["image"].to(device)
        image_orig = batch["image_orig"]
        label = batch["label"].to(device)

        # Get output
        output = model(image)
        output_main = output["out"]
        output_aux = output["aux"]
        output_total = output_main + 0.5 * output_aux
        amax = output_total.argmax(dim=1)
        amax_main = output_main.argmax(dim=1)
        amax_aux = output_aux.argmax(dim=1)

        # Calculate and sum up the loss

        output_main_prob = logsoftmax(output_main)
        output_aux_prob = logsoftmax(output_aux)
        kld_loss_value = kld_loss(output_aux_prob, output_main_prob).sum(dim=1)

        # Label entropy
        if not args.is_hard and args.use_label_ent_weight:
            label_ent = torch.sum(-label * torch.log(label), dim=1) / np.log(
                args.num_classes
            )

            u_weight = (
                kld_loss_value.detach()
                + label_ent.detach() * args.label_weight_temperature
            )
            # print(kld_loss_value.mean(), label_ent.mean())
        else:
            u_weight = kld_loss_value.detach()

        # if args.is_hard:
        #     seg_loss_value = seg_loss(output_total, label, kld_loss_value.detach())
        # else:
        #     seg_loss_value = seg_loss(logsoftmax(output_total), torch.log(label))
        seg_loss_value = seg_loss(
            output_total,
            label,
            u_weight=u_weight,
            is_hard=args.is_hard,
        )

        loss_val = seg_loss_value + weight_kld_loss * kld_loss_value.mean()

        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(
            "==== Epoch {}, iter {}/{}, Cls Loss: {}, Ent Loss: {}====".format(
                epoch,
                i + 1,
                len(s1_loader),
                seg_loss_value.item(),
                kld_loss_value.mean().item(),
            )
        )
        if writer is not None:
            writer.add_scalar(
                "train/cls_loss", seg_loss_value.item(), epoch * len(s1_loader) + i
            )
            writer.add_scalar(
                "train/ent_loss",
                kld_loss_value.mean().item(),
                epoch * len(s1_loader) + i,
            )
            writer.add_scalar(
                "train/total_loss",
                seg_loss_value.item() + kld_loss_value.mean().item(),
                epoch * len(s1_loader) + i,
            )

            if i == 0:
                add_images_to_tensorboard(writer, image_orig, epoch, "train/image")
                if not args.is_hard:
                    label = torch.argmax(label, dim=1)
                add_images_to_tensorboard(
                    writer,
                    label,
                    epoch,
                    "train/label",
                    is_label=True,
                    color_encoding=color_encoding,
                )
                add_images_to_tensorboard(
                    writer,
                    amax,
                    epoch,
                    "train/pred",
                    is_label=True,
                    color_encoding=color_encoding,
                )
                add_images_to_tensorboard(
                    writer,
                    amax_main,
                    epoch,
                    "train/pred_main",
                    is_label=True,
                    color_encoding=color_encoding,
                )

                add_images_to_tensorboard(
                    writer,
                    amax_aux,
                    epoch,
                    "train/pred_aux",
                    is_label=True,
                    color_encoding=color_encoding,
                )

                kld = (
                    torch.exp(-kld_loss_value)
                    # -kld_loss_value / torch.max(kld_loss_value).item() + 1
                )  # * 255# Scale to [0, 255]
                kld = torch.reshape(kld, (kld.size(0), 1, kld.size(1), kld.size(2)))
                add_images_to_tensorboard(
                    writer,
                    kld,
                    epoch,
                    "train/kld",
                )

                if not args.is_hard and args.use_label_ent_weight:
                    pixelwise_weight = (
                        torch.exp(-u_weight)
                        # -u_weight / torch.max(u_weight).item() + 1
                    )  # * 255# Scale to [0, 255]
                    pixelwise_weight = torch.reshape(
                        pixelwise_weight,
                        (
                            pixelwise_weight.size(0),
                            1,
                            pixelwise_weight.size(1),
                            pixelwise_weight.size(2),
                        ),
                    )
                    add_images_to_tensorboard(
                        writer,
                        pixelwise_weight,
                        epoch,
                        "train/pixelwise_weight",
                    )


def val(
    args,
    model: torch.Tensor,
    s1_loader: torch.utils.data.DataLoader,
    a1_loader: Optional[torch.utils.data.DataLoader] = None,
    writer: Optional[torch.utils.tensorboard.SummaryWriter] = None,
    color_encoding: Optional[collections.OrderedDict] = None,
    epoch: int = -1,
    weight_loss_ent: float = 0.1,
    device: str = "cuda",
):
    """Validation

    Parameters
    ----------
    model: `torch.Tensor`
        Model to train
    s1_loader: `torch.utils.data.DataLoader`
        Dataloader for the dataset to train classification
    weight_loss_ent: `float`
        Weight on the entropy loss
    writer: `torch.utils.tensorboard.SummaryWriter`
        SummaryWriter for TensorBoard
    color_encoding: `OrderedDict`
        Mapping from class labels to a corresponding color
    epoch: `int`
        Current epoch number
    device: `str`
        Device on which the optimization is carried out

    Returns
    -------
    metrics: `dict`
        A dictionary that stores metrics as follows:
            "miou": Mean IoU
            "cls_loss": Average classification loss (cross entropy)
            "ent_loss": Average entropy loss (KLD with a uniform dist.)
    """
    # Set the model to 'eval' mode
    model.eval()

    # Loss function
    loss_cls_func = torch.nn.CrossEntropyLoss(
        reduction="mean", ignore_index=args.ignore_index
    )

    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    miou_class = MIOU(num_classes=args.num_classes)
    # Classification for S1
    class_total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(s1_loader):
            # Get input image and label batch
            image = batch["image"].to(device)
            image_orig = batch["image_orig"].to(device)
            label = batch["label"].to(device)

            # Get output
            output = model(image)

            main_output = output["out"]
            aux_output = output["aux"]

            loss_val = loss_cls_func(main_output, label) + 0.5 * loss_cls_func(
                aux_output, label
            )
            class_total_loss += loss_val.item()

            amax_main = main_output.argmax(dim=1)
            amax_aux = aux_output.argmax(dim=1)
            amax_total = (main_output + 0.5 * aux_output).argmax(dim=1)
            inter, union = miou_class.get_iou(amax_total.cpu(), label.cpu())

            inter_meter.update(inter)
            union_meter.update(union)

            # Calculate and sum up the loss
            if i == 0 and writer is not None and color_encoding is not None:
                add_images_to_tensorboard(writer, image_orig, epoch, "val/image")
                add_images_to_tensorboard(
                    writer,
                    label,
                    epoch,
                    "val/label",
                    is_label=True,
                    color_encoding=color_encoding,
                )
                add_images_to_tensorboard(
                    writer,
                    amax_total,
                    epoch,
                    "val/pred",
                    is_label=True,
                    color_encoding=color_encoding,
                )
                add_images_to_tensorboard(
                    writer,
                    amax_main,
                    epoch,
                    "val/pred_main",
                    is_label=True,
                    color_encoding=color_encoding,
                )

                add_images_to_tensorboard(
                    writer,
                    amax_aux,
                    epoch,
                    "val/pred_aux",
                    is_label=True,
                    color_encoding=color_encoding,
                )

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    class_avg_loss = class_total_loss / len(s1_loader)
    avg_iou = iou.mean()

    writer.add_scalar("val/class_avg_loss", class_avg_loss, epoch)
    writer.add_scalar("val/miou", avg_iou, epoch)

    return {
        "miou": avg_iou,
        "cls_loss": class_avg_loss,
    }


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

    #
    # Import datasets (source S1, and the rest A1)
    #
    try:
        from dataset.greenhouse import GreenhouseRGBD, color_encoding

        if args.is_hard:
            dataset_train = GreenhouseRGBD(
                list_name="dataset/data_list/train_greenhouse_a.lst",
                mode="train",
                is_hard_label=args.is_hard,
            )
        else:
            from dataset.greenhouse import GreenhouseRGBDSoftLabel, color_encoding

            dataset_train = GreenhouseRGBDSoftLabel(
                list_name="dataset/data_list/train_greenhouse_a.lst",
                mode="train",
            )

        dataset_pseudo = GreenhouseRGBD(
            list_name="dataset/data_list/train_greenhouse_a.lst",
            mode="val",
            is_hard_label=True,
        )
        dataset_val = GreenhouseRGBD(
            list_name="dataset/data_list/val_greenhouse_a.lst",
            mode="val",
            is_hard_label=True,
            is_old_label=True,
        )
        class_wts = torch.load("./pseudo_labels/mobilenet/class_weights.pt")
        print(class_wts)
        # class_wts = torch.Tensor(class_wts)
        # class_wts.to(args.device)
        # dataset_s1, num_classes, color_encoding, class_wts = import_dataset(
        #     args.s1_name, calc_class_wts=True
        # )
        # dataset_s1_val, _, _, _ = import_dataset(args.s1_name, mode="val")
        args.num_classes = 3

    except Exception as e:
        t, v, tb = sys.exc_info()
        print(traceback.format_exception(t, v, tb))
        print(traceback.format_tb(e.__traceback__))
        print("Dataset '{}' not found".format(args.target))
        sys.exit(1)

    #
    # Define a model
    #
    model = import_model(
        model_name=args.model,
        num_classes=args.num_classes,
        weights=args.resume_from if args.resume_from else None,
        aux_loss=True,
        device=args.device,
    )

    if args.device == "cuda":
        model = torch.nn.DataParallel(model)  # make parallel
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model.to(args.device)
    class_wts.to(args.device)
    print(class_wts)

    #
    # Dataloader
    #
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=True,
    )
    pseudo_loader = torch.utils.data.DataLoader(
        dataset_pseudo,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )

    #
    # Optimizer: Updates
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = get_optimizer(args, model=model)

    #
    # Scheduler: Gradually changes the learning rate
    #
    scheduler = get_scheduler(args, optimizer)
    if args.use_lr_warmup:
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler
        )

    #
    # Loss
    #
    # if args.is_hard:
    #    seg_loss = UncertaintyWeightedSegmentationLoss(
    #        args.num_classes,
    #        class_wts=class_wts,
    #        ignore_index=args.ignore_index,
    #        device=args.device,
    #        temperature=1,
    #        reduction="mean",
    #    )
    # else:
    #    seg_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)
    seg_loss = UncertaintyWeightedSegmentationLoss(
        args.num_classes,
        class_wts=class_wts,
        ignore_index=args.ignore_index,
        device=args.device,
        temperature=1,
        reduction="mean",
    )

    # For estimating pixel-wise uncertainty (KLD between main and aux branches)
    kld_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)

    #
    # Tensorboard writer
    #
    now = datetime.datetime.now() + datetime.timedelta(hours=9)
    condition = "pseudo_" + ("hard" if args.is_hard else "soft")
    save_path = os.path.join(args.save_path, condition, now.strftime("%Y%m%d-%H%M%S"))
    pseudo_save_path = os.path.join(save_path, "pseudo_labels")
    # If the directory not found, create it
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        os.makedirs(pseudo_save_path)

    writer = SummaryWriter(save_path)

    #
    # Training
    #
    current_miou = 0.0

    for ep in range(args.resume_epoch, args.epochs):
        train_pseudo(
            args,
            model,
            train_loader,
            optimizer,
            seg_loss=seg_loss,
            kld_loss=kld_loss,
            class_weights=class_wts,
            weight_kld_loss=args.weight_loss_ent,
            writer=writer,
            color_encoding=color_encoding,
            epoch=ep,
            device=args.device,
        )

        # Update scheduler
        if args.use_lr_warmup:
            scheduler.step(ep)
        else:
            scheduler.step()

        # Update pseudo-labels
        # After the update, usual hard label training is done
        if ep == args.label_update_epoch:
            args.is_hard = True
            args.use_label_ent_weight = False

            class_wts, label_path_list = generate_pseudo_label(
                args,
                model=model,
                testloader=pseudo_loader,
                save_path=pseudo_save_path,
                proto_rect_thresh=args.conf_thresh,
                min_portion=args.sp_label_min_portion,
            )
            class_wts.to(args.device)

            seg_loss.class_wts = class_wts

            dataset_train = GreenhouseRGBD(
                list_name="dataset/data_list/train_greenhouse_a.lst",
                mode="train",
                is_hard_label=args.is_hard,
            )
            train_loader = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers,
                drop_last=True,
            )

            dataset_train.set_label_list(label_path_list)

        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], ep)

        # Validate every 5 epochs
        torch.save(
            model.state_dict(),
            os.path.join(
                save_path, "pseudo_{}_{}_current.pth".format(args.model, args.target)
            ),
        )

        if ep % 100 == 0 and ep != 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_path,
                    "pseudo_{}_{}_ep_{}.pth".format(args.model, args.target, ep),
                ),
            )

        if ep % 5 == 0:
            metrics = val(
                args,
                model,
                s1_loader=val_loader,
                writer=writer,
                color_encoding=color_encoding,
                epoch=ep,
            )

            if current_miou < metrics["miou"]:
                current_miou = metrics["miou"]

                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_path,
                        "pseudo_{}_{}_best_iou.pth".format(args.model, args.target),
                    ),
                )


if __name__ == "__main__":
    main()
