# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

from distutils.fancy_getopt import wrap_text
import os
import sys
import traceback
import argparse
import datetime
import collections
from typing import Optional
import math

import torch
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm

import albumentations as A

from utils.metrics import AverageMeter, MIOU
from utils.visualization import add_images_to_tensorboard
from utils.optim_opt import get_optimizer, get_scheduler
from utils.model_io import import_model
from options.train_options import PreTrainOptions

from utils.dataset_utils import import_dataset, DATASET_LIST


def train(
    args,
    model: torch.Tensor,
    s1_loader: torch.utils.data.DataLoader,
    a1_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: Optional[torch.Tensor] = None,
    weight_loss_ent: float = 0.1,
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
    a1_loader: `torch.utils.data.DataLoader`
        Dataloader for the dataset to train entropy maximization
    optimizer: `torch.optim.Optimizer`
        Optimizer
    class_weights: `torch.Tensor`
        Loss weights per class for classification
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
    loss_cls_func = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        reduction="mean",
        ignore_index=args.ignore_index,
    )

    # Entropy is equivalent to KLD between output and a uniform distribution
    #   Reduction type 'batchmean' is mathematically correct,
    #   while 'mean' is not as of PyTorch 1.11.0.
    #   https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    #   The behavior will be fixed in the future release ('mean' will behave the same as 'batchmean')
    # loss_ent_func = torch.nn.KLDivLoss(reduction="batchmean")
    loss_ent_func = torch.nn.KLDivLoss(reduction="mean")
    log_softmax = torch.nn.LogSoftmax(dim=1)

    optimizer.zero_grad()

    #
    # Training loop
    #
    loss_cls_acc_val = 0.0
    loss_ent_acc_val = 0.0
    loss_cls_acc_val_count = 0.0
    loss_ent_acc_val_count = 0.0

    a1_loader_iter = iter(a1_loader)

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
        loss_cls_acc_val = loss_cls_func(output_main, label) + 0.5 * loss_cls_func(
            output_aux, label
        )
        # loss_cls_acc_val = loss_cls_func(output_total, label)

        if weight_loss_ent > 0.0:
            batch_a = a1_loader_iter.next()
            image_a = batch_a["image"].to(device)

            # Get output and convert it to log probability
            output_a = model(image_a)
            prob_a = log_softmax(output_a["out"])
            prob_a_aux = log_softmax(output_a["aux"])

            # Uniform distribution: the probability of each class is 1/num_classes
            #  The number of classes is the 1st dim of the output
            uni_dist = torch.ones_like(prob_a).to(device) / prob_a.size()[1]
            uni_dist_aux = torch.ones_like(prob_a_aux).to(
                device) / prob_a_aux.size()[1]
            # loss_val = loss_ent_func(output, uni_dist)

            # Calculate and sum up the loss
            # loss_val = weight_loss_ent * loss_val
            loss_ent_acc_val = weight_loss_ent * (
                loss_ent_func(prob_a, uni_dist)
                + 0.5 * loss_ent_func(prob_a_aux, uni_dist_aux)
            )
            loss_val = loss_cls_acc_val + loss_ent_acc_val
        else:
            loss_val = loss_cls_acc_val

        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(
            "==== Epoch {}, iter {}/{}, Cls Loss: {}, Ent Loss: {}====".format(
                epoch,
                i + 1,
                len(s1_loader),
                loss_cls_acc_val.item(),
                loss_ent_acc_val.item() if weight_loss_ent > 0 else 0.0,
            )
        )
        if writer is not None:
            writer.add_scalar(
                "train/cls_loss", loss_cls_acc_val.item(), epoch * len(s1_loader) + i
            )
            writer.add_scalar(
                "train/ent_loss",
                loss_ent_acc_val.item() if weight_loss_ent > 0 else 0.0,
                epoch * len(s1_loader) + i,
            )
            writer.add_scalar(
                "train/total_loss",
                (loss_cls_acc_val.item() + loss_ent_acc_val.item())
                if weight_loss_ent > 0
                else loss_cls_acc_val.item(),
                epoch * len(s1_loader) + i,
            )

            if i == 0:
                add_images_to_tensorboard(
                    writer, image_orig, epoch, "train/image")
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
    a1_loader: `torch.utils.data.DataLoader`
        Dataloader for the dataset to train entropy maximization
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
    loss_ent_func = torch.nn.KLDivLoss(reduction="mean")
    log_softmax = torch.nn.LogSoftmax(dim=1)

    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    miou_class = MIOU(num_classes=args.num_classes)
    # Classification for S1
    class_total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(s1_loader)):
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

            # print("==== Cls Loss: {} ====".format(loss_val.item()))

            if i == 0 and writer is not None and color_encoding is not None:
                add_images_to_tensorboard(
                    writer, image_orig, epoch, "val/image")
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

        if a1_loader is not None:
            ent_total_loss = 0.0
            for i, batch in enumerate(a1_loader):
                # Get input image and label batch
                image = batch["image"].to(device)
                image_orig = batch["image_orig"].to(device)
                label = batch["label"].to(device)

                # Get output
                output = model(image)

                main_output = log_softmax(output["out"])
                aux_output = log_softmax(output["aux"])

                uni_dist = (
                    torch.ones_like(main_output).to(
                        device) / main_output.size()[1]
                )
                uni_dist_aux = (
                    torch.ones_like(aux_output).to(
                        device) / aux_output.size()[1]
                )

                loss_val = loss_ent_func(main_output, uni_dist) + 0.5 * loss_ent_func(
                    aux_output, uni_dist_aux
                )
                ent_total_loss += loss_val.item()

                inter_meter.update(inter)
                union_meter.update(union)

                amax = (main_output + 0.5 * aux_output).argmax(dim=1)
                # Calculate and sum up the loss

                # print("==== Cls Loss: {} ====".format(loss_val.item()))

                if i == 0:
                    add_images_to_tensorboard(
                        writer, image_orig, epoch, "val/a1_image")
                    add_images_to_tensorboard(
                        writer,
                        label,
                        epoch,
                        "val/a1_label",
                        is_label=True,
                        color_encoding=color_encoding,
                    )
                    add_images_to_tensorboard(
                        writer,
                        amax,
                        epoch,
                        "val/a1_pred",
                        is_label=True,
                        color_encoding=color_encoding,
                    )

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    class_avg_loss = class_total_loss / len(s1_loader)
    avg_iou = iou.mean()
    ent_avg_loss = ent_total_loss / \
        len(a1_loader) if a1_loader is not None else 0.0

    writer.add_scalar("val/class_avg_loss", class_avg_loss, epoch)
    writer.add_scalar("val/ent_avg_loss", ent_avg_loss, epoch)
    writer.add_scalar(
        "val/total_avg_loss", class_avg_loss + weight_loss_ent * ent_avg_loss, epoch
    )
    writer.add_scalar("val/miou", avg_iou, epoch)

    return {"miou": avg_iou, "cls_loss": class_avg_loss, "ent_loss": ent_avg_loss}


def main():
    # Get arguments
    # args = parse_arguments()
    args = PreTrainOptions().parse()
    print(args)

    torch.autograd.set_detect_anomaly(True)

    #
    # Import datasets (source S1, and the rest A1)
    #
    transform = A.Compose(
        [
            # A.Resize(width=480, height=256),
            # A.RandomCrop(width=480, height=256),
            A.RandomResizedCrop(
                width=args.train_image_size_w,
                height=args.train_image_size_h,
                scale=(0.5, 2.0),
            ),
            A.HorizontalFlip(p=0.5),
        ]
    )
    max_iter = 3000 if args.weight_loss_ent > 0 else None
    try:
        dataset_s1, num_classes, color_encoding, class_wts = import_dataset(
            args.s1_name,
            mode="train",
            calc_class_wts=(args.class_wts_type != "uniform"),
            is_class_wts_inverse=(args.class_wts_type == "inverse"),
            height=args.train_image_size_h,
            width=args.train_image_size_w,
            transform=transform,
            max_iter=max_iter,
        )
        dataset_s1_val, _, _, _ = import_dataset(
            args.s1_name,
            mode="val",
            height=args.val_image_size_h,
            width=args.val_image_size_w,
        )
        args.num_classes = num_classes
    except Exception as e:
        t, v, tb = sys.exc_info()
        print(traceback.format_exception(t, v, tb))
        print(traceback.format_tb(e.__traceback__))
        print("Dataset '{}' not found".format(args.s1_name))
        sys.exit(1)

    # A1 is a set of datasets other than S1
    dataset_a1_list = []
    dataset_a1_val_list = []

    for ds in DATASET_LIST[:3]:
        # If ds is the name of S1, skip importing
        if ds == args.s1_name:
            continue

        # Import
        try:
            dataset_a_tmp, _, _, _ = import_dataset(
                ds, height=args.train_image_size_h, width=args.train_image_size_w
            )
            dataset_a_val_tmp, _, _, _ = import_dataset(
                ds,
                mode="val",
                height=args.val_image_size_h,
                width=args.val_image_size_w,
                transform=transform,
                max_iter=max_iter,
            )
        except Exception as e:
            t, v, tb = sys.exc_info()
            print(traceback.format_exception(t, v, tb))
            print(traceback.format_tb(e.__traceback__))
            print("Dataset '{}' not found".format(ds))
            sys.exit(1)

        dataset_a1_list.append(dataset_a_tmp)
        dataset_a1_val_list.append(dataset_a_val_tmp)

    # Concatenate the A1 datasets to form a single dataset
    dataset_a1 = torch.utils.data.ConcatDataset(dataset_a1_list)
    dataset_a1_val = torch.utils.data.ConcatDataset(dataset_a1_val_list)
    print(dataset_a1)

    #
    # Dataloader
    #
    print("dataset size: {}".format(len(dataset_s1)))
    train_loader_s1 = torch.utils.data.DataLoader(
        dataset_s1,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader_s1 = torch.utils.data.DataLoader(
        dataset_s1_val,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )

    train_loader_a1 = torch.utils.data.DataLoader(
        dataset_a1,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader_a1 = torch.utils.data.DataLoader(
        dataset_a1_val,
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
        weights=args.resume_from if args.resume_from else None,
        aux_loss=True,
        device=args.device,
    )

    model.to(args.device)
    class_wts.to(args.device)
    print(class_wts)

    #
    # Optimizer: Updates
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("=== Get optimizer ===")
    optimizer = get_optimizer(args, model=model)

    #
    # Scheduler: Gradually changes the learning rate
    #
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #    optimizer, milestones=[50, 100, 200], gamma=0.1
    # )
    print("=== Get scheduler ===")
    scheduler = get_scheduler(args, optimizer)
    if args.use_lr_warmup:
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler
        )

    if args.device == "cuda":
        print("=== Data parallel ===")
        model = torch.nn.DataParallel(model)  # make parallel
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    #
    # Tensorboard writer
    #
    now = datetime.datetime.now() + datetime.timedelta(hours=9)
    save_path = os.path.join(
        args.save_path, args.s1_name, args.model, now.strftime("%Y%m%d-%H%M%S")
    )
    # If the directory not found, create it
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    writer = SummaryWriter(save_path)

    #
    # Training
    #
    current_miou = 0.0

    current_ent_loss = math.inf
    print("=== Start training ===")
    for ep in range(args.resume_epoch, args.epochs):
        if ep % 100 == 0 and ep != 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_path, "{}_{}_ep_{}.pth".format(
                        args.model, args.s1_name, ep)
                ),
            )

        if ep % 5 == 0:
            metrics = val(
                args,
                model,
                s1_loader=val_loader_s1,
                a1_loader=val_loader_a1,
                writer=writer,
                color_encoding=color_encoding,
                epoch=ep,
            )

            if current_miou < metrics["miou"]:
                current_miou = metrics["miou"]

                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_path, "{}_{}_best_iou.pth".format(
                            args.model, args.s1_name)
                    ),
                )
            if current_ent_loss > metrics["ent_loss"]:
                current_ent_loss = metrics["ent_loss"]

                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_path,
                        "{}_{}_best_ent_loss.pth".format(
                            args.model, args.s1_name),
                    ),
                )

        train(
            args,
            model,
            train_loader_s1,
            train_loader_a1,
            optimizer,
            class_weights=class_wts,
            weight_loss_ent=args.weight_loss_ent,
            writer=writer,
            color_encoding=color_encoding,
            epoch=ep,
            device=args.device,
        )

        if args.use_lr_warmup:
            scheduler.step(ep)
        else:
            scheduler.step()

        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], ep)

        # Validate every 5 epochs
        torch.save(
            model.state_dict(),
            os.path.join(
                save_path, "{}_{}_current.pth".format(args.model, args.s1_name)
            ),
        )


if __name__ == "__main__":
    main()
