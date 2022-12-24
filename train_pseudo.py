# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import os
import shutil
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
from utils.calc_prototype import calc_prototype

from utils.dataset_utils import import_dataset, DATASET_LIST

from loss_fns.segmentation_loss import UncertaintyWeightedSegmentationLoss, Entropy
from utils.pseudo_label_generator import generate_pseudo_label

from utils.logger import log_training_conditions


def train_pseudo(
    args,
    model: torch.Tensor,
    s1_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    seg_loss: torch.nn.Module,
    kld_loss: Optional[torch.nn.Module] = None,
    class_weights: Optional[torch.Tensor] = None,
    kld_loss_weight: float = 0.1,
    entropy_loss_weight: float = 0.1,
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
    kld_loss_weight: `float`
        Weight on the KLD loss
    entropy_loss_weight: `float`
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

    optimizer.zero_grad()

    #
    # Training loop
    #
    loss_cls_acc_val = 0.0
    loss_ent_acc_val = 0.0

    logsoftmax = torch.nn.LogSoftmax(dim=1)
    softmax = torch.nn.Softmax(dim=1)
    entropy = Entropy(num_classes=args.num_classes)
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

        output_ent = entropy(softmax(output_total))

        # Calculate and sum up the loss

        output_main_prob = softmax(output_main)
        output_aux_logprob = logsoftmax(output_aux)
        kld_loss_value = kld_loss(output_aux_logprob, output_main_prob).sum(dim=1)

        # Label entropy
        if not args.is_hard and args.use_label_ent_weight:
            # label = softmax(label * 5)
            label_ent = entropy(label)

            kld_weight = torch.exp(-kld_loss_value.detach()) 
            label_ent_weight = torch.exp(-label_ent.detach() * args.label_weight_temperature)
            label_ent_weight[label_ent_weight < args.label_weight_threshold] = 0.0
            u_weight = kld_weight * label_ent_weight
            # print(kld_loss_value.mean(), label_ent.mean())
        else:
            u_weight = torch.exp(-kld_loss_value.detach())

        # if args.is_hard:
        #     seg_loss_value = seg_loss(output_total, label, kld_loss_value.detach())
        # else:
        #     seg_loss_value = seg_loss(logsoftmax(output_total), torch.log(label))
        seg_loss_value = seg_loss(
            output_total,
            label,
            u_weight=u_weight,
        )

        loss_val = seg_loss_value + kld_loss_weight * kld_loss_value.mean() + entropy_loss_weight * output_ent.mean()

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
                add_images_to_tensorboard(
                    writer, image_orig, epoch, "train/image")
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
                kld = torch.reshape(
                    kld, (kld.size(0), 1, kld.size(1), kld.size(2))) / kld.max()
                add_images_to_tensorboard(
                    writer,
                    kld,
                    epoch,
                    "train/kld",
                )

                output_ent = torch.reshape(
                    output_ent, (output_ent.size(0), 1, output_ent.size(1), output_ent.size(2)))
                add_images_to_tensorboard(
                    writer,
                    output_ent,
                    epoch,
                    "train/output_ent",
                )

                if not args.is_hard and args.use_label_ent_weight:
                    label_ent_weight = torch.reshape(
                        label_ent_weight,
                        (
                            label_ent_weight.size(0),
                            1,
                            label_ent_weight.size(1),
                            label_ent_weight.size(2),
                        ),
                    )
                    add_images_to_tensorboard(
                        writer,
                        label_ent_weight,
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
    # Import datasets
    #
    try:
        from dataset.greenhouse import GreenhouseRGBD, color_encoding

        if args.is_hard:
            dataset_train = GreenhouseRGBD(
                list_name="dataset/data_list/train_greenhouse_a.lst",
                label_root=args.pseudo_label_dir,
                mode="train",
                is_hard_label=args.is_hard,
                is_old_label=args.is_old_label,
            )
        else:
            from dataset.greenhouse import GreenhouseRGBDSoftLabel, color_encoding

            dataset_train = GreenhouseRGBDSoftLabel(
                list_name="dataset/data_list/train_greenhouse_a.lst",
                label_root=args.pseudo_label_dir,
                mode="train",
            )

        dataset_pseudo = GreenhouseRGBD(
            list_name="dataset/data_list/train_greenhouse_a.lst",
            label_root=args.pseudo_label_dir,
            mode="val",
            is_hard_label=True,
            load_labels=False,
        )
        dataset_val = GreenhouseRGBD(
            list_name="dataset/data_list/val_greenhouse_a.lst",
            mode="val",
            is_hard_label=True,
            is_old_label=True,
        )

        args.num_classes = 3

    except Exception as e:
        t, v, tb = sys.exc_info()
        print(traceback.format_exception(t, v, tb))
        print(traceback.format_tb(e.__traceback__))
        print("Dataset '{}' not found".format(args.target))
        sys.exit(1)

    if args.class_wts_type == "normal" or args.class_wts_type == "inverse":
        try:
            class_wts = torch.load(
                os.path.join(
                    args.pseudo_label_dir,
                    "class_weights_" +
                    ("hard" if args.is_hard else "soft") + ".pt",
                )
            )
        except Exception as e:
            print(
                "Class weight '{}' not found".format(
                    os.path.join(
                        args.pseudo_label_dir,
                        "class_weights_" +
                        ("hard" if args.is_hard else "soft") + ".pt",
                    )
                )
            )
            sys.exit(1)
    elif args.class_wts_type == "uniform":
        class_wts = torch.ones(args.num_classes).to(args.device)
    else:
        print("Class weight type {} is not supported.".format(args.class_wts_type))
        raise ValueError

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
    # Define a model
    #
    model = import_model(
        model_name=args.model,
        num_classes=args.num_classes,
        weights=args.resume_from if args.resume_from else None,
        aux_loss=True,
        pretrained=False,
        device=args.device,
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

    if args.device == "cuda":
        model = torch.nn.DataParallel(model)  # make parallel
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model.to(args.device)
    class_wts.to(args.device)
    print(class_wts)

    #
    # Loss
    #
    seg_loss = UncertaintyWeightedSegmentationLoss(
        args.num_classes,
        class_wts=class_wts,
        ignore_index=args.ignore_index,
        device=args.device,
        temperature=1,
        reduction="mean",
        is_hard=args.is_hard,
        is_kld=not args.is_hard,
    )

    # For estimating pixel-wise uncertainty
    # (KLD between main and aux branches)
    kld_loss = torch.nn.KLDivLoss(reduction="none")

    #
    # Tensorboard writer
    #
    now = datetime.datetime.now() + datetime.timedelta(hours=9)
    condition = "pseudo_" + ("hard" if args.is_hard else "soft")
    save_path = os.path.join(
        args.save_path, condition, args.model, now.strftime("%Y%m%d-%H%M%S")
    )
    pseudo_save_path = os.path.join(save_path, "pseudo_labels")
    # If the directory not found, create it
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        os.makedirs(pseudo_save_path)

    # SummaryWriter for Tensorboard
    writer = SummaryWriter(save_path)

    # Save the training parameters
    log_training_conditions(args, save_dir=save_path)

    #
    # Training
    #
    current_miou = 0.0

    label_update_times = 0
    for ep in range(args.resume_epoch, args.epochs):
        if ep % 100 == 0 and ep != 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_path,
                    "pseudo_{}_{}_ep_{}.pth".format(
                        args.model, args.target, ep),
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
                        "pseudo_{}_{}_best_iou.pth".format(
                            args.model, args.target),
                    ),
                )

    
        train_pseudo(
            args,
            model,
            train_loader,
            optimizer,
            seg_loss=seg_loss,
            kld_loss=kld_loss,
            class_weights=class_wts,
            kld_loss_weight=args.kld_loss_weight,
            entropy_loss_weight=args.entropy_loss_weight if args.is_hard else 0.0,
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
        # if ep == args.label_update_epoch:
        if ep in args.label_update_epoch:
            args.use_label_ent_weight = False

            # Prototype-based denoising
            prototypes = None
            if args.use_prototype_denoising: 
                prototypes = calc_prototype(
                    model, dataset_pseudo, args.num_classes, args.device)

            class_wts, label_path_list = generate_pseudo_label(
                args,
                model=model,
                testloader=pseudo_loader,
                save_path=pseudo_save_path,
                prototypes=prototypes,
                proto_rect_thresh=args.conf_thresh[label_update_times],
                min_portion=args.sp_label_min_portion,
            )
            label_update_times += 1
            class_wts.to(args.device)

            # Update the configuration of the seg loss
            seg_loss.class_wts = class_wts
            args.is_hard = seg_loss.is_hard = True
            seg_loss.is_kld = False

            dataset_train = GreenhouseRGBD(
                list_name="dataset/data_list/train_greenhouse_a.lst",
                mode="train",
                is_hard_label=args.is_hard,
                load_labels=False,
            )

            dataset_train.set_label_list(label_path_list)

            train_loader = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers,
                drop_last=True,
            )

        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], ep)

        # Validate every 5 epochs
        torch.save(
            model.state_dict(),
            os.path.join(
                save_path, "pseudo_{}_{}_current.pth".format(
                    args.model, args.target)
            ),
        )

    # Remove the pseudo-labels generated during the training
    shutil.rmtree(pseudo_save_path)


if __name__ == "__main__":
    main()
