# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import os
import datetime
from typing import Optional
from dataset.greenhouse import GreenhouseTraversability

import torch
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm

from utils.metrics import AverageMeter, MIOU
from utils.visualization import add_images_to_tensorboard
from utils.optim_opt import get_optimizer, get_scheduler
from utils.model_io import import_model
from options.train_options import TraversabilityTrainOptions


def train(
    model: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    writer: Optional[torch.utils.tensorboard.SummaryWriter] = None,
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
    data_loader: `torch.utils.data.DataLoader`
        Dataloader for the dataset to train classification
    optimizer: `torch.optim.Optimizer`
        Optimizer
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
    loss_cls_func = torch.nn.BCEWithLogitsLoss(
        reduction="mean",
    )

    optimizer.zero_grad()

    #
    # Training loop
    #
    loss_cls_acc_val = 0.0

    # Classification for S1
    for i, batch in enumerate(tqdm(data_loader)):
        # Get input image and label batch
        image = batch["image"].to(device)
        image_orig = batch["image_orig"]
        label = batch["label"].to(device)

        # Get output
        output = model(image)
        output_trav = output["trav"].squeeze(dim=1)

        # Calculate and sum up the loss
        loss_cls_acc_val = loss_cls_func(output_trav, label.float())
        sigmoid = torch.nn.Sigmoid()

        loss_cls_acc_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        if writer is not None:
            writer.add_scalar(
                "train/cls_loss",
                loss_cls_acc_val.item(),
                epoch * len(data_loader) + i
            )

            if i == 0:
                add_images_to_tensorboard(
                    writer,
                    image_orig,
                    epoch,
                    "train/image"
                )
                add_images_to_tensorboard(
                    writer,
                    label,
                    epoch,
                    "train/label",
                    is_label=True,
                )
                output_trav = torch.reshape(
                    sigmoid(output_trav),
                    (output_trav.size(0), 1, output_trav.size(1), output_trav.size(2)))
                add_images_to_tensorboard(
                    writer,
                    output_trav,
                    epoch,
                    "train/pred",
                )


def val(
    model: torch.Tensor,
    data_loader: torch.utils.data.DataLoader,
    writer: Optional[torch.utils.tensorboard.SummaryWriter] = None,
    thresh: float = 0.85,
    epoch: int = -1,
    device: str = "cuda",
):
    """Validation

    Parameters
    ----------
    model: `torch.Tensor`
        Model to train
    data_loader: `torch.utils.data.DataLoader`
        Dataloader for the dataset to train classification
    writer: `torch.utils.tensorboard.SummaryWriter`
        SummaryWriter for TensorBoard
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
    # loss_cls_func = torch.nn.CrossEntropyLoss(
    #     reduction="mean", ignore_index=args.ignore_index
    # )
    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        # Calculate c
        prob_sum_meter = AverageMeter()
        print("=== Calculate c ===")
        for i, batch in enumerate(tqdm(data_loader)):
            # Get input image and label batch
            image = batch["image"].to(device)
            image_orig = batch["image_orig"].to(device)
            label = batch["label"].to(device)

            # Get output
            output = model(image)
            output_trav = output["trav"].squeeze(dim=1)
            # Accumulate probability for positive features
            prob_sum_meter.update(
                sigmoid(output_trav)[label == 1].mean().item(),
                image.size(0))

        c = prob_sum_meter.avg
        print("c = {}".format(c))

        # Evaluate
        inter_meter = AverageMeter()
        union_meter = AverageMeter()
        for i, batch in enumerate(tqdm(data_loader)):
            # Get input image and label batch
            image = batch["image"].to(device)
            image_orig = batch["image_orig"].to(device)
            label = batch["label"].to(device)

            # Get output
            output = model(image)
            output_trav = sigmoid(output["trav"]) / c
            output_trav = torch.clamp(
                output_trav, min=0.0, max=1.0,).squeeze(dim=1)
            # Binary mask
            output_mask = output_trav > thresh

            # Calculate metrics
            union = output_mask | (label == 1)
            inter = output_mask & (label == 1)

            inter_meter.update(inter.sum().item(), inter.size(0))
            union_meter.update(union.sum().item(), union.size(0))

            if i == 0 and writer is not None:
                add_images_to_tensorboard(
                    writer,
                    image_orig,
                    epoch,
                    "val/image")
                add_images_to_tensorboard(
                    writer,
                    label,
                    epoch,
                    "val/label",
                    is_label=True,
                )
                add_images_to_tensorboard(
                    writer,
                    output_mask.long(),
                    epoch,
                    "val/pred_mask",
                    is_label=True,
                )

                output_trav = torch.reshape(
                    output_trav,
                    (output_trav.size(0), 1, output_trav.size(1), output_trav.size(2)))

                add_images_to_tensorboard(
                    writer,
                    output_trav,
                    epoch,
                    "val/pred",
                )

    iou = inter_meter.sum / (union_meter.sum + 1e-10)

    writer.add_scalar(
        "val/miou",
        iou,
        epoch * len(data_loader) + i
    )

    return {"iou": iou}


def main():
    # Get arguments
    # args = parse_arguments()
    args = TraversabilityTrainOptions().parse()
    print(args)

    torch.autograd.set_detect_anomaly(True)

    #
    # Import datasets
    #
    dataset_train = GreenhouseTraversability(
        list_name=args.train_data_list_path, label_root="", mode="train", size=(256, 480),)
    dataset_val = GreenhouseTraversability(
        list_name=args.val_data_list_path, label_root="", mode="val", size=(256, 480),)
    dataset_test = GreenhouseTraversability(
        list_name=args.test_data_list_path, label_root="", mode="test", size=(256, 480),)
    num_classes = 3

    #
    # Dataloader
    #
    print("dataset size: {}".format(len(dataset_train)))
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )
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
        model_name="esptnet",
        num_classes=num_classes,
        weights=args.resume_from if args.resume_from else None,
        aux_loss=True,
        device=args.device,
        use_cosine=args.use_cosine,
    )
    model.to(args.device)
    # Freeze the encoder
    model.freeze_encoder()

    #
    # Optimizer: Updates
    #
    print("=== Get optimizer ===")
    optimizer = get_optimizer(
        optim_name=args.optim,
        model_name=args.model,
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    #
    # Scheduler: Gradually changes the learning rate
    #
    scheduler = get_scheduler(
        scheduler_name=args.scheduler,
        optim_name=args.optim,
        optimizer=optimizer,
        epochs=args.epochs,
        lr=args.lr,
        lr_gamma=args.lr_gamma,
    )
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
    # Dataset name
    dataset_used = args.target + "_trav"

    save_path = os.path.join(
        args.save_path,
        dataset_used,
        now.strftime("%Y%m%d-%H%M%S")
    )
    # If the directory not found, create it
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    writer = SummaryWriter(save_path)

    #
    # Training
    #
    current_miou = 0.0
    print("=== Start training ===")
    for ep in range(args.resume_epoch, args.epochs):
        if ep % 100 == 0 and ep != 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_path, "{}_{}_ep_{}.pth".format(
                        args.model, dataset_used, ep)
                ),
            )

        if ep % 5 == 0:
            metrics = val(
                model,
                data_loader=val_loader,
                writer=writer,
                epoch=ep,
            )

            if current_miou < metrics["iou"]:
                current_miou = metrics["iou"]

                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_path, "{}_{}_best_iou.pth".format(
                            args.model, dataset_used)
                    ),
                )

        train(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            writer=writer,
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
                save_path, "{}_{}_current.pth".format(args.model, dataset_used)
            ),
        )


if __name__ == "__main__":
    main()
