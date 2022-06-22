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

from nbformat import write
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler


from utils.metrics import AverageMeter, MIOU
from utils.visualization import add_images_to_tensorboard
from utils.optim_opt import get_optimizer, get_scheduler

DATASET_LIST = ["camvid", "cityscapes", "forest"]


def parse_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Trainer of domain gap evaluator")

    # Directory
    parser.add_argument("--save-path", help="Save path (tensorboard etc.)")
    # Dataset
    parser.add_argument(
        "--s1-name",
        choices=DATASET_LIST,
        default="camvid",
        help="The dataset used as S1, the main source",
    )

    # GPU/CPU
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Batch size in training",
    )

    # DataLoader
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Whether to use 'pin_memory' in DataLoader",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers used in DataLoader. Should be 4 times the number of GPUs?",
    )

    # Training conditions
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size in training"
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="The number of training epochs"
    )
    # Loss
    parser.add_argument(
        "--weight-loss-ent", type=float, default=0.2, help="Weight on the entropy loss"
    )

    # Optimizer and scheduler
    parser.add_argument(
        "--optim", type=str, default="SGD", help="Optimizer. Default: SGD"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay by L2 normalization",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="cyclic",
        help="Type of the learning rate cheduler. Default: cyclic",
    )
    parser.add_argument(
        "--lr-gamma",
        action="store_true",
        help="Whether to use learning rate warmup",
    )
    parser.add_argument(
        "--use-lr-warmup",
        action="store_true",
        help="Whether to use learning rate warmup",
    )

    return parser.parse_args()


def import_dataset(dataset_name, mode="train"):
    """Import a designated dataset

    Args:
        dataset_name (string):
            Name of the dataset to import

    Returns:
      A list of parsed arguments.
    """
    max_iter = 3000
    if dataset_name == DATASET_LIST[0]:
        from dataset.camvid import CamVidSegmentation, color_encoding

        class_num = 13
        class_wts = torch.ones(13)

        dataset = CamVidSegmentation(
            root="/tmp/dataset/CamVid", mode=mode, max_iter=max_iter
        )
    elif dataset_name == DATASET_LIST[1]:
        from dataset.cityscapes import CityscapesSegmentation, color_encoding

        dataset = CityscapesSegmentation(
            root="/tmp/dataset/cityscapes", mode=mode, max_iter=max_iter
        )
        class_num = 19

        class_wts = torch.ones(19)
        class_wts[0] = 2.8149201869965
        class_wts[1] = 6.9850029945374
        class_wts[2] = 3.7890393733978
        class_wts[3] = 9.9428062438965
        class_wts[4] = 9.7702074050903
        class_wts[5] = 9.5110931396484
        class_wts[6] = 10.311357498169
        class_wts[7] = 10.026463508606
        class_wts[8] = 4.6323022842407
        class_wts[9] = 9.5608062744141
        class_wts[10] = 7.8698215484619
        class_wts[11] = 9.5168733596802
        class_wts[12] = 10.373730659485
        class_wts[13] = 6.6616044044495
        class_wts[14] = 10.260489463806
        class_wts[15] = 10.287888526917
        class_wts[16] = 10.289801597595
        class_wts[17] = 10.405355453491
        class_wts[18] = 10.138095855713
    elif dataset_name == DATASET_LIST[2]:
        from dataset.forest import FreiburgForestDataset, color_encoding

        dataset = FreiburgForestDataset(
            root="/tmp/dataset/freiburg_forest_annotated/", mode=mode, max_iter=max_iter
        )
        class_num = 5
        class_wts = torch.ones(5)
    else:
        raise Exception

    return dataset, class_num, color_encoding, class_wts


def train(
    args,
    model,
    s1_loader,
    a1_loader,
    optimizer,
    class_weights=None,
    weight_loss_ent=0.1,
    writer=None,
    color_encoding=None,
    epoch=-1,
    device="cuda",
):
    """Main training process

    Args:
        model:

        s1_loader:

        a1_loader:

        device:

    """
    # Set the model to 'train' mode
    model.train()

    # Loss function
    class_weights = (
        class_weights.to(device)
        if class_weights is not None
        else torch.ones(args.class_num).to(device)
    )
    loss_cls_func = torch.nn.CrossEntropyLoss(
        weight=class_weights, reduction="mean", ignore_index=255
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
        amax = output_main.argmax(dim=1)
        amax_aux = output_aux.argmax(dim=1)

        # Calculate and sum up the loss
        loss_cls_acc_val = loss_cls_func(output_main, label) + 0.5 * loss_cls_func(
            output_aux, label
        )

        batch_a = a1_loader_iter.next()
        image_a = batch_a["image"].to(device)

        # Get output and convert it to log probability
        output_a = log_softmax(model(image_a)["out"])

        # Uniform distribution: the probability of each class is 1/class_num
        #  The number of classes is the 1st dim of the output
        uni_dist = torch.ones_like(output_a).to(device) / output_a.size()[1]
        # loss_val = loss_ent_func(output, uni_dist)

        # Calculate and sum up the loss
        # loss_val = weight_loss_ent * loss_val
        loss_ent_acc_val = weight_loss_ent * loss_ent_func(output_a, uni_dist)

        loss_val = loss_cls_acc_val + loss_ent_acc_val

        loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(
            "==== Epoch {}, iter {}/{}, Cls Loss: {}, Ent Loss: {}====".format(
                epoch,
                i + 1,
                len(s1_loader),
                loss_cls_acc_val.item(),
                loss_ent_acc_val.item(),
            )
        )
        if writer is not None:
            writer.add_scalar(
                "train/cls_loss", loss_cls_acc_val.item(), epoch * len(s1_loader) + i
            )
            writer.add_scalar(
                "train/ent_loss", loss_ent_acc_val.item(), epoch * len(s1_loader) + i
            )
            writer.add_scalar(
                "train/total_loss",
                loss_cls_acc_val.item() + loss_ent_acc_val.item(),
                epoch * len(s1_loader) + i,
            )

        if i == 0:
            add_images_to_tensorboard(writer, image_orig, epoch, "train/image")
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
                amax_aux,
                epoch,
                "train/pred_aux",
                is_label=True,
                color_encoding=color_encoding,
            )


def val(model, val_loader, writer, color_encoding, epoch, num_classes, device="cuda"):
    """Validation

    Args:
        model:

        s1_loader:

        device:
    """
    # Set the model to 'eval' mode
    model.eval()

    # Loss function
    loss_cls_func = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=255)

    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    miou_class = MIOU(num_classes=num_classes)
    # Classification for S1
    total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
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
            total_loss += loss_val.item()

            amax = main_output.argmax(dim=1)
            amax_aux = aux_output.argmax(dim=1)
            inter, union = miou_class.get_iou(amax.cpu(), label.cpu())

            inter_meter.update(inter)
            union_meter.update(union)

            # Calculate and sum up the loss

            # print("==== Cls Loss: {} ====".format(loss_val.item()))

            if i == 0:
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
                    amax,
                    epoch,
                    "val/pred",
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

    avg_loss = total_loss / len(val_loader)
    avg_iou = iou.mean()

    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/miou", avg_iou, epoch)

    return {"miou": avg_iou, "loss": avg_loss}


def main():
    # Get arguments
    args = parse_arguments()
    print(args)

    torch.autograd.set_detect_anomaly(True)

    #
    # Import datasets (source S1, and the rest A1)
    #
    try:
        dataset_s1, class_num, color_encoding, class_wts = import_dataset(args.s1_name)
        dataset_s1_val, _, _, _ = import_dataset(args.s1_name, mode="val")
        args.class_num = class_num
    except Exception as e:
        t, v, tb = sys.exc_info()
        print(traceback.format_exception(t, v, tb))
        print(traceback.format_tb(e.__traceback__))
        print("Dataset '{}' not found".format(args.s1_name))
        sys.exit(1)

    # A1 is a set of datasets other than S1
    dataset_a1_list = []
    for ds in DATASET_LIST:
        # If ds is the name of S1, skip importing
        if ds == args.s1_name:
            continue

        # Import
        try:
            dataset_a_tmp, _, _, _ = import_dataset(ds)
        except Exception as e:
            t, v, tb = sys.exc_info()
            print(traceback.format_exception(t, v, tb))
            print(traceback.format_tb(e.__traceback__))
            print("Dataset '{}' not found".format(ds))
            sys.exit(1)

        dataset_a1_list.append(dataset_a_tmp)

    # Concatenate the A1 datasets to form a single dataset
    dataset_a1 = torch.utils.data.ConcatDataset(dataset_a1_list)
    print(dataset_a1)

    #
    # Define a model
    #
    #    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False)
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        pretrained=False,
        aux_loss=True,
        num_classes=class_num,
    )
    # Change the classification layer to match the category
    # model.classifier[4] = torch.nn.Conv2d(256, class_num, 1)

    if args.device == "cuda":
        model = torch.nn.DataParallel(model)  # make parallel
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model.to(args.device)
    class_wts.to(args.device)

    #
    # Dataloader
    #
    train_loader_s1 = torch.utils.data.DataLoader(
        dataset_s1,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
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
    )

    #
    # Optimizer: Updates
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = get_optimizer(args, model=model)

    #
    # Scheduler: Gradually changes the learning rate
    #
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #    optimizer, milestones=[50, 100, 200], gamma=0.1
    # )
    scheduler = get_scheduler(args, optimizer)
    if args.use_lr_warmup:
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler
        )

    #
    # Tensorboard writer
    #
    now = datetime.datetime.now() + datetime.timedelta(hours=9)
    save_path = os.path.join(
        args.save_path, args.s1_name, now.strftime("%Y%m%d-%H%M%S")
    )
    # If the directory not found, create it
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    writer = SummaryWriter(save_path)

    #
    # Training
    #
    for ep in range(args.epochs):
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

        current_miou = 0.0
        # Validate every 5 epochs
        torch.save(
            model.state_dict(),
            os.path.join(save_path, "{}_ent_current.pth".format(args.s1_name)),
        )

        if ep % 5 == 0:
            metrics = val(
                model,
                val_loader=val_loader_s1,
                writer=writer,
                color_encoding=color_encoding,
                epoch=ep,
                num_classes=class_num,
            )

            if current_miou < metrics["miou"]:
                current_miou = metrics["miou"]

                torch.save(
                    model.state_dict(),
                    os.path.join(save_path, "{}_ent_best.pth".format(args.s1_name)),
                )


if __name__ == "__main__":
    main()
