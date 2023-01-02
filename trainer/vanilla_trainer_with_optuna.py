import collections
import datetime
import os
import shutil
import sys
import traceback
from typing import Optional
from loss_fns.segmentation_loss import Entropy, UncertaintyWeightedSegmentationLoss
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.calc_prototype import calc_prototype
from utils.logger import log_metrics, log_training_conditions
from utils.metrics import MIOU, AverageMeter
from utils.model_io import import_model
from utils.optim_opt import get_optimizer, get_scheduler
from utils.pseudo_label_generator import generate_pseudo_label
from utils.visualization import add_images_to_tensorboard, assign_label_on_features
from warmup_scheduler import GradualWarmupScheduler

class PseudoTrainer(object):
    """Wrapper of training, validation and hyper-parameter optimization
    
    
    """
    def __init__(self, args,):
        """Initialize the trainer

        Parameters
        ----------
        args: `Namespace`
            A set of parameters from command line arguments
        
        """
        #
        # Parameters
        #
        # self.args = args
        self.resume_epoch = args.resume_epoch
        self.epochs = args.epochs
        self.model_name = args.model
        self.target = args.target
        self.num_classes = args.num_classes
        self.ignore_index = args.ignore_index
        self.device = args.device
        self.is_hard = args.is_hard,
        self.use_kld_class_loss = args.use_kld_class_loss,
        self.use_cosine = args.use_cosine
        self.label_weight_temperature = args.label_weight_temperature
        self.use_label_ent_weight = args.use_label_ent_weight
        self.kld_loss_weight = args.kld_loss_weight
        self.entropy_loss_weight = args.entropy_loss_weight
        self.use_lr_warmup = args.use_lr_warmup
        self.label_update_epoch = args.label_update_epoch
        self.use_prototype_denoising = args.use_prototype_denoising
        self.conf_thresh = args.conf_thresh,
        self.sp_label_min_portion = args.sp_label_min_portion,
        self.batch_size = args.batch_size
        self.pin_memory = args.pin_memory
        self.num_workers = args.num_workers

        #
        # Import datasets
        #
        try:
            from dataset.greenhouse import GreenhouseRGBD, color_encoding
    
            if args.is_hard:
                self.dataset_train = GreenhouseRGBD(
                    list_name="dataset/data_list/train_greenhouse_a.lst",
                    label_root=args.pseudo_label_dir,
                    mode="train",
                    is_hard_label=args.is_hard,
                    is_old_label=args.is_old_label,
                )
            else:
                from dataset.greenhouse import GreenhouseRGBDSoftLabel, color_encoding
    
                self.dataset_train = GreenhouseRGBDSoftLabel(
                    list_name="dataset/data_list/train_greenhouse_a.lst",
                    label_root=args.pseudo_label_dir,
                    mode="train",
                )

            self.color_encoding = color_encoding
    
            self.dataset_pseudo = GreenhouseRGBD(
                list_name="dataset/data_list/train_greenhouse_a.lst",
                label_root=args.pseudo_label_dir,
                mode="val",
                is_hard_label=True,
                load_labels=False,
            )
            self.dataset_val = GreenhouseRGBD(
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
            print("Dataset '{}' not found".format(self.target))
            sys.exit(1)
    
        if args.class_wts_type == "normal" or args.class_wts_type == "inverse":
            try:
                self.class_wts = torch.load(
                    os.path.join(
                        args.pseudo_label_dir,
                        "class_weights_" +
                        ("hard" if self.is_hard else "soft") + ".pt",
                    )
                )
            except Exception as e:
                print(
                    "Class weight '{}' not found".format(
                        os.path.join(
                            args.pseudo_label_dir,
                            "class_weights_" +
                            ("hard" if self.is_hard else "soft") + ".pt",
                        )
                    )
                )
                sys.exit(1)
        elif args.class_wts_type == "uniform":
            self.class_wts = torch.ones(self.num_classes).to(self.device)
        else:
            print("Class weight type {} is not supported.".format(args.class_wts_type))
            raise ValueError
    
        #
        # Dataloader
        #
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            drop_last=True,
        )
        self.pseudo_loader = torch.utils.data.DataLoader(
            self.dataset_pseudo,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
        )
    
        #
        # Define a model
        #
        self.model = import_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            weights=args.resume_from if args.resume_from else None,
            aux_loss=True,
            pretrained=False,
            device=self.device,
            use_cosine=self.use_cosine,
        )
    
        #
        # Optimizer: Updates
        #
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        self.optimizer = get_optimizer(args, model=self.model)
    
        #
        # Scheduler: Gradually changes the learning rate
        #
        self.scheduler = get_scheduler(args, self.optimizer)
        if args.use_lr_warmup:
            self.scheduler = GradualWarmupScheduler(
                self.optimizer, multiplier=1, total_epoch=5, after_scheduler=self.scheduler
            )
    
        if args.device == "cuda":
            self.model = torch.nn.DataParallel(self.model)  # make parallel
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
    
        self.model.to(args.device)
        self.class_wts.to(args.device)
        print(self.class_wts)
    
        #
        # Loss
        #
        self.seg_loss = UncertaintyWeightedSegmentationLoss(
            self.num_classes,
            class_wts=self.class_wts,
            ignore_index=self.ignore_index,
            device=self.device,
            temperature=1,
            reduction="mean",
            is_hard=self.is_hard,
            is_kld=self.use_kld_class_loss,
        )
    
        # For estimating pixel-wise uncertainty
        # (KLD between main and aux branches)
        self.kld_loss = torch.nn.KLDivLoss(reduction="none")

        self.entropy = Entropy(num_classes=self.num_classes)
    
        #
        # Tensorboard writer
        #
        now = datetime.datetime.now() + datetime.timedelta(hours=9)
        condition = "pseudo_" + ("hard" if args.is_hard else "soft")
        self.save_path = os.path.join(
            self.save_path, condition, self.model_name, now.strftime("%Y%m%d-%H%M%S")
        )
        self.pseudo_save_path = os.path.join(self.save_path, "pseudo_labels")
        # If the directory not found, create it
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
            os.makedirs(self.pseudo_save_path)
    
        # SummaryWriter for Tensorboard
        self.writer = SummaryWriter(self.save_path)
    
        # Save the training parameters
        log_training_conditions(args, save_dir=self.save_path)

        # Optuna parameters
        self.n_trials = 100
        self.timeout = 600
    
    def train(self, epoch: int = -1,) -> None:
        """Main training process for one epoch
    
        Parameters
        ----
        epoch: `int`
            Current epoch number
    
        Returns
        -------
        `None`
        """
        # Set the model to 'train' mode
        self.model.train()
    
        # Loss function
        self.class_weights = (
            self.class_weights.to(self.device)
            if self.class_weights is not None
            else torch.ones(self.num_classes).to(self.device)
        )
    
        self.optimizer.zero_grad()
    
        #
        # Training loop
        #
        for i, batch in enumerate(self.train_loader):
            # Get input image and label batch
            image = batch["image"].to(self.device)
            image_orig = batch["image_orig"]
            label = batch["label"].to(self.device)
    
            # Get output
            if self.use_cosine and self.is_hard:
                output = self.model(image, label)
            else:
                output = self.model(image,)
    
            output_main = output["out"]
            output_aux = output["aux"]
            output_total = output_main + 0.5 * output_aux
            amax = output_total.argmax(dim=1)
            amax_main = output_main.argmax(dim=1)
            amax_aux = output_aux.argmax(dim=1)
    
            # Calculate output probability etc.
            output_main_prob = F.softmax(output_main, dim=1)
            output_aux_logprob = F.log_softmax(output_aux, dim=1)
            kld_loss_value = self.kld_loss(
                output_aux_logprob, output_main_prob).sum(dim=1)
    
            # Entropy
            output_ent = self.entropy(F.softmax(output_total, dim=1))
    
            # Label entropy
            if not self.is_hard and self.use_label_ent_weight:
                # label = softmax(label * 5)
                label_ent = self.entropy(label)
    
                kld_weight = torch.exp(-kld_loss_value.detach())
                label_ent_weight = torch.exp(-label_ent.detach()
                                             * self.label_weight_temperature)
                # label_ent_weight[label_ent_weight < args.label_weight_threshold] = 0.0
                u_weight = kld_weight * label_ent_weight
                # print(kld_loss_value.mean(), label_ent.mean())
            else:
                u_weight = torch.exp(-kld_loss_value.detach())
    
            # Classification loss
            seg_loss_value = self.seg_loss(
                output_total,
                label,
                u_weight=u_weight,
            )
    
            entropy_loss_weight = self.entropy_loss_weight if self.is_hard else 0.0,
            loss_val = seg_loss_value + self.kld_loss_weight * kld_loss_value.mean() + entropy_loss_weight * output_ent.mean()
    
            loss_val.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
            print(
                "==== Epoch {}, iter {}/{}, Cls Loss: {}, Ent Loss: {}====".format(
                    epoch,
                    i + 1,
                    len(self.train_loader),
                    seg_loss_value.item(),
                    kld_loss_value.mean().item(),
                )
            )
            if self.writer is not None:
                self.writer.add_scalar(
                    "train/cls_loss", seg_loss_value.item(), epoch * len(self.train_loader) + i
                )
                self.writer.add_scalar(
                    "train/ent_loss",
                    kld_loss_value.mean().item(),
                    epoch * len(self.train_loader) + i,
                )
                self.writer.add_scalar(
                    "train/total_loss",
                    seg_loss_value.item() + kld_loss_value.mean().item(),
                    epoch * len(self.train_loader) + i,
                )
    
                if i == 0:
                    add_images_to_tensorboard(
                        self.writer, image_orig, epoch, "train/image")
                    if not self.is_hard:
                        label = torch.argmax(label, dim=1)
    
                    add_images_to_tensorboard(
                        self.writer,
                        label,
                        epoch,
                        "train/label",
                        is_label=True,
                        color_encoding=self.color_encoding,
                    )
                    add_images_to_tensorboard(
                        self.writer,
                        amax,
                        epoch,
                        "train/pred",
                        is_label=True,
                        color_encoding=self.color_encoding,
                    )
                    add_images_to_tensorboard(
                        self.writer,
                        amax_main,
                        epoch,
                        "train/pred_main",
                        is_label=True,
                        color_encoding=self.color_encoding,
                    )
    
                    add_images_to_tensorboard(
                        self.writer,
                        amax_aux,
                        epoch,
                        "train/pred_aux",
                        is_label=True,
                        color_encoding=self.color_encoding,
                    )
    
                    kld = (
                        torch.exp(-kld_loss_value)
                        # -kld_loss_value / torch.max(kld_loss_value).item() + 1
                    )  # * 255# Scale to [0, 255]
                    kld = torch.reshape(
                        kld, (kld.size(0), 1, kld.size(1), kld.size(2))) / kld.max()
                    add_images_to_tensorboard(
                        self.writer,
                        kld,
                        epoch,
                        "train/kld",
                    )
    
                    output_ent = torch.reshape(
                        output_ent, (output_ent.size(0), 1, output_ent.size(1), output_ent.size(2)))
                    add_images_to_tensorboard(
                        self.writer,
                        output_ent,
                        epoch,
                        "train/output_ent",
                    )
    
                    if not self.is_hard and self.use_label_ent_weight:
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
                            self.writer,
                            label_ent_weight,
                            epoch,
                            "train/pixelwise_weight",
                        )


    def val(self, epoch: int = -1,) -> dict:
        """Validation

        Parameters
        ----------
        epoch: `int`
            Current epoch number

        Returns
        -------
        metrics: `dict`
            A dictionary that stores metrics as follows:
                "miou": Mean IoU
                "cls_loss": Average classification loss (cross entropy)
                "ent_loss": Average entropy loss (KLD with a uniform dist.)
        """
        # Set the model to 'eval' mode
        self.model.eval()

        # Loss function
        loss_cls_func = torch.nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.ignore_index
        )

        inter_meter = AverageMeter()
        union_meter = AverageMeter()
        miou_class = MIOU(num_classes=self.num_classes)
        # Classification for S1
        class_total_loss = 0.0
        feature_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                # Get input image and label batch
                image = batch["image"].to(self.device)
                image_orig = batch["image_orig"].to(self.device)
                label = batch["label"].to(self.device)

                # Get output
                output = self.model(image)

                main_output = output["out"]
                aux_output = output["aux"]
                feature = output["feat"]

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

                # Visualize features
                features, labels = assign_label_on_features(
                    feature,
                    label,
                    label_type='object',
                    scale_factor=16,
                    ignore_index=self.ignore_index,
                )
                feature_list += features
                label_list += labels

                # Calculate and sum up the loss
                if i == 0 and self.writer is not None and self.color_encoding is not None:
                    add_images_to_tensorboard(
                        self.writer, image_orig, epoch, "val/image")
                    add_images_to_tensorboard(
                        self.writer,
                        label,
                        epoch,
                        "val/label",
                        is_label=True,
                        color_encoding=self.color_encoding,
                    )
                    add_images_to_tensorboard(
                        self.writer,
                        amax_total,
                        epoch,
                        "val/pred",
                        is_label=True,
                        color_encoding=self.color_encoding,
                    )
                    add_images_to_tensorboard(
                        self.writer,
                        amax_main,
                        epoch,
                        "val/pred_main",
                        is_label=True,
                        color_encoding=self.color_encoding,
                    )

                    add_images_to_tensorboard(
                        self.writer,
                        amax_aux,
                        epoch,
                        "val/pred_aux",
                        is_label=True,
                        color_encoding=self.color_encoding,
                    )

        iou = inter_meter.sum / (union_meter.sum + 1e-10)
        class_avg_loss = class_total_loss / len(self.val_loader)
        avg_iou = iou.mean()

        self.writer.add_scalar("val/class_avg_loss", class_avg_loss, epoch)
        self.writer.add_scalar("val/miou", avg_iou, epoch)
        self.writer.add_embedding(
            torch.Tensor(features),
            metadata=labels,
            global_step=epoch,
        )

        return {
            "miou": avg_iou,
            "plant_iou": iou[0],
            "artificial_iou": iou[1],
            "ground_iou": iou[2],
            "cls_loss": class_avg_loss,
        }

    def test(self,):
        """
        
        """
        pass

    def fit(self,):
        """
        
        """
        current_miou = 0.0

        label_update_times = 0
        for ep in range(self.resume_epoch, self.epochs):
            if ep % 100 == 0 and ep != 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.save_path,
                        "pseudo_{}_{}_ep_{}.pth".format(
                            self.model_name, self.target, ep),
                    ),
                )

            if ep % 5 == 0:
                metrics = self.val(epoch=ep,)

                # Log the metric values in a text file
                log_metrics(
                    metrics=metrics,
                    epoch=ep,
                    save_dir=self.save_path,
                    write_header=(ep == 0)
                )

                if current_miou < metrics["miou"]:
                    current_miou = metrics["miou"]

                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            self.save_path,
                            "pseudo_{}_{}_best_iou.pth".format(
                                self.model_name, self.target),
                        ),
                    )

            # Training step
            self.train(epoch=ep,)

            # Update scheduler
            if self.use_lr_warmup:
                self.scheduler.step(ep)
            else:
                self.scheduler.step()

            # Update pseudo-labels
            # After the update, usual hard label training is done
            # if ep == args.label_update_epoch:
            if ep in self.label_update_epoch:
                self.use_label_ent_weight = False

                # Prototype-based denoising
                prototypes = None
                if self.use_prototype_denoising:
                    prototypes = calc_prototype(
                        self.model, self.dataset_pseudo, self.num_classes, self.device)

                self.class_wts, label_path_list = generate_pseudo_label(
                    model=self.model,
                    testloader=self.pseudo_loader,
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                    save_path=self.pseudo_save_path,
                    prototypes=prototypes,
                    proto_rect_thresh=self.conf_thresh[label_update_times],
                    min_portion=self.sp_label_min_portion,
                )
                label_update_times += 1
                self.class_wts.to(self.device)

                # Update the configuration of the seg loss
                self.seg_loss.class_wts = self.class_wts
                self.is_hard = self.seg_loss.is_hard = True
                self.seg_loss.is_kld = False

                from dataset.greenhouse import GreenhouseRGBD, color_encoding
                dataset_train = GreenhouseRGBD(
                    list_name="dataset/data_list/train_greenhouse_a.lst",
                    mode="train",
                    is_hard_label=self.is_hard,
                    load_labels=False,
                )

                dataset_train.set_label_list(label_path_list)

                self.train_loader = torch.utils.data.DataLoader(
                    dataset_train,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=self.pin_memory,
                    num_workers=self.num_workers,
                    drop_last=True,
                )

            self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], ep)

            # Validate every 5 epochs
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.save_path, "pseudo_{}_{}_current.pth".format(
                        self.model_name, self.target_name)
                ),
            )

        # Remove the pseudo-labels generated during the training
        shutil.rmtree(self.pseudo_save_path)


    def optuna_init_optimizer(self, trial):
        """
        
        """
        pass

    def optuna_init_training(self, trial):
        """
        
        """
        pass

    def optuna_objective(self, trial):
        """Objective function for hyper-parameter tuning by Optuna

        Parameters
        ----------
        trial: 

        
        """
        self.optuna_init_training()
        pass

    def optuna_optimize(self, n_trials: int=100, timeout=600):
        """Optimize hyper-parameters by Optuna

        Parameters
        ----------
        n_trials
        
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(self.optuna_objective, n_trials=n_trials, timeout=timeout)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))