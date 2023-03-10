import datetime
import os
import random
import sys
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from utils.logger import log_metrics
from utils.metrics import MIOU, AverageMeter
from utils.model_io import import_model
from tqdm import tqdm
from utils.optim_opt import get_optimizer, get_scheduler
from utils.visualization import add_images_to_tensorboard
from warmup_scheduler import GradualWarmupScheduler
from utils.pseudo_label_generator import class_balanced_pseudo_label_selection
from utils.dataset_utils import import_dataset, import_target_dataset


class MSDACLTrainer(object):
    def __init__(self, args):
        self.args = args

        #
        self.rand_seed = args.rand_seed
        self.resume_from = args.resume_from
        self.batch_size = args.batch_size
        self.use_cosine = args.use_cosine
        self.device = args.device
        self.pin_memory = args.pin_memory
        self.num_workers = args.num_workers
        self.target_name = args.target
        self.num_classes = 5 if self.target_name == "sakaki" or self.target_name == "imo" else 3
        self.model_name = args.model
        self.train_data_list_path = args.train_data_list_path
        self.val_data_list_path = args.val_data_list_path
        self.test_data_list_path = args.test_data_list_path
        self.val_every_epochs = args.val_every_epochs
        self.vis_every_vals = args.vis_every_vals
        self.save_path_root = args.save_path
        self.train_image_size_h = 256
        self.train_image_size_w = 480

        # Datasets
        self.source_train_loaders = []
        self.target_train_loader = None
        self.target_pseudo_loader = None
        self.target_val_loader = None
        self.target_test_loader = None
        self.models = []
        self.optimizers = []
        self.schedulers = []

        # Parameters
        self.lambda_col = 1.0
        self.lambda_seg_t = 0.1
        self.device = "cuda"
        self.alpha = 0.5
        self.tau = 0.9
        # Optimizer and learning rate
        self.scheduler_name = args.scheduler
        self.optimizer_name = args.optim
        self.lr = args.lr
        self.lr_gamma = args.lr_gamma
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.use_lr_warmup = args.use_lr_warmup

        self.ignore_idx = args.ignore_index
        self.epochs_source = 20
        self.epochs_target = 100
        self.max_iter = self.epochs_target * 3000 / self.batch_size

        # Loss
        self.loss_seg = torch.nn.CrossEntropyLoss(
            ignore_index=self.ignore_idx,
            reduction="mean",
        )
        self.loss_col = torch.nn.KLDivLoss(reduction="batchmean")

        #
        # Tensorboard writer
        #
        now = datetime.datetime.now() + datetime.timedelta(hours=9)
        condition = "MSDA_CL"
        self.save_path = os.path.join(
            self.save_path_root,
            self.target_name,
            condition,
            self.model_name,
            now.strftime("%Y%m%d-%H%M%S"),
        )

        self.pseudo_save_path = os.path.join(self.save_path, "pseudo_labels")
        # If the directory not found, create it
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
            os.makedirs(self.pseudo_save_path)

        self.writer = SummaryWriter(self.save_path)

        # Initialize training
        self._init_training()

        # Training settings
        self.current_iters = [0] * len(self.models)

        for i, model in enumerate(self.models):
            model = torch.nn.DataParallel(model)  # make parallel
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

            self.models[i] = model

    def source_train(self,):
        """Train the source models"""
        for ep in range(self.epochs_source):
            for i in range(len(self.models)):
                with tqdm(self.source_train_loaders[i]) as pbar_loader:
                    pbar_loader.set_description(
                        "Epoch {:<3d}/{:<3d}".format(ep+1, self.epochs_source))

                    # Data loaders for computing collaborative loss
                    loader_iters = [
                        iter(loader)
                        for loader in self.source_train_loaders
                    ]
                    for iter_i, batch_i in enumerate(self.source_train_loaders[i]):
                        loss = 0.0
                        # Get input image and label batch from dataset i (main dataset)
                        image_i = batch_i["image"].to(self.device)
                        image_orig_i = batch_i["image_orig"]
                        label_i = batch_i["label"].to(self.device)

                        # Label conversion
                        # label_i = label_conversions[i][label_i]
                        output = self.models[i](image_i)
                        output_i = output["out"] + 0.5 * output["aux"]

                        loss += self.loss_seg(output_i, label_i)

                        amax_i = torch.argmax(output_i, dim=1)

                        # Collaborative learning with other sources
                        for k, iter_k in enumerate(loader_iters):
                            # Skip the computation of loss_col for the same dataset as i
                            if i != k:
                                continue

                            batch_k = next(iter_k)
                            image_k = batch_k["image"].to(self.device)
                            output_k_i = F.log_softmax(
                                self.models[i](image_k)["out"], dim=1)
                            output_k_k = F.softmax(
                                self.models[k](image_k)["out"], dim=1)

                            loss += self.lambda_col * \
                                self.loss_col(output_k_i, output_k_k)

                            # Update tqdm
                            pbar_loader.set_postfix(
                                cls="{:.4f}".format(loss.item()),
                            )
                            pbar_loader.update()

                        loss.backward()
                        self.optimizers[i].step()
                        self.optimizers[i].zero_grad()

                        namespace = "MSDACL/stage1"
                        if self.writer is not None:
                            self.writer.add_scalar(
                                namespace + "/train/{}/loss".format(i),
                                loss.item(),
                                ep *
                                len(self.source_train_loaders[i]) + iter_i,
                            )

                            if iter_i == 0:
                                add_images_to_tensorboard(
                                    self.writer,
                                    image_orig_i,
                                    ep,
                                    namespace + "/train/{}/image".format(i)
                                )

                                add_images_to_tensorboard(
                                    self.writer,
                                    label_i,
                                    ep,
                                    namespace + "/train/{}/label".format(i),
                                    is_label=True,
                                    color_encoding=self.color_encoding,
                                )
                                add_images_to_tensorboard(
                                    self.writer,
                                    amax_i,
                                    ep,
                                    namespace + "/train/{}/pred".format(i),
                                    is_label=True,
                                    color_encoding=self.color_encoding,
                                )

    def generate_pseudo_labels(self, ):
        """Generate pseudo-labels

        Parameters
        ----------
        """
        print("Generate pseudo-labels")

        self._load_target_dataset(pseudo_only=True)

        for batch in self.target_pseudo_loader:
            image_t = batch["image"].to(self.device)
            name = batch["name"]

            P = 0
            for i in range(len(self.models)):
                P += self.models[i](image_t)["out"]

            # Normalize the summed score
            P = F.softmax(P, dim=1)
            # Y = torch.argmax(P, dim=1)
            Y = class_balanced_pseudo_label_selection(
                P,
                num_classes=self.num_classes,
                ignore_idx=self.ignore_idx,
                alpha=self.alpha,
                tau=self.tau
            )

            # Save the pseudo-labels
            for i in range(image_t.size(0)):
                label_i = Y[i]
                name_i = name[i]

                label_i = label_i.cpu().byte().numpy()
                label_i = Image.fromarray(
                    label_i.astype(np.uint8)).convert('P')
                label_i.putpalette(self.color_palette)
                label_i.save(
                    os.path.join(
                        self.pseudo_save_path,
                        "{}.png".format(
                            name_i.replace(".png", "").replace(".jpg", "")
                        )
                    )
                )

    def target_train(self,):
        """Training using target dataset"""

        best_miou = 0.0

        for ep in range(self.epochs_target):
            # Validation
            metrics = self.val(ep, visualize=True)
            # Log the metric values in a text file
            log_metrics(
                metrics=metrics,
                epoch=ep,
                save_dir=self.save_path,
                write_header=(ep == 0)
            )

            # Update best mIoU
            if best_miou < metrics["miou"]:
                best_miou = metrics["miou"]
                for i, source in enumerate(self.source_dataset_name_list):
                    torch.save(
                        self.models[i].state_dict(),
                        os.path.join(
                            self.save_path,
                            "pseudo_{}_{}_{}_best_iou.pth".format(
                                self.model_name,
                                self.target_name,
                                source,
                            ),
                        ),
                    )

            # Train
            for i in range(len(self.source_train_loaders)):
                with tqdm(self.source_train_loaders[i]) as pbar_loader:
                    pbar_loader.set_description(
                        "Epoch {:<3d}/{:<3d}".format(ep+1, self.epochs_target))

                    # Data loaders for computing collaborative loss
                    iter_t = iter(self.target_train_loader)
                    for iter_i, batch_i in enumerate(self.source_train_loaders[i]):
                        loss = 0.0
                        # Get input image and label batch from dataset i (main dataset)
                        image_i = batch_i["image"].to(self.device)
                        image_orig_i = batch_i["image_orig"]
                        label_i = batch_i["label"].to(self.device)

                        # Label conversion
                        # label_i = label_conversions[i][label_i]
                        output = self.models[i](image_i)
                        output_i = output["out"] + 0.5 * output["aux"]

                        loss += self.loss_seg(output_i, label_i)

                        amax_i = torch.argmax(output_i, dim=1)

                        # Collaborative learning with other sources
                        try:
                            # Samples the batch
                            batch_t = next(iter_t)
                        except StopIteration:
                            # restart the generator if the previous generator is exhausted.
                            iter_t = iter(self.target_train_loader)
                            batch_t = next(iter_t)

                        image_t = batch_t["image"].to(self.device)
                        image_orig_t = batch_t["image_orig"].to(self.device)
                        label_t = batch_t["label"].to(self.device)
                        output = self.models[i](image_t)
                        output_t_i = output["out"] + 0.5 * output["aux"]
                        # output_t_i = self.models[i](image_t)

                        loss += self.current_iters[i] / self.max_iter * \
                            self.lambda_seg_t * \
                            self.loss_seg(output_t_i, label_t)

                        amax_t = torch.argmax(output_t_i, dim=1)

                        self.current_iters[i] += 1

                        loss.backward()
                        self.optimizers[i].step()
                        self.optimizers[i].zero_grad()

                        namespace = "MSDACL/stage2"
                        if self.writer is not None:
                            self.writer.add_scalar(
                                namespace + "/train/{}/loss".format(i),
                                loss.item(),
                                ep *
                                len(self.source_train_loaders[i]) + iter_i,
                            )

                            if iter_i == 0:
                                add_images_to_tensorboard(
                                    self.writer,
                                    image_orig_i,
                                    ep,
                                    namespace + "/train/{}/image".format(i)
                                )

                                add_images_to_tensorboard(
                                    self.writer,
                                    label_i,
                                    ep,
                                    namespace + "/train/{}/label".format(i),
                                    is_label=True,
                                    color_encoding=self.color_encoding,
                                )
                                add_images_to_tensorboard(
                                    self.writer,
                                    amax_i,
                                    ep,
                                    namespace + "/train/{}/pred".format(i),
                                    is_label=True,
                                    color_encoding=self.color_encoding,
                                )

                                # Targes
                                add_images_to_tensorboard(
                                    self.writer,
                                    image_orig_t,
                                    ep,
                                    namespace +
                                    "/train/{}/target/image".format(i)
                                )

                                add_images_to_tensorboard(
                                    self.writer,
                                    label_t,
                                    ep,
                                    namespace +
                                    "/train/{}/target/label".format(i),
                                    is_label=True,
                                    color_encoding=self.color_encoding,
                                )
                                add_images_to_tensorboard(
                                    self.writer,
                                    amax_t,
                                    ep,
                                    namespace +
                                    "/train/{}/target/pred".format(i),
                                    is_label=True,
                                    color_encoding=self.color_encoding,
                                )

                        # Update tqdm
                        pbar_loader.set_postfix(
                            cls="{:.4f}".format(loss.item()),
                        )
                        pbar_loader.update()

    def val(self, epoch: int = -1, visualize=False) -> dict:
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
        for model in self.models:
            model.eval()

        # Loss function
        loss_cls_func = torch.nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.ignore_idx
        )

        inter_meter = AverageMeter()
        union_meter = AverageMeter()
        miou_class = MIOU(
            num_classes=self.num_classes)
        # Classification for S1
        class_total_loss = 0.0
        feature_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(self.target_val_loader):
                # Get input image and label batch
                image = batch["image"].to(self.device)
                image_orig = batch["image_orig"].to(self.device)
                label = batch["label"].to(self.device)

                # Get output
                output = 0.0
                class_total_loss = 0.0
                for model in self.models:
                    output_tmp = model(image)
                    main_output = output_tmp["out"]
                    aux_output = output_tmp["aux"]

                    output += (main_output + 0.5 * aux_output)

                    loss_val = loss_cls_func(main_output, label) + 0.5 * loss_cls_func(
                        aux_output, label
                    )

                    class_total_loss += loss_val.item()

                amax_total = torch.argmax(output, dim=1)

                inter, union = miou_class.get_iou(
                    amax_total.cpu(), label.cpu())

                inter_meter.update(inter)
                union_meter.update(union)

                # Calculate and sum up the loss
                namespace = "MSDACL/target"
                if visualize and i == 0 and self.writer is not None and self.color_encoding is not None:
                    add_images_to_tensorboard(
                        self.writer,
                        image_orig,
                        epoch,
                        namespace + "/val/image")
                    add_images_to_tensorboard(
                        self.writer,
                        label,
                        epoch,
                        namespace + "/val/label",
                        is_label=True,
                        color_encoding=self.color_encoding,
                    )
                    add_images_to_tensorboard(
                        self.writer,
                        amax_total,
                        epoch,
                        namespace + "/val/pred",
                        is_label=True,
                        color_encoding=self.color_encoding,
                    )

        iou = inter_meter.sum / (union_meter.sum + 1e-10)
        class_avg_loss = class_total_loss / len(self.target_val_loader)
        avg_iou = iou.mean()

        self.writer.add_scalar(
            namespace + "/val/class_avg_loss", 
            class_avg_loss, 
            epoch
        )
        self.writer.add_scalar(namespace + "/val/miou", avg_iou, epoch)

        # Set model mode back
        for model in self.models:
            model.train()

        metrics = {self.class_list[i]: iou[i] for i in range(iou.shape[0])}
        metrics["miou"] = avg_iou
        metrics["cls_loss"] = class_avg_loss

        return metrics

    def test(self,):
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
        # Load the best weights
        for i, source in enumerate(self.source_dataset_name_list):
            state_dict = torch.load(
                os.path.join(
                    self.save_path,
                    "pseudo_{}_{}_{}_best_iou.pth".format(
                        self.model_name,
                        self.target_name,
                        source,
                    ),
                ),
            )
            self.models[i].load_state_dict(state_dict)
            self.models[i].to(self.device)
            # Set the model to 'eval' mode
            self.models[i].eval()

        inter_meter = AverageMeter()
        union_meter = AverageMeter()
        miou_class = MIOU(num_classes=self.num_classes)
        # Classification for S1
        class_total_loss = 0.0
        test_save_path = os.path.join(
            self.save_path,
            "test",
        )

        if not os.path.isdir(test_save_path):
            os.makedirs(test_save_path)

        with torch.no_grad():
            for _, batch in enumerate(self.target_test_loader):
                # Get input image and label batch
                image = batch["image"].to(self.device)
                label = batch["label"].to(self.device)
                name = batch["name"]

                # Get output
                output = 0.0
                for model in self.models:
                    output_tmp = model(image)
                    main_output = output_tmp["out"]
                    aux_output = output_tmp["aux"]

                    output += (main_output + 0.5 * aux_output)

                amax_total = torch.argmax(output, dim=1)

                inter, union = miou_class.get_iou(
                    amax_total.cpu(), label.cpu())

                inter_meter.update(inter)
                union_meter.update(union)

                amax_total_np = amax_total[0].cpu().numpy().astype(np.uint8)
                # File name ('xxx.png')
                filename = name[0].split(
                    "/")[-1].replace(".png", "").replace(".jpg", "")
                label = Image.fromarray(amax_total_np).convert("P")
                label.putpalette(self.color_palette)
                label.save(
                    os.path.join(test_save_path, filename + ".png",)
                )

        iou = inter_meter.sum / (union_meter.sum + 1e-10)
        class_avg_loss = class_total_loss / len(self.target_val_loader)
        avg_iou = iou.mean()

        # Logging
        metrics = {self.class_list[i]: iou[i] for i in range(iou.shape[0])}
        metrics["miou"] = avg_iou
        metrics["cls_loss"] = class_avg_loss
        log_metrics(
            metrics=metrics,
            epoch=0,
            save_dir=test_save_path,
            write_header=True,
            name="metrics_test.txt",
        )

    def msdacl_main(self):
        self.source_train()
        self.generate_pseudo_labels()
        self._load_target_dataset()
        self._load_optimizers(
            max_epochs=self.epochs_target)
        self.target_train()
        self.test()

    def _init_training(self):
        """Initialize training settings"""

        # Manually set the seeds of random values
        # https://qiita.com/north_redwing/items/1e153139125d37829d2d
        torch.manual_seed(self.rand_seed)
        random.seed(self.rand_seed)
        np.random.seed(self.rand_seed)

        self._load_source_datasets()
        self._load_models()
        self._load_optimizers(
            max_epochs=self.epochs_source)

        if self.target_name == "greenhouse":
            from dataset.greenhouse import color_encoding
        elif self.target_name == "sakaki" or self.target_name == "imo":
            from dataset.sakaki import color_encoding
        else:
            raise ValueError

        self.color_encoding = color_encoding

    def _load_source_datasets(self,):
        """_summary_"""
        self.source_dataset_name_list = self.args.source_dataset_names.split(
            ",")

        #
        # Import source datasets
        #
        for dataset_name in self.source_dataset_name_list:
            try:
                dataset_s1, _, _, _ = import_dataset(
                    # self.s1_name,
                    dataset_name=dataset_name,
                    mode="train",
                    calc_class_wts=False,
                    is_class_wts_inverse=False,
                    height=self.train_image_size_h,
                    width=self.train_image_size_w,
                    max_num=3000,
                    label_conversion_to=self.target_name,
                )
            except Exception as e:
                t, v, tb = sys.exc_info()
                print(traceback.format_exception(t, v, tb))
                print(traceback.format_tb(e.__traceback__))
                print("Dataset '{}' not found".format(dataset_name))
                sys.exit(1)

            self.source_train_loaders.append(
                torch.utils.data.DataLoader(
                    dataset_s1,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=self.pin_memory,
                    num_workers=self.num_workers,
                    drop_last=True,
                )
            )

    def _load_target_dataset(self, pseudo_only: bool = False):
        """_summary_

        Parameters
        ----------
        pseudo_only : `bool`, optional
            `True` to import only dataset for pseudo-label generation. 
            Default: `False`
        """
        #
        # Import target datasets
        #
        try:
            target_dataset_pseudo, _, self.color_encoding, self.color_palette, _ = import_target_dataset(
                dataset_name=self.target_name,
                mode="pseudo",
                data_list_path=self.train_data_list_path,
                pseudo_label_dir=self.pseudo_save_path,
                is_hard=True,
                load_labels=not pseudo_only,
            )

            if not pseudo_only:
                target_dataset_train, self.num_classes, self.color_encoding, self.color_palette, self.class_list = import_target_dataset(
                    dataset_name=self.target_name,
                    mode="train",
                    data_list_path=self.train_data_list_path,
                    pseudo_label_dir=self.pseudo_save_path,
                    is_hard=True,
                    is_old_label=False,
                    max_iter=3000,
                )

                target_dataset_val, _, _, _, _ = import_target_dataset(
                    dataset_name=self.target_name,
                    mode="val",
                    data_list_path=self.val_data_list_path,
                )

                target_dataset_test, _, _, _, _ = import_target_dataset(
                    dataset_name=self.target_name,
                    mode="test",
                    data_list_path=self.test_data_list_path,
                )

                self.batch_size = min(
                    self.batch_size, len(target_dataset_train))

        except Exception as e:
            t, v, tb = sys.exc_info()
            print(traceback.format_exception(t, v, tb))
            print(traceback.format_tb(e.__traceback__))
            print("Dataset '{}' not found".format(self.target_name))
            sys.exit(1)

        #
        # Dataloader
        #
        self.target_pseudo_loader = torch.utils.data.DataLoader(
            target_dataset_pseudo,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

        if not pseudo_only:
            self.target_train_loader = torch.utils.data.DataLoader(
                target_dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                drop_last=True,
            )
            self.target_val_loader = torch.utils.data.DataLoader(
                target_dataset_val,
                batch_size=64,
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )
            self.target_test_loader = torch.utils.data.DataLoader(
                target_dataset_test,
                batch_size=1,
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )

    def _load_optimizers(self, max_epochs: int) -> None:
        """Import optimizer and scheduler"""
        #
        # Optimizer: Updates
        #
        for model in self.models:
            optimizer = get_optimizer(
                optim_name=self.optimizer_name,
                model_name=self.model_name,
                model=model,
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )

            self.optimizers.append(optimizer)

            #
            # Scheduler: Gradually changes the learning rate
            #
            scheduler = get_scheduler(
                scheduler_name=self.scheduler_name,
                optim_name=self.optimizer_name,
                optimizer=optimizer,
                epochs=max_epochs,
                lr=self.lr,
                lr_gamma=self.lr_gamma,
            )
            if self.use_lr_warmup:
                scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=1,
                    total_epoch=max_epochs // 10,
                    after_scheduler=scheduler,
                )

            self.schedulers.append(scheduler)

    def _load_models(self) -> None:
        """Load models"""
        source_model_name_list = self.args.source_model_names.split(",")
        source_weight_name_list = self.args.source_weight_names.split(",")
        for model_name, weight_name in zip(source_model_name_list, source_weight_name_list):
            model = import_model(
                model_name=model_name,
                num_classes=self.num_classes,
                # weights=self.resume_from if self.resume_from else None,
                weights=weight_name,
                aux_loss=True,
                pretrained=False,
                device=self.device,
                use_cosine=self.use_cosine,
            )
            model.to(self.device)

            self.models.append(model)


if __name__ == "__main__":
    # Pseudo-label test
    from options.train_options import MSDACLTrainOptions
    args = MSDACLTrainOptions().parse()

    msdacl = MSDACLTrainer(args)

    # msdacl.generate_pseudo_labels()
