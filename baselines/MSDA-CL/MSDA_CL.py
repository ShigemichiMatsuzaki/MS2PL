import os
import numpy as np
import torch
from PIL import Image
from loss_fns.segmentation_loss import UncertaintyWeightedSegmentationLoss
from utils.model_io import import_model
from tqdm import tqdm
import torch.nn.functional as F

class MSDA_CL(object):
    def __init__(self, args):
        self.datasets = []
        self.train_loaders = []
        self.target_loader = None
        self.source_models = []
        self.target_model = None
        self.lambda_col = 0.5
        self.source_optimizers = []
        self.device = "cuda"
        self.num_classes = 3
        self.alpha = 0.7
        self.ignore_idx = 3


        # Loss
        self.loss_seg = UncertaintyWeightedSegmentationLoss(
            self.num_classes,
            class_wts=self.class_wts,
            ignore_index=self.ignore_index,
            device=self.device,
            temperature=1,
            reduction="mean",
            is_hard=self.params.is_hard,
            is_kld=self.params.use_kld_class_loss,
        )
        self.loss_col = torch.nn.KLDivLoss(reduction="batchmean")

        # Optimizer

        # Save path
        self.pseudo_save_path = ""

    def source_train(self, epoch: int):
        """Train the source models

        Parameters
        ----------
        epoch : `int`
            The epoch number
        """
        for i in len(self.source_models):
            with tqdm(self.train_loaders[i]) as pbar_loader:
                pbar_loader.set_description("Epoch {:<3d}".format(epoch+1))

                # Data loaders for computing collaborative loss
                loader_iters = [iter(loader) 
                    for j, loader in enumerate(self.train_loaders[i]) 
                ]
                for batch_i in self.train_loaders[i]:
                    # Get input image and label batch from dataset i (main dataset)
                    image_i = batch_i["image"].to(self.device)
                    image_orig_i = batch_i["image_orig"]
                    label_i = batch_i["label"].to(self.device)

                    # Label conversion
                    # label_i = label_conversions[i][label_i]
                    output_i = self.source_models[i](image_i)

                    loss = self.loss_seg(output_i, label_i)

                    # Collaborative learning with other sources
                    for k, iter_k in enumerate(len(loader_iters)):
                        # Skip the computation of loss_col for the same dataset as i
                        if i != k:
                            continue

                        batch_k = next(iter_k)
                        image_k = batch_k["image"].to(self.device)
                        output_k_i = self.source_models[i](image_k)
                        output_k_k = self.source_models[k](image_k)

                        loss += self.lambda_col * self.loss_col(output_k_k, output_k_i)
                
                loss.backward()
                self.source_optimizers[i].step()
                self.source_optimizers[i].zero_grad()

    def generate_pseudo_labels(self, tau: float=0.9):
        """Generate pseudo-labels

        Parameters
        ----------
        tau : `float`, optional
            Minimum value for the ratio of pseudo-labels to be selected per class. Default: `0.9`
        """
        for batch in self.target_loader:
            image_t = batch["image"].to(self.device)
            name = batch["name"]

            P = 0
            for i in len(self.source_models):
                P += self.target_model(image_t)

            # Normalize the summed score
            P = F.softmax(P, dim=1)
            Y = torch.argmax(P, dim=1)

            # Select the labels
            for c in range(self.num_classes):
                P_c = sorted(P[:, c, :, :], reverse=True)

                # Number of pixels predicted as class c
                n_c = (Y == c).sum()

                # Confidence threshold for pseudo-label c
                th = min(P_c[int(n_c * self.alpha)], tau)

                # Replace with 'ignore_idx' with the labels 
                # with confidence below the threshold
                mask1 = (Y == c)
                mask2 = (P[:, c, :, :] <= th)

                Y[mask1 & mask2] = self.ignore_idx

            # Save the pseudo-labels
            for i in range(image_t.size(0)):
                label_i = Y[i]
                name_i = name[i]

                label_i = label_i.cpu().byte().numpy()
                label_i = Image.fromarray(label_i.astype(np.uint8))
                label_i.save(
                    os.path.join(
                        self.pseudo_save_path, 
                        "{}_argmax.png".format(name_i)
                    )
                )

    def target_train(self,):
        pass

    def msdacl_main(self):
        self.source_train()
        self.generate_pseudo_labels()
        self.target_train()
        
    