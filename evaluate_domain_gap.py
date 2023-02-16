import os

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from options.pseudo_label_options import PseudoLabelOptions
# from domain_gap_evaluator.domain_gap_evaluator import calculate_domain_gap
from domain_gap_evaluator.domain_gap_evaluator import calculate_domain_gap, calc_norm_ent
from utils.model_io import import_model
import matplotlib.pyplot as plt

from tqdm import tqdm

def main():
    args = PseudoLabelOptions().parse()
    print(args)

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
    for os_m, os_w, os_d in zip(
        source_model_name_list, source_weight_name_list, source_dataset_name_list
    ):
        if os_d == "camvid":
            os_seg_classes = 13
        elif os_d == "cityscapes":
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
        source_model_list.append(os_model)

    if args.target == "greenhouse":
        from dataset.greenhouse import GreenhouseRGBD, color_encoding

        target_dataset = GreenhouseRGBD(
            list_name=args.target_data_list, mode="val", load_labels=False
        )
    elif args.target == "sakaki":
        from dataset.sakaki import SakakiDataset, color_encoding

        target_dataset = SakakiDataset(
            list_name=args.target_data_list, mode="val", load_labels=False
        )

    elif args.target == "cityscapes":
        from dataset.cityscapes import CityscapesSegmentation

        target_dataset = CityscapesSegmentation(
            root="/tmp/dataset/cityscapes", mode="val"
        )
    elif args.target == "camvid":
        from dataset.camvid import CamVidSegmentation, color_encoding

        target_dataset = CamVidSegmentation(
            root="/tmp/dataset/CamVid", mode="val")
    elif args.target == "forest":
        from dataset.forest import FreiburgForestDataset, color_encoding

        target_dataset = FreiburgForestDataset(
            root="/tmp/dataset/freiburg_forest_annotated", mode="val"
        )

    else:
        print("Target {} is not supported.".format(args.target))
        raise ValueError

    target_loader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
    )

    for i, batch in enumerate(target_loader):
        image = batch["image"].to('cuda')
        print(batch['name'])
        name = batch["name"]

        with torch.no_grad():
            weight_list = []
            for model in source_model_list:
                output_tmp = model(image)
                output = output_tmp["out"] + 0.5 * output_tmp["aux"]
                output_target = F.softmax(output, dim=1)
                domain_gap_w = calc_norm_ent(
                    output_target,
                    reduction="per_sample",
                )["ent"]

                weight_list.append(1 / domain_gap_w.cpu().item())

            weight_list = np.array(weight_list)
            weight_list /= weight_list.sum()

            plt.cla()
            plt.figure()
            # Show image
            img_cv = batch["image_orig"][0].permute(1, 2, 0).numpy()
            img_cv = cv2.resize(
                img_cv, dsize=(img_cv.shape[1] * 2, img_cv.shape[0] * 2), 
                interpolation=cv2.INTER_CUBIC
            )
            plt.subplot(121).imshow(img_cv,)

            # Show chart
            plt.subplot(122)
            plt.title("Domain similarity")
            plt.ylim([0.0, 0.7])
            plt.bar(source_dataset_name_list, weight_list)
            plt.savefig(
                os.path.join(
                    args.save_path, 
                    name[0].replace(".png", "_gap.png")
                )
            )

if __name__ == "__main__":
    main()
