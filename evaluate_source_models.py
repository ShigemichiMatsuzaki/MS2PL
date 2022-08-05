import os
import torch

from options.pseudo_label_options import PseudoLabelOptions
from utils.model_io import import_model
from domain_gap_evaluator.domain_gap_evaluator import get_output
from tqdm import tqdm
from PIL import Image
import numpy as np


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
    label_conversions = []
    for os_m, os_w, os_d in zip(
        source_model_name_list, source_weight_name_list, source_dataset_name_list
    ):
        if os_d == "camvid":
            os_seg_classes = 13
            from dataset.camvid import id_camvid_to_greenhouse as label_conversion
        elif os_d == "cityscapes":
            os_seg_classes = 19
            from dataset.cityscapes import (
                id_cityscapes_to_greenhouse as label_conversion,
            )
        elif os_d == "forest" or os_d == "greenhouse":
            os_seg_classes = 5
            from dataset.forest import id_forest_to_greenhouse as label_conversion
        else:
            print("{} is not supported.".format(os_d))
            raise ValueError

        label_conversions.append(label_conversion)

        os_model = import_model(
            model_name=os_m,
            num_classes=os_seg_classes,
            weights=os_w,
            aux_loss=True,
            device=args.device,
        )
        os_model.eval()
        source_model_list.append(os_model)

    if args.target == "greenhouse":
        from dataset.greenhouse import GreenhouseRGBD, color_palette

        target_dataset = GreenhouseRGBD(
            list_name=args.target_data_list, mode="val", load_labels=False
        )
    elif args.target == "cityscapes":
        from dataset.cityscapes import CityscapesSegmentation

        target_dataset = CityscapesSegmentation(
            root="/tmp/dataset/cityscapes", mode="val"
        )
    elif args.target == "camvid":
        from dataset.camvid import CamVidSegmentation, color_encoding

        target_dataset = CamVidSegmentation(root="/tmp/dataset/CamVid", mode="val")
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

    with torch.no_grad():
        with tqdm(total=len(target_loader)) as pbar:
            for i, batch in enumerate(tqdm(target_loader)):
                image = batch["image"].to(args.device)
                name = batch["name"]
                # output = model(image)

                # # pred = model(image)["out"]
                # pred = output["out"] + 0.5 * output["aux"]
                for j, model in enumerate(source_model_list):
                    pred = get_output(model, image, aux_weight=0.5, device=args.device)
                    amax = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
                    amax = label_conversions[j][amax]

                    # File name ('xxx.png')
                    filename = name[0].split("/")[-1].replace(".png", "")
                    label = Image.fromarray(amax.astype(np.uint8)).convert("P")
                    label.putpalette(color_palette)
                    # Save the predicted images (+ colorred images for visualization)
                    # label.save("%s/%s.png" % (save_pred_path, image_name))
                    label.save(
                        os.path.join(
                            args.save_path,
                            filename + "_" + source_dataset_name_list[j] + ".png",
                        )
                    )


if __name__ == "__main__":
    main()
