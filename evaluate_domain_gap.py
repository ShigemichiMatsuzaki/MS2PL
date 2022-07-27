import torch

from options.pseudo_label_options import PseudoLabelOptions
from domain_gap_evaluator.domain_gap_evaluator import calculate_domain_gap
from utils.model_io import import_model


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
            os_seg_classes = 19
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

        target_dataset = GreenhouseRGBD(list_name=args.target_data_list, train=False)
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

    domain_gap_list = calculate_domain_gap(
        source_model_list, target_loader, args.device
    )["domain_gap_list"]

    print(domain_gap_list)


if __name__ == "__main__":
    main()
