from multiprocessing.sharedctypes import Value
import torch
from options.pseudo_label_options import PseudoLabelOptions
from utils.pseudo_label_generator import generate_pseudo_label_multi_model
from utils.model_io import import_model


def main():
    args = PseudoLabelOptions().parse()

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
            aux_loss=False,
            device=args.device,
        )
        source_model_list.append(os_model)

    if args.target == "greenhouse":
        from dataset.greenhouse import GreenhouseRGBD, color_encoding

        pseudo_dataset = GreenhouseRGBD(list_name=args.target_data_list, train=False)
        pseudo_loader = torch.utils.data.DataLoader(
            pseudo_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
        )
        num_classes = 3
    else:
        print("Target {} is not supported.".format(args.target))
        raise ValueError

    class_weights = generate_pseudo_label_multi_model(
        model_list=source_model_list,
        source_dataset_name_list=source_dataset_name_list,
        target_dataset_name="greenhouse",
        data_loader=pseudo_loader,
        num_classes=num_classes,
        device=args.device,
        save_path=args.save_path,
        min_portion=args.superpixel_pseudo_min_portion,
        ignore_index=args.ignore_index,
    )


if __name__ == "__main__":
    main()
