import os
import torch
from options.pseudo_label_options import PseudoLabelOptions
from utils.pseudo_label_generator import generate_pseudo_label_multi_model
from utils.pseudo_label_generator import generate_pseudo_label_multi_model_domain_gap
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
    dg_model_list = []
    for os_m, os_w, os_d in zip(
        source_model_name_list, source_weight_name_list, source_dataset_name_list
    ):
        if os_d == "camvid":
            os_seg_classes = 13
        elif os_d == "cityscapes":
            # os_seg_classes = 19
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
        os_model_dg = import_model(
            model_name=os_m,
            num_classes=os_seg_classes,
            weights=os_w.replace("best_iou", "best_ent_loss"),
            aux_loss=True,
            device=args.device,
        )

        # Model to evaluate domain gap
        source_model_list.append(os_model)
        dg_model_list.append(os_model_dg)

    if args.target == "greenhouse":
        from dataset.greenhouse import GreenhouseRGBD, color_encoding

        pseudo_dataset = GreenhouseRGBD(
            list_name=args.target_data_list, mode="val", load_labels=False
        )
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

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    #
    # Generate pseudo-labels
    #
    if args.is_hard:
        class_wts = generate_pseudo_label_multi_model(
            model_list=source_model_list,
            source_dataset_name_list=source_dataset_name_list,
            target_dataset_name="greenhouse",
            data_loader=pseudo_loader,
            num_classes=num_classes,
            device=args.device,
            save_path=args.save_path,
            min_portion=args.sp_label_min_portion,
            ignore_index=args.ignore_index,
        )
    else:
        class_wts = generate_pseudo_label_multi_model_domain_gap(
            model_list=source_model_list,
            dg_model_list=dg_model_list,
            source_dataset_name_list=source_dataset_name_list,
            target_dataset_name="greenhouse",
            data_loader=pseudo_loader,
            num_classes=num_classes,
            save_path=args.save_path,
            device=args.device,
            use_domain_gap=args.use_domain_gap,
            label_normalize="softmax" if args.is_softmax_normalize else "L1",
            is_per_pixel=args.is_per_pixel,
            is_per_sample=args.is_per_sample,
            ignore_index=args.ignore_index,
        )

    # class_wts = torch.Tensor(class_wts)
    filename = "class_weights_{}.pt".format("hard" if args.is_hard else "soft")
    torch.save(class_wts, os.path.join(args.save_path, filename))


if __name__ == "__main__":
    main()
