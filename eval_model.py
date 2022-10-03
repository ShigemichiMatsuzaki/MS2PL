# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import torch

from utils.model_io import import_model
from options.train_options import TrainOptions
from fvcore.nn import FlopCountAnalysis, flop_count_table


def main():
    # Get arguments
    # args = parse_arguments()
    args = TrainOptions().parse()
    print(args)
    num_classes = 3

    #
    # Define a model
    #
    model = import_model(
        model_name=args.model,
        num_classes=num_classes,
        weights=args.resume_from if args.resume_from else None,
        aux_loss=True,
        device=args.device,
    )

    # if args.device == "cuda":
    #     model = torch.nn.DataParallel(model)  # make parallel
    #     torch.backends.cudnn.enabled = True
    #     torch.backends.cudnn.benchmark = True

    model.to(args.device)
    model.eval()

    # Output FLOPs
    input = torch.rand((1, 3, 224, 224)).to(args.device)
    flops = FlopCountAnalysis(model, input)

    print(flop_count_table(flops))


if __name__ == "__main__":
    main()
