# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import sys
import traceback
import torch
import argparse

DATASET_LIST = ['camvid', 'cityscapes', 'forest']

def parse_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Trainer of domain gap evaluator")
    # Dataset
    parser.add_argument('--s1-name', choices=DATASET_LIST, default='camvid', 
        help='The dataset used as S1, the main source')

    # GPU/CPU
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
        help='Batch size in training')

    # Training conditions
    parser.add_argument('--batch-size', type=int, default=32,
        help='Batch size in training')
    parser.add_argument('--ent-weight', type=float, default=0.1, 
        help='A weight for the entropy loss')
    parser.add_argument('--epochs', type=float, default=500,
        help='The number of training epochs')

    return parser.parse_args()


def import_dataset(dataset_name):
    """Import a designated dataset

    Args:
        dataset_name (string): Name of the dataset to import

    Returns:
      A list of parsed arguments.
    """
    if dataset_name == DATASET_LIST[0]:
        from dataset.camvid import CamVidSegmentation
        dataset = CamVidSegmentation(root='/tmp/dataset/CamVid', size=(480, 256))
        class_num = 13
    elif dataset_name == DATASET_LIST[1]:
        from dataset.cityscapes import CityscapesSegmentation
        dataset = CityscapesSegmentation(root='/tmp/dataset/cityscapes', size=(480, 256))
        class_num = 20
    elif dataset_name == DATASET_LIST[2]:
        from dataset.forest import FreiburgForestDataset
        dataset = FreiburgForestDataset(root='/tmp/dataset/freiburg_forest_annotated/', size=(480, 256))
        class_num = 5
    else:
        raise Exception

    return dataset, class_num


def train(model, s1_loader, a1_loader, optimizer, weight_loss_ent=0.1, device='cuda'):
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
    loss_cls_func = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_ent_func = torch.nn.KLDivLoss(reduction='mean') # Entropy is equivalent to KLD between output and a uniform distribution

    optimizer.zero_grad()

    #
    # Training loop
    #
    loss_cls_acc_val = 0.0
    loss_ent_acc_val = 0.0

    # Classification for S1
    for batch in s1_loader:
        # Get input image and label batch
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        # Get output
        output = model(image)['out']

        loss_val = loss_cls_func(output, label)

        # Calculate and sum up the loss
        loss_cls_acc_val = loss_val

        print("==== Cls Loss: {} ====".format(loss_val.item()))

        loss_cls_acc_val.backward()

    log_softmax = torch.nn.LogSoftmax(dim=1)
    # Entropy maximization for A1
    for batch in a1_loader:
        # Get input image and label batch
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        # Get output and convert it to log probability
        output = log_softmax(model(image)['out'])

        # Uniform distribution: the probability of each class is 1/class_num
        #  The number of classes is the 1st dim of the output
        uni_dist = torch.ones_like(output).to(device) / output.size()[1]
        loss_val = loss_ent_func(output, uni_dist)

        # Calculate and sum up the loss
        loss_ent_acc_val = weight_loss_ent * loss_val

        print("==== Ent Loss: {} ====".format(loss_val.item()))

        loss_ent_acc_val.backward()
    
    loss_ent_acc_val /= len(a1_loader)

    # Calculate overall loss
#    loss_all_val = loss_cls_acc_val + weight_loss_ent * loss_ent_acc_val

    optimizer.step()


def val(model, val_loader, device='cuda'):
    """Validation

    Args:
        model:
        s1_loader:
        device:
    """
    # Set the model to 'train' mode
    model.eval()

    # Loss function
    loss_cls_func = torch.nn.CrossEntropyLoss(reduction='mean')

    # Classification for S1
    for batch in val_loader:
        # Get input image and label batch
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        # Get output
        output = model(image)['out']

        loss_val = loss_cls_func(output, label)

        # Calculate and sum up the loss
        loss_cls_acc_val = loss_val

        print("==== Cls Loss: {} ====".format(loss_val.item()))

        loss_cls_acc_val.backward()


def main():
    print()
    # Get arguments
    args = parse_arguments()


    #
    # Import datasets (source S1, and the rest A1)
    #
    try:
        dataset_s1, class_num = import_dataset(args.s1_name)
    except Exception as e:
        t, v, tb = sys.exc_info()
        print(traceback.format_exception(t,v,tb))
        print(traceback.format_tb(e.__traceback__))
        print("Dataset '{}' not found".format(ds))
 
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
            dataset_a_tmp, _ = import_dataset(ds)
        except Exception as e:
            t, v, tb = sys.exc_info()
            print(traceback.format_exception(t,v,tb))
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
    model = torch.hub.load('pytorch/vision:v0.7.0', 'deeplabv3_resnet50', pretrained=False)
    # Change the classification layer to match the category 
    model.classifier[4] = torch.nn.Conv2d(256, class_num, 1)
    model.to(args.device)

    #
    # Dataloader
    #
    train_loader_s1 = torch.utils.data.DataLoader(dataset_s1, batch_size=args.batch_size, shuffle=True)
    train_loader_a1 = torch.utils.data.DataLoader(dataset_a1, batch_size=args.batch_size, shuffle=True)

    #
    # Optimizer: Updates 
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    #
    # Scheduler: Gradually changes the learning rate
    #
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200], gamma=0.1)

   #
   # Training
   #
    for ep in range(args.epochs):
        train(model, train_loader_s1, train_loader_a1, optimizer, device=args.device)
        scheduler.step()


if __name__=='__main__':
    main()
