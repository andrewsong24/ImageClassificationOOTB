import torch
import torch.nn as nn
import torch.optim as optim
import os

from Models.VGG import VGG16
from Models.Custom import CustomNetworkWrapper
import Utils.Dataset as ds
import Utils.utils as utils

import argparse


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_classes_paths, classes = utils.get_classes_and_paths(args.dataFolder)

    train_indices = []
    test_indices = []

    for i_class in image_classes_paths:
        class_len = len(os.listdir(i_class))

        train, val, test = \
            utils.get_indices(class_len, train_percent=args.train_percent, val_percent=args.val_percent, test_percent=args.test_percent)

        train_indices.append(train)
        test_indices.append(test)

    train_set = ds.DataSetCreator(image_classes_paths, train_indices, input_dim=args.input_dim)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=2)

    test_set = ds.DataSetCreator(image_classes_paths, test_indices, augment=False, input_dim=args.input_dim)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=True, num_workers=2)

    data_loaders = {'train': train_loader, 'test': test_loader}

    if args.custom:

        config = os.path.join(os.path.join(os.getcwd(), 'Models'), 'custom.txt')
        net = CustomNetworkWrapper(args.input_dim, len(classes), config, data_loaders)
        print(net.net)

    else:
        net = VGG16(len(classes), data_loaders, freeze_layers=args.num_layers_frozen)

    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.net.parameters()), lr=args.lr)

    net.train(criterion, optimizer, num_epochs=args.epochs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataFolder', type=str, default='Data', help='Name of root data folder')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=4, help='Batch Size')
    parser.add_argument('--train_percent', type=float, default=0.7, help='Percent of training set')
    parser.add_argument('--val_percent', type=float, default=0.15, help='Percent of validation set')
    parser.add_argument('--test_percent', type=float, default=0.15, help='Percent of test set')
    parser.add_argument('--custom', type=int, default=0, help='Using custom network or pre-trained VGG16')
    parser.add_argument('--num_layers_frozen', type=int, default=40, help='Number of layers to freeze for pre-trained VGG16')
    parser.add_argument('--input_dim', type=int, default=224, help='Input dimension for custom nets (224 for pre-trained)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam')

    args = parser.parse_args()

    main(args)



