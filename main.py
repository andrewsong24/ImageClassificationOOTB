import torch
import torch.nn as nn
import torch.optim as optim
import os

from Models.VGG import VGG16
import Utils.Dataset as ds
import Utils.utils as utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_classes_paths, classes = utils.get_classes_and_paths()

train_indices = []
test_indices = []

for i_class in image_classes_paths:
    class_len = len(os.listdir(i_class))

    train, val, test = \
        utils.get_indices(class_len, train_percent=0.8, val_percent=0.0, test_percent=0.2)

    train_indices.append(train)
    test_indices.append(test)

train_set = ds.DataSetCreator(image_classes_paths, train_indices)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

test_set = ds.DataSetCreator(image_classes_paths, test_indices, augment=False)
test_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

data_loaders = {'train': train_loader, 'test': test_loader}

net = VGG16(len(classes), data_loaders)

net.to(device)

if torch.cuda.is_available():
    net = torch.nn.DataParallel(net.net)
    criterion = nn.CrossEntropyLoss().cuda()
    dtype = torch.cuda.FloatTensor

else:
    criterion = nn.CrossEntropyLoss()
    dtype = torch.FloatTensor

optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.net.parameters()))

net.train(criterion, optimizer, num_epochs=10)

