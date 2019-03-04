import torch
import torch.nn as nn
from Models.VGG import VGG16
import Utils.Dataset as ds
import Utils.utils as utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_classes_paths, classes = utils.get_classes_and_paths()
print(f'Image Classes Paths: {image_classes_paths}, Classes: {classes}')

# TODO: Get indicies for train and test

dataset = ds.DataSetCreator(image_classes_paths)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
data_loaders = {'train': train_loader}

net = VGG16(len(classes), data_loaders)

if torch.cuda.is_available():
    net = torch.nn.DataParallel(net.net)
    criterion = nn.CrossEntropyLoss().cuda()
    dtype = torch.cuda.FloatTensor

else:
    criterion = nn.CrossEntropyLoss()
    dtype = torch.FloatTensor




