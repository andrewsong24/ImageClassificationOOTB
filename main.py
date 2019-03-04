import torch
import torch.nn as nn
import Models.Models
import Utils.Dataset as ds
import Utils.utils as utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_classes_paths, classes = utils.get_classes_and_paths()
print(f'Image Classes Paths: {image_classes_paths}, Classes: {classes}')

net = Models.Models.vgg16(len(classes))

dataset = ds.DataSetCreator(image_classes_paths)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
dataiter = iter(data_loader)

net.to(device)

if torch.cuda.is_available():
    net = torch.nn.DataParallel(net)
    criterion = nn.CrossEntropyLoss().cuda()
    dtype = torch.cuda.FloatTensor

else:
    criterion = nn.CrossEntropyLoss()
    dtype = torch.FloatTensor


# TODO: Get indicies for train, val, and test


