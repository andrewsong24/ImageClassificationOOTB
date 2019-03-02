import torch
import torchvision
import os
import numpy as np

import Models.Models
import Utils.Dataset as ds
import Utils.utils as utils

image_classes_paths, classes = utils.get_classes_and_paths()
print(f'Image Classes Paths: {image_classes_paths}, Classes: {classes}')

net = Models.Models.vgg16(len(classes))

dataset = ds.DataSetCreator(image_classes_paths)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
dataiter = iter(data_loader)


# TODO: Get indicies for train, val, and test


