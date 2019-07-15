import torchvision.models as models
import torch.nn as nn
import os


def vgg16(num_classes, freeze_layers=40):
    model = models.vgg16_bn(pretrained=True)
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, num_classes)])
    model.classifier = nn.Sequential(*features)
    for i, param in enumerate(model.parameters()):
        if i < freeze_layers:
            param.requires_grad = False
    return model


