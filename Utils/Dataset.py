import torchvision.transforms as transforms
from torch.utils import data
import os
import numpy as np
from PIL import Image

transform = transforms.Compose(

    [
        transforms.Resize(250),
        transforms.RandomCrop(224, pad_if_needed=True),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

)

non_augment_transform = transforms.Compose(

    [
        transforms.Resize(312),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ]

)


class DataSetCreator(data.Dataset):

    def __init__(self, image_paths, augment=True):

        self.image_paths = image_paths
        self.npLabels = np.array([1])
        self.files = []
        labels = []

        for i, image_class in enumerate(image_paths):

            labels.append(np.full((len(os.listdir(image_class))), i))

            for image in reversed(os.listdir(image_class)):
                self.files.append(os.path.join(image_class, image))

        for label in labels:
            self.npLabels = np.concatenate((self.npLabels, np.array(label)), axis=None)

        self.npLabels = np.delete(self.npLabels, [0])

        print(self.files)
        print(self.npLabels)

        if augment:
            self.tsfm = transform
        else:
            self.tsfm = non_augment_transform

    def __getitem__(self, item):

        y = self.npLabels[item]

        x = Image.open(self.files[item]).convert('RGB')
        x = self.tsfm(x)

        return x, y.astype(int)

    def __len__(self):
        return len(self.files)

