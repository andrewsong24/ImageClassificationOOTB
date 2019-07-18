import torchvision.transforms as transforms
from torch.utils import data
import os
import numpy as np
from PIL import Image


def get_transform(input_dim, augment):

    if augment:
        transform = transforms.Compose(

            [
                transforms.Resize(input_dim),
                transforms.RandomCrop(input_dim, pad_if_needed=True),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]

        )
        return transform
    else:
        non_augment_transform = transforms.Compose(

            [
                transforms.Resize(input_dim),
                transforms.CenterCrop(input_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ]

        )
        return non_augment_transform


class DataSetCreator(data.Dataset):

    def __init__(self, image_paths, indices, input_dim, augment=True):

        self.image_paths = image_paths
        self.npLabels = np.array([1])
        self.files = []
        labels = []

        for i, image_class in enumerate(image_paths):

            class_indices = indices[i]
            images = []
            all_images = os.listdir(image_class)

            for index in class_indices:
                images.append(all_images[index])

            labels.append(np.full((len(images)), i))

            for image in reversed(images):
                self.files.append(os.path.join(image_class, image))

        for label in labels:
            self.npLabels = np.concatenate((self.npLabels, np.array(label)), axis=None)

        self.npLabels = np.delete(self.npLabels, [0])

        self.tsfm = get_transform(input_dim, augment)

    def __getitem__(self, item):

        y = self.npLabels[item]

        x = Image.open(self.files[item]).convert('RGB')
        x = self.tsfm(x)

        return x, y.astype(int)

    def __len__(self):
        return len(self.files)

