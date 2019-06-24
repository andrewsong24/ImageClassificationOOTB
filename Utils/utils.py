import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def get_indices(n, train_percent=0.8, val_percent=0.1, test_percent=0.1):

    assert train_percent <= 1 and val_percent <= 1 and test_percent <= 1
    assert train_percent + val_percent + test_percent == 1
    assert train_percent >= 0 and val_percent >= 0 and test_percent >= 0

    non_test_indices = np.random.choice(n, int(n * (1-test_percent)), replace=False)

    test_indices = np.setdiff1d(np.arange(n), non_test_indices, assume_unique=False)

    len_non_train_indices = len(non_test_indices)

    val_indices = np.random.choice(non_test_indices,
                                   int((len_non_train_indices - 1) * val_percent/(1-test_percent)),
                                   replace=False)

    train_indices = np.setdiff1d(non_test_indices, val_indices)

    return train_indices, val_indices, test_indices


def get_classes_and_paths(data_folder_path):
    current_dr = os.path.join(os.path.join(os.getcwd(), data_folder_path))
    directories = os.listdir(current_dr)

    image_classes = []
    classes = []

    for directory in directories:
        if '.' not in directory:
            image_classes.append(os.path.join(current_dr, directory))
            classes.append(directory)

    return image_classes, classes


def show_image(image):

    img = image / 2 + .5

    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def update_progress(i, total):

    progress = i / total

    bar_length = 30

    block = int(round(bar_length*progress))

    text = f'\rPercent: [{"█"*block + " "*(bar_length-block)} {round(progress*100, 2)}% {i}/{total}] '
    sys.stdout.write(text)
    sys.stdout.flush()

