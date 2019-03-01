import matplotlib.pyplot as plt
import numpy as np


def show_image(image):

    img = image / 2 + .5

    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

