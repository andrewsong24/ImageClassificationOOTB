import os
import network

current_dr = os.path.join(os.getcwd(), 'Data')

directories = os.listdir(current_dr)

image_classes = []
classes = []

for direct in directories:
    if '.' not in direct:
        image_classes.append(os.path.join(current_dr, direct))
        classes.append(direct)


def get_classes():
    return classes


def get_class_paths():
    return image_classes



