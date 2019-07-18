from Models import train_model
from Models.Models import vgg16


class VGG16:

    def __init__(self, num_classes, data_loaders, freeze_layers=40):
        if freeze_layers <= 0:
            freeze_layers = 0
        self.net = vgg16(num_classes, freeze_layers)
        self.data_loaders = data_loaders
        self.train_loss_history = None
        self.test_loss_history = None

    def train(self, criterion, optim, scheduler=None, num_epochs=50):

        self.net, self.train_loss_history, self.test_loss_history = \
            train_model.train(self.net, criterion, optim, self.data_loaders, scheduler, num_epochs)

    def single_test(self, inputs):
        return self.net(inputs)

    def to(self, device):
        self.net.to(device)
