from Models import train_model
import Models.Models as models


class VGG16:

    def __init__(self, num_classes, data_loaders):
        self.net = models.vgg16(num_classes)
        self.data_loaders = data_loaders
        self.loss_history = None

    def train(self, criterion, optim, scheduler=None, num_epochs=50):

        self.net, self.loss_history = \
            train_model.train(self.net, criterion, optim, self.data_loaders, scheduler, num_epochs)

    def test(self, inputs):
        return self.net(inputs)

    def to(self, device):
        self.net.to(device)

