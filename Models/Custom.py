import torch.nn as nn
import torch.nn.functional as F
from Models.train_model import train


class CustomNetworkWrapper:

    def __init__(self, input_dim, output_dim, config, data_loaders):
        self.net = CustomNetwork(input_dim, output_dim, config)

        self.train_loss_history = None
        self.test_loss_history = None
        self.data_loaders = data_loaders

    def to(self, device):
        self.net.to(device)

    def train(self, criterion, optim, scheduler=None, num_epochs=50):

        self.net, self.train_loss_history, self.test_loss_history = \
            train(self.net, criterion, optim, self.data_loaders, scheduler, num_epochs)


class CustomNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, config):

        super(CustomNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.non_lin = None

        self.fc_size = input_dim
        last_out = 0
        firstFC = True
        last_fc_size = 0

        with open(config, 'r') as f:

            for i, line in enumerate(f):
                line = line.strip().lower()

                if line == 'conv2d':
                    next_line = next(f).strip()
                    next_line = next_line.split(' ')
                    next_line = list(map(int, next_line))

                    layer = nn.Conv2d(in_channels=next_line[0], out_channels=next_line[1], kernel_size=next_line[2],
                                      stride=next_line[3], padding=next_line[4])

                    self.layers.append(layer)

                    self.fc_size = (self.fc_size - next_line[2] + 2 * next_line[4]) / next_line[3] + 1
                    last_out = next_line[1]

                elif line == 'fc':

                    if firstFC:
                        self.fc_size = int(self.fc_size ** 2) * last_out
                        firstFC = False
                        in_dim = self.fc_size
                    else:
                        in_dim = last_fc_size

                    next_line = next(f).strip()
                    next_line = next_line.split(' ')

                    assert len(next_line) == 1, 'FC only has one number'

                    if next_line[0] == 'OUT':
                        out_dim = output_dim
                    else:
                        out_dim = int(next_line[0])

                    layer = nn.Linear(in_dim, out_dim)
                    self.layers.append(layer)
                    last_fc_size = out_dim

                elif line == 'pool2d':

                    next_line = next(f).strip()
                    next_line = next_line.split(' ')
                    next_line = list(map(int, next_line))

                    layer = nn.MaxPool2d(next_line[0], next_line[1])
                    self.layers.append(layer)

                    self.fc_size = (self.fc_size - next_line[0]) / next_line[1] + 1

                elif line == 'dropout2d':

                    next_line = next(f).strip()
                    next_line = next_line.split(' ')

                    assert len(next_line) == 1, 'Dropout only has 1 parameter'

                    layer = nn.Dropout2d(float(next_line[0]))
                    self.layers.append(layer)

                elif line == 'non_lin':

                    next_line = next(f).lower().strip()
                    if next_line == 'relu':
                        self.non_lin = F.relu
                    elif next_line == 'tanh':
                        self.non_lin = F.tanh
                    elif next_line == 'lrelu':
                        self.non_lin = F.leaky_relu
                else:
                    assert line == '', f'Name {line} not found'

    def forward(self, x):

        firstFC = True
        last_layer_type = None

        for layer in self.layers:

            # this is used to apply non-linearity s.t. it is not applied in last linear layer
            if last_layer_type == nn.Conv2d or last_layer_type == nn.Linear:
                x = self.non_lin(x)

            if firstFC and isinstance(layer, nn.Linear):
                firstFC = False
                x = x.view(-1, self.fc_size)

            last_layer_type = type(layer)

            x = layer(x)

        return x



