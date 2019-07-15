import torch.nn as nn


class CustomNetworkWrapper:

    def __init__(self, input_dim, output_dim, config):
        self.net = CustomNetwork(input_dim, output_dim, config)

    def to(self, device):
        self.net.to(device)

    def train(self, criterion, optimizer, num_epochs):
        pass


class CustomNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, config):

        super(CustomNetwork, self).__init__()

        self.layers = nn.ModuleList()

        self.fc_size = input_dim
        last_out = 0
        firstFC = True

        with open(config, 'r') as f:

            for i, line in enumerate(f):
                line = line.strip()

                if line == 'conv2d':
                    next_line = next(f)
                    next_line = next_line[:-1].split(' ')
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

                    next_line = next(f)
                    next_line = next_line[:-1].split(' ')

                    if next_line[0] == 'IN':
                        next_line[0] = self.fc_size
                    if next_line[1] == 'OUT':
                        next_line[1] = output_dim

                    next_line = list(map(int, next_line))

                    layer = nn.Linear(next_line[0], next_line[1])
                    self.layers.append(layer)

                elif line == 'pool':

                    # TODO: Implement pool

                    pass

                elif line == 'dropout':
                    pass
                    # TODO: Implement dropout

                else:
                    assert False, f'Name {line} not found'

    def forward(self, x):

        firstFC = True
        for layer in self.layers:
            if firstFC and isinstance(layer, type(nn.Linear(1, 1))):
                firstFC = False
                x = x.view(-1, self.fc_size)
            x = layer(x)



