import torch
from torch import nn

from applications.connect_four.connect_four import ConnectFourState
from deep_search.nn.deep_heuristic import GameNetwork


class BasicCNN(GameNetwork):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # TODO: parameterize
        self.cnn_layers = nn.Sequential(
            # 1st conv
            nn.Conv2d(2, 32, ConnectFourState.connect_num, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # 2nd conv
            nn.Conv2d(32, 16, ConnectFourState.connect_num // 2, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        )
        conv_out_size = self._get_conv_output_shape()
        self.dense_layers = nn.Sequential(
            nn.Linear(conv_out_size, conv_out_size // 2),
            nn.LeakyReLU(),
            nn.Linear(conv_out_size // 2, 1)
        )

    def _get_conv_output_shape(self):  # TODO: test
        # easiest way to get conv layer's output to be always correct is by trying a forward pass of a random input
        image_dim = (1, 2, ConnectFourState.nrows, ConnectFourState.ncols)
        conv_output = self.cnn_layers(torch.rand(*(image_dim))).view(1, -1).shape[1]
        return int(conv_output)

    def forward(self, state_rep: torch.Tensor, with_sigmoid: bool = True):
        x = self.cnn_layers(state_rep)
        x = x.view(x.size(0), -1)
        x = self.dense_layers(x)
        # TODO: sigmoid or tanh? --> teachers use [-1, 1], not [0, 1], should also use another loss
        return torch.sigmoid(x) if with_sigmoid else x

    def get_model_parameters(self):
        # TODO: parameterize
        return {}
