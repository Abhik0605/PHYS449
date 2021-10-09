import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    '''
    Neural network class.
    Architecture:
        Three fully-connected layers fc1 and fc2.
        non linear activation function is relu.
        The last output is 5 dim vector for the 5 classes
    '''
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(196, 98)
        self.fc2 = nn.Linear(98, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()




