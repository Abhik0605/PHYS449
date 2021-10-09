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
        self.fc1 = nn.Linear(196, 96)
        self.fc2 = nn.Linear(96, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    # Reset function for the training weights
    # Use if the same network is trained multiple times.



