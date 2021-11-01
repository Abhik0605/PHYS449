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
        The last output is 2 dim vector
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 196)
        self.fc2 = nn.Linear(196, 196)
        self.fc3 = nn.Linear(196, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x