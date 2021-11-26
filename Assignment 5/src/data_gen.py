import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.images =data.iloc[:,:-1]
        self.X =  torch.from_numpy(np.array(self.images)).float()
        self.labels = data.iloc[:,-1:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = int((np.array(self.labels)[idx][0])/2)
        image = torch.from_numpy(np.expand_dims(np.array(self.images.iloc[idx]).reshape(14,14), 0)).float()

        return image, label


