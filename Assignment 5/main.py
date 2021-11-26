import json, argparse, torch, sys
import torch.optim as optim
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.train import *
from src.data_gen import CustomImageDataset
from src.autoencoder import VAE

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def plot_results(test_loss, train_loss):
    y_train = np.linspace(0,len(train_loss),len(train_loss))
    y_test = np.linspace(0,len(test_loss), len(test_loss))
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax2.plot(y_train, train_loss)
    # ax2.set_ylim(0,1)
    ax2.set_title('Training Loss')
    ax1.plot(y_test, test_loss)
    ax1.set_title('Test Loss')
    plt.savefig('results/plot.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML with PyTorch')
    parser.add_argument('--input_csv', help = 'file_path_to_csv')
    parser.add_argument('--param', help='parameter file name')
    args = parser.parse_args()

    csv_path = args.input_csv

    data = pd.read_csv(csv_path, header = None, sep = ' ')/256
    f = open(args.param, "r")
    hparams = json.loads(f.read())

    device = torch.device("cuda:0") #change to gpu if need be
    c = 64

    model = VAE(features=2, c=c).to(torch.device(device))

    learning_rate = hparams['exec']['learning_rate']
    epochs = hparams['exec']['num_epochs']
    batch_size = hparams['exec']['batch_size']

    test_data = data.sample(n=3000, random_state=1)
    train_data = data.drop(test_data.index)

    img = data.iloc[0,:-1].to_numpy()
    # img = data.iloc[0,:].to_numpy()

    # plt.imshow(img.reshape(14,14))
    # plt.show()
    custom_dataset_train = CustomImageDataset(train_data)
    custom_dataset_test = CustomImageDataset(test_data)

    train_loader = DataLoader(custom_dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=True)

    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction='sum')
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_loss_arr = fit(model, train_loader, optimizer, criterion, train_data, device, batch_size)
        val_epoch_loss, val_loss_arr = validate(model, val_loader, optimizer, criterion, test_data, device, batch_size, epoch+1)
        train_loss.extend(train_loss_arr)
        val_loss.extend(val_loss_arr)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}")


    plot_results(val_loss, train_loss)
