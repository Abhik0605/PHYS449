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

from torchvision.utils import save_image



def plot_results(train_loss, output_path):
    y_train = np.linspace(0,len(train_loss),len(train_loss))

    plt.title('Training Loss')
    plt.plot(y_train, train_loss)
    plt.xlabel('Num Batches')
    plt.ylabel('Loss')
    plt.savefig(f'{out_path}/loss.pdf')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML with PyTorch')
    parser.add_argument('--input_csv', help = 'file_path_to_csv', default=f'inputs/even_mnist.csv')
    parser.add_argument('--param', help='parameter file name', default='inputs/hyper_param.json')
    parser.add_argument('-o', metavar='out_dir', help = 'filepath to results', default='outputs')
    parser.add_argument('-n',metavar='num_samples', help = 'number of samples', type=int, default=100)
    parser.add_argument('--animate',metavar='animate', help = 'Boolean to show animation of result', type=bool, default=False)


    args = parser.parse_args()

    csv_path = args.input_csv

    out_path = args.o

    n = args.n

    data = pd.read_csv(csv_path, header = None, sep = ' ')/256
    f = open(args.param, "r")
    hparams = json.loads(f.read())

    device = torch.device("cuda:0") #change to gpu if need be
    c = hparams['exec']['capacity']

    #latent_dims
    features = 14

    model = VAE(features=features, c=c).to(torch.device(device))

    learning_rate = hparams['exec']['learning_rate']
    epochs = hparams['exec']['num_epochs']
    batch_size = hparams['exec']['batch_size']

    test_data = data.sample(n=3000, random_state=1)
    train_data = data.drop(test_data.index)

    custom_dataset_train = CustomImageDataset(train_data)
    custom_dataset_test = CustomImageDataset(test_data)
    foo = custom_dataset_train[0]
    # plt.imshow(foo[0][0].numpy())
    # plt.show()
    # print()
    train_loader = DataLoader(custom_dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=True)
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss(reduction='sum')
    train_loss = []
    val_loss = []
    final_ims = []
    fig, ax = plt.subplots()

        # print(x.shape)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_loss_arr, ims = fit(model, train_loader, optimizer, criterion, train_data, device, batch_size, fig, ax, foo[0], args.animate)
        val_epoch_loss, val_loss_arr = validate(model, val_loader, optimizer, criterion, test_data, device, batch_size, epoch+1)
        train_loss.extend(train_loss_arr)
        val_loss.extend(val_loss_arr)
        final_ims.extend(ims)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}")
    # val_loss.insert(0, train_loss[0])
    if args.animate:
        ani = animation.ArtistAnimation(fig, final_ims, interval=1, blit=True, repeat_delay=1000)
        # ani.save("VAE.mp4")
        plt.show()

    with torch.no_grad():
        for i in range(n):
            sample = torch.randn(1, features)
            sample = sample.to(device)
            predict = model.decoder(sample)
            # plt.imshow(predict.cpu().numpy()[0][0])
            save_image(predict.cpu()[0][0], f"{args.o}/{i}.pdf")
            # plt.show()

    # print(val_loss)

    plot_results(train_loss, args.o)
