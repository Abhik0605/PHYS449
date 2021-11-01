import json, argparse, sys

import os
import numpy as np
import random

from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader


from src.data_gen import ODE, generate_data, CustomDataset
from util.utils import  plot_results
from src.nn_gen import Net
from src.train import train_loop, test_loop

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML with PyTorch')
    parser.add_argument('--param', help='parameter file name')
    parser.add_argument('-v', help = 'verbosity')
    parser.add_argument('--res-path', metavar='res_path', help='path to save the test plots at')
    parser.add_argument('--x-field', metavar = '--x_field', help='expression of the x-component of the vector field')
    parser.add_argument('--y-field', metavar= '--y_field', help='expression of the y-component of the vector field')
    parser.add_argument('--lb', help='lower bound for initial conditions')
    parser.add_argument('--ub', help='upper bound for initial conditions')
    parser.add_argument('--n-tests',metavar='--n_tests', help='number of test trajectories to plot')
    args = parser.parse_args()

    x_field = args.x_field
    y_field = args.y_field
    u = lambda x, y: eval(x_field)
    v = lambda x, y: eval(y_field)

    ode_obj = ODE(u, v)

    f = ode_obj.func()

    ub = int(float(args.ub))
    lb = int(float(args.lb))
    n = 250

    file = open(args.param, "r")
    hparams = json.loads(file.read())

    learning_rate = hparams['exec']['learning_rate']
    epochs = hparams['exec']['num_epochs']
    batch_size = hparams['exec']['batch_size']

    data = generate_data(f, lb, ub, n)
    device = torch.device("cpu") #change to gpu if need be
    model = Net().to(torch.device(device))

    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

    # print(type(train_dataset[0][0]))
    custom_dataset_train = CustomDataset(train_dataset)
    custom_dataset_test = CustomDataset(test_dataset)

    train_dataloader = DataLoader(custom_dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    big_train_loss = []
    big_test_loss = []
    for t in range(epochs):
      print(f"Epoch {t+1}\n-------------------------------")
      train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
      test_loss = test_loop(test_dataloader, model, loss_fn)
      big_train_loss.extend(train_loss)
      big_test_loss.extend(test_loss)

    model.eval()
    with torch.no_grad():
    	n_path = 30
    	path = np.zeros((int(args.n_tests), n_path , 2))
    	for j in range(int(args.n_tests)):
          start_point = [random.uniform(lb, ub), random.uniform(lb, ub)]
          x = torch.tensor([start_point,])
          y = model(x)

          path[j][0] = start_point
          path[j][1] = y.numpy()
          for i in range(n_path - 2):
            y_new = model(y)
            path[j][i+2] = y_new.numpy()
            y = y_new

    plot_results(path, lb, ub, x_field, y_field, args.res_path)

    print(data)


