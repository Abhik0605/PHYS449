import json, argparse, sys

import os
import numpy as np
from scipy.integrate import solve_ivp

# import torch
# import torch.nn as nn
# import torch.nn.functional as f
# import torch.optim as optim

from src.data_gen import ODE, generate_mesh
from util.utils import eval_expr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML with PyTorch')
    #parser.add_argument('--param', help='parameter file name')
    #parser.add_argument('-v', help = 'file_path_to_csv')
    #parser.add_argument('--res-path', help='path to save the test plots at')
    parser.add_argument('--x-field', metavar = '--x_field', help='expression of the x-component of the vector field')
    parser.add_argument('--y-field', metavar= '--y_field', help='expression of the y-component of the vector field')
    parser.add_argument('--lb', help='lower bound for initial conditions')
    parser.add_argument('--ub', help='upper bound for initial conditions')
    # parser.add_argument('--n-tests', help='number of test trajectories to plot')
    args = parser.parse_args()

    u = lambda x, y: -y/np.sqrt(x**2 + y**2)
    v = lambda x, y: x/np.sqrt(x**2 + y**2)

    ode_obj = ODE(u, v)

    f = ode_obj.func()

    ub = int(float(args.ub))
    lb = int(float(args.lb))
    n = 10

    foo = generate_mesh(f, lb, ub, n)

    print(foo)
