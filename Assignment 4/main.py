from src.data_gen import extract_data
from util.util import sigmoid, banded
import numpy as np
from scipy.sparse import diags
from scipy.stats import entropy
import argparse


def sample(x, lambda_mat):
  x = x.reshape(-1,1)
  x_tilde = sigmoid(lambda_mat@x)
  # gibbs sampling
  tilde_energy = x_tilde * lambda_mat@x_tilde
  x_energy = x * lambda_mat@x
  x_tilde = np.where(tilde_energy<x_energy, x_tilde, x_sample2)




  for j in range(k):
    # x_energy = x * lambda_mat@x    #x.T@lambda_mat@x
    tilde_energy = x_tilde * lambda_mat@x_tilde

    x_sample1 = np.where(np.random.uniform(size=x.shape)<sigmoid(lambda_mat@x_tilde),1,0)
    x_sample2 = np.where(np.random.uniform(size=x.shape)<sigmoid(lambda_mat@x_tilde),1,0)

    sample_energy = x_sample1 * lambda_mat@x_sample1



  return x_tilde


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='ML with PyTorch')
  parser.add_argument('--input_file', help='input file name')
  args = parser.parse_args()

  with open(args.input_file) as f:
      lines = f.readlines()
  data = extract_data(lines)
  input_size = len(data[0])
  #initialize the weight matrix

  foo = np.random.uniform(low=-1,high=1,size=4)
  sp = diags([[0]*4,foo,foo],[0,1,-1])

  # for printing
  lambda_mat = sp.toarray()

  lambda_mat[0][-1] = foo[-1]
  lambda_mat[-1][0] = foo[-1]
  print(lambda_mat)

  # iteration for gibbs sampling
  k = 5
  # learning rate
  lr = 0.000001
  # epochs
  epochs = 50

  last_sample = data[-1]
  # training loop
  for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    for x in data:
      x = x.reshape(-1,1)
      x_tilde = sigmoid(lambda_mat@x)
      # gibbs sampling
      for j in range(k):
        x_tilde = np.where(np.random.uniform(size=x.shape)<sigmoid(lambda_mat@x_tilde),1,0)
      # # print(x_tilde, x)
      # x_tilde = sample(x,lambda_mat)

      lambda_grad = (x@x.T - x_tilde@x_tilde.T)
      # print(np.sum(np.abs(lambda_grad)))
      lambda_mat = lambda_mat + lr*lambda_grad

  output = {}
  indices = []
  for i in range(4):
    indices.append(i)
  temp = []
  for j in range(len(indices)):
    temp.append((indices[j], indices[j-1]))

  for k in range(len(temp)):
    output[f'{temp[k]}'] = f'{lambda_mat[temp[k]]}'

  print(temp)


