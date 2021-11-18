from src.data_gen import extract_data
from util.util import sigmoid, banded
import numpy as np
from scipy.sparse import diags
from scipy.stats import entropy
import argparse
import matplotlib.pyplot as plt


def KL_div(N, x, y, lambdas):
  return np.sum((-1/N)*(np.log(np.sum(energy(x, y, lambdas))) + np.log(np.exp(energy(x, y, lambdas)))))


def energy(x, y, lambdas):
  return lambdas*x.dot(np.transpose(y))


def energy_change(sigma,spin_index,J,h):
    """
    Computes the energy change if spin at
    position spin_index is flipped.
    Parameters:
    sigma: configuration
    spin_index: spin to be flipped
    J, h : additional parameters
    Returns:
    ec: energy change
    """
    N_spins = len(sigma)
    ec = 2*J/N_spins*sum(sigma[spin_index]
                         *np.delete(sigma, spin_index))
    +2*h*sigma[spin_index]
    return(ec)


def metropolis(N,beta,J,h,iterations):
    """
    Metroplis-Hastings algorithm for Curie-Weiss model
    Parameters:
    N: number of spins
    iterations: number of steps that the Markov chain takes
    beta, J, h : additional parameters
    Returns:
    total_mag: empirical magnetization at each step i,
    vector of length iterations
    sigma: configuration at final step
    """

    # random initial spin assignment
    sigma = np.random.choice([-1,1],N)

    # preallocate variable
    emp_mag = np.zeros(iterations)

    for i in range(iterations):
        # choose a random spin
        spin_index = np.random.randint(N)

        # draw a random uniform number between 0 and 1
        x = np.random.random()
        ec = energy_change(sigma,spin_index,J,h)

        if x < np.exp(-beta*ec):
            sigma[spin_index] = sigma[spin_index]*-1

        emp_mag[i] = sum(sigma)/len(sigma)

    return sigma


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='ML with PyTorch')
  parser.add_argument('--input_file', help='input file name')
  args = parser.parse_args()

  with open(args.input_file) as f:
      lines = f.readlines()
  data = extract_data(lines)
  input_size = len(data[0])

  pos_phase = np.zeros(input_size)


  # iteration for gibbs sampling
  k = 100
  # learning rate
  lr = 0.001
  # epochs
  epochs = 50

  last_sample = data[-1]
  # training loop
  pos_phase = np.zeros(input_size)
  lambdas = np.zeros((epochs, input_size))
  loss_arr = []
  for l in range(1, epochs):
    print(f"Epoch {l}\n-------------------------------")
    neg_phase = np.zeros(input_size)

    temp_data = np.zeros((k, input_size))

    for j in range(k):
      temp_data[j,:] = metropolis(input_size,1,1,1,10)

    for i in range(input_size):
      pos_phase[i] = data[:,i].dot(np.transpose(data[:,i-1]))
      neg_phase[i] = temp_data[:,i].dot(np.transpose(temp_data[:,i-1]))

    #print('neg_phase', pos_phase)
    lambdas[l,:] = lambdas[l-1,:] + lr*(1/len(data))*(pos_phase - neg_phase)
    for i in range(input_size):
      loss = KL_div(len(temp_data), temp_data[:,i], temp_data[:,i-1], lambdas[l,:])
    loss_arr.append(loss)
    print(loss)
  output = {}
  indices = []
  for i in range(4):
    indices.append(i)
  temp = []
  for j in range(len(indices)):
    temp.append((indices[j-1], indices[j]))

  for k in range(len(temp)):
    output[f'{temp[k]}'] = f'{2*lambdas[-1,k]}'

  plt.plot(loss_arr)
  plt.savefig('plot.png')
  plt.show()
  print(output)


