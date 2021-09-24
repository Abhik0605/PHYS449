import numpy as np
import argparse
import json

# class to extract data
class data_extraction():
  def __init__(self, input_file, json_file):
    self.input_file = input_file
    self.json_file = json_file

  def extract_input_file(self):
    f = np.loadtxt(self.input_file)
    return f

  def extract_json(self):
    f = open(self.json_file, "r")
    data = json.loads(f.read())
    return data

# class for regression
class Regression():
  def __init__(self, data):
    self.data = data

  def analytical_linear_reg(self):
    phi = self.data[:,:-1]
    y = self.data[:,-1]

    phiTphi = np.matmul(np.transpose(phi), phi)
    phiTphi_inv = np.linalg.inv(phiTphi)
    phiT_y = np.matmul(np.transpose(phi), y)
    w = np.matmul(phiTphi_inv, phiT_y)
    #output are the wieghts of the model
    return np.around(w, decimals=4)

  def grad_descent_linear_reg(self, json_data):
    learning_rate = json_data['learning rate']
    num_iter = json_data['num iter']
    phi = self.data[:,:-1]
    y = self.data[:,-1]

    w = np.ones(len(phi[0]))

    for _ in range(num_iter):
      for i in range(len(phi)):
        w_new = w + (learning_rate * ((y[i] - np.dot(w, phi[i,:])))) * phi[i,:]
        w = w_new
    # output are the weights of the model
    return np.around(w, decimals = 4)

# function for writing the output file
def write_to_file(args, analytical_sol, grad_descent):
  x = [i for i in args.inp_file_name if i.isdigit()]
  f = open(f'data/{x[0]}.out', 'w')
  for i in range(len(analytical_sol)):
    f.write(f'{str(analytical_sol[i])}\n')
  f.write('\n')
  for i in range(len(grad_descent)):
    f.write(f'{str(grad_descent[i])}\n')
  f.close()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('inp_file_name', type = str)
  parser.add_argument('json_file_name', type = str)

  args = parser.parse_args()

  data_object = data_extraction(args.inp_file_name, args.json_file_name)
  json_data = data_object.extract_json()
  input_data = data_object.extract_input_file()

  regression_obj = Regression(input_data)
  analytical_sol = regression_obj.analytical_linear_reg()
  grad_descent = regression_obj.grad_descent_linear_reg(json_data)

  write_to_file(args, analytical_sol, grad_descent)
