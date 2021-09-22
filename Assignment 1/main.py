import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('inp_file_name', type = str)
parser.add_argument('json_file_name', type = str)

args = parser.parse_args()

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


class Regression():
  def __init__(self, data):
    self.data = data

  def analytical_linear_reg(self):
    phi = self.data[:,:2]
    y = self.data[:,2]

    phiTphi = np.matmul(np.transpose(phi), phi)
    phiTphi_inv = np.linalg.inv(phiTphi)
    phiT_y = np.matmul(np.transpose(phi), y)
    w = np.matmul(phiTphi_inv, phiT_y)
    #have output as files
    return w

  def grad_descent_linear_reg(input):
    #have output as files
    return None

data_object = data_extraction(args.inp_file_name, args.json_file_name)
json_data = data_object.extract_json()
# fee = Regression(data).analytical_linear_reg()

print(json_data['learning rate'])