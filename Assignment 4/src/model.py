import numpy as np


# class Boltzman():
#   def __init__(self, lambda_matrix, sigmoid):
#     self.lambda_matrix = lambda_matrix
#     self.sigmoid = sigmoid

#   def grad(self, x, k):
#     x_tilde = self.sigmoid(self.lambda_matrix@x)
#     lambda_matrix = self.lambda_matrix
#     for i in range(k):
#       x_tilde =  self.sigmoid(self.lambda_matrix@x_tilde)
#     lambda_matrix_grad = x@x.T - x_tilde@x_tilde.T

#     return lambda_matrix_grad

#   def update(self, lambda_matrix, lambda_matrix_grad, lr):
#     self.lambda_matrix = self.lambda_matrix + lr*lambda_matrix_grad
#     return lambda_matrix




