import numpy as np
import os
from scipy.integrate import solve_ivp


class ODE():
	def __init__(self, u, v):
		self.u = u
		self.v = v

	def func(self):
		f = lambda t, y: (self.u(y[0], y[1]), self.v(y[0], y[1]))
		return f



def generate_mesh(f, lb, ub, n):
	nx, ny = (n, n) #n is the number of data points
	x = np.linspace(lb, ub, nx)
	y = np.linspace(lb, ub, ny)
	xv, yv = np.meshgrid(x, y)
	test = np.zeros((n*n, 4))
	epsilon = 0.1
	z = 0
	for i in range(nx):
	    for j in range(ny):
	        sol = solve_ivp(f, [0, epsilon], [xv[i, j], yv[i, j]])
	        test[z] = xv[i, j], yv[i, j], sol.y[0][1], sol.y[1][1]
	        z += 1

	return test


