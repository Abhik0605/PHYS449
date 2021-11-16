import numpy as np

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def banded(g, N):
    """Creates a `g` generated banded matrix with 'N' rows"""
    n=len(g)
    T = np.zeros((N,N+n-1))
    for x in range(N):
        T[x][x:x+n]=g
    return T
