import numpy as np


def matrix_completion_gradient(x, m, omega):
    shape = np.shape(x)
    grad = np.zeros(shape)
    for ind in omega:
        grad[ind] = 2 * (x[ind] - m[ind])
    return grad
