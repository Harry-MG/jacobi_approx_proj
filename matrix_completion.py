import numpy as np
import random

# generate data
from jacobi_fast_projected import jacobi_fast_projected
from matrix_completion_gradient import matrix_completion_gradient
from mat_com import mat_com

n = 100

c = 1

m = np.random.rand(n, n)

m = 0.5 * (m + np.transpose(m))

indices = [(i, j) for i in range(n) for j in range(n) if i <= j]

inds = random.sample(indices, c * n)

inds2 = [(i, j) for (j, i) in inds if j < i]

omega = list(set(inds + inds2))


# set up problem for use with solver

def objective(x):
    return mat_com(x, m, omega)


def gradient(x):
    return matrix_completion_gradient(x, m, omega)


max_iter = 1000

dimension = n

step_size = 0.5

reltol = 1e-3

# run the solver

report = jacobi_fast_projected(objective, gradient, max_iter, dimension, step_size, reltol, tol=1e-3)


