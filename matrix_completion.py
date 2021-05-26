import numpy as np
import random
import matplotlib.pyplot as plt

# generate data
from jacobi_fast_projected import jacobi_fast_projected
from jacobi_fast_prox_nuc_norm import jacobi_fast_prox_nuc_norm
from matrix_completion_gradient import matrix_completion_gradient
from matrix_completion_objective import matrix_completion_objective

# generate a symmetric sparse low rank matrix

n = 60
m = np.random.rand(n, n)

m = 0.5 * (m + np.transpose(m))

p = 0.2

u, s, v = np.linalg.svd(m)

nonzeros = [i for i in range(n) if s[i] != 0]

rands = random.sample(nonzeros, int(p*n))

for i in range(n):
    if i in rands:
        s[i] = s[i]
    else:
        s[i] = 0

d = np.diag(s)

m = u @ d @ v

print("The rank of the data matrix is", np.linalg.matrix_rank(m))

# generate a subset of indices

indices = [(i, j) for i in range(n) for j in range(n) if i <= j]

inds = random.sample(indices, int(0.05 * n ** 2))

inds2 = [(i, j) for (j, i) in inds if j < i]

omega = list(set(inds + inds2))


# set up problem for use with solver

def objective(x):
    return matrix_completion_objective(x, m, omega)


def gradient(x):
    return matrix_completion_gradient(x, m, omega)


max_iter = 1000

dimension = n

step_size = 0.5

reltol = 1e-3

lambd = 0.5

# run the solver

report = jacobi_fast_prox_nuc_norm(objective, gradient, max_iter, dimension, step_size, reltol, lambd, tol=1e-3)

print("the total time taken was", report.total_time, "seconds")

print("the minimum objective value is", report.minimum)

# reporting - plots

iter_times = report.iter_times

sweep_list = report.sweeps_per_iter

ranks = report.ranks

obj_vals = report.objective_values

plt.plot(iter_times)
plt.xlabel("k")
plt.ylabel("time taken for iteration k")
plt.show()

plt.plot(sweep_list)
plt.xlabel("k")
plt.ylabel("number of sweeps at iteration k")
plt.show()

plt.plot(ranks)
plt.hlines(np.linalg.matrix_rank(m), 0, len(ranks), linestyles='dashed')
plt.xlabel("k")
plt.ylabel("rank of kth iterate")
plt.show()

plt.plot(obj_vals)
plt.xlabel("k")
plt.ylabel("objective function value")
plt.show()