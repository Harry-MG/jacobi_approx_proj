import numpy as np
import time
from cyclic_jacobi_sweeps import cyclic_jacobi_sweeps
from cyclic_jacobi_tol import cyclic_jacobi_tol
from prox_l1_norm import prox_l1_norm


def jacobi_fast_prox_nuc_norm(f, grad, max_iter, dimension, step_size, reltol, lambd, tol=None, num_sweeps=None):
    if tol is not None and num_sweeps is not None:
        raise Exception(
            "Set only one of tol and num_sweeps. If using cyclic_jacobi_tol use tol. If using cyclic_jacobi_sweeps "
            "use num_sweeps")

    # Initialise
    n = dimension

    x = np.eye(n)

    U = np.eye(n)

    t = step_size

    sweeps_list = np.zeros(max_iter)

    times_list = np.zeros(max_iter)

    rank_list = np.zeros(max_iter)

    objective_value = np.zeros(max_iter)

    start_main = time.time()

    for k in range(2, max_iter + 2):
        start = time.time()

        x_old = x
        beta = (k - 1) / (k + 2)
        y = x + beta * (x - x_old)
        z = y - t * grad(y)

        if tol is not None:
            D = cyclic_jacobi_tol(np.transpose(U) @ z @ U, tol).diagonal
            J = cyclic_jacobi_tol(np.transpose(U) @ z @ U, tol).eigenvectors
            nsweeps = cyclic_jacobi_tol(np.transpose(U) @ z @ U, tol).sweeps
            sweeps_list[k] = nsweeps

        else:
            D = cyclic_jacobi_sweeps(np.transpose(U) @ z @ U, num_sweeps).diagonal
            J = cyclic_jacobi_sweeps(np.transpose(U) @ z @ U, num_sweeps).eigenvectors

        U = U @ J

        prox = prox_l1_norm(np.abs(np.diag(D)), t*lambd)

        Dplus = np.diag(prox)

        x = U @ Dplus @ np.transpose(U)

        end = time.time()

        times_list[k] = end - start

        rank_list[k] = np.linalg.matrix_rank(x)

        objective_value[k] = f(x)

        if np.abs(f(x) - f(x_old)) < reltol:
            final_iter = k
            break

    end_main = time.time()

    class outputs:
        argmin = x
        minimum = f(x)
        objective_values = objective_value[:final_iter]
        sweeps_per_iter = sweeps_list[:final_iter]
        iter_times = times_list[:final_iter]
        ranks = rank_list[:final_iter]
        total_time = end_main - start_main

    return outputs

