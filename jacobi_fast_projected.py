import numpy as np
import time
from cyclic_jacobi_sweeps import cyclic_jacobi_sweeps
from cyclic_jacobi_tol import cyclic_jacobi_tol


def jacobi_fast_projected(f, grad, max_iter, dimension, step_size, reltol, tol=None, num_sweeps=None):
    if tol is not None and num_sweeps is not None:
        raise Exception(
            "Set only one of tol and num_sweeps. If using cyclic_jacobi_tol use tol. If using cyclic_jacobi_sweeps "
            "use num_sweeps")

    # Initialise
    n = dimension

    x = np.eye(n)

    U = np.eye(n)

    t = step_size

    sweeps_list = np.empty(max_iter)

    times_list = np.empty(max_iter)

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

        Dplus = np.diag(np.maximum(np.diag(D), 0))

        x = U @ Dplus @ np.transpose(U)

        end = time.time()

        times_list[k] = end - start

        if np.abs(f(x) - f(x_old)) < reltol:
            break

    end_main = time.time()

    class outputs:
        argmin = x
        minimum = f(x)
        sweeps_per_iter = sweeps_list
        iter_times = times_list
        total_time = end_main - start_main

    return outputs
