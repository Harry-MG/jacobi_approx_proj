import numpy as np

from cyclic_jacobi_sweeps import cyclic_jacobi_sweeps


def jacobi_proj_numits(A, numits):
    D = cyclic_jacobi_sweeps(A, numits).diagonal
    P = cyclic_jacobi_sweeps(A, numits).eigenvectors
    Dplus = np.diag(np.maximum(np.diag(D), 0))

    return P @ Dplus @ np.transpose(P)
