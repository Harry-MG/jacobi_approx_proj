import numpy as np

from cyclic_jacobi_tol import cyclic_jacobi_tol


def jacobi_proj_tol(A, tol):
    D = cyclic_jacobi_tol(A, tol).diagonal
    P = cyclic_jacobi_tol(A, tol).eigenvectors
    Dplus = np.diag(np.maximum(np.diag(D), 0))

    return P @ Dplus @ np.transpose(P)
