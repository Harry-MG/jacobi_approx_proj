import numpy as np

from symSchur2 import symSchur2


def cyclic_jacobi_tol(A, tol):
    n = np.shape(A)[0]
    V = np.eye(n)
    delta = tol * np.linalg.norm(A, 'fro')

    nsweeps = 0

    while np.linalg.norm(A - np.diag(np.diag(A)), 'fro') > delta:
        nsweeps = nsweeps + 1
        for p in range(n - 1):
            for q in range(p + 1, n):
                c, s = symSchur2(A, p, q)
                J = np.eye(n)
                J[p, p] = c
                J[p, q] = s
                J[q, p] = -s
                J[q, q] = c
                A = np.matmul(np.matmul(np.transpose(J), A), J)
                V = V * J

    class outputs():
        diagonal = A
        eigenvectors = V
        sweeps = nsweeps

    return outputs
