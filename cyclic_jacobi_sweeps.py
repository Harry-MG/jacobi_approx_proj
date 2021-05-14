import numpy as np

from symSchur2 import symSchur2


def cyclic_jacobi_sweeps(A, sweeps):
    n = np.shape(A)[0]
    V = np.eye(n)

    for i in range(sweeps):
        for p in range(n - 1):
            for q in range(p + 1, n):
                c, s = symSchur2(A, p, q)
                J = np.eye(n)
                J[p, p] = c
                J[p, q] = s
                J[q, p] = -s
                J[q, q] = c
                A = np.transpose(J) @ A @ J
                V = V * J

    class outputs:
        def __init__(self):
            pass

        diagonal = A
        eigenvectors = V

    return outputs
