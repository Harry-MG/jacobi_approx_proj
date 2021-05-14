import numpy as np


def symSchur2(A, p, q):
    if A[p, q] != 0:
        tau = (A[q, q] - A[p, p]) / (2 * A[p, q])

        if tau >= 0:
            t = 1 / (tau + np.sqrt(1 + tau ** 2))
        else:
            t = 1 / (tau - np.sqrt(1 + tau ** 2))

        c = 1 / np.sqrt(1 + t ** 2)
        s = t * c

    else:
        c = 1
        s = 0
    return c, s
