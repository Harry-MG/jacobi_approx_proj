import numpy as np


def matrix_completion_objective(x, m, omega):
    n = len(omega)

    s = 0

    for i in omega:
        s += (x[i] - m[i]) ** 2

    return s
