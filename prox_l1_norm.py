import numpy as np

def prox_l1_norm(x, lambd):
    return np.sign(x) * np.fmax(np.zeros(x.shape), np.fabs(x) - lambd)