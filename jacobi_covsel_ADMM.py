import numpy as np
import time

from cyclic_jacobi_sweeps import cyclic_jacobi_sweeps
from cyclic_jacobi_tol import cyclic_jacobi_tol
from shrinkage import shrinkage


def jacobi_covsel_ADMM(D, mu, alpha, beta, tau, tol=None, nsweeps=None):
    t_start = time.time()

    MAX_ITER = 1000
    ABSTOL = 1e-4
    RELTOL = 1e-2

    C = np.cov(D)
    n = np.shape(C)[0]

    R = np.zeros((n, n))
    S = np.zeros((n, n))
    L = np.zeros((n, n))
    lambd = np.zeros((n, n))

    if tol is not None and nsweeps is not None:
        raise Exception(
            'Set only one of tol and nsweeps. If using cyclic_jacobi_tol use tol. If using cyclic_jacobi_nsweeps use '
            'nsweeps')

    class history:

        objval = np.empty(MAX_ITER)
        r_norm = np.empty(MAX_ITER)
        s_norm = np.empty(MAX_ITER)
        eps_pri = np.empty(MAX_ITER)
        eps_dual = np.empty(MAX_ITER)

        R = np.zeros((n, n))
        S = np.zeros((n, n))
        L = np.zeros((n, n))
        lambd = np.zeros((n, n))

        for k in range(MAX_ITER):

            # R update
            if tol is not None:
                Q, q, throwaway = cyclic_jacobi_tol(mu * (C - lambd) - S + L, tol)
                es = np.diag(q)
                xi = 0.5 * (-es + np.sqrt(es ** 2 + 4 * mu))
                R = Q @ np.diag(xi) @ np.transpose(Q)

                # Optional over-relaxation. Boyd suggests choosing relax between 1.5
                # and 1.8.
                # R_relax = relax*R - (1-relax)*[np.eye(n), -np.eye(n)] @ W
                G = R - S + L - mu * lambd

                # S update
                S = shrinkage(S - tau * G, alpha * mu * tau)

                # L update
                Lold = L
                U, u, throwawat2 = cyclic_jacobi_tol(L - tau * G, tol)
                evs = np.diag(u)
                gi = np.max(evs - mu * tau * beta * np.ones((n, 1)), np.zeros((n, 1)))
                L = U @ np.diag(gi) @ np.transpose(U)

            if nsweeps is not None:
                Q, q = cyclic_jacobi_sweeps(mu * (C - lambd) - S + L, nsweeps)
                es = np.diag(q)
                xi = 0.5 * (-es + np.sqrt(es ** 2 + 4 * mu))
                R = Q @ np.diag(xi) @ np.transpose(Q)

                # Optional over-relaxation. Boyd suggests choosing relax between 1.5
                # and 1.8.
                # R_relax = relax*R - (1-relax)*[np.eye(n), -np.eye(n)] @ W
                G = R - S + L - mu * lambd

                # S update
                S = shrinkage(S - tau * G, alpha * mu * tau)

                # L update
                Lold = L
                U, u = cyclic_jacobi_sweeps(L - tau * G, nsweeps)
                evs = np.diag(u)
                gi = np.max(evs - mu * tau * beta * np.ones((n, 1)), np.zeros((n, 1)))
                L = U @ np.diag(gi) @ np.transpose(U)

            # Lagrangian variable update (later - add relaxation)
            lambd = lambd - (R - S + L) / mu

            history.objval[k] = objective(R, S, L, alpha, beta, C)

            history.r_norm[k] = np.linalg.norm(R - S + L, 'fro')
            history.s_norm[k] = np.linalg.norm((L - Lold) / mu, 'fro')

            history.eps_pri[k] = n * ABSTOL + RELTOL * max(max(norm(R, 'fro'), norm(S, 'fro')), norm(L, 'fro'))
            history.eps_dual[k] = n * ABSTOL + RELTOL * np.linalg.norm(lambd, 'fro')

            if history.r_norm[k] < history.eps_pri[k] and history.s_norm[k] < history.eps_dual[k]:
                break

    return S, L, history

    print(time.time() - t_start)