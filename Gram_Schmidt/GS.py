import numpy as np


def classicGramSchmidt(A: np.matrix) -> tuple:
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = np.matrix(A, copy=True)
    for i in range(n):
        pass


def modifiedGramSchmidt(A: np.matrix) -> tuple:
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = np.matrix(A, copy=True)
    for i in range(n):
        R[i, i] = np.linalg.norm(V[i])
        Q[i] = V[i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = Q[i].T * V[j]
            V[j] = V[j] - R[i, j] * Q[i]
    return Q, R
