import numpy as np


def Householder(A: np.matrix) -> tuple:
    m, n = A.shape
    # check for complex numbers, if so then convert all to complex
    complex_values = np.iscomplex(A).any()
    if complex_values:
        Q = np.identity(m, dtype=complex)
        R = np.matrix(A, copy=True, dtype=complex)
        A = np.matrix(A, dtype=complex)
    else:
        Q = np.identity(m)
        R = np.matrix(A, copy=True)

    for i in range(n):
        x = R[i:, i]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        Q_i = np.identity(m, dtype=complex if complex_values else None)
        if complex_values:
            Q_i[i:, i:] -= (2.0 + 0j) * np.outer(v, v)
        else:
            Q_i[i:, i:] -= 2.0 * np.outer(v, v)
        Q = Q @ Q_i
        R = Q_i @ R
    return Q, R


if __name__ == '__main__':
    A = np.matrix([[1 + 1j,2,3], [4,5,6], [7,8,8], [7,8,8]])
    Q, R = Householder(A)
    print(Q)
    print(R)
    print(Q @ R)
