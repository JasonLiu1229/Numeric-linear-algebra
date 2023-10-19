import numpy as np

from Gram_Schmidt import GS

if __name__ == '__main__':
    A = np.matrix([[1,2,3], [4,5,6], [7,8,8], [7,8,8]])
    Q, R = GS.classicGramSchmidt(A)
    # print(Q)
    # print(R)
    print(Q @ R)