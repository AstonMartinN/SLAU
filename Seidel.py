import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.linalg as sla
import time

def seidel(A, F, N, X):
    xn = np.zeros(N)
    for i in range(0, N):
        s = 0
        for j in range(0, i):
            s += A[i][j] * xn[j]
        for j in range(i + 1, N):
            s +=  A[i][j] * X[j]
        xn[i] = (F[i] - s) / A[i][i]
    return xn

def solve(A, F, N):
    xnew = np.zeros(N)
    x = np.eye(N)
    while np.allclose(x, xnew) == 0:
        print("*")
        x = xnew
        xnew = seidel(A, F, N, x)
    return xnew

file1 = open('plot_my_seidel.txt', 'w')
file2 = open('plot_seidel.txt', 'w')

for n in range(50, 150, 10):
    M1 = np.random.rand(n,n)
    B1 = np.random.rand(n)
    for i in range(0, n):
        M1[i][i] = abs(M1[i][i])
        for j in range(0, n):
            M1[i][i] += abs(M1[i][j])
    M2 = np.array(M1)
    B2 = np.array(B1)
    start = time.time()
    ANS1 = np.linalg.solve(M1, B1)
    stop = time.time()
    file2.write(str(n) + ' ' + str(stop - start) + '\n')
    start = time.time()
    ANS2 = solve(M2, B2, n)
    stop = time.time()
    file1.write(str(n) + ' ' + str(stop - start) + '\n')
    print(np.allclose(ANS1, ANS2))
