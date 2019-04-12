import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.linalg as sla
import time

def init_matrix(A, B, C, F, N):
    A *= 10
    B *= 10
    C *= 10
    F *= 10
    #for i in range(0, N - 1):
     #   randon.seed()
      #  A[i] = random.randint(-40, 40)
       # B[i] = random.randint(-40, 40)
       # C[i] = random.randint(-40, 40)
       # F[i] = random.randint(-40, 40)
    #A[N - 1] = random.randint(-40, 40)

def running(A, B, C, F, N):
    alpha = np.empty(N + 1)
    beta = np.empty(N + 1)
    alpha[0] = 0
    beta[0] = 0
    for i in range(0, N):
        d = A[i]*alpha[i] + B[i]
        alpha[i + 1] = (-1 * C[i]) / d
        beta[i + 1] = (F[i] - A[i]*beta[i]) / d
    X = np.empty(N + 1)
    X[N] = 0
    for i in range(0, N):
        X[N - i - 1] = alpha[N - i]*X[N - i] + beta[N - i]
    return X

file1 = open('plot_my_running.txt', 'w')
file2 = open('plot_running.txt', 'w')

for n in range(15000, 19000, 250):
    a = np.random.rand(n)
    b = np.random.rand(n)
    c = np.random.rand(n)
    f = np.random.rand(n)
    init_matrix(a, b, c, f, n)
    c[n - 1] = 0
    a[0] = 0
    a2 = np.concatenate((a[1:], np.zeros(1)))
    c2 = np.concatenate((np.zeros(1) , c[:n-1]))
    A = np.array([c2] + [b] + [a2])
    start = time.time()
    X1 = running(a, b, c, f, n)
    stop = time.time()
    file1.write(str(n) + ' ' + str(stop - start) + '\n');
    start2 = time.time()
    X2 = sla.solve_banded((1, 1), A, f)
    stop2 = time.time()
    file2.write(str(n) + ' ' + str(stop2 - start2) + '\n')
    print(np.allclose(X1[:n], X2))
file1.close()
file2.close()
