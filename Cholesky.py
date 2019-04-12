import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.linalg as sla
import time

def Cholesky(A, N):
    L = np.zeros((N, N));
    L[0][0] = A[0][0] ** 0.5
    for j in range(1, n):
        L[j][0] = A[j][0] / L[1][1]
    for i in range(0, N):
        SUM = 0
        for k in range(0, i):
            SUM += (L[k][i]) ** 2
        L[i][i] = (A[i][i] - SUM) ** 0.5
        for j in range(i + 1, N):
            SUM = 0
            for k in range(0, i):
                SUM += L[k][i] * L[k][j]
            L[i][j] = (A[i][j] - SUM) / L[i][i]
    print(L)
    return L

def init_matrix(M, N):
    for i in range(0, N):
        for j in range(i + 1, N):
            M[j][i] = M[i][j]

def init_low_tr(M, N):
    for i in range(0, N):
        for j in range(i + 1, N):
            M[i][j] = 0
        M[i][i] = abs(M[i][i]) + 1

def Cholesky2(A, N):
    L = np.zeros((N, N))
    L[0][0] = A[0][0] ** 0.5
    for i in range(1, N):
        L[i][0] = A[i][0] / L[0][0]
    for i in range(1, N):
        SUM = 0
        for p in range(0, i):
            SUM += (L[i][p] ** 2)
        L[i][i] = (A[i][i] - SUM) ** 0.5
        for j in range(i + 1, N):
            SUM = 0
            for p in range(0, i):
                SUM += L[i][p] * L[j][p]
            L[j][i] = (A[j][i] - SUM) / L[i][i]
    #print(L)
    return L

def Ch_ans1(M, B, N):
    ANS = np.zeros(N)
    for i in range(0, n):
        ANS[i] = B[i]
        for j in range(0, i):
            ANS[i] -= ANS[j]*L[i][j]
        ANS[i] /= L[i][i]
    return ANS

def Ch_ans2(M, B, N):
    ANS = np.zeros(N)
    for i in range(0, N):
        ANS[N - i - 1] = B[N - i - 1]
        for j in range(N - i, N):
            ANS[N - i - 1] = ANS[N - i - 1] - M[N - i -1][j]*ANS[j]
        ANS[N - i - 1] /= M[N - i - 1][N - i - 1]
    return ANS


file1 = open('plot_my_cholesky.txt', 'w')
file2 = open('plot_cholesky.txt', 'w')

for n in range(20, 150, 10):
    L = np.random.randint(-15, 15, (n, n))
    B = np.random.randint(-15, 15, n)
    init_low_tr(L, n)
    A = L.dot(np.transpose(L))
    start = time.time()
    L1 = Cholesky2(A, n)
    stop = time.time()
    file1.write(str(n) + ' ' + str(stop - start) + '\n');
    start = time.time()
    L2 = sla.cholesky(A, lower = True)
    stop = time.time()
    file2.write(str(n) + ' ' + str(stop - start) + '\n');
    print(np.allclose(L1, L2))
    #ANS = Ch_ans1(L1, B, n)
    #ANS2 = Ch_ans2(np.transpose(L1), ANS, n)
    #ANS_G = np.linalg.solve(A, B)
    #print(np.allclose(ANS2, ANS_G))
    #print(ANS2, ANS_G)
file1.close()
file2.close()
