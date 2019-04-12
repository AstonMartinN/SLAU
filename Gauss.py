import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.linalg as sla
import time

def diag(M, N, B):
    for i in range(0, N):
        B[i] = random.randint(-40, 40)
        for j in range(0, N):
            random.seed()
            M[i][j] = random.randint(-40, 40)
    for i in range(0, N):
        M[i][i] = abs(M[i][i])
        for j in range(0, N):
            M[i][i] += abs(M[i][j])

def Gauss(matrix, n, b, ans):
    for i in range(0, n):
        for j in range(i + 1, n):
            matrix[i][j] = matrix[i][j] / matrix[i][i]
        b[i] = b[i] / matrix[i][i]
        matrix[i][i] = 1
        for j in range(i + 1, n):     # fix ->
            for k in range(i + 1, n):     # run ->
                matrix[j][k] = matrix[j][k] - matrix[i][k]*matrix[j][i]
            b[j] = b[j] - b[i] * matrix[j][i]
            matrix[j][i] = 0
    for i in range(0, n):
        ans[n - i - 1] = b[n - i - 1]
        for j in range(n - i, n):
            ans[n - i - 1] = ans[n - i - 1] - matrix[n - i -1][j]*ans[j] 

def Gauss_new(matrix, n, b, ans):
    for i in range(0, n):
        b[i] = b[i] / matrix[i][i]
        matrix[i] /= matrix[i][i]
        for j in range(i + 1, n):     # fix ->
            b[j] = b[j] - b[i] * matrix[j][i]
            matrix[j] -= matrix[i]*matrix[j][i]
    for i in range(0, n):
        ans[n - i - 1] = b[n - i - 1]
        for j in range(n - i, n):
            ans[n - i - 1] = ans[n - i - 1] - matrix[n - i -1][j]*ans[j] 

file1 = open('plot_my_gauss.txt', 'w')
file2 = open('plot_gauss.txt', 'w')
for n in range(10, 200, 10):
    ANS = np.empty(n)
    MAT = np.random.rand(n,n)
    BB = np.random.rand(n)
    for i in range(0, n):
        MAT[i][i] = abs(MAT[i][i])
        for j in range(0, n):
            MAT[i][i] += abs(MAT[i][j])
    MAT2 = np.array(MAT)
    BB2 = np.array(BB)
    start = time.time()
    Gauss_new(MAT, n, BB, ANS)
    stop = time.time()
    file1.write(str(n) + ' ' + str(stop - start) + '\n')
    start2 = time.clock()
    x = np.linalg.solve(MAT2, BB2)
    stop2 = time.clock()
    file2.write(str(n) + ' ' + str(stop2 - start2) + '\n')
    print(np.allclose(x, ANS))
file1.close()
file2.close()
