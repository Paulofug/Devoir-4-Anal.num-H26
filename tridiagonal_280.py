import numpy as np
 
 
def tridiagonal(N, D, I, S, b):
   
    D = D.astype(float).copy()
    b = b.astype(float).copy()
    I = I.astype(float).copy()
    S = S.astype(float).copy()
 
    y = np.zeros(N)
    x = np.zeros(N)
 
    y[0] = b[0] / D[0]
 
    for i in range(1, N):
        S[i - 1] = S[i - 1] / D[i - 1]
        D[i]     = D[i] - I[i - 1] * S[i - 1]
        y[i]     = (b[i] - I[i - 1] * y[i - 1]) / D[i]
 
    x[N - 1] = y[N - 1]
 
    for i in range(N - 2, -1, -1):
        x[i] = y[i] - S[i] * x[i + 1]
 
    return x