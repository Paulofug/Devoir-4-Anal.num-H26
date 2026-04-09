import numpy as np
from tridiagonal_280 import tridiagonal
 
 
def problimite(N, Q, R, a, b, alpha, beta):
   
    h = (b - a) / (N + 1)
 
    D = -2.0 * np.ones(N) - Q * h**2
 
    I_diag = np.ones(N - 1)
    S_diag = np.ones(N - 1)
 
    b_vec = R * h**2
 
    b_vec[0]  -= alpha
    b_vec[-1] -= beta
 
    y_int = tridiagonal(N, D, I_diag, S_diag, b_vec)
 
    y = np.empty(N + 2)
    y[0]    = alpha
    y[1:-1] = y_int
    y[-1]   = beta
 
    return y