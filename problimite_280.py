import numpy as np
from tridiagonal import solve_tridiagonal

def problimite(h, P, Q, R, a, b, alpha, beta):
    """
    Résout le problème aux limites pour l'équation différentielle linéaire du second ordre.
    
    Arguments:
    - h : pas de discrétisation.
    - P : vecteur contenant les évaluations de la fonction p(x) aux noeuds xi.
    - Q : vecteur contenant les évaluations de la fonction q(x) aux noeuds xi.
    - R : vecteur contenant les évaluations de la fonction r(x) aux noeuds xi.
    - a : borne inférieure de l'intervalle [a, b].
    - b : borne supérieure de l'intervalle [a, b].
    - alpha : condition limite y(a) = alpha.
    - beta : condition limite y(b) = beta.
    
    Retourne:
    - y : vecteur solution approximant y(x) aux noeuds xi.
    """
    N = len(P)  # Nombre de noeuds intérieurs
    D = np.zeros(N)  # Diagonale principale
    I = np.zeros(N - 1)  # Diagonale inférieure
    S = np.zeros(N - 1)  # Diagonale supérieure
    b_vec = np.zeros(N)  # Vecteur du côté droit

    # Construction des coefficients de la matrice tridiagonale et du vecteur b
    for i in range(N):
        D[i] = 2 + Q[i] * h**2
        if i > 0:
            I[i - 1] = -1 - P[i] * h / 2
        if i < N - 1:
            S[i] = -1 + P[i] * h / 2
        b_vec[i] = -R[i] * h**2

    # Ajout des conditions limites dans le vecteur b
    b_vec[0] += (1 + P[0] * h / 2) * alpha
    b_vec[-1] += (1 - P[-1] * h / 2) * beta

    # Résolution du système tridiagonal
    y_interior = solve_tridiagonal(D, I, S, b_vec)

    # Ajout des conditions limites à la solution
    y = np.zeros(N + 2)
    y[0] = alpha
    y[1:-1] = y_interior
    y[-1] = beta

    return y