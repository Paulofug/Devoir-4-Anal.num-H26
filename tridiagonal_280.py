import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time

def solve_tridiagonal(D, I, S, b):
    """
    Résout le système linéaire Ax = b où A est une matrice tridiagonale.
    
    Arguments:
    - D : vecteur contenant la diagonale principale de la matrice A.
    - I : vecteur contenant la diagonale inférieure de la matrice A.
    - S : vecteur contenant la diagonale supérieure de la matrice A.
    - b : vecteur du côté droit de l'équation Ax = b.
    
    Retourne:
    - x : solution du système linéaire.
    """
    # Construction de la matrice tridiagonale A
    n = len(D)
    diagonals = [I, D, S]
    offsets = [-1, 0, 1]
    A = diags(diagonals, offsets, shape=(n, n), format='csr')  # Matrice sparse

    # Résolution du système
    x = spsolve(A, b)
    return x

# Test de la fonction avec une matrice de grande taille
if __name__ == "__main__":
    # Taille de la matrice
    N = 15000

    # Construction des diagonales
    D = np.full(N, 4)  # Diagonale principale
    I = np.full(N - 1, 1)  # Diagonale inférieure
    S = np.full(N - 1, 1)  # Diagonale supérieure
    b = np.random.rand(N)  # Vecteur b aléatoire

    # Résolution sans matrice sparse (pour comparaison)
    A_dense = np.diag(D) + np.diag(I, k=-1) + np.diag(S, k=1)
    start_time = time.time()
    x_dense = np.linalg.solve(A_dense, b)
    dense_time = time.time() - start_time
    print(f"Temps de résolution avec matrice dense : {dense_time:.4f} secondes")

    # Résolution avec matrice sparse
    start_time = time.time()
    x_sparse = solve_tridiagonal(D, I, S, b)
    sparse_time = time.time() - start_time
    print(f"Temps de résolution avec matrice sparse : {sparse_time:.4f} secondes")

    # Vérification des résultats
    print(f"Les solutions sont-elles proches ? {np.allclose(x_dense, x_sparse)}")