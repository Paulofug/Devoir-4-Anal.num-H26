import numpy as np
import matplotlib.pyplot as plt
from problimite_280 import problimite
 
# ============================================================
# Paramètres du problème (équation (4))
# ============================================================
L   = 6       # longueur de la barre
Ta  = 20      # température ambiante
T   = 350     # température à x=0
k   = 1.2     # coefficient de diffusion
 
a, b   = 0, L
alpha  = T    # y(0) = T
beta   = Ta   # y(L) = Ta
 
# L'équation y'' - k²(y - Ta) = 0  s'écrit  y'' - k²y = -k²Ta
# donc  q(x) = k²  et  r(x) = -k²Ta  (constantes)
def q_func(x):
    return k**2 * np.ones_like(x)
 
def r_func(x):
    return -k**2 * Ta * np.ones_like(x)
 
# ============================================================
# Solution exacte  y(x) = Ta + β1·e^(kx) + β2·e^(-kx)
# Conditions :
#   y(0) = Ta + β1 + β2 = T          => β1 + β2 = T - Ta
#   y(L) = Ta + β1·e^(kL) + β2·e^(-kL) = Ta  => β1·e^(kL) + β2·e^(-kL) = 0
# D'où β2 = (T-Ta)/(1 - e^(-2kL))   et   β1 = (T-Ta) - β2
# ============================================================
delta  = T - Ta
beta2  = delta / (1.0 - np.exp(-2.0 * k * L))
beta1  = delta - beta2
 
def exact(x):
    return Ta + beta1 * np.exp(k * x) + beta2 * np.exp(-k * x)
 
 
# ============================================================
# Question 2 a) — Figures pour h = 2 et h = 1
# ============================================================
fig1, ax1 = plt.subplots(figsize=(8, 5))
 
x_fine = np.linspace(a, b, 500)
ax1.plot(x_fine, exact(x_fine), 'k-', linewidth=2, label='Solution exacte')
 
couleurs = ['tab:blue', 'tab:orange']
for h, couleur in zip([2, 1], couleurs):
    N = int(round((b - a) / h)) - 1          # h = (b-a)/(N+1)  =>  N = (b-a)/h - 1
    x_nodes = a + np.arange(N + 2) * h       # x_0, x_1, ..., x_{N+1}
    Q = q_func(x_nodes[1:-1])
    R = r_func(x_nodes[1:-1])
    y = problimite(N, Q, R, a, b, alpha, beta)
    ax1.plot(x_nodes, y, 'o--', color=couleur, label=f'h = {h}  (N = {N})')
 
ax1.set_xlabel('x')
ax1.set_ylabel('y(x)  [°C]')
ax1.set_title('Distribution de température — approximation vs solution exacte\n'
              r'$y\'\'(x) - k^2(y - T_a) = 0$,  L=6, Ta=20, T=350, k=1.2')
ax1.legend()
ax1.grid(True)
plt.tight_layout()
plt.savefig('figure1.png', dpi=150)
 
 
# ============================================================
# Question 2 b) — Erreur E(h) en échelle log-log
# ============================================================
denominateurs = [3, 6, 100, 1_000, 10_000]
h_vals  = [L / d for d in denominateurs]
E_vals  = []
 
for h in h_vals:
    N = int(round((b - a) / h)) - 1
    x_nodes = a + np.arange(N + 2) * h
    Q = q_func(x_nodes[1:-1])
    R = r_func(x_nodes[1:-1])
    y = problimite(N, Q, R, a, b, alpha, beta)
    # Erreur sur les noeuds intérieurs uniquement
    E = np.max(np.abs(y[1:-1] - exact(x_nodes[1:-1])))
    E_vals.append(E)
 
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.loglog(h_vals, E_vals, 'bo-', label='E(h)')
 
# Droite de référence de pente 2
h_arr = np.array(h_vals)
ref   = E_vals[0] * (h_arr / h_vals[0])**2
ax2.loglog(h_vals, ref, 'r--', label='Pente 2 (référence)')
 
ax2.set_xlabel('h')
ax2.set_ylabel('E(h)')
ax2.set_title("Erreur maximale E(h) en fonction du pas de discrétisation h\n"
              "(échelle log-log — ordre de convergence = 2)")
ax2.legend()
ax2.grid(True, which='both', ls=':')
plt.tight_layout()
plt.savefig('figure2.png', dpi=150)
plt.show()