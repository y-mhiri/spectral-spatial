import torch
import torch.nn as nn
from tqdm.auto import tqdm
from grad import gradient, div_3d

class ProximalGradient(nn.Module):
    """
    Algorithme de gradient proximal pour résoudre le problème de pansharpening hyperspectral.
    """

    def __init__(self, max_iter=100, lmbda=1.0, lmbda_m=1.0, tau=0.1, tol=1e-7, verbose=True):
        super(ProximalGradient, self).__init__()
        self.max_iter = max_iter
        self.lmbda = lmbda  # Paramètre de régularisation pour la variation totale
        self.lmbda_m = lmbda_m  # Paramètre de régularisation pour l'attache aux données multispectrales
        self.tau = tau  # Pas de gradient
        self.tol = tol  # Tolérance pour la convergence
        self.verbose = verbose  # Affichage des informations

    def grad_f(self, U, Y_H, Y_M,B, B_t, R):
        """
        Calcule le gradient de la fonction f(U).
        """
        # Terme 1 : Gradient de 1/2 ||Y_H - H U B||_F^2
        grad1 = (U @ B - Y_H) @ B_t

        # Terme 2 : Gradient de (lambda_m / 2) ||Y_M - R H U||_F^2
        grad2 = self.lmbda_m * (R.T @ (R @ U - Y_M))

        return grad1 + grad2

    def norm_221(self, A):
        """
        Calcule la norme collaborative L2,2,1 d'un tenseur A.
        A : Tensor de forme (2, N, M, C).
        """
        # Étape 1 : Calculer la somme des carrés sur les bandes spectrales (dim=3)
        sum_squares = torch.sum(A**2, dim=3)  # Forme (2, N, M)

        # Étape 2 : Somme sur la largeur (dim=2)
        sum_width = torch.sum(sum_squares, dim=2)  # Forme (2, N)

        # Étape 3 : Prendre la racine carrée
        sqrt_sum = torch.sqrt(sum_width)  # Forme (2, N)

        # Étape 4 : Somme sur la hauteur (dim=1)
        norm_height = torch.sum(sqrt_sum, dim=1)  # Forme (2,)

        return norm_height

    def proj(self, z):
        """
        Projette les valeurs de z sur la boule unité en utilisant la norme L2,2,1.
        z : Tensor de forme (2, hauteur, largeur, bandes)
        """
        # Calcul de la norme L2,2,1 pour z
        norm = self.norm_221(z)

        # Appliquer la projection uniquement si la norme L2,2,1 > 1
        mask = norm > 1
        z_proj = torch.where(mask.unsqueeze(1).unsqueeze(2).unsqueeze(3), 
                             z / norm.unsqueeze(1).unsqueeze(2).unsqueeze(3), z)

        return z_proj

    def grad_proj(self, x, tau, max_iter):
        """
        Calcule le gradient de la projection.
        x : Tensor de forme (hauteur, largeur, bandes)
        tau : Pas de gradient
        max_iter : Nombre maximal d'itérations
        """
        n, m, c = x.shape
        z0 = torch.ones((2, n, m, c), device=x.device)  # Initialisation de z0

        for _ in range(max_iter):
            # Calcul du gradient de la divergence
            div_z0 = div_3d(z0)  # Divergence de z0
            grad_z = -2 * gradient(div_z0 + x / self.lmbda)  # Gradient de la fonction

            # Mise à jour de z
            z = self.proj(z0 - tau * grad_z)
            z0 = z

        return z

    def proxg(self, x, tau, max_iter):
        """
        Calcule le proximal.
        x : Tensor de forme (hauteur, largeur, bandes)
        tau : Pas de gradient
        max_iter : Nombre maximal d'itérations
        """
        z = self.grad_proj(x, tau, max_iter)
        y = x + self.lmbda * div_3d(z)
        return y
    def forward(self, Y_H, Y_M, B, B_t, R, U_init=None):
        """
        Résout le problème d'optimisation.
        """
        # Initialisation
        if U_init is None:
            U = torch.zeros_like(Y_H)
        else:
            U = torch.clone(U_init)

        # Boucle d'optimisation
        for it in tqdm(range(self.max_iter)):
            # Gradient de f(U)
            grad_U = self.grad_f(U, Y_H, Y_M, B, B_t, R)

            # Mise à jour de U
            U = self.proxg(U - self.lmbda * grad_U, self.tau, self.max_iter)

            # Affichage des informations
            if self.verbose and (it % 10 == 0):
                print(f'Iteration {it}: Loss = {torch.norm(Y_H - U @ B)}')

        return U