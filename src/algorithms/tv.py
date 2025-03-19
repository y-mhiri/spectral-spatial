import torch
import torch.nn as nn
from tqdm.auto import tqdm



class ProximalGradient(nn.Module):
    """
    Algorithme de gradient proximal pour résoudre le problème de pansharpening hyperspectral.
    """
    def __init__(self, max_iter=100, lmbda=1.0, lmbda_m=1.0, h=0.1, tol=1e-7, verbose=True):
        super(ProximalGradient, self).__init__()
        self.max_iter = max_iter
        self.lmbda = lmbda  # Paramètre de régularisation pour la variation totale
        self.lmbda_m = lmbda_m  # Paramètre de régularisation pour l'attache aux données multispectrales
        self.h = h  # Pas de gradient
        self.tol = tol  # Tolérance pour la convergence
        self.verbose = verbose  # Affichage des informations

    def grad_f(self, U, Y_H, Y_M, H, B, S, R):
        """
        Calcule le gradient de la fonction f(U).
        """
        # Terme 1 : Gradiant de 1/2 ||Y_H - H U B S||_F^2
        grad1 = H.T @ (H @ U @ B @ S - Y_H) @ (B @ S).T

        # Terme 2 : Gradiant de (lambda_m / 2) ||Y_M - R H U||_F^2
        grad2 = self.lmbda_m * (H.T @ R.T @ (R @ H @ U - Y_M))

        return grad1 + grad2

    def prox_g(self, U, h):
        """
        Calcule le proximal de la fonction g(U) = lambda ||grad U||_{2,2,1}.
        """
        # Calcul du gradient de U
        grad_U_h = torch.gradient(U, dim=2)[0]  # Gradient horizontal
        grad_U_v = torch.gradient(U, dim=3)[0]  # Gradient vertical

        # Norme du gradient
        norm_grad_U = torch.sqrt(grad_U_h**2 + grad_U_v**2)

        # Proximal de la norme l2,2,1
        U_prox = U / torch.clamp(norm_grad_U, min=1)

        return U_prox

    def forward(self, Y_H, Y_M, H, B, S, R, U_init=None):
        """
        Résout le problème d'optimisation.
        """
        # Initialisation
        if U_init is None:
            U = torch.zeros_like(Y_H)
        else:
            U = torch.clone(U_init)

        # Boucle d'optimisation
        loss_history = []
        for it in tqdm(range(self.max_iter)):
            # Gradient de f(U)
            grad_U = self.grad_f(U, Y_H, Y_M, H, B, S, R)

            # Mise à jour de U
            U = self.prox_g(U - self.h * grad_U, self.h * self.lmbda)

            # Calcul de la perte
            loss = 0.5 * torch.norm(Y_H - H @ U @ B @ S)**2 + \
                   self.lmbda_m * 0.5 * torch.norm(Y_M - R @ H @ U)**2 + \
                   self.lmbda * torch.sum(torch.sqrt(torch.gradient(U, dim=[2, 3])[0]**2 + torch.gradient(U, dim=[2, 3])[1]**2))
            loss_history.append(loss.item())

            # Affichage des informations
            if self.verbose and (it % 10 == 0):
                print(f'Iteration {it}: Loss = {loss.item()}')

            # Critère d'arrêt
            if it > 0 and abs(loss_history[-1] - loss_history[-2]) < self.tol:
                print(f'Converged after {it} iterations.')
                break

        return U, loss_history