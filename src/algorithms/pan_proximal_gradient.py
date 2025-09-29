import sys
import torch
import math 
import logging
import torch.nn as nn
from tqdm.auto import tqdm
from nabla import nabla
from datetime import datetime

sys.path.append('src/datasets')
sys.path.append('src/algorithms')

class PANProximalGradient(nn.Module):
    """
    Algorithme de gradient proximal pour le pansharpening hyperspectral.
    
    Attributes:
        max_iter (int): Nombre maximal d'itérations
        lmbda (float): Paramètre de régularisation TV
        lmbda_m (float): Poids de l'attache aux données multispectrales
        tol (float): Tolérance de convergence
        scale (int): Facteur d'échelle
        verbose (bool): Affichage des informations
        A (function): Opérateur de sous-échantillonnage
        Aadj (function): Adjoint de l'opérateur A
        R (torch.Tensor): Matrice de projection panchromatique
    """

    def __init__(self, A, Aadj,spectral_op,spectral_op_t,max_iter, lmbda,alpha, lmbda_m, tol,scale,p,q,r,verbose):
        super().__init__()
        self.max_iter = max_iter
        self.scale = scale
        self.lmbda = lmbda
        self.alpha = alpha
        self.lmbda_m = lmbda_m
        self.tol = tol
        self.verbose = verbose
        self.A = A
        self.Aadj = Aadj
        self.spectral_op = spectral_op
        self.spectral_op_t = spectral_op_t
        self.p = p
        self.q = q 
        self.r = r

        # Configuration du logger simple vers stdout
        self.logger = logging.getLogger('PANProximalGradient')
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(ch)


    def ctv_norm(self, U, eps=1e-8):
        """
        Calcule la norme CTV l^{p,q,r} avec support pour p,q,r = infini.
        
        Args:
            U (torch.Tensor): Tenseur de gradients [B, C, H, W, 2]
            eps (float): Petite valeur pour stabilité numérique
            
        Returns:
            torch.Tensor: Norme CTV [B,1,1,1]
        """
        # ---- Norme p sur les canaux ----
        if torch.isinf(torch.tensor(self.p)):
            norm_p = torch.amax(torch.abs(U), dim=1, keepdim=True)  # l^infini
        else:
            norm_p = (torch.sum(torch.abs(U) ** self.p, dim=1, keepdim=True) + eps) ** (1.0 / self.p)

        # ---- Norme q sur les directions ----
        if torch.isinf(torch.tensor(self.q)):
            norm_q = torch.amax(norm_p, dim=-1, keepdim=True)  # l^infini
        else:
            norm_q = (torch.sum(norm_p ** self.q, dim=-1, keepdim=True) + eps) ** (1.0 / self.q)

        # ---- Norme r sur les pixels ----
        if torch.isinf(torch.tensor(self.r)):
            norm_r = torch.amax(norm_q, dim=(2, 3), keepdim=True)  # l^infini
        else:
            norm_r = (torch.sum(norm_q ** self.r, dim=(2, 3), keepdim=True) + eps) ** (1.0 / self.r)

        # ---- Retour : somme totale par image ----
        #   Dans l'article, CTV = somme sur tous les pixels des normes locales
        return norm_r
        

    def convergence_criteria(self, U0, U1):
        """
        Critère de convergence basé sur la variation relative.
        
        Args:
            U0 (torch.Tensor): Image à l'itération précédente [b,c,h,w]
            U1 (torch.Tensor): Image courante [b,c,h,w]
            
        Returns:
            bool: True si convergence atteinte
        """
        return (torch.linalg.norm(U1-U0)/torch.linalg.norm(U0)) < self.tol
    
    def compute_cost(self, U, Y_H, Y_M):
        """
        Calcule le coût total de la fonction objective.
        """ 

        raise NotImplementedError('compute_cost is not implemented in abstract class.')       
       
    def grad_f(self, U, Y_H, Y_M):
        """
        Calcule le gradient de la fonction objective.
        
        Args:
            U (torch.Tensor): Image estimée [b,c,h,w]
            Y_H (torch.Tensor): Données hyperspectrales [b,c,h//scale,w//scale]
            Y_M (torch.Tensor): Données panchromatiques [b,1,h,w]
            
        Returns:
            torch.Tensor: Gradient combiné [b,c,h,w]
        """
        # Terme 1: Gradient de 1/2 ||Y_H - A(U)||²
        grad1 = self.Aadj((self.A(U) - Y_H))
        
        # Terme 2: Gradient de (λ_m/2) ||Y_M - RU||²
        subtracted = self.spectral_op(U) - Y_M
        RT_subtracted = self.spectral_op_t(subtracted)
        grad2 = self.lmbda_m * RT_subtracted
        
        return grad1 + grad2
    
    def proxg(self, x):
        """Opérateur proximal (à implémenter)."""
        raise NotImplementedError("prox() is not implemented in abstract class.")
    

        
    def forward(self, Y_H, Y_M):
        """
        Proximal gradient : U_{k+1} = prox_g(U_k - alpha * grad_f(U_k))
        """
        # ---- Calcul de L et alpha ----
        c = Y_H.shape[1]   # nombre de bandes
        L = 1.0 + self.lmbda_m * (1.0 / c)
        self.alpha = 1.0 / L
        self.logger.info(f"[Proximal Gradient] L = {L:.6f}  |  alpha = {self.alpha:.6f}")

        # ---- Initialisation ----
        U = self.Aadj(Y_H).clone()
        cost_history = torch.zeros(self.max_iter, device=U.device)
        relval = torch.zeros(self.max_iter, device=U.device)

        for it in range(self.max_iter):
            U_prev = U.clone()

            # Gradient
            grad = self.grad_f(U, Y_H, Y_M)

            # Mise à jour ISTA
            U = self.proxg(U - self.alpha * grad)

            # Calcul du coût
            total_cost, data_term_h, data_term_m, tv_term = self.compute_cost(U, Y_H, Y_M)
            cost_history[it] = total_cost.item()

            # Critère d'arrêt
            delta_U = torch.norm(U - U_prev).item() / (torch.norm(U).item() + 1e-8)
            relval[it] = delta_U
            if it % 10 == 0 or delta_U < self.tol:
                self.logger.info(
                    f"{it:<5} | {total_cost.item():<12.3e} | {data_term_h.item():<12.3e} | "
                    f"{data_term_m.item():<12.3e} | {tv_term.item():<12.3e} | {delta_U:<12.3e}"
                )
                if delta_U < self.tol:
                    break

        return U, cost_history, relval

