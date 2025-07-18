import sys
import torch
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

    def __init__(self, A, Aadj,spectral_op,spectral_op_t,max_iter, step_size, lmbda, lmbda_m, tol,scale,p,q,r,verbose):
        super().__init__()
        self.max_iter = max_iter
        self.scale = scale
        self.step_size = step_size
        self.lmbda = lmbda
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

    



    def ctv_norm(self, U,eps=1e-8):
        """
        Calcule la norme CTV l^p,q,r avec support pour p,q,r = infini.
        
        Args:
            U (torch.Tensor): Tenseur de gradients [b,c,h,w,2]
            p, q, r (float or torch.inf): Exposants de la norme
            eps (float): Petite valeur pour stabilité numérique
            
        Returns:
            torch.Tensor: Norme CTV [b,1,1,1]
        """
        # Norme p sur les canaux (axis=1)
        if torch.isinf(torch.tensor(self.p)):
            norm_p = torch.amax(torch.abs(U), dim=1, keepdim=True)  # l^infini
        else:
            norm_p = torch.sum(torch.abs(U)**self.p, dim=1, keepdim=True)**(1/(self.p + eps))

        # Norme q sur les dérivées (axis=-1)
        if torch.isinf(torch.tensor(self.q)):
            norm_q = torch.amax(torch.abs(norm_p), dim=-1, keepdim=True)  # l^infini
        else:
            norm_q = torch.sum(norm_p**self.q, dim=-1, keepdim=True)**(1/(self.q + eps))

        # Norme r sur les pixels (axis=(2,3))
        if torch.isinf(torch.tensor(self.r)):
            norm_r = torch.amax(torch.abs(norm_q), dim=(2,3), keepdim=True)  # l^infini
        else:
            norm_r = torch.sum(norm_q**self.r, dim=(2,3), keepdim=True)**(1/(self.r + eps))

        return norm_r
        
    
    
    
    def ctv_norm1(self,U, p, q, r):
        """Calcule la norme CTV l^p,q,r d'un tenseur A (shape: b x c x hx w x 2 )."""
        # Ordre: p sur canaux (axis=1), q sur dérivées (axis=-1), r sur pixels (axis=(2,3))
        norm_p = torch.sum(torch.abs(U)**p, dim=1, keepdim=True)**(1/p)
        norm_q = torch.sum(norm_p**q, dim=-1, keepdim=True)**(1/q)
        norm_r = torch.sum(norm_q**r, dim=(2,3), keepdim=True)**(1/r)
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
        
        Args:
            U (torch.Tensor): Image estimée [b,c,h,w]
            Y_H (torch.Tensor): Données hyperspectrales [b,c,h//scale,w//scale]
            Y_M (torch.Tensor): Données panchromatiques [b,1,h,w]
            
        Returns:
            float: Coût total
        """
        # Terme d'attache aux données hyperspectrales
        data_term_h = 0.5 * torch.norm(self.A(U)-Y_H)**2
        
        # Terme d'attache aux données panchromatiques
        
        data_term_m = 0.5 * self.lmbda_m * torch.norm(self.spectral_op(U)-Y_M)**2
        
        # Terme de régularisation TV
        grad_U = nabla(U)
        #tv_per_pixel = torch.sqrt(torch.sum(grad_U**2, dim=(1,4)))
        #torch.sum(tv_per_pixel)
        tv_term = self.lmbda * self.ctv_norm(grad_U,eps=1e-8)
        
        return data_term_h + data_term_m + tv_term ,data_term_h,data_term_m,tv_term

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
        raise NotImplementedError("L'opérateur proximal doit être implémenté")
        
    def forward(self, Y_H, Y_M):
        """
        Résout le problème d'optimisation complet.
        
        Args:
            Y_H (torch.Tensor): Données hyperspectrales [b,c,h//scale,w//scale]
            Y_M (torch.Tensor): Données panchromatiques [b,1,h,w]
            
        Returns:
            tuple: (Image estimée [b,c,h,w], historique des coûts)
        """
        U = self.Aadj(Y_H).clone()
        #U = torch.zeros_like(self.Aadj(Y_H)) 
        cost_history = []
        
        # En-tête du tableau
        self.logger.info("\nDébut de l'optimisation:")
        self.logger.info(f"{'It':<5} | {'Coût total':<12} | {'Data H':<12} | {'Data M':<12} | {'TV':<12} | {'ΔU':<12}")
        self.logger.info("-" * 80)
        
        cost_history = torch.zeros(self.max_iter)
        for it in range(self.max_iter):
            U_prev = U.clone()
            
            # Étape de gradient
            grad = self.grad_f(U, Y_H, Y_M)
            U = self.proxg(U - self.step_size * grad)
            
            # Calcul des métriques
            total_cost,data_term_h,data_term_m,tv_term = self.compute_cost(U, Y_H, Y_M)
            delta_U = torch.norm(U - U_prev).item() / (torch.norm(U_prev).item() + 1e-8)
            cost_history[it] = total_cost.item()
            
            # Affichage conditionnel
            if it % 10 == 0 or it == self.max_iter - 1 or delta_U < self.tol:
                log_message = (f"{it:<5} | {total_cost.item():<12.3e} | {data_term_h.item():<12.3e} | "
                             f"{data_term_m.item():<12.3e} | {tv_term.item():<12.3e} | {delta_U:<12.3e}")
                self.logger.info(log_message)
                
                if delta_U < self.tol:
                    self.logger.info(f"\nConvergence atteinte à l'itération {it} (ΔU = {delta_U:.3e} < {self.tol})")
                    break
        
        return U, cost_history