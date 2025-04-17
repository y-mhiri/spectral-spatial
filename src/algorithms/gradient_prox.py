import sys
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from nabla import nabla, nabla_adjoint

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

    def __init__(self, A, Aadj,max_iter, lmbda, lmbda_m, tol, scale, verbose):
        super().__init__()
        self.max_iter = max_iter
        self.scale = scale
        self.lmbda = lmbda
        self.lmbda_m = lmbda_m
        self.tol = tol
        self.verbose = verbose
        self.A = A
        self.Aadj = Aadj
        

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
    
    def compute_cost(self, U, Y_H, Y_M,R):
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
        U_flat = U.view(1, 31, -1)
        RU = torch.matmul(R, U_flat.squeeze(0)).unsqueeze(0)
        RU_reshaped = RU.view(1, 1, U.shape[2], U.shape[3])
        data_term_m = 0.5 * self.lmbda_m * torch.norm(RU_reshaped-Y_M)**2
        
        # Terme de régularisation TV
        grad_U = nabla(U)
        tv_per_pixel = torch.sqrt(torch.sum(grad_U**2, dim=(1,4)))
        tv_term = self.lmbda * torch.sum(tv_per_pixel)
        
        return data_term_h + data_term_m + tv_term

    def grad_f(self, U, Y_H, Y_M ,R):
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
        U_flat = U.view(1, 31, -1)
        RU = torch.matmul(R, U_flat.squeeze(0)).unsqueeze(0)
        subtracted = RU.view(1, 1, U.shape[2], U.shape[3]) - Y_M
        grad2_flat = torch.matmul(R.t(), subtracted.view(1, 1, -1))
        grad2 = self.lmbda_m * grad2_flat.view(1, 31, U.shape[2], U.shape[3])
        
        return grad1 + grad2
    
    def proxg(self, x):
        """Opérateur proximal (à implémenter)."""
        raise NotImplementedError("L'opérateur proximal doit être implémenté")
        
    def forward(self, Y_H, Y_M,R):
        """
        Résout le problème d'optimisation complet.
        
        Args:
            Y_H (torch.Tensor): Données hyperspectrales [b,c,h//scale,w//scale]
            Y_M (torch.Tensor): Données panchromatiques [b,1,h,w]
            
        Returns:
            tuple: (Image estimée [b,c,h,w], historique des coûts)
        """
        U = self.Aadj(Y_H).clone()
        cost_history = []
        
        if self.verbose:
            print("\nDébut de l'optimisation:")
            print(f"{'It':<5} | {'Coût total':<12} | {'Data H':<12} | {'Data M':<12} | {'TV':<12} | {'ΔU':<12}")
            print("-" * 80)
        
        for it in range(self.max_iter):
            U_prev = U.clone()
            
            # Étape de gradient
            grad = self.grad_f(U, Y_H, Y_M,R)
            U = self.proxg(U - self.lmbda * grad)
            
            # Calcul des métriques
            total_cost = self.compute_cost(U, Y_H, Y_M,R)
            delta_U = torch.norm(U - U_prev).item() / (torch.norm(U_prev).item() + 1e-8)
            cost_history.append(total_cost.item())
            
            # Affichage conditionnel
            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1 or delta_U < self.tol):
                data_term_h = 0.5 * torch.norm(Y_H - self.A(U))**2
                RU = torch.matmul(R, U.view(1, 31, -1)).view(1, 1, *U.shape[2:])
                data_term_m = 0.5 * self.lmbda_m * torch.norm(Y_M - RU)**2
                tv_term = self.lmbda * torch.sum(torch.norm(nabla(U), dim=-1))
                
                print(f"{it:<5} | {total_cost.item():<12.3e} | {data_term_h.item():<12.3e} | "
                      f"{data_term_m.item():<12.3e} | {tv_term.item():<12.3e} | {delta_U:<12.3e}")
                
                if delta_U < self.tol:
                    print(f"\nConvergence atteinte à l'itération {it} (ΔU = {delta_U:.3e} < {self.tol})")
                    break
        
        return U, cost_history