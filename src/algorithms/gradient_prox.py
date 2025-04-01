import sys
sys.path.append('src/datasets')
sys.path.append('src/algorithms')

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from nabla import nabla, nabla_adjoint
from pansharpening import PANDataset

class ProximalGradient(nn.Module):
    """
    Algorithme de gradient proximal pour résoudre le problème de pansharpening hyperspectral.
    se que l'image de sortie sera blanche aussi
    Attributs:
        max_iter (int): Nombre maximal d'itérations
        lmbda (float): Paramètre de régularisation pour la variation totale
        lmbda_m (float): Paramètre de régularisation pour l'attache aux données multispectrales
        tau (float): Pas de descente
        tol (float): Critère de convergence
        A (function): Opérateur de sous-échantillonnage plus flou gaussien
        Aadj (function): Opérateur adjoint de A
        scale (int): Facteur d'échelle
        verbose (bool): Affichage des informations
    """

    def __init__(self, A, Aadj, max_iter, lmbda, lmbda_m, tau, tol, scale, verbose):
        super(ProximalGradient, self).__init__()
        self.max_iter = max_iter
        self.scale = scale
        self.lmbda = lmbda
        self.lmbda_m = lmbda_m
        self.tau = tau
        self.tol = tol
        self.verbose = verbose
        self.A = A
        self.Aadj = Aadj

    def convergence_criteria(self, U0, U1):
        """
        Critère de convergence comparant la norme avec la valeur de tolérance.
        
        Args:
            U1 (torch.Tensor): Image estimée à l'itération i [b,c,h,w]
            U0 (torch.Tensor): Image estimée à l'itération i-1 [b,c,h,w]
            
        Returns:
            bool: True si la condition est vérifiée, False sinon
        """
        return (torch.linalg.norm(U1-U0)/torch.linalg.norm(U0)) < self.tol
    
    def grad_f(self, U, Y_H, Y_M):
        """
        Calcule le gradient de la fonction f(U).
        
        Args:
            U (torch.Tensor): Image hyperspectrale estimée [b,c,h,w]
            Y_H (torch.Tensor): Image hyperspectrale basse résolution [b,c,h//scale,w//scale]
            Y_M (torch.Tensor): Image panchromatique [b,1,h,w]
            
        Returns:
            torch.Tensor: Gradient combiné [h,w,c]
        """
        R = PANDataset.spectral(U)
        #R_T = R.t()
        
        # Terme 1: Gradient de 1/2 ||Y_H - A(U)||_F^2
        grad1 = self.Aadj((self.A(U) - Y_H))

        # Terme 2: Gradient de (lambda_m/2) ||Y_M - R H U||_F^2
        #U_flat = U.view(1, 31, -1)
        #RU = torch.matmul(R, U_flat.squeeze(0)).unsqueeze(0) 
        #RU_reshaped = RU.view(1, 1, 1040, 1392)
        #subtracted = RU_reshaped - Y_M
        #subtracted_flat = subtracted.view(1, 1, -1).float()
        #grad2_flat = torch.matmul(R_T, subtracted_flat)
        #grad2 = self.lmbda_m * grad2_flat.view(1, 31, 1040, 1392)
        #R = PANDataset.spectral(U)  # [1, C]
        residual_pan = torch.einsum('ij,jklm->iklm', R, U) - Y_M  # [1,1,h,w]
        grad2 = self.lmbda_m * torch.einsum('ij,jklm->iklm', R.T, residual_pan)

        return grad1 + grad2
    
    def proj(self, z):
        """
        Projection sur la boule unité pour la norme l221.
        
        Args:
            z (torch.Tensor): Tenseur [b,c,h,w,2]
            
        Returns:
            torch.Tensor: Image projetée [b,c,h,w,2]
        """
        return z / torch.maximum(torch.norm(z, dim=-1, keepdim=True), torch.ones_like(z))
    
    def grad_proj(self, x):
        """
        Descente de gradient projeté.
        
        Args:
            x (torch.Tensor): Tenseur [b,c,h,w]
            
        Returns:
            torch.Tensor: Minimum [b,c,h,w]
        """
        b, c, h, w = x.shape
        z0 = torch.ones((b, c, h, w, 2))
        
        for i in range(self.max_iter):
            grad_z = -2 * nabla(nabla_adjoint(z0) + x / self.lmbda)
            z = self.proj(z0 - self.tau * grad_z)
            
            if self.convergence_criteria(z, z0):
                break
                
            z0 = z
            
        return z
    
    def proxg(self, x):
        """
        Opérateur proximal de x.
        
        Args:
            x (torch.Tensor): Tenseur [b,c,h,w]
            
        Returns:
            torch.Tensor: Image projetée [b,c,h,w]
        """
        z = torch.clone(x)
        z = self.grad_proj(z)
        return x + self.lmbda * nabla_adjoint(z)
    
    def forward(self, Y_H, Y_M):
        """
        Résout le problème d'optimisation.
        
        Args:
            Y_H (torch.Tensor): Image hyperspectrale basse résolution [b,c,h//scale,w//scale]
            Y_M (torch.Tensor): Image panchromatique [b,1,h,w]
            
        Returns:
            torch.Tensor: Image estimée [b,c,h,w]
        """
        # Initialisation
        #Y_H = (Y_H - Y_H.min()) / (Y_H.max() - Y_H.min())
        #Y_M = (Y_M - Y_M.min()) / (Y_M.max() - Y_M.min())
        #_, _, h, w = Y_M.shape
        #b, c, _, _ = Y_H.shape
        #U = torch.zeros((b, c, h, w))
        U = self.Aadj(Y_H).clone()

        
        for it in tqdm(range(self.max_iter), disable=not self.verbose):
            # Gradient de f(U)
            grad_U = self.grad_f(U, Y_H, Y_M)
            
            # Mise à jour de U
            U_new = self.proxg(U - self.lmbda * grad_U)
            
            # Calcul des termes du coût
            # 1. Terme d'attache aux données hyperspectrales
            data_term_h = 0.5 * torch.norm(Y_H - self.A(U_new))**2
            
            # 2. Terme d'attache aux données panchromatiques
            R = PANDataset.spectral(U_new)
            #U_flat = U_new.view(1, 31, -1)
            #RU = torch.matmul(R, U_flat.squeeze(0)).unsqueeze(0)
            #RU_reshaped = RU.view(1, 1, 1040, 1392)
            data_term_m = 0.5 * self.lmbda_m * torch.norm(Y_M - torch.einsum('ij,jklm->iklm', R, U))**2
            
            # 3. Terme de régularisation TV
            grad_U = nabla(U_new)  # [b,c,h,w,2]
            tv_per_pixel = torch.sqrt(torch.sum(grad_U**2, dim=(1,4)))  # [b,h,w]
            tv_term = self.lmbda * torch.sum(tv_per_pixel)
            
            # Coût total
            total_cost = data_term_h + data_term_m + tv_term
            
            # Affichage du coût
            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1):
                print(f"Iter {it:4d} | Coût total: {total_cost.item():.3e} | "
                      f"Data H: {data_term_h.item():.3e} | "
                      f"Data M: {data_term_m.item():.3e} | "
                      f"TV: {tv_term.item():.3e}")
            
            # Vérification de la convergence
            if it > 0 and self.convergence_criteria(U_new, U):
                if self.verbose:
                    print(f"Convergence atteinte à l'itération {it}")
                break
                
            U = U_new
        
        return U