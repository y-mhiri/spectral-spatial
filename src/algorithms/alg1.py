from algorithms.pan_gradient_prox import PANProximalGradient
import torch
from gradient import gradient, divergence
import numpy as np


class PANTVGradProj(PANProximalGradient):
    """
    Calcul de l'opérateur proximal de la TV vectorielle par descente de gradient projeté.
    
    Attributs:
        max_iter_gp (int): Nombre maximal d'itérations pour la descente de gradient
        tau (float): Pas de descente
    """

    def __init__(self, A, Aadj,spectral_op,spectral_op_t, max_iter, lmbda, lmbda_m, tau, tol, scale, verbose, max_iter_gp):
        """
        Initialise les paramètres de l'algorithme.
        
        Args:
            max_iter_gp (int): Nombre max d'itérations pour la sous-optimisation
            tau (float): Pas de descente pour le gradient projeté
        """
        super().__init__(A, Aadj,spectral_op,spectral_op_t,max_iter, lmbda, lmbda_m, tol, scale, verbose)
        self.max_iter_gp = max_iter_gp
        self.tau = tau

    def proj(self, z):
        """
        Projection sur la boule unité.
        
        Args:
            z (torch.Tensor): Tenseur [b,c,h,w,2]
            
        Returns:
            torch.Tensor: Tenseur projeté [b,c,h,w,2]
        """
        #norm_z = torch.sqrt(torch.sum(z**2, dim=-1, keepdim=True))
        return z / torch.maximum(torch.norm(z, dim=-1, keepdim=True), torch.ones_like(z))
    

    def grad_proj(self, x):
        """
        Descente de gradient projeté avec accélération FISTA pour résoudre le sous-problème TV.
        
        Args:
            x (torch.Tensor): Tenseur [b,c,h,w]
            
        Returns:
            torch.Tensor: Solution [b,c,h,w,2]
    """
        b, c, h, w = x.shape
        w = torch.ones((b, c, h, w, 2), device=x.device, dtype=x.dtype)
        y = w.clone()
        t = 1.0
    
        # FISTA acceleration
        for i in range(self.max_iter_gp):
            w_prev = w.clone()
            
            # Calcul du gradient
            grad_z = -2 * gradient(divergence(y) + x / self.lmbda)
            
            # Mise à jour avec projection
            w = self.proj(y - self.tau * grad_z)
            
            # Mise à jour de l'accélération FISTA
            t_prev = t
            t = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = w + ((t_prev - 1) / t) * (w - w_prev)
            
            # Critère de convergence
            if self.convergence_criteria(w, w_prev):
                break
            
        return w

    def proxg(self, x):
        """
        Opérateur proximal avec régularisation TV.
        
        Args:
            x (torch.Tensor): Image d'entrée [b,c,h,w]
            
        Returns:
            torch.Tensor: Image régularisée [b,c,h,w]
        """
        z = self.grad_proj(x)
        return x + self.lmbda * divergence(z)