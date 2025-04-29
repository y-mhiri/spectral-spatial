from pan_gradient_prox import PANProximalGradient
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

    def __init__(self, A, Aadj,spectral_op,spectral_op_t, max_iter, lmbda, lmbda_m, tau, tol, scale, verbose,max_iter_gp,p,q,r):
        """
        Initialise les paramètres de l'algorithme.
        
        Args:
            max_iter_gp (int): Nombre max d'itérations pour la sous-optimisation
            tau (float): Pas de descente pour le gradient projeté
        """
        super().__init__(A, Aadj,spectral_op,spectral_op_t,max_iter, lmbda, lmbda_m, tol, scale,p,q,r,verbose)
        self.max_iter_gp = max_iter_gp
        self.tau = tau
        self.p = p 
        self.q = q 
        self.r = r

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
    

    def project_dual_ball(self,A, p_star, q_star, r_star):
        """Projection sur la boule duale l^{p*,q*,r*} <= 1."""
        # Normalisation par la norme duale
        norm_p_star = torch.sum(torch.abs(A)**p_star, dim=1, keepdim=True)**(1/p_star)
        norm_q_star = torch.sum(norm_p_star**q_star, dim=-1, keepdim=True)**(1/q_star)
        norm_r_star = torch.sum(norm_q_star**r_star, dim=(2,3), keepdim=True)**(1/r_star)
        
        # Scaling pour respecter la contrainte
        scaling = torch.maximum(torch.tensor(1.0, device= A.device), norm_r_star)
        return A / scaling
    

    def project_dual_ball_r(self, U, p_star, q_star, r_star, eps=1e-8):
        """
        Projection sur la boule duale l^{p*,q*,r*} <= 1.
        Gère explicitement p*, q*, r* = infinity.
        """
        # Étape 1: Norme p* sur les canaux (axis=1)
        if torch.isinf(torch.tensor(p_star, device=U.device)):
            norm_p_star= torch.amax(torch.abs(U), dim=1, keepdim=True) # l^infini
        else:
            norm_p_star = torch.sum(torch.abs(U)**p_star, dim=1, keepdim=True)**(1/(p_star + eps))

        # Étape 2: Norme q* sur les dérivées (axis=-1)
        if torch.isinf(torch.tensor(q_star, device=U.device)):
            norm_q_star= torch.amax(torch.abs(norm_p_star), dim=-1, keepdim=True)   # l^infini
        else:
            norm_q_star = torch.sum(norm_p_star**q_star, dim=-1, keepdim=True)**(1/(q_star + eps))

        # Étape 3: Norme r* sur les pixels (axis=(2,3))
        if torch.isinf(torch.tensor(r_star, device=U.device)):
            norm_r_star= torch.amax(torch.abs(norm_q_star), dim=(2,3), keepdim=True)  # l^infini
        else:
            norm_r_star = torch.sum(norm_q_star**r_star, dim=(2,3), keepdim=True)**(1/(r_star + eps))

        # Scaling pour respecter ||U||_{p*,q*,r*} <= 1
        scaling = torch.maximum(torch.tensor(1.0, device=U.device), norm_r_star)
        return U / (scaling + eps)
    

    def prox_ctv(self,x):
        """Opérateur proximal pour la norme CTV l^p,q,r."""
        # Étape 1: Calcul de la projection duale
        # Calcul des exposants duaux (gère p,q,r=1 et p,q,r=infini)
        def get_dual_exponent(val):
            if val == 1:
                return torch.inf
            elif torch.isinf(torch.tensor(val)):
                return 1.0
            else:
                return 1 / (1 - 1/val)

        p_star = get_dual_exponent(self.p)
        q_star = get_dual_exponent(self.q)
        r_star = get_dual_exponent(self.r)  # Dual exponents

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
            w = self.project_dual_ball_r(y - self.tau * grad_z, p_star, q_star, r_star)
            
            # Mise à jour de l'accélération FISTA
            t_prev = t
            t = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = w + ((t_prev - 1) / t) * (w - w_prev)
            
            # Critère de convergence
            if self.convergence_criteria(w, w_prev):
                break
            
        return w
    

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
        z = self.prox_ctv(x)
        return x + self.lmbda * divergence(z)