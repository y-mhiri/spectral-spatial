from pan_gradient_prox import PANProximalGradient
import torch
from gradient import gradient, divergence
import numpy as np

class PANTVGradProj(PANProximalGradient):
    """
    Implémentation de l'opérateur proximal de la TV vectorielle avec :
    - Descente de gradient projeté accélérée (FISTA)
    - Recherche linéaire adaptative (Armijo line search)
    - Support des normes mixtes l^p,q,r
    """
    
    def __init__(self, A, Aadj, spectral_op, spectral_op_t, max_iter, lmbda, lmbda_m, tau, tol, scale, verbose, max_iter_gp, p, q, r):
        super().__init__(A, Aadj, spectral_op, spectral_op_t, max_iter, lmbda, lmbda_m, tol, scale, p, q, r, verbose)
        self.max_iter_gp = max_iter_gp  # Max iterations for gradient projection
        self.tau = tau                  # Initial step size
        self.p = p                      # Norm parameters
        self.q = q
        self.r = r
        
        # Line search parameters
        self.alpha = 0.5     # Step reduction factor
        self.beta = 0.3      # Sufficient decrease parameter
        self.max_ls = 10      # Max line search attempts

    def proj(self, z):
        """Projection sur la boule unité l2."""
        return z / torch.maximum(torch.norm(z, dim=-1, keepdim=True), torch.ones_like(z))

    def project_dual_ball_r(self, U, p_star, q_star, r_star):
        """Projection sur la boule duale l^{p*,q*,r*} <= 1 avec gestion des infinis."""
        # Étape 1: Norme p* sur les canaux
        if torch.isinf(torch.tensor(p_star)):
            norm_p_star = torch.amax(torch.abs(U), dim=1, keepdim=True)
        else:
            norm_p_star = torch.sum(torch.abs(U)**p_star, dim=1, keepdim=True)**(1/p_star)

        # Étape 2: Norme q* sur les dérivées
        if torch.isinf(torch.tensor(q_star)):
            norm_q_star = torch.amax(norm_p_star, dim=-1, keepdim=True)
        else:
            norm_q_star = torch.sum(norm_p_star**q_star, dim=-1, keepdim=True)**(1/q_star)

        # Étape 3: Norme r* sur les pixels
        if torch.isinf(torch.tensor(r_star)):
            norm_r_star = torch.amax(norm_q_star, dim=(2,3), keepdim=True)
        else:
            norm_r_star = torch.sum(norm_q_star**r_star, dim=(2,3), keepdim=True)**(1/r_star)

        scaling = torch.maximum(torch.tensor(1.0, device=U.device), norm_r_star)
        return U / scaling

    def tv_objective(self, w, x):
        """Fonction objectif: ||div(w) + x/λ||²_F (norme de Frobenius au carré)"""
        div_w = divergence(w)
        return torch.sum((div_w + x / self.lmbda)**2)

    def prox_ctv(self, x):
        """Opérateur proximal avec line search adaptative."""
        # Calcul des exposants duaux
        def get_dual_exponent(val):
            return torch.inf if val == 1 else (1.0 if torch.isinf(torch.tensor(val)) else 1/(1-1/val))
        
        p_star = get_dual_exponent(self.p)
        q_star = get_dual_exponent(self.q)
        r_star = get_dual_exponent(self.r)

        # Initialisation FISTA
        b, c, h, w = x.shape
        w = torch.ones((b, c, h, w, 2), device=x.device, dtype=x.dtype)
        y = w.clone()
        t = 1.0
        tau = self.tau  # Initial step size

        for i in range(self.max_iter_gp):
            w_prev = w.clone()
            div_y = divergence(y)  # Pré-calcul
            
            # Gradient computation
            grad_f = -2 * gradient(div_y + x / self.lmbda)
            
            # Line search d'Armijo
            for _ in range(self.max_ls):
                w_candidate = self.project_dual_ball_r(y - tau * grad_f, p_star, q_star, r_star)
                
                # Calcul des termes pour Armijo
                f_y = self.tv_objective(y, x)
                f_w = self.tv_objective(w_candidate, x)
                grad_term = torch.sum(grad_f * (w_candidate - y))
                
                if f_w <= f_y + self.beta * grad_term:
                    break
                    
                tau *= self.alpha
            
            # Mise à jour des variables
            w = w_candidate
            t_prev = t
            t = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = w + ((t_prev - 1) / t) * (w - w_prev)
            
            # Critère de convergence
            if i > 0 and self.convergence_criteria(w_prev, w):
                if self.verbose:
                    print(f"CTV proximal converged at iter {i}")
                break
                
        return w

    def proxg(self, x):
        """Application du proximal TV complet."""
        z = self.prox_ctv(x)
        return x + self.lmbda * divergence(z)