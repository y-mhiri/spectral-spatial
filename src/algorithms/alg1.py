from pan_gradient_prox import PANProximalGradient
import torch
from gradient import gradient, divergence

class PANTVGradProj(PANProximalGradient):
    """
    Implémentation simplifiée de l'opérateur proximal de la TV vectorielle avec :
    - Descente de gradient projeté basique
    - Pas fixe (sans line search)
    - Support des normes mixtes l^p,q,r
    """
    
    def __init__(self, A, Aadj, spectral_op, spectral_op_t, max_iter, lmbda, lmbda_m, tau, tol, scale, verbose, max_iter_gp, p, q, r):
        super().__init__(A, Aadj, spectral_op, spectral_op_t, max_iter, lmbda, lmbda_m, tol, scale, p, q, r, verbose)
        self.max_iter_gp = max_iter_gp  # Max iterations for gradient projection
        self.tau = tau                  # Fixed step size
        self.p = p                      # Norm parameters
        self.q = q
        self.r = r

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

    def prox_ctv(self, x):
        """Opérateur proximal simplifié avec pas fixe."""
        # Calcul des exposants duaux
        def get_dual_exponent(val):
            return torch.inf if val == 1 else (1.0 if torch.isinf(torch.tensor(val)) else 1/(1-1/val))
        
        p_star = get_dual_exponent(self.p)
        q_star = get_dual_exponent(self.q)
        r_star = get_dual_exponent(self.r)

        # Initialisation
        b, c, h, w = x.shape
        w = torch.zeros((b, c, h, w, 2), device=x.device, dtype=x.dtype)

        for i in range(self.max_iter_gp):
            w_prev = w.clone()
            div_w = divergence(w)
            
            # Calcul du gradient
            grad_f = -2 * gradient(div_w + x / self.lmbda)
            
            # Mise à jour avec pas fixe et projection
            w = self.project_dual_ball_r(w - self.tau * grad_f, p_star, q_star, r_star)
            
            # Critère de convergence simple
            if i > 0 and torch.norm(w - w_prev) < self.tol * (1 + torch.norm(w_prev)):
                if self.verbose:
                    print(f"CTV proximal converged at iter {i}")
                break
                
        return w

    def proxg(self, x):
        """Application du proximal TV complet (inchangé)."""
        z = self.prox_ctv(x)
        return x + self.lmbda * divergence(z)