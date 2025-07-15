from pan_gradient_prox import PANProximalGradient 
from nabla import nabla,nabla_adjoint
import torch




class PANTVCTV(PANProximalGradient):
    """
    Calcul de l'opérateur proximal de la TV vectorielle par descente de gradient projeté.
    
    Attributs:
        max_iter_gp (int): Nombre maximal d'itérations pour la descente de gradient
        tau (float): Pas de descente
    """

    def __init__(self, A, Aadj,spectral_op,spectral_op_t, max_iter, lmbda, lmbda_m, tol, scale,p ,q ,r,verbose):
        """
        Initialise les paramètres de l'algorithme.
        
        Args:
            max_iter_gp (int): Nombre max d'itérations pour la sous-optimisation
            tau (float): Pas de descente pour le gradient projeté
        """
        super().__init__(A, Aadj,spectral_op,spectral_op_t,max_iter, lmbda, lmbda_m, tol, scale,p,q,r,verbose)
        self.p = p 
        self.q = q
        self.r = r
        
    
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


    def proxg(self, x):
        """Opérateur proximal pour la norme CTV l^p,q,r avec gestion de p,q,r=infini."""
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
        r_star = get_dual_exponent(self.r)

        # Projection duale + formule de Moreau
        grad = nabla(x)
        x_tilde = grad / self.lmbda
        proj = self.project_dual_ball_r(x_tilde, p_star, q_star, r_star)
        return x - self.lmbda * nabla_adjoint(proj)