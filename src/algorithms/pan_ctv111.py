from pan_gradient_prox import PANProximalGradient
from nabla import nabla 
import torch




class PANTVCTV111(PANProximalGradient):
    """
    Calcul de l'opérateur proximal de la TV vectorielle par descente de gradient projeté.
    
    Attributs:
        max_iter_gp (int): Nombre maximal d'itérations pour la descente de gradient
        tau (float): Pas de descente
    """

    def __init__(self, A, Aadj,spectral_op,spectral_op_t, max_iter, lmbda, lmbda_m, tol,scale,p,q,r,verbose):
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




    def proxg(self,U):
     U = nabla (U)
     return torch.sign(U) * torch.maximum(torch.abs(U) - self.lmbda, torch.tensor(0.0, device=U.device))
    
        