from pan_gradient_prox import PANProximalGradient
from grad_proj import GradProj

class PANTVGP(PANProximalGradient):
    """
    Implémentation du pansharpening avec régularisation TV vectorielle utilisant l'algorithme de gradient projeté.
    Hérite de PANProximalGradient et remplace seulement le calcul de l'opérateur proximal.
    
    Attributes:
        optim (GradProj): Instance du solveur de gradient projeté pour la TV vectorielle

    Methods:
        proxg(x): Calcule l'opérateur proximal de la TV vectorielle via gradient projeté
    """

    def __init__(self, params, A, Aadj, spectral_op, spectral_op_t, 
                 max_iter=2000, lmbda=1, lmbda_m=1, tol=1e-7, 
                 scale=4, p=1, q=1, r=1, verbose=True):
        """
        Initialise le solveur avec les paramètres donnés.
        
        Args:
            params (dict): Paramètres pour GradProj
            A, Aadj: Opérateurs de sous-échantillonnage et son adjoint
            spectral_op, spectral_op_t: Opérateurs spectraux et son adjoint
            max_iter: Nombre max d'itérations
            lmbda: Paramètre de régularisation TV
            lmbda_m: Poids de l'attache aux données MS
            tol: Tolérance de convergence
            scale: Facteur d'échelle
            p, q, r: Paramètres de la norme TV vectorielle
            verbose: Affichage des informations
        """
        # Initialisation de la classe parente
        super().__init__(A=A, Aadj=Aadj, spectral_op=spectral_op, spectral_op_t=spectral_op_t,
                        max_iter=max_iter, lmbda=lmbda, lmbda_m=lmbda_m, tol=tol,
                        scale=scale, p=p, q=q, r=r, verbose=verbose)
        
        # Configuration du solveur GradProj
        gp_params = {
            'max_iter_gp': params.get('max_iter_gp', 2000),
            'lmbda': self.lmbda,  # Utilise le même lambda que pour le pansharpening
            'tau': params.get('tau', 0.08),
            'tol': self.tol,
            'p': self.p,
            'q': self.q,
            'r': self.r,
            'verbose': False
        }
        
        self.optim = GradProj(**gp_params)

    def proxg(self, x):
        """
        Calcule l'opérateur proximal de la TV vectorielle via gradient projeté.
        
        Args:
            x (torch.Tensor): Image d'entrée [b,c,h,w]
            
        Returns:
            torch.Tensor: Résultat du proximal [b,c,h,w]
        """
        return self.optim(x, verbose=False, return_loss=False)