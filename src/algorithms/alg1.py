
from gradient_prox import PANProximalGradient
import torch

from nabla import nabla, nabla_adjoint

class PANTVGradProj(PANProximalGradient):
    """
    
    ....
    
    """
    def __init__(self, A, Aadj, R, max_iter, lmbda, lmbda_m, tau, tol, scale, verbose, max_iter_gp):
        super().__init__(A, Aadj, R, max_iter, lmbda, lmbda_m, tau, tol, scale, verbose)

        self.max_iter_gp = max_iter_gp


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
    
