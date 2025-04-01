from gradient_prox import PANProximalGradient
from tvprior import TVPrior

class PANTVCB(PANProximalGradient):
    """ 
     
    ....
    
    """
    def __init__(self, A, Aadj, R, max_iter, lmbda, lmbda_m, tau, tol, scale, verbose, **params):
        super().__init__(A, Aadj, R, max_iter, lmbda, lmbda_m, tau, tol, scale, verbose)

        self.solver = TVPrior(**params)


    def proxg(self,x):
        """
        ....
        
        """
        return self.solver(x)