from gradient_prox import PANProximalGradient
from tvprior import TVPrior

class PANTVCB(PANProximalGradient):
    """
    Calcul de l'opérateur proximale de la TV vectorielle en utilisant chamboll pock.
    Attributs:
        solver (class): résoud l'algo de chamboll pock avec les paramétre nécessaire.

    Methods:
    proxg(input_image)
       cette fonction donne l'opérateur proximale de l'image d'entrée en faisant un algo de chamboll pock

    """
    def __init__(self, A,Aadj,spectral_op,spectral_op_t, max_iter, lmbda, lmbda_m, tol, scale, verbose, params):
        super().__init__(A, Aadj,spectral_op,spectral_op_t ,max_iter, lmbda, lmbda_m, tol, scale, verbose)

        self.optim = TVPrior(**params)


    def proxg(self,x):
        """
        Donne l'opérateur proximale de la tv vectorielle avec chamboll pock ....
        
        """
        params = {}

        params['compute_L'] = {'nband': 31}
        params['K'] = {}
        params['K_adjoint'] = {}
        params['prox_sigma_g_conj'] = {}
        params['prox_tau_f'] = {'y': x, 'sigma2': 1}
        params['loss_fn'] = {}


        return self.optim(x,init=None, verbose=False, params=params, return_loss=False)
    

   