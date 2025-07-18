from pan_gradient_prox import PANProximalGradient
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
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        params['p'] = self.p
        params['q'] = self.q
        params['r'] = self.r

        self.optim = TVPrior(**params)


    def proxg(self,x):
        """
        Donne l'opérateur proximale de la tv vectorielle avec chamboll pock ....
        
        """
        params = {}

        params['compute_L'] = {'nband': x.shape[1]}
        params['K'] = {}
        params['K_adjoint'] = {}
        params['prox_sigma_g_conj'] = {}
        params['prox_tau_f'] = {'y': x, 'sigma2': self.step_size*self.lmbda}
        params['loss_fn'] = {}


        return self.optim(x,init=None, verbose=False, params=params, return_loss=False)
    

   