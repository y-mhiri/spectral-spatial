import torch
from .nabla import nabla
from .pan_proximal_gradient import PANProximalGradient
from .prox_tv_grad_align import TVGradAlignment

class PANCTVGradAlignment(PANProximalGradient):
    """
    Calcul de l'opérateur proximale de la TV vectorielle en utilisant chamboll pock.
    Attributs:
        solver (class): résoud l'algo de chamboll pock avec les paramétre nécessaire.

    Methods:
    proxg(input_image)
       cette fonction donne l'opérateur proximale de l'image d'entrée en faisant un algo de chamboll pock

    """
    def __init__(self, init_params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        init_params['p'] = self.p
        init_params['q'] = self.q
        init_params['r'] = self.r
 

        self.optim = TVGradAlignment(**init_params)
        self.W = self.optim.W


    def proxg(self,x, gamma=1):
        """
        Donne l'opérateur proximale de la tv vectorielle avec chamboll pock ....
        
        """
        params = {}

        params['compute_L'] = {'nband': x.shape[1]}
        params['K'] = {}
        params['K_adjoint'] = {}
        params['prox_sigma_g_conj'] = {}
        params['prox_tau_f'] = {'y': x, 'sigma2': 1}
        params['loss_fn'] = {}

        # self.optim.lmbda = self.optim.lmbda * gamma
        return self.optim(x,init=None, verbose=False, params=params, return_loss=False)
    


    def compute_cost(self, U, Y_H, Y_M):
        """
        Calcule le coût total de la fonction objective.
        
        Args:
            U (torch.Tensor): Image estimée [b,c,h,w]
            Y_H (torch.Tensor): Données hyperspectrales [b,c,h//scale,w//scale]
            Y_M (torch.Tensor): Données panchromatiques [b,1,h,w]
            
        Returns:
            float: Coût total
        """
        # Terme d'attache aux données hyperspectrales
        data_term_h = 0.5 * torch.norm(self.A(U)-Y_H)**2
        
        # Terme d'attache aux données panchromatiques
        
        data_term_m = 0.5 * self.lmbda_m * torch.norm(self.spectral_op(U)-Y_M)**2
        
        # Terme de régularisation TV
        grad_U = torch.matmul(self.W,nabla(U).unsqueeze(-1)).squeeze(-1)
        #tv_per_pixel = torch.sqrt(torch.sum(grad_U**2, dim=(1,4)))
        #torch.sum(tv_per_pixel)
        tv_term = self.lmbda * self.ctv_norm(grad_U,eps=1e-8)
        
        return data_term_h + data_term_m + tv_term ,data_term_h,data_term_m,tv_term
    

