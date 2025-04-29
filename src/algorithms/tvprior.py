
import torch
from chambolle_pock import ChambollePock
from math import sqrt
from nabla import nabla, nabla_adjoint



class TVPrior(ChambollePock):

    def __init__(self, p, q, r, *args, **kwargs):
        super(TVPrior, self).__init__(*args, **kwargs)
        self.p = p 
        self.q = q
        self.r = r

 

    def compute_L(self, nband):
        return sqrt(8)*self.lmbda*nband
    

    def K(self, u, **kwargs):
        """
            Define the linear operator associated to the primal dual formulation of the problem

            $$
            Ku(x,y) = \sum_{sigma} \nabla(u(x,y,\sigma)) 
            $$

            :param Torch tensor u: Input hyper-spectral tensor of shape (batch, channels, height, width)
            :return: The output of the linear operator

        """

        return nabla(u) 

    def K_adjoint(self, q):

        """
            Define the adjoint operator associated to the primal dual formulation of the problem

            $$
                K^*q = (\nabla^*q, \nabla^*q, ..., \nabla^*q)^t
            $$

            :param Torch tensor q: Input tensor of shape (batch, height, width, 2)
            :return: The output of the adjoint operator

        """

        return nabla_adjoint(q)
        
        

    def prox_tau_f(self, u, tau, y, sigma2=1):
        r"""
        Proximal operator of the function :math:`\frac{1}{2\sigma^2}\|x-y\|_2^2`.
        """
        return (sigma2*u + tau * y) / (sigma2 + tau)


    # def prox_sigma_g_conj(self, q, sigma):
    #     r"""
        
    #     Proximal operator of TV.


    #     """

    #     return q / torch.maximum(torch.norm(q, dim=-1, keepdim=True), torch.ones_like(q))
    


    def prox_sigma_g_conj(self, U, eps=1e-8):
        """
        Projection sur la boule duale l^{p*,q*,r*} <= 1.
        Gère explicitement p*, q*, r* = infinity.
        """

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
    
        

    def loss_fn(self, u, y, lmbda):
        r"""
        Compute the loss function of the problem
        """
        f = lambda u: 0.5*torch.norm(u - y)**2
        reg = lambda u: lmbda*torch.sum(
                                torch.norm(nabla(u), dim=-1)
                                )
        return f(u) + reg(u)
    


