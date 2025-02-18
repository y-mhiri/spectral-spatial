
import torch
from chambolle_pock import ChambollePock
from math import sqrt
from nabla import nabla, nabla_adjoint



class TVPrior(ChambollePock):

    def __init__(self, *args, **kwargs):
        super(TVPrior, self).__init__(*args, **kwargs)

 

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


    def prox_sigma_g_conj(self, q, sigma):
        r"""
        
        Proximal operator of TV.


        """

        return q / torch.maximum(torch.norm(q, dim=-1, keepdim=True), torch.ones_like(q))
        

    def loss_fn(self, u, y, lmbda):
        r"""
        Compute the loss function of the problem
        """
        f = lambda u: 0.5*torch.norm(u - y)**2
        reg = lambda u: lmbda*torch.sum(
                                torch.norm(nabla(u), dim=-1)
                                )
        return f(u) + reg(u)
    


