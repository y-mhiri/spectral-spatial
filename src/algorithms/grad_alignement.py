import torch

from chambolle_pock import ChambollePock
from math import sqrt
from torch.linalg import svd
from nabla import nabla, nabla_adjoint



class GradAlignement(ChambollePock):

    def __init__(self, *args, **kwargs):
        super(GradAlignement, self).__init__(*args, **kwargs)

 

    def compute_L(self, nband):
        return sqrt(8*nband)*self.lmbda #sqrt(band)?
    

    def K(self, u, **kwargs):
        """
            Define the linear operator associated to the primal dual formulation of the problem

            $$
            Ku(x,y) = \sum_{sigma} \nabla(u(x,y,\sigma)) 
            $$

            :param Torch tensor u: Input hyper-spectral tensor of shape (batch, channels, height, width)
            :return: The output of the linear operator

        """

        grads = nabla(u) # Compute the gradient of u 

        # sum along the channel axis
        # return torch.sum(grads, dim=1).unsqueeze(1)
        return grads

    def K_adjoint(self, q):

        """
            Define the adjoint operator associated to the primal dual formulation of the problem

            $$
                K^*q = (\nabla^*q, \nabla^*q, ..., \nabla^*q)^t
            $$

            :param Torch tensor q: Input tensor of shape (batch, height, width, 2)
            :return: The output of the adjoint operator

        """

        # q_adj = nabla_adjoint(q)
        # return torch.repeat_interleave(q_adj, nband, dim=1) # repeat the input tensor along the channel axis and apply the adjoint of the gradient operator
        return nabla_adjoint(q)
        

    def prox_tau_f(self, u, tau, y, sigma2=1):
        r"""
        Proximal operator of the function :math:`\frac{1}{2\sigma^2}\|x-y\|_2^2`.
        """
        return (sigma2*u + tau * y) / (sigma2 + tau)


    def prox_sigma_g_conj(self, q, sigma, alpha, eps=1e-7):
        r"""
        
        Proximal operator of the indicator function of the set :math:`\{q \mid \|q\|_2 = \|\alpha\|_2 , <q,\alpha>=0\}`.


        """

        norm_alpha = torch.norm(alpha, dim=-1, keepdim=True)
        # norm_alpha = torch.repeat_interleave(norm_alpha, q.shape[-1], dim=-1)

        q_orth = q - alpha*torch.sum(q*alpha, dim=-1).unsqueeze(-1) / (norm_alpha**2 + eps)
        
        norm_qorth = torch.norm(q_orth, dim=-1, keepdim=True)
        # norm_qorth = torch.repeat_interleave(norm_qorth, q.shape[-1], dim=-1)

        q_orth = q_orth / (torch.maximum(norm_qorth/(norm_alpha + eps), torch.ones_like(q_orth)))

        return q_orth
        

    def loss_fn(self, u, y, lmbda, grad_panc, sigma2=1): #dépendance en sigma2
        r"""
        Compute the loss function of the problem
        """
        f = lambda u: (1/(2*sigma2))*torch.norm(u - y)**2
        # reg = lambda u: lmbda*torch.sum(
        #                         torch.sqrt(
        #                             torch.abs(
        #                             torch.norm(nabla(u), dim=-1)**2 * torch.norm(grad_panc, dim=-1)**2 
        #                             - (torch.sum(nabla(u)*grad_panc, dim=-1))**2
        #                             )
        #                         )
        #                         )

        grad_panc_orth = torch.zeros_like(grad_panc).to(grad_panc.device).type(grad_panc.dtype)
        grad_panc_orth[...,0] = grad_panc[...,1]
        grad_panc_orth[...,1] = -grad_panc[...,0]

        reg = lambda u: lmbda*torch.sum(torch.abs(torch.sum(nabla(u)*grad_panc_orth, dim=-1)))

        return f(u) + reg(u)
    

    def hsi_viz(x):

        x_mat = x.reshape(x.shape[1], -1)

        U, s, V = svd(x_mat, full_matrices=False)
        

        Z_mat = torch.diag(s) @ V
        Z = Z_mat.reshape(x.shape)

        return Z, s 

    

