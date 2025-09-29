import torch

from chambolle_pock import ChambollePock
from math import sqrt
from torch.linalg import svd, norm
from nabla import nabla, nabla_adjoint


class TVGradAlignement(ChambollePock):

    def __init__(self, grad_panc, thresh=0, weight_fun=None, *args, **kwargs):
        super(TVGradAlignement, self).__init__(*args, **kwargs)

        if weight_fun is None:
            self.weight_fun = lambda normp, mu: torch.stack((2 - torch.exp(-mu*normp), torch.exp(-mu*normp)), dim=-1).transpose(4,5).squeeze(-1)
        else:
            self.weight_fun = weight_fun # weight_fun should return weights of size [batch, channel, height, width, 2]

        self.W = self.weight(grad_panc, thresh=thresh)

    def compute_L(self, nband):
        return sqrt(8)*self.lmbda*nband #sqrt(band)?
    


    def weight(self,grad_panc, thresh):

        grad_panc_orth = torch.zeros_like(grad_panc).to(grad_panc.device).type(grad_panc.dtype)
        grad_panc_orth[...,0] = grad_panc[...,1]
        grad_panc_orth[...,1] = -grad_panc[...,0]
        norm_grad_panc = norm(grad_panc, dim=-1, keepdim=True) 

        grads = torch.stack((grad_panc_orth/(norm_grad_panc + 1e-7), grad_panc/(norm_grad_panc+ 1e-7)), dim=-1).transpose(-2,-1)
        thresh = thresh / (torch.mean(norm(grad_panc, ord=2, dim=-1)) + 1e-7)

        weights = self.weight_fun(norm_grad_panc, thresh).unsqueeze(-1) 

        return (weights*grads)

    def K(self, u, **kwargs):
        """
            Define the linear operator associated to the primal dual formulation of the problem

            $$
                Ku(x,y) = \sum_{sigma} W(x,y) \nabla(u(x,y,\sigma)) 
            $$

            :param Torch tensor u: Input hyper-spectral tensor of shape (batch, channels, height, width)
            :return: The output of the linear operator

        """

        grads = nabla(u) # Compute the gradient of u 
        return torch.matmul(self.W,grads.unsqueeze(-1)).squeeze(-1)

    def K_adjoint(self, q):

        """
            Define the adjoint operator associated to the primal dual formulation of the problem

            $$
                K^*q = ( \nabla^* W(x,y)^t q, \nabla^* W(x,y)^t q, ..., \nabla^* W(x,y)^t q)^t
            $$

            :param Torch tensor q: Input tensor of shape (batch, height, width, 2)
            :return: The output of the adjoint operator

        """

        return nabla_adjoint(torch.matmul(self.W.transpose(4,5),q.unsqueeze(-1)).squeeze(-1))        

    def prox_tau_f(self, u, tau, y, sigma2=1):
        r"""
        Proximal operator of the function :math:`\frac{1}{2\sigma^2}\|x-y\|_2^2`.
        """
        return (sigma2*u + tau * y) / (sigma2 + tau)


    def prox_sigma_g_conj(self, q, sigma, eps=1e-7):
        r"""
        
        Proximal operator of the indicator function of the set :math:`\{q \mid \|q\|_2 = \|\alpha\|_2 , <q,\alpha>=0\}`.


        """
        return q / torch.maximum(norm(q, dim=-1, keepdim=True), torch.ones_like(q))
    
        

    def loss_fn(self, u, y, lmbda, sigma2=1): #dépendance en sigma2
        r"""
        Compute the loss function of the problem
        """
        f = lambda u: (1/(2*sigma2))*norm(u - y)**2
        reg = lambda p: lmbda*torch.sum(norm(torch.matmul(self.W,nabla(p).unsqueeze(-1)).squeeze(-1),dim=-1))#.squeeze(-1), dim=-1)

        return f(u) + reg(u)
    

    def hsi_viz(x):

        x_mat = x.reshape(x.shape[1], -1)

        U, s, V = svd(x_mat, full_matrices=False)
        

        Z_mat = torch.diag(s) @ V
        Z = Z_mat.reshape(x.shape)

        return Z, s 

    
