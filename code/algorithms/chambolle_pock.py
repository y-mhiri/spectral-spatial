"""
Defines a base class to implement Chambolle-Pock type algorithms. 

"""
import torch
from torch import nn

class ChambollePock(nn.Module):
    """
    Chambolle-Pock algorithm for solving the optimization problem:
    min_u 1/2 ||Ku - y||^2 + lmbda * g(u)
    where g is a convex function and K is a linear operator.

    The class inherits from nn.Module and uses torch tensors.

    Attributes:
    - max_iter: maximum number of iterations
    - lmbda: regularization parameter
    - theta: relaxation parameter
    - sigma: step size for the primal variable u
    - tau: step size for the dual variable

    Methods:
    - K: linear operator K
    - K_adjoint: adjoint operator of K
    - prox_sigma_g_conj: proximal operator of sigma * g^*
    - prox_tau_f: proximal operator of tau * f
    - compute_L: compute the Lipschitz constant of K
    - loss_fn: loss function to be minimized
    - forward: run the Chambolle-Pock algorithm

    """

    def __init__(self, max_iter=100, lmbda=1, theta=1, sigma=.99, tau=0.99):
        super(ChambollePock, self).__init__()
        self.max_iter = max_iter
        self.lmbda = lmbda
        self.theta = theta
        self.sigma = sigma
        self.tau = tau
        self.L = None


    def K(self, u, **kwargs):
        """
            Define the linear operator associated to the primal dual formulation of the problem

            Parameters:
            - u: input tensor of shape (batch, channels, height, width)
            - kwargs: additional parameters
        """

        pass

    def K_adjoint(self, q, **kwargs):
        """
            Define the adjoint operator associated to the primal dual formulation of the problem

            Parameters:
            - q: input tensor of shape (batch, height, width, 2)
            - kwargs: additional parameters
        """
        pass


    def prox_sigma_g_conj(self, q, sigma, **kwargs):
        """
            Define the proximal operator of sigma * g^*

            Parameters:
            - q: input tensor of shape (batch, height, width, 2)
            - sigma: step size
            - kwargs: additional parameters
        """
        pass

    def prox_tau_f(self, u, tau, **kwargs):
        """
            Define the proximal operator of tau * f

            Parameters:
            - u: input tensor of shape (batch, channels, height, width)
            - tau: step size
            - kwargs: additional parameters
        """

        pass


    def compute_L(self, **kwargs):
        """
            Compute the Lipschitz constant of K

            Parameters:
            - kwargs: additional parameters
        """
        pass
    
    def loss_fn(self, u, y, lmbda, **kwargs):
        """
            Define the loss function to be minimized

            Parameters:
            - u: (estimate) input tensor of shape (batch, channels, height, width)
            - y: (observation) input tensor of shape (batch, channels, height, width)
            - lmbda: regularization parameter
            - kwargs: additional parameters
        """
        pass

    def forward(self, y, init=None, verbose=True, params={}):
        """
            Solve the optimization problem using the Chambolle-Pock algorithm

            Parameters:
            - y: input tensor of shape (batch, channels, height, width)
            - init: initial estimate. If None, set to K^*y
            - verbose: print the progress of the algorithm
            - params: dictionary of additional parameters

            Returns:
            - u: estimate of the solution
            - loss: loss function at each iteration

        """

        b,c,h,w = y.shape

        L = self.compute_L(**params['compute_L'])
        sigma = self.sigma/L
        tau = self.tau/L 


        if init is not None:
            u = torch.clone(init)
        else:
            u = torch.clone(y)

        q = self.K(u, **params['K'])
        v = torch.clone(u) 

  
        loss = torch.zeros(self.max_iter)

        print(f'Chambolle Pock algorithm starting...')
        for it in range(self.max_iter):
            
            u_old = torch.clone(u)

            q = self.prox_sigma_g_conj(q + self.lmbda * sigma * self.K(v,**params['K']), sigma, **params['prox_sigma_g_conj'])
            
            u = self.prox_tau_f(u - self.lmbda * tau * self.K_adjoint(q, **params['K_adjoint']), tau, **params['prox_tau_f'])

            v = u + self.theta * (u - u_old)

      
            loss[it] = self.loss_fn(u, y, self.lmbda, **params['loss_fn'])

            if verbose:
                print('Iteration: ', it, 'relative variation: ', torch.norm(u - u_old)/torch.norm(u_old))

                print('Cost function: ', loss[it])

        return u, loss
        

    