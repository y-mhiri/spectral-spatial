
"""
Defines a base class to implement gradient projetée type algorithms. 

"""
import torch
from torch import nn
from tqdm.auto import tqdm
from math import sqrt
from gradient import gradient ,divergence

class GradProj(nn.Module):
    """
    gradient projeté algorithm for solving the optimization problem:
    min_u 1/2 ||y-u||^2 + lmbda * g(u)
    where g is a convex function .

    The class inherits from nn.Module and uses torch tensors.

    Attributes:
    - max_iter: maximum number of iterations
    - lmbda: regularization parameter
    - tau: step size for the gradient projeté

    Methods:
    - gradF: compute the gradient of F
    - proj:  projection dans la boule unité
    - grad_projet : Descente de gradient projeté pour résoudre le sous-problème TV.
    - proxg: proximal operator of lambda * g 
    - loss_fn: loss function to be minimized
    - forward: run the proximal algorithm

    """

    def __init__(self,max_iter_gp = 50, lmbda=1,tau=0.08,tol = 1e-7,verbose = None):
        super(GradProj, self).__init__()
        self.max_iter_gp = max_iter_gp
        self.lmbda = lmbda
        self.tau = tau
        self.tol = tol
        self.verbose = verbose


    def grad(self,z0,x):

        return -2*gradient(divergence(z0) + x / self.lmbda )

    def proj(self, z):
        """
        Projection sur la boule unité.
        
        Args:
            z (torch.Tensor): Tenseur [b,c,h,w,2]
            
        Returns:
            torch.Tensor: Tenseur projeté [b,c,h,w,2]
        """
        #norm_z = torch.sqrt(torch.sum(z**2, dim=-1, keepdim=True))
        return z / torch.maximum(torch.norm(z, dim=-1, keepdim=True), torch.ones_like(z))
    

    def convergence_criteria(self, U0, U1):
        """
        Critère de convergence comparant la norme avec la valeur de tolérance.
        
        Args:
            U1 (torch.Tensor): Image estimée à l'itération i [b,c,h,w]
            U0 (torch.Tensor): Image estimée à l'itération i-1 [b,c,h,w]
            
        Returns:
            bool: True si la condition est vérifiée, False sinon
        """
        return (torch.linalg.norm(U1-U0)/torch.linalg.norm(U0)) < self.tol
    

    def loss_fn(self,u,y):
        r"""
        Compute the loss function of the problem
        """
        f = lambda u: 0.5*torch.norm(u-y)**2
        reg = lambda u: self.lmbda*torch.sum(
                                torch.norm(gradient(u), dim=-1)
                                )
        return f(u) + reg(u)
    

    def forward(self, y, init=None, verbose=True, return_loss=True):
        """
            Solve the optimization problem using the gradient proximal algorithm

            Parameters:
            - y: input tensor of shape (batch, channels, height, width)
            - init: initial estimate.
            - verbose: print the progress of the algorithm

            Returns:
            - u: estimate of the solution
            - loss: loss function at each iteration

        """


        if init is not None:
            u = torch.clone(init)

        else:
            u = torch.clone(y)

        b, c, h, w = y.shape
        W = torch.ones((b, c, h, w, 2), device=y.device, dtype=y.dtype)
        loss = torch.zeros(self.max_iter_gp)
        rel = torch.zeros(self.max_iter_gp)

        if verbose:
            print(f'gradient proximal algorithm starting...')
        # for it in tqdm(range(self.max_iter)):
        for it in range(self.max_iter_gp):
            
            u_old = torch.clone(u)

            grad = self.grad(W,y)
            
            W = self.proj(W - self.tau * grad)

            u = y + self.lmbda*(divergence(W))
            loss[it] = self.loss_fn(u,y)
            rel[it] = torch.norm(u - u_old)/torch.norm(u_old)


            if verbose:
                print('Iteration: ', it, 'relative variation: ', torch.norm(u - u_old)/torch.norm(u_old))

                print('Cost function: ', loss[it])

            
            if rel[it] < self.tol:
                print('Iteration: ', it, 'relative variation: ', torch.norm(u - u_old)/torch.norm(u_old))
                print(f'Converged after {it+1} iterations.')
                break

        if return_loss:
            return u, loss, rel
        else:
            return u

    
