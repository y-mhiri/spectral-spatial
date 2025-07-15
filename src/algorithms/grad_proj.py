
"""
Defines a base class to implement gradient projetée type algorithms. 

"""
import torch
import numpy as np
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

    def __init__(self,max_iter_gp = 2000, lmbda=1,tau=0.08,tol = 1e-7,verbose = None,p = 1,q = 1,r = 1):
        super(GradProj, self).__init__()
        self.max_iter_gp = max_iter_gp
        self.lmbda = lmbda
        self.tau = tau
        self.tol = tol
        self.verbose = verbose
        self.p = p
        self.q = q
        self.r = r


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
    
    def project_dual_ball_r(self, U, p_star, q_star, r_star, eps=1e-8):
        """
        Projection sur la boule duale l^{p*,q*,r*} <= 1.
        Gère explicitement p*, q*, r* = infinity.
        """
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
    

    def get_dual_exponent(self,val):
            if val == 1:
                return torch.inf
            elif torch.isinf(torch.tensor(val)):
                return 1.0
            else:
                return 1 / (1 - 1/val)
    

    def prox_ctv(self,x):
        """Opérateur proximal pour la norme CTV l^p,q,r."""
        # Étape 1: Calcul de la projection duale
        # Calcul des exposants duaux (gère p,q,r=1 et p,q,r=infini)
        def get_dual_exponent(val):
            if val == 1:
                return torch.inf
            elif torch.isinf(torch.tensor(val)):
                return 1.0
            else:
                return 1 / (1 - 1/val)

        p_star = get_dual_exponent(self.p)
        q_star = get_dual_exponent(self.q)
        r_star = get_dual_exponent(self.r)  # Dual exponents

        b, c, h, w = x.shape
        w = torch.ones((b, c, h, w, 2), device=x.device, dtype=x.dtype)
        y = w.clone()
        t = 1.0
    
        # FISTA acceleration
        for i in range(self.max_iter_gp):
            w_prev = w.clone()
            
            # Calcul du gradient
            grad_z = -2 * gradient(divergence(y) + x / self.lmbda)
            
            # Mise à jour avec projection
            w = self.project_dual_ball_r(y - self.tau * grad_z, p_star, q_star, r_star)
            
            # Mise à jour de l'accélération FISTA
            t_prev = t
            t = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = w + ((t_prev - 1) / t) * (w - w_prev)
            
            # Critère de convergence
            if self.convergence_criteria(w, w_prev):
                break
            
        return w
    
    

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
        reg = lambda u: self.lmbda*self.ctv_norm(u)
        return f(u) + reg(u)
    


    def ctv_norm(self, U,eps=1e-8):
        """
        Calcule la norme CTV l^p,q,r avec support pour p,q,r = infini.
        
        Args:
            U (torch.Tensor): Tenseur de gradients [b,c,h,w,2]
            p, q, r (float or torch.inf): Exposants de la norme
            eps (float): Petite valeur pour stabilité numérique
            
        Returns:
            torch.Tensor: Norme CTV [b,1,1,1]
        """
        # Norme p sur les canaux (axis=1)
        if torch.isinf(torch.tensor(self.p)):
            norm_p = torch.amax(torch.abs(U), dim=1, keepdim=True)  # l^infini
        else:
            norm_p = torch.sum(torch.abs(U)**self.p, dim=1, keepdim=True)**(1/(self.p + eps))

        # Norme q sur les dérivées (axis=-1)
        if torch.isinf(torch.tensor(self.q)):
            norm_q = torch.amax(torch.abs(norm_p), dim=-1, keepdim=True)  # l^infini
        else:
            norm_q = torch.sum(norm_p**self.q, dim=-1, keepdim=True)**(1/(self.q + eps))

        # Norme r sur les pixels (axis=(2,3))
        if torch.isinf(torch.tensor(self.r)):
            norm_r = torch.amax(torch.abs(norm_q), dim=(2,3), keepdim=True)  # l^infini
        else:
            norm_r = torch.sum(norm_q**self.r, dim=(2,3), keepdim=True)**(1/(self.r + eps))

        return norm_r
    

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

        #b, c, h, w = y.shape
        W = torch.zeros_like(gradient(y))
        p_star = self.get_dual_exponent(self.p)
        q_star = self.get_dual_exponent(self.q)
        r_star = self.get_dual_exponent(self.r) 
        loss = torch.zeros(self.max_iter_gp)
        rel = torch.zeros(self.max_iter_gp)

        if verbose:
            print(f'gradient proximal algorithm starting...')
        # for it in tqdm(range(self.max_iter)):
        for it in range(self.max_iter_gp):
            
            u_old = torch.clone(u)

            grad = self.grad(W,y)
            
            W = self.project_dual_ball_r(W - self.tau * grad,p_star,q_star,r_star)

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

    
