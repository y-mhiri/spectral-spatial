import torch

from skimage.filters import threshold_otsu
from math import sqrt
from torch.linalg import svd, norm
from torch.nn.functional import sigmoid

from .chambolle_pock import ChambollePock
from .nabla import nabla, nabla_adjoint
from .dual_projections import select_dual_projection

def compute_alpha_from_pan(pan_image):
    """
    Calcule le seuil alpha à partir d'une image panchromatique
    en suivant exactement le code donné.
    
    Args:
        pan_image (torch.Tensor): Image panchromatique [1,1,H,W]
    
    Returns:
        float: seuil alpha
    """
    grad_norm = torch.norm(pan_image.squeeze(), dim=-1)  # Shape [H,W]
    
    # Critère c_n
    criterion = grad_norm / (grad_norm.sum() + 1e-7)
    c_n = criterion.cpu().numpy()
    
    alpha = threshold_otsu(c_n)
    
    return alpha


class GradientWeights:
    """
    Factory class to create weight functions for TVGradAlignment
    Supports both hard and soft thresholding strategies
    """
    @staticmethod
    def hard_threshold(epsilon=1e-7):
        """Hard thresholding weight function"""
        def weight_fn(c, alpha):
            return torch.stack((torch.ones_like(c),torch.where(c < alpha, torch.ones_like(c), torch.ones_like(c) * epsilon)
    ), dim=-1).transpose(-2, -1)
        return weight_fn

    @staticmethod
    def soft_threshold(tau=1.0):
        """Soft thresholding with sigmoid transition"""
        def weight_fn(c, alpha):
            return torch.stack((torch.ones_like(c), torch.ones_like(c) - sigmoid((c - alpha)/tau)), dim=-1).transpose(-2, -1)
        return weight_fn


class TVGradAlignment(ChambollePock):
    def __init__(self, grad_panc, p, q, r, 
                 threshold_softness=1.0, threshold=None,  # tune how smooth the sigmoid threshold is
                 *args, **kwargs):
        """
        Args:
            grad_panc: Panchromatic image gradients
            p, q, r: Norm parameters
            threshold_param: epsilon (for hard) or tau (for soft)
        """
        super().__init__(*args, **kwargs)

        self.weight_fun = GradientWeights.soft_threshold(tau=threshold_softness)

        # Compute threshold using Otsu on the gradient of the PAN image.
        self.alpha = self.compute_alpha_from_pan(grad_panc) if threshold is None else threshold    
        self.W = self._compute_weights(grad_panc)
        self.p = p
        self.q = q
        self.r = r

    def _compute_weights(self, grad_panc):
        """Compute the weight matrix from panchromatic gradients"""
        # Create orthogonal gradients
        grad_panc_orth = torch.zeros_like(grad_panc).to(grad_panc.device).type(grad_panc.dtype)
        grad_panc_orth[...,0] = grad_panc[...,1]
        grad_panc_orth[...,1] = -grad_panc[...,0]
        norm_grad_panc = norm(grad_panc, dim=-1, keepdim=True) 
        c = norm_grad_panc/(torch.sum(norm(grad_panc, dim=-1)) + 1e-7)

        grads = torch.stack((grad_panc_orth/(norm_grad_panc + 1e-7), grad_panc/(norm_grad_panc+ 1e-7)), dim=-1).transpose(-2,-1)

        weights = self.weight_fun(c, self.alpha)
        # Normalize weights
        return weights * grads

    def compute_L(self, nband=None):
        """
        Retourne une borne sur ||K|| avec K = W * nabla.

        On utilise: ||K|| <= (max_x ||W(x)||_2) * ||nabla||, 
        et ||nabla|| ~= sqrt(8) pour un gradient 2D (forward diffs).
        """
        # W attendu de forme [B, 1, H, W, 2, 2] (broadcast sur les canaux)
        W = self.W
        if W.dim() != 6 or W.size(-1) != 2 or W.size(-2) != 2:
            raise ValueError(f"Expected W of shape [B,1,H,W,2,2], got {tuple(W.shape)}")

        # norme spectrale (valeur singulière max) de chaque matrice 2x2
        # reshape -> (N, 2, 2), SVD -> (N, 2), on prend la première colonne (s_max)
        B, One, H, Wd, _, _ = W.shape
        W_mats = W.reshape(B * One * H * Wd, 2, 2)
        # torch.linalg.svd renvoie U, S, Vh ; S = (N, 2) triées décroissantes
        S = torch.linalg.svdvals(W_mats)          # (N, 2)
        s_max = S[:, 0]                           # (N,)
        s_max_global = torch.max(s_max)           # scalaire

        # ||nabla|| ~= sqrt(8) pour 2D; on prend une marge légère
        grad_norm = sqrt(8.0)

        L = (s_max_global * grad_norm).item()
        # marge de sécu si on veut être très prudent
        L = max(L, 1e-8)  # éviter 0
        return L

    
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


    def prox_sigma_g_conj(self, Q, sigma=None, **kwargs):
        """
        Pour g(z) = λ ||z||_{p,q,r}, prox_{σ g^*}(Q) = Proj_{||.||_{(p,q,r)^*} ≤ λ}(Q)
        """
        lam = self.lmbda
        proj = select_dual_projection(self.p, self.q, self.r)
        
        return proj(Q, radius=1/lam)


    def loss_fn(self, u, y, lmbda, sigma2=1): #dépendance en sigma2
        r"""
        Compute the loss function of the problem
        """
        return 0 # unused ?