import torch

from skimage.filters import threshold_otsu
from math import sqrt
from torch.linalg import norm
from torch.nn.functional import sigmoid

from ..chambolle_pock import ChambollePock
from ..utils.nabla import nabla, nabla_adjoint
from ..utils.dual_projections import select_dual_projection


def compute_alpha_from_pan(pan_image):
    """
    Compute the Otsu threshold alpha from a panchromatic image.

    Args:
        pan_image (torch.Tensor): Panchromatic image [1, 1, H, W]

    Returns:
        float: Otsu threshold on the normalized gradient criterion c(x,y)
    """
    grad_norm = torch.norm(pan_image.squeeze(), dim=-1)  # [H, W]
    criterion = grad_norm / (grad_norm.sum() + 1e-7)
    c_n = criterion.cpu().numpy()
    return threshold_otsu(c_n)


class GradientWeights:
    """
    Factory class to create weight functions for TVGradAlignment.
    Supports hard and soft thresholding strategies.
    """
    @staticmethod
    def hard_threshold(epsilon=1e-7):
        """Hard thresholding weight function."""
        def weight_fn(c, alpha):
            return torch.stack((
                torch.ones_like(c),
                torch.where(c < alpha, torch.ones_like(c), torch.ones_like(c) * epsilon)
            ), dim=-1).transpose(-2, -1)
        return weight_fn

    @staticmethod
    def soft_threshold(tau=1.0):
        """Soft thresholding with sigmoid transition."""
        def weight_fn(c, alpha):
            return torch.stack((
                torch.ones_like(c),
                torch.ones_like(c) - sigmoid((c - alpha) / tau)
            ), dim=-1).transpose(-2, -1)
        return weight_fn


class TVGradAlignment(ChambollePock):
    def __init__(self, grad_panc, p, q, r,
                 threshold_softness=1.0, threshold=None,
                 *args, **kwargs):
        """
        Args:
            grad_panc (torch.Tensor): Panchromatic image gradients [1, 1, H, W, 2]
            p, q, r (float): CTV norm parameters
            threshold_softness (float): Sigmoid smoothness tau for soft thresholding
            threshold (float or None): Fixed threshold; if None, computed via Otsu
        """
        super().__init__(*args, **kwargs)

        self.weight_fun = GradientWeights.soft_threshold(tau=threshold_softness)
        self.alpha = compute_alpha_from_pan(grad_panc) if threshold is None else threshold
        self.W = self._compute_weights(grad_panc)
        self.p = p
        self.q = q
        self.r = r
        self._proj = select_dual_projection(p, q, r)   # cached — fixed for given (p,q,r)
        self._L    = self._cache_compute_L()            # cached — W is fixed after __init__

    def _compute_weights(self, grad_panc):
        """Compute the weight matrix from panchromatic gradients."""
        norm_grad_panc = norm(grad_panc, dim=-1, keepdim=True)
        c = norm_grad_panc / (norm_grad_panc.sum() + 1e-7)

        grad_panc_orth = torch.zeros_like(grad_panc)
        grad_panc_orth[..., 0] =  grad_panc[..., 1]
        grad_panc_orth[..., 1] = -grad_panc[..., 0]

        safe_norm = norm_grad_panc + 1e-7
        grads = torch.stack((
            grad_panc_orth / safe_norm,
            grad_panc       / safe_norm,
        ), dim=-1).transpose(-2, -1)

        weights = self.weight_fun(c, self.alpha)
        return weights * grads

    def _cache_compute_L(self):
        """
        Compute and cache the Lipschitz constant of K = W * nabla.

        ||K|| <= max_x ||W(x)||_2 * ||nabla||,  with ||nabla|| ~= sqrt(8) in 2D.
        """
        W = self.W
        if W.dim() != 6 or W.size(-1) != 2 or W.size(-2) != 2:
            raise ValueError(f"Expected W of shape [B,1,H,W,2,2], got {tuple(W.shape)}")

        B, One, H, Wd, _, _ = W.shape
        W_mats = W.reshape(B * One * H * Wd, 2, 2)
        S = torch.linalg.svdvals(W_mats)   # (N, 2), sorted descending
        s_max = S[:, 0].max()

        L = (s_max * sqrt(8.0)).item()
        return max(L, 1e-8)

    def compute_L(self, **kwargs):
        return self._L

    def K(self, u, **kwargs):
        """
        K u(x) = W(x) nabla u(x)

        Args:
            u (torch.Tensor): Hyperspectral image [B, C, H, W]

        Returns:
            torch.Tensor: [B, C, H, W, 2]
        """
        grads = nabla(u)
        return torch.matmul(self.W, grads.unsqueeze(-1)).squeeze(-1)

    def K_adjoint(self, q, **kwargs):
        """
        K^* q = nabla^* (W^T q)

        Args:
            q (torch.Tensor): Dual variable [B, C, H, W, 2]

        Returns:
            torch.Tensor: [B, C, H, W]
        """
        return nabla_adjoint(torch.matmul(self.W.transpose(4, 5), q.unsqueeze(-1)).squeeze(-1))

    def prox_tau_f(self, u, tau, y, sigma2=1):
        r"""Proximal operator of (1/2 sigma^2) ||u - y||^2."""
        return (sigma2 * u + tau * y) / (sigma2 + tau)

    def prox_sigma_g_conj(self, Q, sigma=None, **kwargs):
        """
        For g(z) = lambda ||z||_{p,q,r}:
        prox_{sigma g^*}(Q) = Proj_{ ||.||_{(p,q,r)^*} <= lambda }(Q),
        independent of sigma.
        """
        return self._proj(Q, radius=self.lmbda)

    def loss_fn(self, u, y, lmbda, sigma2=1):
        return 0
