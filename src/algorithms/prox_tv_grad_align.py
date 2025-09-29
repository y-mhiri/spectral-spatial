import torch

from chambolle_pock import ChambollePock
from skimage.filters import threshold_otsu
from math import sqrt
from torch.linalg import svd, norm
from nabla import nabla, nabla_adjoint
from torch.nn.functional import sigmoid


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
            return torch.stack((torch.ones_like(c),sigmoid((c - alpha)/tau)), dim=-1).transpose(-2, -1)
        return weight_fn


class TVGradAlignment(ChambollePock):
    def __init__(self, grad_panc, p, q, r, 
                 threshold_type='soft',  # 'soft' or 'hard'
                 alpha=0.1,             # threshold value
                 threshold_param=1.0,   # epsilon (hard) or tau (soft)
                 *args, **kwargs):
        """
        Args:
            grad_panc: Panchromatic image gradients
            p, q, r: Norm parameters
            threshold_type: 'soft' or 'hard' thresholding strategy
            alpha: Threshold value
            threshold_param: epsilon (for hard) or tau (for soft)
        """
        super().__init__(*args, **kwargs)

        # Select weight function based on threshold type
        if threshold_type == 'hard':
            self.weight_fun = GradientWeights.hard_threshold(epsilon=threshold_param)
        elif threshold_type == 'soft':
            self.weight_fun = GradientWeights.soft_threshold(tau=threshold_param)
        else:
            raise ValueError("threshold_type must be 'hard' or 'soft'")

        # Normalize alpha relative to gradient magnitudes
        #self.alpha = alpha / (torch.mean(norm(grad_panc, dim=-1)) + 1e-7)  
        self.alpha = self.compute_alpha_from_pan(grad_panc)      
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
    

    def compute_alpha_from_pan(self,pan_image):
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


        # =======================
    #   Projections duales
    # =======================
    @staticmethod
    def _proj_inf_inf_inf(z, radius):
        return torch.clamp(z, -radius, radius)

    @staticmethod
    def _proj_2_2_inf(z, radius):
        # clip Frobenius par pixel (C x 2)
        B, C, H, W, D = z.shape
        z_flat = z.permute(0, 2, 3, 1, 4).reshape(-1, C * D)  # (BHW, C*2)
        nF = torch.linalg.norm(z_flat, ord=2, dim=1, keepdim=True)  # (BHW,1)
        scale = torch.clamp(nF / radius, min=1.0)
        z_flat = z_flat / scale
        return z_flat.reshape(B, H, W, C, D).permute(0, 3, 1, 2, 4)

    @staticmethod
    def _proj_l1_ball_rows(V, radius=1.0):
        absV = V.abs()
        s, _ = torch.sort(absV, dim=1, descending=True)
        cssv = torch.cumsum(s, dim=1)
        r = torch.arange(1, V.shape[1] + 1, device=V.device, dtype=V.dtype).view(1, -1)
        cond = s > (cssv - radius) / r
        rho = cond.sum(dim=1) - 1
        theta = (cssv[torch.arange(V.shape[0]), rho] - radius) / (rho.to(V.dtype) + 1.0)
        theta = theta.unsqueeze(1)
        return torch.sign(V) * torch.clamp(absV - theta, min=0.0)

    @staticmethod
    def _proj_1_inf_inf(z, radius):
        B, C, H, W, D = z.shape
        out = torch.empty_like(z)
        for j in range(D):
            Zj = z[..., j]                              # (B,C,H,W)
            V = Zj.permute(0, 2, 3, 1).reshape(-1, C)   # (BHW, C)
            Vp = TVGradAlignment._proj_l1_ball_rows(V, radius=radius)
            out[..., j] = Vp.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

    @staticmethod
    def _select_dual_projection(p, q, r):
        if (p, q, r) == (1, 1, 1):
            return TVGradAlignment._proj_inf_inf_inf
        if (p, q, r) == (2, 2, 1):
            return TVGradAlignment._proj_2_2_inf
        if p in (float('inf'), torch.inf) and (q, r) == (1, 1):
            return TVGradAlignment._proj_1_inf_inf
        raise NotImplementedError("Configs supportées : (1,1,1), (2,2,1), (inf,1,1).")

    def prox_sigma_g_conj(self, Q, sigma=None, **kwargs):
        """
        Pour g(z) = λ ||z||_{p,q,r}, prox_{σ g^*}(Q) = Proj_{||.||_{(p,q,r)^*} ≤ λ}(Q)
        """
        lam = self.lmbda
        if (self.p, self.q, self.r) == (1, 1, 1):
            return self._proj_inf_inf_inf(Q, radius=lam)
        if (self.p, self.q, self.r) == (2, 2, 1):
            return self._proj_2_2_inf(Q, radius=lam)
        if self.p in (float('inf'), torch.inf) and (self.q, self.r) == (1, 1):
            return self._proj_1_inf_inf(Q, radius=lam)
        # fallback générique
        proj = self._select_dual_projection(self.p, self.q, self.r)
        return proj(Q, radius=lam)


    def loss_fn(self, u, y, lmbda, sigma2=1): #dépendance en sigma2
        r"""
        Compute the loss function of the problem
        """
        f = lambda u: (1/(2*sigma2))*norm(u - y)**2
        reg = lambda p: lmbda*torch.sum(norm(torch.matmul(self.W,nabla(p).unsqueeze(-1)).squeeze(-1),dim=-1))#.squeeze(-1), dim=-1)

        return f(u) + reg(u)