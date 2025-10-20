import torch
from math import sqrt
from chambolle_pock import ChambollePock 
from nabla import nabla, nabla_adjoint        
 
class TVPrior(ChambollePock):
    """
    Prior CTV l^{p,q,r} pour Chambolle–Pock avec projections duales:
      (1,1,1) -> (inf,inf,inf) : clamp
      (2,2,1) -> (2,2,inf)     : clip Frobenius par pixel (C x 2)
      (inf,1,1)-> (1,inf,inf)  : proj L1 par pixel et par direction
    """

    def __init__(self, p, q, r, *args, **kwargs):
        super(TVPrior, self).__init__(*args, **kwargs)
        self.p = p
        self.q = q
        self.r = r

    # --- Opérateurs K et K* ---
    def K(self, u, **kwargs):
        return nabla(u)

    def K_adjoint(self, q, **kwargs):
        return nabla_adjoint(q)

    # --- Constante de Lipschitz de ∇ (2D) ---
    def compute_L(self, **kwargs):
        # ||∇|| = sqrt(8) pour diff. finies avant + Neumann en 2D
        return sqrt(8.0)

    # --- Prox de f : fidélité quadratique ---
    def prox_tau_f(self, u, tau, y, sigma2=1.0, **kwargs):
        return (sigma2 * u + tau * y) / (sigma2 + tau)

    # =======================
    #   Projections duales
    # =======================
    @staticmethod
    def _proj_inf_inf_inf(z, radius):
        # Proj sur {||.||_{∞,∞,∞} ≤ radius} : clamp élément par élément
        return torch.clamp(z, -radius, radius)

    @staticmethod
    def _proj_2_2_inf(z, radius):
        """
        Proj sur {||.||_{2,2,∞} ≤ radius} :
          pour chaque pixel (H,W), clip Frobenius sur le bloc (C x 2)
        """
        # z : (B,C,H,W,2)
        B, C, H, W, D = z.shape
        z_flat = z.permute(0, 2, 3, 1, 4).reshape(-1, C * D)         # (B*H*W, C*2)
        nF = torch.linalg.norm(z_flat, ord=2, dim=1, keepdim=True)   # (BHW,1)
        # scale = max(1, nF / radius)
        scale = torch.clamp(nF / radius, min=1.0)
        z_flat = z_flat / scale
        return z_flat.reshape(B, H, W, C, D).permute(0, 3, 1, 2, 4)

    @staticmethod
    def _proj_l1_ball_rows(V, radius=1.0):
        """
        Projection L1 par ligne sur une boule de rayon 'radius'.
        V : (N, C)
        """
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
        """
        Proj sur {||.||_{1,∞,∞} ≤ radius} :
          pour chaque direction (2) et chaque pixel (B,H,W),
          on projette le vecteur (C,) sur la boule L1 de rayon 'radius'.
        """
        B, C, H, W, D = z.shape
        out = torch.empty_like(z)
        for j in range(D):
            Zj = z[..., j]                               # (B,C,H,W)
            V = Zj.permute(0, 2, 3, 1).reshape(-1, C)    # (B*H*W, C)
            Vp = TVPrior._proj_l1_ball_rows(V, radius=radius)
            out[..., j] = Vp.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

    @staticmethod
    def _select_dual_projection(p, q, r):
        """
        Sélectionne la projection sur la boule unitaire de la norme duale (rayon = 1).
        (1,1,1)     -> dual = (∞,∞,∞)  -> clamp
        (2,2,1)     -> dual = (2,2,∞)  -> clip Frobenius par pixel
        (inf,1,1)   -> dual = (1,∞,∞)  -> proj L1 par pixel et par direction
        """
        if (p, q, r) == (1, 1, 1):
            return TVPrior._proj_inf_inf_inf
        if (p, q, r) == (2, 2, 1):
            return TVPrior._proj_2_2_inf
        if p in (float('inf'), torch.inf) and (q, r) == (1, 1):
            return TVPrior._proj_1_inf_inf
        raise NotImplementedError("Configs supportées : (1,1,1), (2,2,1), (inf,1,1).")

    # --- Prox de g* (projection sur la boule duale de rayon λ) ---
    def prox_sigma_g_conj(self, Q, sigma=None, **kwargs):
        """
        Pour g(z) = λ ||z||_{p,q,r} :
        prox_{σ g^*}(Q) = Proj_{ ||.||_{(p,q,r)^*} ≤ λ } (Q),
        indépendant de σ.
        """
        proj_unit = self._select_dual_projection(self.p, self.q, self.r)
        lam = self.lmbda

        # Applique la projection unitaire puis adapte le rayon λ,
        # ou directement utiliser la variante à rayon λ :
        # - pour clamp : clamp(Q, -lam, lam)
        # - pour L2 bloc : scale = max(1, ||bloc||/lam)
        # - pour L1 : proj_l1_ball_rows(..., radius=lam)
        if (self.p, self.q, self.r) == (1, 1, 1):
            return self._proj_inf_inf_inf(Q, radius=lam)
        if (self.p, self.q, self.r) == (2, 2, 1):
            return self._proj_2_2_inf(Q, radius=lam)
        if self.p in (float('inf'), torch.inf) and (self.q, self.r) == (1, 1):
            return self._proj_1_inf_inf(Q, radius=lam)

        # fallback (ne devrait pas arriver car _select_dual_projection leve déjà)
        return proj_unit(Q, radius=lam)

    # (Optionnel) Pour du logging/visualisation :
    def loss_fn(self, u, y, lmbda, **kwargs):
        """
        f(u) = 0.5 ||u - y||^2 + λ * TV_{p,q,r}(u)  (valeur diagnostique)
        """
        data = 0.5 * torch.norm(u - y) ** 2
        # NB: calculer exactement ||∇u||_{p,q,r} coûte ; à n'utiliser qu'en debug
        return data
