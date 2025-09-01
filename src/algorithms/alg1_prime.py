from pan_gradient_prox import PANProximalGradient
import torch
from gradient import gradient, divergence

class PANTVGradProj(PANProximalGradient):
    """
    Implémentation simplifiée de l'opérateur proximal de la TV vectorielle avec :
    - Descente de gradient projeté basique
    - Pas fixe (sans line search)
    - Support des normes mixtes l^p,q,r
    """
    def __init__(self, A, Aadj, spectral_op, spectral_op_t, max_iter, lmbda,alpha,lmbda_m, tau, tol, scale, verbose, max_iter_gp, p, q, r):
        super().__init__(A, Aadj, spectral_op, spectral_op_t, max_iter, lmbda,alpha,lmbda_m, tol, scale, p, q, r, verbose)
        self.max_iter_gp = max_iter_gp  # Max iterations for gradient projection
        self.tau = tau                  # Fixed step size
        self.p = p                      # Norm parameters
        self.q = q
        self.r = r

    @staticmethod
    def _proj_inf_inf_inf(z):
        # ||.||_{∞,∞,∞} <= 1 : clamp composante-par-composante
        return torch.clamp(z, -1.0, 1.0)

    @staticmethod
    def _proj_2_2_inf(z):
        # ||.||_{2,2,∞} <= 1 : clip Frobenius par pixel (C x 2)
        # nF = sqrt(sum_{c,dir} z^2)
        nF = torch.sqrt((z**2).sum(dim=(1, -1), keepdim=True))  # (B,1,H,W,1)
        scale = torch.clamp(nF, min=1.0)
        return z / scale

    @staticmethod
    def _proj_l1_ball_rows(V, radius=1.0):
        """
        projection L1 sur la boule de rayon 'radius' pour chaque ligne.
        V: (N, C)
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
    def _proj_1_inf_inf(z):
        # ||.||_{1,∞,∞} <= 1 : L1 sur canaux (C) indépendamment pour chaque pixel (B,H,W) et chaque direction (2)
        B, C, H, W, D = z.shape  # D=2
        out = torch.empty_like(z)
        for j in range(D):
            Zj = z[..., j]                           # (B,C,H,W)
            V = Zj.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            Vp = PANTVGradProj._proj_l1_ball_rows(V, radius=1.0)  # proj L1
            out[..., j] = Vp.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

    @staticmethod
    def _select_dual_projection(p, q, r):
        """
        renvoie la proj sur la boule unitaire de la norme duale pour (p,q,r)
        (1,1,1)     -> dual = (∞,∞,∞)  -> clamp
        (2,2,1)     -> dual = (2,2,∞)  -> clip Frobenius par pixel
        (inf,1,1)   -> dual = (1,∞,∞)  -> proj L1 par pixel et par direction
        """
        if (p, q, r) == (1, 1, 1):
            return PANTVGradProj._proj_inf_inf_inf
        if (p, q, r) == (2, 2, 1):
            return PANTVGradProj._proj_2_2_inf
        if p in (float('inf'), torch.inf) and (q, r) == (1, 1):
            return PANTVGradProj._proj_1_inf_inf
        raise NotImplementedError("Configs supportées : (1,1,1), (2,2,1), (inf,1,1).")

    
    def proxg(self, u):
        if self.lmbda <= 0:
            raise ValueError("lambd > 0 requis.")
        if self.max_iter_gp <= 0:
            raise ValueError("K > 0 requis.")

        proj_dual = PANTVGradProj._select_dual_projection(self.p, self.q, self.r)

        z = torch.zeros(u.shape + (2,), device=u.device, dtype=u.dtype)  # (B,C,H,W,2)

        for _ in range(self.max_iter_gp):
            g = divergence(z) + u / self.lmbda
            grad_z = -2.0 * gradient(g)

            # --- Pas adaptatif ---
            tau = self.tau
            c = 1e-4  # coefficient d’Armijo
            obj_old = torch.norm(g) ** 2  # objectif dual approx.

            while True:
                z_new = z - tau * grad_z
                z_new = proj_dual(z_new)

                g_new = divergence(z_new) + u / self.lmbda
                obj_new = torch.norm(g_new) ** 2

                if obj_new <= obj_old - c * tau * torch.norm(grad_z) ** 2:
                    break  # critère satisfait
                tau *= 0.5  # réduction du pas si pas satisfaisant

            z = z_new

        return u + self.lmbda * divergence(z)
