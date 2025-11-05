import torch
from math import sqrt
from .chambolle_pock import ChambollePock 
from .nabla import nabla, nabla_adjoint        
from .dual_projections import select_dual_projection
 
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



    # --- Prox de g* (projection sur la boule duale de rayon λ) ---
    def prox_sigma_g_conj(self, Q, sigma=None, **kwargs):
        """
        Pour g(z) = λ ||z||_{p,q,r} :
        prox_{sigma g^*}(Q) = Proj_{ ||.||_{(p,q,r)^*} ≤ λ } (Q),
        indépendant de sigma.
        """
        proj_unit = select_dual_projection(self.p, self.q, self.r)
        lam = self.lmbda

        return proj_unit(Q, radius=lam)


    def loss_fn(self, u, y, lmbda, sigma2=1): #dépendance en sigma2
        r"""
        Compute the loss function of the problem
        """
        return 0 # unused ?