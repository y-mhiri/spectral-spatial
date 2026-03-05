import torch
from math import sqrt
from ..chambolle_pock import ChambollePock
from ..utils.nabla import nabla, nabla_adjoint
from ..utils.dual_projections import select_dual_projection


class TVPrior(ChambollePock):
    """
    CTV l^{p,q,r} prior for Chambolle-Pock with dual ball projections.
    Supported configurations (primal -> dual):
      (1,1,1)   -> (inf,inf,inf) : clamp
      (2,2,1)   -> (2,2,inf)     : per-pixel Frobenius clip (C x 2)
      (inf,1,1) -> (1,inf,inf)   : per-pixel per-direction L1 projection
    """

    def __init__(self, p, q, r, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.q = q
        self.r = r
        self._proj = select_dual_projection(p, q, r)  # cached — fixed for given (p,q,r)

    def K(self, u, **kwargs):
        return nabla(u)

    def K_adjoint(self, q, **kwargs):
        return nabla_adjoint(q)

    def compute_L(self, **kwargs):
        # ||nabla|| = sqrt(8) for forward differences + Neumann BC in 2D
        return sqrt(8.0)

    def prox_tau_f(self, u, tau, y, sigma2=1.0, **kwargs):
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
