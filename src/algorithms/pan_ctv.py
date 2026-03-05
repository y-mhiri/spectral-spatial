import torch
from .utils.nabla import nabla
from .pan_proximal_gradient import PANProximalGradient
from .prox.tv_prior import TVPrior


class PANCTV(PANProximalGradient):
    """
    Pansharpening with CTV regularization. Proximal operator solved via Chambolle-Pock.

    Args:
        init_params (dict): Parameters for TVPrior (max_iter, lmbda, theta, sigma, tau).
        *args, **kwargs: Forwarded to PANProximalGradient.
    """

    def __init__(self, init_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_params['p'] = self.p
        init_params['q'] = self.q
        init_params['r'] = self.r
        self.optim = TVPrior(**init_params)

    def proxg(self, x, gamma=1):
        """Proximal operator of the CTV prior, solved by Chambolle-Pock."""
        params = {'prox_tau_f': {'y': x, 'sigma2': 1}}
        return self.optim(x, init=None, verbose=False, params=params, return_loss=False)

    def compute_cost(self, U, Y_H, Y_M):
        """
        Total objective: HSI data fidelity + PAN data fidelity + CTV regularization.

        Args:
            U   (torch.Tensor): Current estimate          [b, c, h, w]
            Y_H (torch.Tensor): Low-resolution HSI        [b, c, h//scale, w//scale]
            Y_M (torch.Tensor): Panchromatic observation  [b, 1, h, w]

        Returns:
            Tuple: (total, data_term_h, data_term_m, tv_term)
        """
        data_term_h = 0.5 * torch.norm(self.A(U) - Y_H) ** 2
        data_term_m = 0.5 * self.lmbda_m * torch.norm(self.spectral_op(U) - Y_M) ** 2
        tv_term     = self.lmbda * self.ctv_norm(nabla(U), eps=1e-8)
        return data_term_h + data_term_m + tv_term, data_term_h, data_term_m, tv_term