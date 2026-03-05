"""
Base class for Chambolle-Pock primal-dual algorithms.
"""
import torch
from torch import nn
from math import sqrt


class ChambollePock(nn.Module):
    """
    Chambolle-Pock algorithm solving:
        min_u  f(u) + g(K u)
    via the primal-dual iteration:
        q_{n+1} = prox_{sigma g*}(q_n + sigma K v_n)
        u_{n+1} = prox_{tau  f }(u_n - tau  K* q_{n+1})
        v_{n+1} = u_{n+1} + theta (u_{n+1} - u_n)

    Subclasses must implement: K, K_adjoint, prox_sigma_g_conj, prox_tau_f, compute_L.
    Optionally: loss_fn (for monitoring; returning 0 disables it).

    Attributes:
        max_iter (int):   Maximum iterations.
        lmbda (float):    Regularization weight (passed to prox_sigma_g_conj).
        theta (float):    Relaxation parameter.
        sigma, tau (float): Step sizes (scaled by 1/L in forward).
        tol (float):      Convergence tolerance on relative primal change.
        accelerate (bool): Enable accelerated variant (requires gamma).
    """

    def __init__(self, max_iter=100, lmbda=1, theta=1, sigma=0.99, tau=0.99,
                 tol=1e-7, accelerate=False, gamma=None):
        super().__init__()
        self.max_iter   = max_iter
        self.lmbda      = lmbda
        self.theta      = theta
        self.sigma      = sigma
        self.tau        = tau
        self.tol        = tol
        self.accelerate = accelerate
        self.gamma      = gamma

        if self.accelerate and self.gamma is None:
            raise ValueError('gamma must be provided when accelerate=True')

    # ── abstract interface ────────────────────────────────────────────────────

    def K(self, u, **kwargs):
        """Linear operator K. Must be overridden."""
        pass

    def K_adjoint(self, q, **kwargs):
        """Adjoint K*. Must be overridden."""
        pass

    def prox_sigma_g_conj(self, q, sigma, **kwargs):
        """Proximal operator of sigma * g*. Must be overridden."""
        pass

    def prox_tau_f(self, u, tau, **kwargs):
        """Proximal operator of tau * f. Must be overridden."""
        pass

    def compute_L(self, **kwargs):
        """Lipschitz constant of K. Must be overridden."""
        pass

    def loss_fn(self, u, y, lmbda, **kwargs):
        """Objective value for monitoring. Return 0 to disable tracking."""
        pass

    # ── algorithm ─────────────────────────────────────────────────────────────

    def forward(self, y, init=None, verbose=False, params=None, return_loss=False):
        """
        Run Chambolle-Pock.

        Args:
            y (torch.Tensor): Input / observation [b, c, h, w].
            init:             Initial primal variable (defaults to y).
            verbose (bool):   Print iteration info.
            params (dict):    Per-method keyword arguments, keyed by method name.
            return_loss (bool): If True, also return loss and rel arrays.

        Returns:
            u                        if return_loss=False
            (u, loss, rel)           if return_loss=True
        """
        if params is None:
            params = {}

        L     = self.compute_L(**params.get('compute_L', {}))
        sigma = self.sigma / L
        tau   = self.tau   / L

        u = torch.clone(init if init is not None else y)
        q = self.K(u, **params.get('K', {}))
        v = torch.clone(u)

        rel  = torch.zeros(self.max_iter, device=y.device)
        loss = torch.zeros(self.max_iter, device=y.device) if return_loss else None

        if verbose:
            print('Chambolle-Pock starting...')

        for it in range(self.max_iter):
            u_old = torch.clone(u)

            q = self.prox_sigma_g_conj(
                    q + sigma * self.K(v, **params.get('K', {})),
                    sigma, **params.get('prox_sigma_g_conj', {}))
            u = self.prox_tau_f(
                    u - tau * self.K_adjoint(q, **params.get('K_adjoint', {})),
                    tau, **params.get('prox_tau_f', {}))

            if self.accelerate:
                self.theta = 1 / sqrt(1 + 2 * self.gamma * tau)
                tau   = self.gamma * self.theta
                sigma = sigma / self.theta

            v = u + self.theta * (u - u_old)

            rel[it] = torch.norm(u - u_old) / torch.norm(u_old)
            if return_loss:
                loss[it] = self.loss_fn(u, y, self.lmbda, **params.get('loss_fn', {}))

            if verbose:
                print(f'[CP] it={it:4d}  rel={rel[it].item():.3e}')

            if rel[it] < self.tol:
                if verbose:
                    print(f'[CP] Converged after {it + 1} iterations.')
                break

        if return_loss:
            return u, loss, rel
        return u
