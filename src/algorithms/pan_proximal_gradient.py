import sys
import math
import torch
import logging
import torch.nn as nn


class PANProximalGradient(nn.Module):
    """
    Proximal gradient algorithm (ISTA) for hyperspectral pansharpening.

    Minimizes: 1/2 ||A(U) - Y_H||^2 + lmbda_m/2 ||R(U) - Y_M||^2 + lmbda * g(U)
    where g is the CTV regularization term solved by a subclass via proxg().

    Attributes:
        max_iter (int):   Maximum number of outer ISTA iterations.
        lmbda (float):    TV regularization weight.
        lmbda_m (float):  Panchromatic data fidelity weight.
        tol (float):      Convergence tolerance on relative iterate change.
        scale (int):      Spatial downsampling factor.
        A (callable):     Degradation operator (blur + downsample).
        Aadj (callable):  Adjoint of A.
        spectral_op (callable):   Spectral averaging operator R.
        spectral_op_t (callable): Adjoint of R.
    """

    def __init__(self, A, Aadj,spectral_op,spectral_op_t,max_iter, lmbda, lmbda_m, tol,scale,p,q,r,verbose):
        super().__init__()
        self.max_iter = max_iter
        self.scale = scale
        self.lmbda = lmbda
        self.lmbda_m = lmbda_m
        self.tol = tol
        self.verbose = verbose
        self.A = A
        self.Aadj = Aadj
        self.spectral_op = spectral_op
        self.spectral_op_t = spectral_op_t
        self.p = p
        self.q = q 
        self.r = r

        self.logger = logging.getLogger('PANProximalGradient')
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)


    def ctv_norm(self, U, eps=1e-8):
        """
        Compute the l^{p,q,r} CTV norm, supporting p, q, r = inf.

        Args:
            U   (torch.Tensor): Gradient tensor [B, C, H, W, 2]
            eps (float):        Small constant for numerical stability.

        Returns:
            torch.Tensor: CTV norm value [B, 1, 1, 1]
        """
        # l^p norm over spectral channels
        if math.isinf(self.p):
            norm_p = torch.amax(torch.abs(U), dim=1, keepdim=True)
        else:
            norm_p = (torch.sum(torch.abs(U) ** self.p, dim=1, keepdim=True) + eps) ** (1.0 / self.p)

        # l^q norm over gradient directions
        if math.isinf(self.q):
            norm_q = torch.amax(norm_p, dim=-1, keepdim=True)
        else:
            norm_q = (torch.sum(norm_p ** self.q, dim=-1, keepdim=True) + eps) ** (1.0 / self.q)

        # l^r norm over pixels
        if math.isinf(self.r):
            norm_r = torch.amax(norm_q, dim=(2, 3), keepdim=True)
        else:
            norm_r = (torch.sum(norm_q ** self.r, dim=(2, 3), keepdim=True) + eps) ** (1.0 / self.r)

        return norm_r
        

    def compute_cost(self, U, Y_H, Y_M):
        """Total objective value. Must be implemented by subclasses."""
        raise NotImplementedError('compute_cost is not implemented in abstract class.')

    def grad_f(self, U, Y_H, Y_M):
        """
        Gradient of the smooth part: A^T(A(U) - Y_H) + lmbda_m * R^T(R(U) - Y_M).

        Args:
            U   (torch.Tensor): Current estimate          [b, c, h, w]
            Y_H (torch.Tensor): Low-resolution HSI        [b, c, h//scale, w//scale]
            Y_M (torch.Tensor): Panchromatic observation  [b, 1, h, w]

        Returns:
            torch.Tensor: Gradient [b, c, h, w]
        """
        grad1 = self.Aadj(self.A(U) - Y_H)
        grad2 = self.lmbda_m * self.spectral_op_t(self.spectral_op(U) - Y_M)
        return grad1 + grad2

    def proxg(self, x, gamma=1):
        """Proximal operator of g. Must be implemented by subclasses."""
        raise NotImplementedError("proxg() is not implemented in abstract class.")
    

        
    def forward(self, Y_H, Y_M):
        """
        Proximal gradient : U_{k+1} = prox_g(U_k - alpha * grad_f(U_k))
        """
        # Lipschitz constant of grad_f and resulting step size
        c = Y_H.shape[1]
        L = 1.0 + self.lmbda_m * (1.0 / c)
        self.alpha = 1.0 / L
        if self.verbose:
            self.logger.info(f"[Proximal Gradient] L = {L:.6f}  |  alpha = {self.alpha:.6f}")

        U = self.Aadj(Y_H).clone()
        cost_history = torch.zeros(self.max_iter, device=U.device)
        relval = torch.zeros(self.max_iter, device=U.device)

        for it in range(self.max_iter):
            U_prev = U.clone()
            grad   = self.grad_f(U, Y_H, Y_M)
            U      = self.proxg(U - self.alpha * grad, gamma=self.alpha)

            total_cost, data_term_h, data_term_m, tv_term = self.compute_cost(U, Y_H, Y_M)
            cost_history[it] = total_cost.item()

            delta_U    = torch.norm(U - U_prev).item() / (torch.norm(U).item() + 1e-8)
            relval[it] = delta_U
            if self.verbose and (it % 10 == 0 or delta_U < self.tol):
                self.logger.info(
                    f"{it:<5} | {total_cost.item():<12.3e} | {data_term_h.item():<12.3e} | "
                    f"{data_term_m.item():<12.3e} | {tv_term.item():<12.3e} | {delta_U:<12.3e}"
                )
            if delta_U < self.tol:
                break

        return U, cost_history, relval

