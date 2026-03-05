import torch

def nabla(x):
    r"""
    Applies the finite differences operator associated with tensors of the same shape as x.
    """
    b, c, h, w = x.shape
    u = torch.zeros((b, c, h, w, 2), device=x.device, dtype=x.dtype)
    u[:, :, :-1, :, 0] = u[:, :, :-1, :, 0] - x[:, :, :-1]
    u[:, :, :-1, :, 0] = u[:, :, :-1, :, 0] + x[:, :, 1:]
    u[:, :, :, :-1, 1] = u[:, :, :, :-1, 1] - x[..., :-1]
    u[:, :, :, :-1, 1] = u[:, :, :, :-1, 1] + x[..., 1:]
    return u

def nabla_adjoint(x):
    r"""
    Applies the adjoint of the finite difference operator.
    """
    b, c, h, w = x.shape[:-1]
    # note: transposed operator — each line has left/right sides swapped
    u = torch.zeros((b, c, h, w), device=x.device, dtype=x.dtype)
    u[:, :, :-1] = u[:, :, :-1] - x[:, :, :-1, :, 0]
    u[:, :, 1:] = u[:, :, 1:] + x[:, :, :-1, :, 0]
    u[..., :-1] = u[..., :-1] - x[..., :-1, 1]
    u[..., 1:] = u[..., 1:] + x[..., :-1, 1]

    return u