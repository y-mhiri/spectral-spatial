import torch

def nabla(x):
    r"""
    Applies the finite differences operator associated with tensors of the same shape as x.
    """
    b, c, h, w = x.shape
    u = torch.zeros((b, c, h, w, 2), device=x.device).type(x.dtype)
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
    u = torch.zeros((b, c, h, w), device=x.device).type(
        x.dtype
    )  # note that we just reversed left and right sides of each line to obtain the transposed operator
    u[:, :, :-1] = u[:, :, :-1] - x[:, :, :-1, :, 0]
    u[:, :, 1:] = u[:, :, 1:] + x[:, :, :-1, :, 0]
    u[..., :-1] = u[..., :-1] - x[..., :-1, 1]
    u[..., 1:] = u[..., 1:] + x[..., :-1, 1]

    return u