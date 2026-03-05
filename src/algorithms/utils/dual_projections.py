import torch


def proj_inf_inf_inf(z, radius):
    """Project onto { ||.||_{inf,inf,inf} <= radius }: elementwise clamp."""
    return torch.clamp(z, -radius, radius)


def proj_2_2_inf(z, radius):
    """
    Project onto { ||.||_{2,2,inf} <= radius }:
    for each pixel (H, W), clip the (C x 2) block by its Frobenius norm.

    Args:
        z (torch.Tensor): [B, C, H, W, 2]
        radius (float):   Ball radius.
    """
    B, C, H, W, D = z.shape
    z_flat = z.permute(0, 2, 3, 1, 4).reshape(-1, C * D)          # (B*H*W, C*2)
    nF     = torch.linalg.norm(z_flat, ord=2, dim=1, keepdim=True) # (B*H*W, 1)
    scale  = torch.clamp(nF / radius, min=1.0)
    z_flat = z_flat / scale
    return z_flat.reshape(B, H, W, C, D).permute(0, 3, 1, 2, 4)


def proj_l1_ball_rows(V, radius):
    """
    Project each row of V onto the L1 ball of given radius.

    Args:
        V (torch.Tensor): (N, C)
        radius (float):   Ball radius.
    """
    absV = V.abs()
    s, _ = torch.sort(absV, dim=1, descending=True)
    cssv = torch.cumsum(s, dim=1)
    r    = torch.arange(1, V.shape[1] + 1, device=V.device, dtype=V.dtype).view(1, -1)
    cond = s > (cssv - radius) / r
    rho  = cond.sum(dim=1) - 1
    theta = (cssv[torch.arange(V.shape[0]), rho] - radius) / (rho.to(V.dtype) + 1.0)
    return torch.sign(V) * torch.clamp(absV - theta.unsqueeze(1), min=0.0)


def proj_1_inf_inf(z, radius):
    """
    Project onto { ||.||_{1,inf,inf} <= radius }:
    for each gradient direction and each pixel, project the spectral vector (C,)
    onto the L1 ball of given radius.

    Args:
        z (torch.Tensor): [B, C, H, W, 2]
        radius (float):   Ball radius.
    """
    B, C, H, W, D = z.shape
    # Batch all directions together: (B*H*W*D, C)
    V  = z.permute(0, 2, 3, 4, 1).reshape(-1, C)
    Vp = proj_l1_ball_rows(V, radius=radius)
    return Vp.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)


def select_dual_projection(p, q, r):
    """
    Select the projection onto the dual ball of the l^{p,q,r} CTV norm.

    Supported configurations (primal -> dual):
        (1,   1, 1) -> (inf, inf, inf) : elementwise clamp
        (2,   2, 1) -> (2,   2, inf)   : per-pixel Frobenius clip
        (inf, 1, 1) -> (1,   inf, inf) : per-pixel per-direction L1 projection
    """
    if (p, q, r) == (1, 1, 1):
        return proj_inf_inf_inf
    elif (p, q, r) == (2, 2, 1):
        return proj_2_2_inf
    elif p in (float('inf'), torch.inf) and (q, r) == (1, 1):
        return proj_1_inf_inf
    else:
        raise NotImplementedError(f"Unsupported (p,q,r)=({p},{q},{r}). Supported: (1,1,1), (2,2,1), (inf,1,1).")
