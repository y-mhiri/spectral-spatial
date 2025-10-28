import torch

def proj_inf_inf_inf(z, radius):
    # Proj sur {||.||_{inf,inf,inf} ≤ radius} : clamp élément par élément
    return torch.clamp(z, -radius, radius)

def proj_2_2_inf(z, radius):
    """
    Proj sur {||.||_{2,2,inf} ≤ radius} :
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

def proj_l1_ball_rows(V, radius):
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

def proj_1_inf_inf(z, radius):
    """
    Proj sur {||.||_{1,inf,inf} ≤ radius} :
        pour chaque direction (2) et chaque pixel (B,H,W),
        on projette le vecteur (C,) sur la boule L1 de rayon 'radius'.
    """
    B, C, H, W, D = z.shape
    out = torch.empty_like(z)
    for j in range(D):
        Zj = z[..., j]                               # (B,C,H,W)
        V = Zj.permute(0, 2, 3, 1).reshape(-1, C)    # (B*H*W, C)
        Vp = proj_l1_ball_rows(V, radius=radius)
        out[..., j] = Vp.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return out

def select_dual_projection(p, q, r):
    """
    Sélectionne la projection sur la boule unitaire de la norme duale (rayon = 1).
    (1,1,1)     -> dual = (inf,inf,inf)  -> clamp
    (2,2,1)     -> dual = (2,2,inf)  -> clip Frobenius par pixel
    (inf,1,1)   -> dual = (1,inf,inf)  -> proj L1 par pixel et par direction
    """
    if (p, q, r) == (1, 1, 1):
        return proj_inf_inf_inf
    elif (p, q, r) == (2, 2, 1):
        return proj_2_2_inf
    elif p in (float('inf'), torch.inf) and (q, r) == (1, 1):
        return proj_1_inf_inf
    else:
        raise NotImplementedError("Configs supportées : (1,1,1), (2,2,1), (inf,1,1).")