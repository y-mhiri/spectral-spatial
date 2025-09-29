import torch

def gradient(u):
    """
    Calcule le gradient d'un tableau en utilisant les différences finies.

    Parameters:
    - u : array_like, tableau 4D de forme (b, c, m, n).

    Returns:
    - grad_u : array_like, gradient de u de taille (b, c, m, n, 2).
    """
    if u.ndim != 4:
        raise ValueError("L'entrée u doit être un tableau 4D de forme (b, c, m, n).")
    
    b, c, m, n = u.shape
    grad_u = torch.zeros((b, c, m, n, 2))

    # Gradient vertical
    grad_u[:, :, :-1, :, 0] = u[:, :, 1:, :] - u[:, :, :-1, :]  # Gradient vertical

    # Gradient horizontal
    grad_u[:, :, :, :-1, 1] = u[:, :, :, 1:] - u[:, :, :, :-1]  # Gradient horizontal

    return grad_u

def divergence(p):
    """
    Calcule la divergence d'un champ vectoriel en suivant votre style.
    Adapté pour un tableau 5D (b, c, m, n, 2).

    Parameters:
    - p : array_like, champ vectoriel de taille (b, c, m, n, 2).

    Returns:
    - div_p : array_like, divergence de p de taille (b, c, m, n).
    """
    if p.ndim != 5 or p.shape[4] != 2:
        raise ValueError("L'entrée p doit être un tableau de taille (b, c, m, n, 2).")
    
    b, c, m, n = p.shape[:4]
    div_p = torch.zeros((b, c, m, n))
    
    # Divergence pour la composante verticale (axe 0)
    div_p[:, :, :-1, :] += p[:, :, :-1, :, 0]
    div_p[:, :, 1:, :] -= p[:, :, :-1, :, 0]
    
    # Divergence pour la composante horizontale (axe 1)
    div_p[:, :, :, :-1] += p[:, :, :, :-1, 1]
    div_p[:, :, :, 1:] -= p[:, :, :, :-1, 1]
    
    return div_p