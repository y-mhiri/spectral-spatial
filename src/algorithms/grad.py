import torch

def gradient(u):
    """
    Calcule le gradient spatial d'un tenseur 3D (hauteur, largeur, bandes).
    u : Tensor de forme (N, M, C) (hauteur, largeur, bandes spectrales).
    Retourne un tenseur de forme (2, N, M, C) représentant le gradient horizontal et vertical.
    """
    N, M, C = u.shape
    grad_u = torch.zeros((2, N, M, C), device=u.device)  # Initialisation du tenseur de gradient

    # Gradient horizontal (différence entre les lignes)
    grad_u[0, :-1, :, :] = u[1:, :, :] - u[:-1, :, :]

    # Gradient vertical (différence entre les colonnes)
    grad_u[1, :, :-1, :] = u[:, 1:, :] - u[:, :-1, :]

    return grad_u


def div_3d(p):
    """
    Calcule la divergence spatiale d'un tenseur 3D (2, N, M).
    p : Tensor de forme (2, N, M) (2 composantes, hauteur, largeur).
    Retourne un tenseur de forme (N, M) représentant la divergence.
    """
    _, N, M = p.shape
    div_p = torch.zeros((N, M), device=p.device)  # Initialisation du tenseur de divergence

    # Divergence pour la première composante (horizontal)
    div_p[:-1, :] += p[0, :-1, :]  # Ajouter les valeurs de p[0] à gauche
    div_p[1:, :] -= p[0, :-1, :]   # Soustraire les valeurs de p[0] à droite

    # Divergence pour la deuxième composante (vertical)
    div_p[:, :-1] += p[1, :, :-1]  # Ajouter les valeurs de p[1] en haut
    div_p[:, 1:] -= p[1, :, :-1]   # Soustraire les valeurs de p[1] en bas

    return div_p