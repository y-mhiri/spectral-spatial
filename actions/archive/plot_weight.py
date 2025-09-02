import sys
import os

data_path = os.getenv("DATA_PATH")

path = '/home/ndiayem/Documents/spectral-spatial/src'
sys.path.append(os.path.join(path, 'datasets'))
sys.path.append(os.path.join(path, 'algorithms'))
sys.path.append(os.path.join(path, 'metrics'))


import argparse     
import numpy as np
import zarr
import torch
from PIL import Image
import matplotlib.pyplot as plt
from nabla import nabla
from torch.linalg import norm

def generate_figure(image_array, filename, folder):
    """Génère et sauvegarde une image à partir d'un tableau numpy"""
    if image_array.ndim == 2:
        mode = 'L'
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        mode = 'RGB'
    else:
        raise ValueError("Le tableau doit être 2D (niveaux de gris) ou 3D (RGB)")

    pil_image = Image.fromarray((image_array * 255).astype(np.uint8), mode=mode)
    save_path = os.path.join(folder, f'{filename}.png')
    pil_image.save(save_path)
    print(f'Sauvegarde de {save_path}')

def apply_threshold_to_gradient(pan_image, thresh, weight_type, epsilon=None, tau=None):
    """Applique un seuillage sur le gradient de l'image panchromatique."""
    grad_panc = nabla(pan_image)

    if weight_type == "hard":
        weight_fun = lambda c: hard_threshold(c, thresh, epsilon)
    else:
        weight_fun = lambda c: soft_threshold(c, thresh, tau)

    grad_panc_orth = torch.zeros_like(grad_panc).to(grad_panc.device).type(grad_panc.dtype)
    grad_panc_orth[...,0] = grad_panc[...,1]
    grad_panc_orth[...,1] = -grad_panc[...,0]
    norm_grad_panc = norm(grad_panc, dim=-1, keepdim=True)

    grads = torch.stack((grad_panc_orth/(norm_grad_panc + 1e-7), grad_panc/(norm_grad_panc+ 1e-7)), dim=-1).transpose(-2,-1)
    thresh = thresh / (torch.mean(norm(grad_panc, ord=2, dim=-1)) + 1e-7)

    c = norm_grad_panc
    weights = weight_fun(c)
    # Normalisation des poids pour que la somme des W2,n soit égale à 1
    weights_sum = torch.sum(weights, dim=-1, keepdim=True)
    weights = weights / (weights_sum + 1e-7)

    return (weights*grads)

def hard_threshold(c, alpha, epsilon):
    return torch.stack((
        torch.ones_like(c),
        torch.where(c < alpha, torch.ones_like(c), torch.ones_like(c) * epsilon)
    ), dim=-1).transpose(-2, -1)

def soft_threshold(c, alpha, tau):
    sigmoid = lambda x: 1 / (1 + torch.exp(-x))
    return torch.stack((
        torch.ones_like(c),
        sigmoid(tau * (c - alpha))
    ), dim=-1).transpose(-2, -1)

def plot_gradient(grad, title, folder):
    """Visualise et sauvegarde le gradient seuillé."""
    plt.figure(figsize=(10, 5))
    plt.imshow(grad.squeeze().cpu().numpy(), cmap='gray')
    plt.title(title)
    plt.colorbar()
    save_path = os.path.join(folder, f'{title}.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Sauvegarde de {save_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Application des poids sur le gradient de l'image panchromatique")

    # Arguments principaux
    parser.add_argument('--storage_path', type=str, required=True, help='Chemin des résultats')
    parser.add_argument('--idxs', type=int, nargs='*', help='Indices des images à visualiser')

    # Options de seuillage
    parser.add_argument('--thresh', type=float, default=0.5, help='Seuil pour le seuillage')
    parser.add_argument('--weight_type', type=str, default="hard", choices=["hard", "soft"],
                       help="Type de fonction weight: hard (thresholding) ou soft (sigmoid)")
    parser.add_argument('--epsilon', type=float, default=1e-7,
                       help="Valeur epsilon pour hard thresholding (uniquement pour weight_type=hard)")
    parser.add_argument('--tau', type=float, default=10.0,
                       help="Paramètre tau pour soft thresholding (uniquement pour weight_type=soft)")

    args = parser.parse_args()

    try:
        # Chargement des images panchromatiques
        root = zarr.open(f'{args.storage_path}/results.zarr', mode='r')
        pan_noisy = root['pan_noise ']

        # Création du dossier de sortie
        fig_folder = os.path.join(args.storage_path, 'figures')
        os.makedirs(fig_folder, exist_ok=True)

        # Traitement des images
        idxs = args.idxs if args.idxs else range(pan_noisy.shape[0])
        for idx in idxs:
            pan_image = pan_noisy[idx]
            if pan_image.ndim == 3 and pan_image.shape[0] == 1:
                pan_image = pan_image[0]
            pan_image = torch.tensor(pan_image)
            pan_image = (pan_image - torch.min(pan_image)) / (torch.max(pan_image) - torch.min(pan_image) + 1e-7)  # Ajout de 1e-7 pour éviter la division par zéro

            weighted_grad = apply_threshold_to_gradient(
                pan_image, args.thresh, args.weight_type, args.epsilon, args.tau
            )
            plot_gradient(weighted_grad, f'weighted_grad_{idx}', fig_folder)

        print('Application des poids sur le gradient terminée!')

    except Exception as e:
        print(f"Une erreur s'est produite : {e}") 
