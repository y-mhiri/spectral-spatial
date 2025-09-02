import sys
import os

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

    # Normalisation et conversion en uint8
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-7)
    pil_image = Image.fromarray((image_array * 255).astype(np.uint8), mode=mode)
    save_path = os.path.join(folder, f'{filename}.png')
    pil_image.save(save_path)
    print(f'Sauvegarde de {save_path}')

def apply_threshold_to_gradient(pan_image, thresh, weight_type, epsilon=None, tau=None):
    """Applique un seuillage sur le gradient de l'image panchromatique."""
    # Conversion en tensor si nécessaire
    if not isinstance(pan_image, torch.Tensor):
        pan_image = torch.tensor(pan_image)
    
    # Ajout des dimensions batch et channel si nécessaire
    if pan_image.ndim == 2:
        pan_image = pan_image.unsqueeze(0).unsqueeze(0)
    elif pan_image.ndim == 3:
        pan_image = pan_image.unsqueeze(0)
    
    # Calcul du gradient
    grad_panc = nabla(pan_image)
    
    # Calcul de la norme du gradient
    norm_grad_panc = norm(grad_panc, dim=-1, keepdim=True)
    
    # Application du seuillage
    if weight_type == "hard":
        weights = hard_threshold(norm_grad_panc, thresh, epsilon)
    else:
        weights = soft_threshold(norm_grad_panc, thresh, tau)
    
    # Calcul du gradient orthogonal
    grad_panc_orth = torch.zeros_like(grad_panc)
    grad_panc_orth[...,0] = grad_panc[...,1]
    grad_panc_orth[...,1] = -grad_panc[...,0]
    
    # Normalisation
    grad_panc_norm = grad_panc / (norm_grad_panc + 1e-7)
    grad_panc_orth_norm = grad_panc_orth / (norm_grad_panc + 1e-7)
    
    # Application des poids
    weighted_grad = weights * torch.stack((grad_panc_orth_norm, grad_panc_norm), dim=-1)
    
    return weighted_grad.squeeze().cpu().numpy()

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

def visualize_gradient_results(pan_image, grad, weighted_grad, idx, folder):
    """Visualisation complète des résultats"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image panchromatique
    axes[0].imshow(pan_image.squeeze(), cmap='gray')
    axes[0].set_title('Image Panchromatique')
    
    # Gradient original
    grad_magnitude = np.linalg.norm(grad, axis=-1)
    im = axes[1].imshow(grad_magnitude, cmap='jet')
    axes[1].set_title('Gradient Original')
    plt.colorbar(im, ax=axes[1])
    
    # Gradient pondéré
    weighted_magnitude = np.linalg.norm(weighted_grad, axis=-1)
    im = axes[2].imshow(weighted_magnitude, cmap='jet')
    axes[2].set_title('Gradient Pondéré')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    save_path = os.path.join(folder, f'gradient_comparison_{idx}.png')
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
                       help="Valeur epsilon pour hard thresholding")
    parser.add_argument('--tau', type=float, default=10.0,
                       help="Paramètre tau pour soft thresholding")

    args = parser.parse_args()

    try:
        # Chargement des données
        root = zarr.open(f'{args.storage_path}/results.zarr', mode='r')
        pan_noisy = root['pan_noise ']
        
        # Création du dossier de sortie
        fig_folder = os.path.join(args.storage_path, 'gradient_analysis')
        os.makedirs(fig_folder, exist_ok=True)

        # Traitement des images
        idxs = args.idxs if args.idxs else range(len(pan_noisy))
        
        for idx in idxs:
            # Chargement de l'image
            pan_image = pan_noisy[idx]
            if pan_image.ndim == 3 and pan_image.shape[0] == 1:
                pan_image = pan_image[0]
            
            # Conversion et normalisation
            pan_tensor = torch.tensor(pan_image, dtype=torch.float32)
            pan_tensor = (pan_tensor - torch.min(pan_tensor)) / (torch.max(pan_tensor) - torch.min(pan_tensor) + 1e-7)
            
            # Calcul du gradient original
            grad_panc = nabla(pan_tensor.unsqueeze(0).unsqueeze(0))
            
            # Application du seuillage
            weighted_grad = apply_threshold_to_gradient(
                pan_tensor, 
                args.thresh, 
                args.weight_type, 
                args.epsilon, 
                args.tau
            )
            
            # Visualisation
            visualize_gradient_results(
                pan_tensor.numpy(),
                grad_panc.squeeze().cpu().numpy(),
                weighted_grad,
                idx,
                fig_folder
            )

        print('Traitement terminé avec succès!')

    except Exception as e:
        print(f"Erreur lors du traitement: {str(e)}")