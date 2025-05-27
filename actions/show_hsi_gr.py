import sys
import os
import argparse
import yaml
import numpy as np
import zarr
import rich
import torch
from sklearn.decomposition import PCA
from scipy.linalg import svd
from PIL import Image
from torchvision import transforms

# Chemin des modules
path = "/home/ndiayem/Documents/spectral-spatial/src"
sys.path.append(f'{path}/algorithms')
sys.path.append(f'{path}/datasets')
sys.path.append(f'{path}/metrics')

from pansharpening import PANDataset

def load_dataset(data_path, data_idx, crop, crop_size, scale, sigma, noise_level, device, seed):
    """Charge le dataset hyperspectral avec option de recadrage"""
    crop_transform = transforms.Compose([transforms.CenterCrop(crop_size)])
    if crop:
        dataset = PANDataset(root_dir=data_path, split='train', transform=crop_transform,
                           normalize=True, scale=scale, sigma=sigma, sigma1=noise_level,
                           device=device, size=crop_size, seed=seed)
    else:
        dataset = PANDataset(root_dir=data_path, split='train', transform=None,
                           normalize=True, scale=scale, sigma=sigma, sigma1=noise_level,
                           device=device, size=None, seed=seed)
    subset = torch.utils.data.Subset(dataset, data_idx)
    return subset, dataset.rgb_index

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

def get_eigenimages(hsi_data, return_eigenvalues=False):
    """Calcule les eigenimages via SVD"""
    assert len(hsi_data.shape) == 3, "Input must be [bands, height, width]"
    num_bands, height, width = hsi_data.shape
    hsi_data_squeezed = hsi_data.reshape(num_bands, -1)
    _, evalue, V = svd(hsi_data_squeezed, full_matrices=False)
    eimage = V.reshape(num_bands, height, width)
    return (eimage, evalue) if return_eigenvalues else eimage

def visualize_hyperspectral_image(hsi_cube, name='image', eigen_indices=None, 
                                  rgb_indices=None, band_indices=None, 
                                  show_eigenimage=False, folder='.'):
    """Visualise un cube hyperspectral selon différents modes"""
    num_bands, height, width = hsi_cube.shape

    # Visualiser l'image propre
    #clean_image = hsi_cube  # Utiliser l'image propre
    #clean_image_normalized = (clean_image - np.min(clean_image)) / (np.max(clean_image) - np.min(clean_image))
    #generate_figure(clean_image_normalized, f'{name}_clean_image', folder)

    # Visualiser toutes les eigenimages
    if show_eigenimage:
        eigenimage1 = get_eigenimages(hsi_cube)[0:3].swapaxes(0,2).swapaxes(0,1)
        eigenimage1 = (eigenimage1 - np.min(eigenimage1)) / (np.max(eigenimage1) - np.min(eigenimage1))
        eigenimages = get_eigenimages(hsi_cube).swapaxes(0, 2).swapaxes(0, 1)
        for idx in range(num_bands):
            eigenimage = eigenimages[:, :, idx]
            eigenimage = (eigenimage - np.min(eigenimage)) / (np.max(eigenimage) - np.min(eigenimage))
            generate_figure(eigenimage, f'{name}_eigenimage_{idx}', folder)
        generate_figure(eigenimage1, f'{name}_eigenimage', folder)

    if rgb_indices:
        rgb_image = hsi_cube[rgb_indices, :, :].swapaxes(0, 2).swapaxes(0, 1)
        rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
        generate_figure(rgb_image, f'{name}_rgb', folder)

    if band_indices:
        for idx in band_indices:
            band_image = hsi_cube[idx, :, :]
            band_image = (band_image - np.min(band_image)) / (np.max(band_image) - np.min(band_image))
            generate_figure(band_image, f'{name}_band_{idx}', folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation d'images hyperspectrales")
    
    # Arguments principaux
    parser.add_argument('--storage_path', type=str, required=True, help='Chemin des résultats')
    parser.add_argument('--idxs', type=int, nargs='*', help='Indices des images à visualiser')
    
    # Options de visualisation
    parser.add_argument('--hsi_low_noisy', action='store_true', help='Visualiser HSI bruité')
    parser.add_argument('--reconstructed', action='store_true', help='Visualiser les reconstructions')
    parser.add_argument('--ground_truth', action='store_true', help='Visualiser la vérité terrain')
    parser.add_argument('--panchromatic_noisy', action='store_true', 
                       help='Visualiser les images panchromatiques bruitées')
    parser.add_argument('--rgb', action='store_true', help='Afficher en RGB')
    parser.add_argument('--eigenimage', action='store_true', 
                       help='Afficher toutes les eigenimages')
    parser.add_argument('--band_indices', type=int, nargs='+', 
                       help='Indices des bandes à visualiser')
    
    # Options supplémentaires
    parser.add_argument('--dataset', type=str, help='Nom du dataset à utiliser')

    args = parser.parse_args()
    rich.print('[bold green]Début de la visualisation...')

    # Lister tous les groupes
    group_dirs = [d for d in os.listdir(args.storage_path) if os.path.isdir(os.path.join(args.storage_path, d))]

    for group in group_dirs:
        folder = os.path.join(args.storage_path, group)
        rich.print(f'[bold green]Traitement du groupe : {group}...[/bold green]')
        
        # Vérifiez si le fichier info.yaml existe
        info_path = os.path.join(folder, 'info.yaml')
        if not os.path.exists(info_path):
            print(f"[red]Fichier info.yaml introuvable dans {folder}[/red]")
            continue
        
        with open(info_path, 'r') as f:
            info = yaml.safe_load(f)
        
        dataset_path = info['datasets'][args.dataset] if args.dataset else next(iter(info['datasets'].values()))
        
        # Chargement des paramètres
        root = zarr.open(f'{folder}/results.zarr', mode='r')
        attrs = root.attrs
        ds, rgb_index = load_dataset(
            dataset_path, attrs['data_idx'], attrs['crop'], attrs['crop_size'],
            attrs['scale'], attrs['sigma'], attrs['noise_level'],
            attrs['device'], attrs['seed']
        )
        
        # Création du dossier de sortie
        fig_folder = os.path.join(folder, 'figures')
        os.makedirs(fig_folder, exist_ok=True)
        
        # Configuration des visualisations
        viz_args = {
            'rgb_indices': rgb_index if args.rgb else None,
            'show_eigenimage': args.eigenimage,
            'band_indices': args.band_indices,
            'folder': fig_folder
        }
        
        # Traitement des images
        idxs = args.idxs if args.idxs else range(len(ds))
        for idx in idxs:
            # Nouvelle visualisation panchromatique
            if args.panchromatic_noisy:
                pan_noisy = root['pan_noise '][idx]
                if pan_noisy.ndim == 3 and pan_noisy.shape[0] == 1:
                    pan_noisy = pan_noisy[0]
                pan_noisy = (pan_noisy - np.min(pan_noisy)) / (np.max(pan_noisy) - np.min(pan_noisy))
                generate_figure(pan_noisy, f'pan_noisy_{idx}', fig_folder)
            
            # Visualisations existantes
            if args.hsi_low_noisy:
                hsi_low_noisy = root['hsi_noise'][idx]  # Format [C, H, W]
                visualize_hyperspectral_image(hsi_low_noisy, f'hsi_low_noisy_{idx}', **viz_args)
                
            if args.reconstructed:
                reconstructed = root['reconstructed'][idx]
                visualize_hyperspectral_image(reconstructed, f'reconstructed_{idx}', **viz_args)
                
            if args.ground_truth:
                gt = ds[idx].numpy()
                visualize_hyperspectral_image(gt, f'ground_truth_{idx}', **viz_args)

    rich.print('[bold green]Visualisation terminée![/bold green]')