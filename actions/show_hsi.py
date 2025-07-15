import sys
import os
# Chemin des modules
path = "/home/ndiayem/Documents/spectral-spatial/src"
sys.path.append(f'{path}/algorithms')
sys.path.append(f'{path}/datasets')
sys.path.append(f'{path}/metrics')

from pansharpening import PANDataset

import argparse
import yaml
import numpy as np
import zarr
import rich
import torch
from PIL import Image
from sklearn.decomposition import PCA
from scipy.linalg import svd
from math import sqrt
from torchvision import transforms

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

    if show_eigenimage:
        eigenimage = get_eigenimages(hsi_cube)[0:3].swapaxes(0,2).swapaxes(0,1)
        eigenimage = (eigenimage - np.min(eigenimage)) / (np.max(eigenimage) - np.min(eigenimage))
        generate_figure(eigenimage, f'{name}_eigenimage', folder)

    if rgb_indices:
        rgb_image = hsi_cube[rgb_indices, :, :].swapaxes(0,2).swapaxes(0,1)
        rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
        generate_figure(rgb_image, f'{name}_rgb', folder)

    if band_indices:
        for idx in band_indices:
            band_image = hsi_cube[idx, :, :]
            band_image = (band_image - np.min(band_image)) / (np.max(band_image) - np.min(band_image))
            generate_figure(band_image, f'{name}_band_{idx}', folder)

    if eigen_indices:
        eigenimages = get_eigenimages(hsi_cube).swapaxes(0,2).swapaxes(0,1)
        for idx in eigen_indices:
            if idx < num_bands:
                eigenimage = eigenimages[:, :, idx]
                eigenimage = (eigenimage - np.min(eigenimage)) / (np.max(eigenimage) - np.min(eigenimage))
                generate_figure(eigenimage, f'{name}_eigenimage_{idx}', folder)
            else:
                print(f"Index {idx} hors limites (max={num_bands-1})")

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
                       help='Afficher les 3 premières eigenimages en RGB')
    parser.add_argument('--band_indices', type=int, nargs='+', 
                       help='Indices des bandes à visualiser')
    parser.add_argument('--eigen_indices', type=int, nargs='+', 
                       help='Indices des eigenimages à visualiser')
    
    # Options supplémentaires
    parser.add_argument('--dataset', type=str, help='Nom du dataset à utiliser')
    parser.add_argument('--group', type=int, nargs='*', help='Numéro(s) de groupe à traiter')
    parser.add_argument('--all_groups', action='store_true', 
                       help='Traiter tous les groupes disponibles')

    args = parser.parse_args()
    rich.print('[bold green]Début de la visualisation...')

    # Déterminer les groupes à traiter
    if args.all_groups:
        # Trouver tous les groupes dans le dossier de stockage
        groups = []
        for item in os.listdir(args.storage_path):
            if item.startswith('group_') and os.path.isdir(os.path.join(args.storage_path, item)):
                try:
                    groups.append(int(item.split('_')[1]))
                except ValueError:
                    continue
        groups = sorted(groups)
    elif args.group:
        groups = args.group
    else:
        groups = [0]  # Par défaut, traiter le groupe 0

    # Traiter chaque groupe
    for group_num in groups:
        group_path = os.path.join(args.storage_path, f'group_{group_num}')
        if not os.path.exists(group_path):
            rich.print(f"[red]Le groupe {group_num} n'existe pas dans {args.storage_path}[/red]")
            continue

        rich.print(f"\n[bold]Traitement du groupe {group_num}...")
        
        try:
            #Charger la configuration
            info_path = os.path.join(group_path, 'info.yaml')
            if not os.path.exists(info_path):
               rich.print(f"[red]Fichier info.yaml introuvable dans {group_path}[/red]")
               continue

            with open(info_path, 'r') as f:
                 info = yaml.safe_load(f)
            
            dataset_path = info['datasets'][args.dataset] if args.dataset else next(iter(info['datasets'].values()))
            #dataset_path = "/uds_data/listic/mhiriy/data/harvard.zarr"
            info['datasets'][args.dataset] if args.dataset else next(iter(info['datasets'].values()))
            
            # Chargement des paramètres
            zarr_path = os.path.join(group_path, 'results.zarr')
            if not os.path.exists(zarr_path):
                rich.print(f"[red]Fichier results.zarr introuvable dans {group_path}[/red]")
                continue

            root = zarr.open(zarr_path, mode='r')
            attrs = root.attrs
            ds, rgb_index = load_dataset(
                dataset_path, attrs['data_idx'], attrs['crop'], attrs['crop_size'],
                attrs['scale'], attrs['sigma'], attrs['noise_level'],
                attrs['device'], attrs['seed']
            )
            
            # Création du dossier de sortie
            fig_folder = os.path.join(group_path, 'figures')
            os.makedirs(fig_folder, exist_ok=True)
            
            # Configuration des visualisations
            viz_args = {
                'rgb_indices': rgb_index if args.rgb else None,
                'show_eigenimage': args.eigenimage,
                'band_indices': args.band_indices,
                'eigen_indices': args.eigen_indices,
                'folder': fig_folder
            }
            
            # Traitement des images
            idxs = args.idxs if args.idxs else range(len(ds))
            for idx in idxs:
                try:
                    if args.panchromatic_noisy and 'pan_noise ' in root:
                        pan_noisy = root['pan_noise '][idx]
                        if pan_noisy.ndim == 3 and pan_noisy.shape[0] == 1:
                            pan_noisy = pan_noisy[0]
                        pan_noisy = (pan_noisy - np.min(pan_noisy)) / (np.max(pan_noisy) - np.min(pan_noisy))
                        generate_figure(pan_noisy, f'pan_noisy_{idx}', fig_folder)
                    
                    if args.hsi_low_noisy and 'hsi_noise' in root:
                        hsi_low_noisy = root['hsi_noise'][idx]
                        visualize_hyperspectral_image(hsi_low_noisy, f'hsi_low_noisy_{idx}', **viz_args)
                        
                    if args.reconstructed and 'reconstructed' in root:
                        reconstructed = root['reconstructed'][idx]
                        visualize_hyperspectral_image(reconstructed, f'reconstructed_{idx}', **viz_args)
                        
                    if args.ground_truth:
                        gt = ds[idx].numpy()
                        visualize_hyperspectral_image(gt, f'ground_truth_{idx}', **viz_args)
                except Exception as e:
                    rich.print(f"[red]Erreur lors du traitement de l'image {idx} du groupe {group_num}: {str(e)}[/red]")
        
        except Exception as e:
            rich.print(f"[red]Erreur lors du traitement du groupe {group_num}: {str(e)}[/red]")

    rich.print('[bold green]Visualisation terminée!')