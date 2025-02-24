import sys
import os
import argparse
import numpy as np
import zarr
import rich
import torch
from PIL import Image
from sklearn.decomposition import PCA
from math import sqrt
from torchvision import transforms

# Assuming the necessary modules are in the specified path
path = '/home/mhiriy/projects/spectral-spatial/src'
sys.path.append(f'{path}/algorithms')
sys.path.append(f'{path}/datasets')
sys.path.append(f'{path}/metrics')

from datasets import HSIDataset

def load_dataset(data_path, data_idx, crop, crop_size):
    """Load the hyperspectral dataset with optional cropping."""
    val_transform = transforms.Compose([transforms.ToTensor()])
    crop_transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(crop_size)])

    if crop:
        dataset = HSIDataset(root_dir=data_path, split='train', transform=crop_transform, normalize=True)
    else:
        dataset = HSIDataset(root_dir=data_path, split='train', transform=val_transform, normalize=True)

    subset = torch.utils.data.Subset(dataset, data_idx)
    return subset

def generate_figure(image_array, filename, folder):
    """Generate and save a figure from an image array."""
    if image_array.ndim == 2:
        mode = 'L'
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        mode = 'RGB'
    else:
        raise ValueError("Input array must be either a 2D grayscale image or a 3D RGB image.")

    pil_image = Image.fromarray((image_array * 255).astype(np.uint8), mode=mode)
    save_path = os.path.join(folder, f'{filename}.png')
    pil_image.save(save_path)
    print(f'Saved image to {save_path}')

def visualize_hyperspectral_image(hsi_cube, name='image', eigen_indices=None, rgb_indices=None, band_indices=None, use_pca=False, folder='.'):
    """
    Visualize a hyperspectral image cube.

    Parameters:
    - hsi_cube: 3D NumPy array (height, width, bands) representing the hyperspectral image.
    - name: Base name for saving the images.
    - eigen_indices: List of indices for specific eigenimages to display.
    - rgb_indices: List of three indices corresponding to the RGB bands.
    - band_indices: List of indices for specific bands to display.
    - use_pca: Boolean flag to use PCA for visualization.
    - folder: Directory to save the images.
    """
    height, width, num_bands = hsi_cube.shape

    if use_pca:
        reshaped_hsi = hsi_cube.reshape(-1, num_bands)
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(reshaped_hsi)
        pca_image = pca_result.reshape(height, width, 3)
        pca_image = (pca_image - np.min(pca_image)) / (np.max(pca_image) - np.min(pca_image))
        filename = f'{name}_pca'
        generate_figure(pca_image, filename, folder)

    elif rgb_indices:
        rgb_image = hsi_cube[:, :, rgb_indices]
        rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
        filename = f'{name}_rgb'
        generate_figure(rgb_image, filename, folder)

    elif band_indices:
        for idx in band_indices:
            filename = f'{name}_band_{idx}'
            band_image = hsi_cube[:, :, idx]
            band_image = (band_image - np.min(band_image)) / (np.max(band_image) - np.min(band_image))
            generate_figure(band_image, filename, folder)

    elif eigen_indices:
        reshaped_hsi = hsi_cube.reshape(-1, num_bands)
        pca = PCA(n_components=num_bands)
        pca_result = pca.fit_transform(reshaped_hsi)
        eigenimages = pca_result.reshape(height, width, num_bands)
        for idx in eigen_indices:
            if idx < num_bands:
                eigenimage = eigenimages[:, :, idx]
                eigenimage = (eigenimage - np.min(eigenimage)) / (np.max(eigenimage) - np.min(eigenimage))
                filename = f'{name}_eigenimage_{idx}'
                generate_figure(eigenimage, filename, folder)
            else:
                print(f"Eigenindex {idx} is out of bounds for the number of bands {num_bands}.")

    else:
        raise ValueError("Specify either 'rgb_indices', 'band_indices', 'eigen_indices', or set 'use_pca' to True.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hyperspectral images.")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--storage_path', type=str, required=True, help='Path to store results.')
    parser.add_argument('--idxs', type=int, nargs='*', help='Indices of the data to visualize.')
    parser.add_argument('--noisy', action='store_true', help='Visualize noisy images.')
    parser.add_argument('--reconstructed', action='store_true', help='Visualize reconstructed images.')
    parser.add_argument('--ground_truth', action='store_true', help='Visualize ground truth images.')
    parser.add_argument('--rgb', type=bool, default=False, help='Show RGB if set to True.')
    parser.add_argument('--show_pca',type=bool, default=False, help='Show the 3 first eigenimages as an RGB image.')
    parser.add_argument('--band_indices', type=int, help='Indices of the bands to show. No band images are displayed if None.')
    parser.add_argument('--eigen_indices', type=int, help='Indices of the eigenimages to show. No eigenimages are displayed if None.')

    args = parser.parse_args()

    rich.print('[bold green]Starting visualization process...')

    folder = args.storage_path
    root = zarr.open(f'{folder}/results.zarr', mode='r')
    data_idx = root['attrs']['data_idx']
    crop = root['attrs']['crop']
    crop_size = root['attrs']['crop_size']

    ds = load_dataset(args.data_path, data_idx, crop, crop_size)
    idxs = args.idxs if args.idxs else range(len(ds))

    viz_args = {
            'rgb_indices': ds.rgb_index if args.rgb else None,
            'use_pca': args.show_pca,
            'band_indices': args.band_indices,
            'eigen_indices': args.eigen_indices,
            'folder': folder
        }

    for idx in idxs:
        if args.noisy:
            residual_noise = root["residual_noise"]
            noise_map = root["noise_map"]
            noise_level = root.attrs["noise_level"]
            gt = ds[idx].numpy()
            gt += 1e-2 * residual_noise / np.linalg.norm(gt)
            noisy = gt + sqrt(noise_level) * noise_map
            visualize_hyperspectral_image(noisy, name=f'noisy_{idx}', **viz_args)

        if args.reconstructed:
            reconstructed = root['reconstructed'][idx]
            visualize_hyperspectral_image(reconstructed, name=f'reconstructed_{idx}', **viz_args)

        if args.ground_truth:
            gt = ds[idx].numpy()
            visualize_hyperspectral_image(gt, name=f'ground_truth_{idx}', **viz_args)
