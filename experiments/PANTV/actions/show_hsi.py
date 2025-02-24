import sys
import os
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
    return subset, dataset.rgb_index

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

def get_eigenimages(hsi_data, return_eigenvalues=False):

    assert len(hsi_data.shape) == 3, "Input must be a 3D tensor of shape [channels, height, width]"

    num_bands, height, width = hsi_data.shape

    # Reshape HSI cubes to matrices of size [number of bands, number of pixels]
    hsi_data_squeezed = hsi_data.reshape(num_bands, -1)
    # Singular Value Decomposition of noisy and true HSI
    _, evalue, V = svd(hsi_data_squeezed, full_matrices=False)
    # Eigenimages are the coefficient images of each HSI in the basis formed by its eigenvectors
    eimage_squeezed = V
    # eimage_squeezed = np.diag(evalue) @ V
    eimage = eimage_squeezed.reshape(num_bands, height,width)

    if return_eigenvalues:
        return eimage, evalue

    return eimage

def visualize_hyperspectral_image(hsi_cube, name='image', eigen_indices=None, rgb_indices=None, band_indices=None, show_eigenimage=False, folder='.'):
    """
    Visualize a hyperspectral image cube.

    Parameters:
    - hsi_cube: 3D NumPy array (bands, height, width) representing the hyperspectral image.
    - name: Base name for saving the images.
    - eigen_indices: List of indices for specific eigenimages to display.
    - rgb_indices: List of three indices corresponding to the RGB bands.
    - band_indices: List of indices for specific bands to display.
    - show_eigenimage: Boolean flag to use PCA for visualization.
    - folder: Directory to save the images.
    """
    num_bands, height, width = hsi_cube.shape

    if show_eigenimage:
        eigenimage = get_eigenimages(hsi_cube)[0:3].swapaxes(0,2).swapaxes(0,1)
        eigenimage = (eigenimage - np.min(eigenimage)) / (np.max(eigenimage) - np.min(eigenimage))
        filename = f'{name}_eigenimage'
        generate_figure(eigenimage, filename, folder)

    if rgb_indices:
        rgb_image = hsi_cube[rgb_indices, :, :].swapaxes(0,2).swapaxes(0,1)
        rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
        filename = f'{name}_rgb'
        generate_figure(rgb_image, filename, folder)

    if band_indices:
        for idx in band_indices:
            filename = f'{name}_band_{idx}'
            band_image = hsi_cube[idx, :, :]
            band_image = (band_image - np.min(band_image)) / (np.max(band_image) - np.min(band_image))
            generate_figure(band_image, filename, folder)

    if eigen_indices:
        eigenimages = get_eigenimages(hsi_cube).swapaxes(0,2).swapaxes(0,1)

        for idx in eigen_indices:
            if idx < num_bands:
                eigenimage = eigenimages[:, :, idx]
                eigenimage = (eigenimage - np.min(eigenimage)) / (np.max(eigenimage) - np.min(eigenimage))
                filename = f'{name}_eigenimage_{idx}'
                generate_figure(eigenimage, filename, folder)
            else:
                print(f"Eigenindex {idx} is out of bounds for the number of bands {num_bands}.")

    if not (show_eigenimage and eigen_indices and rgb_indices and band_indices):
        print("To plot somethin, specify either 'rgb_indices', 'band_indices', 'eigen_indices', or set 'show_eigenimage' to True.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hyperspectral images.")

    parser.add_argument('--storage_path', type=str, required=True, help='Path to store results.')
    parser.add_argument('--idxs', type=int, nargs='*', help='Indices of the data to visualize.')
    parser.add_argument('--noisy', action='store_true', help='Visualize noisy images.')
    parser.add_argument('--reconstructed', action='store_true', help='Visualize reconstructed images.')
    parser.add_argument('--ground_truth', action='store_true', help='Visualize ground truth images.')
    parser.add_argument('--rgb', action='store_true', help='Show RGB if set to True.')
    parser.add_argument('--eigenimage',action='store_true', help='Show the 3 first eigenimages as an RGB image.')
    parser.add_argument('--band_indices', type=int, nargs='+', help='Indices of the bands to show. No band images are displayed if None.')
    parser.add_argument('--eigen_indices', type=int, nargs='+', help='Indices of the eigenimages to show. No eigenimages are displayed if None.')
    parser.add_argument('--dataset', type=str, help='Name of the dataset to use. Default take the first dataset in the list.')

    args = parser.parse_args()

    rich.print('[bold green]Starting visualization process...')

    folder = args.storage_path

    with open(os.path.join(folder, 'info.yaml'),'r') as f:
        info = yaml.load(f, yaml.SafeLoader )
    
    
    datasets = info['datasets']

    if args.dataset:
        dataset_path = datasets['dataset']
    else:
        ds_keys = list(datasets.keys())
        dataset_path = datasets[ds_keys[0]]

    root = zarr.open(f'{folder}/results.zarr', mode='r')
    data_idx = root.attrs['data_idx']
    crop = root.attrs['crop']
    crop_size = root.attrs['crop_size']

    ds, rgb_index = load_dataset(dataset_path, data_idx, crop, crop_size)
    idxs = args.idxs if args.idxs else range(len(ds))

    viz_args = {
            'rgb_indices': rgb_index if args.rgb else None,
            'show_eigenimage': args.eigenimage,
            'band_indices': args.band_indices,
            'eigen_indices': args.eigen_indices,
            'folder': folder
        }

    for idx in idxs:
        if args.noisy:
            residual_noise = root["residual_noise"][:]
            noise_map = root["noise_map"][:]
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
