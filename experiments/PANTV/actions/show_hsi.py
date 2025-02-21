import sys
path = '/home/mhiriy/projects/spectral-spatial/src'
sys.path.append(f'{path}/algorithms')
sys.path.append(f'{path}/datasets')
sys.path.append(f'{path}/metrics')

import torch
import argparse
import os
import numpy as np
import zarr
import rich

from datasets import HSIDataset
from sklearn.decomposition import PCA
from math import sqrt
from torchvision import transforms
from PIL import Image



def load_dataset(data_path, data_idx):

    val_transform = transforms.Compose([transforms.ToTensor()]) # Transforms a the input data to torch tensors
    crop_transform = transforms.Compose([ transforms.ToTensor(), transforms.CenterCrop(crop_size)])
    if crop:
        dataset = HSIDataset(root_dir=data_path, split='train', transform=crop_transform, normalize=True)
    else:
        dataset = HSIDataset(root_dir=data_path, split='train', transform=val_transform, normalize=True)

    subset = torch.utils.data.Subset(dataset, data_idx)

    return subset

def generate_figure(image_array,
                    filename,
                    folder,
                    **kwargs):

    if image_array.ndim == 2:
        # Grayscale image
        mode = 'L'
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        # RGB image
        mode = 'RGB'
    else:
        raise ValueError("Input array must be either a 2D grayscale image or a 3D RGB image.")

    # Create PIL Image object
    pil_image = Image.fromarray(image_array, mode=mode)
    
    pil_image.save(os.path.join(folder, f'{filename}.png'))


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
        # Reshape the cube for PCA
        reshaped_hsi = hsi_cube.reshape(-1, num_bands)

        # Perform PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(reshaped_hsi)

        # Reshape back to image shape
        pca_image = pca_result.reshape(height, width, 3)

        # Normalize for display
        pca_image = (pca_image - np.min(pca_image)) / (np.max(pca_image) - np.min(pca_image))

        # Display PCA result as RGB
        filename = f'{name}_pca'
        generate_figure(pca_image, filename, folder)

    elif rgb_indices:
        # Extract RGB bands
        rgb_image = hsi_cube[:, :, rgb_indices]

        # Normalize for display
        rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))

        # Display RGB image
        filename = f'{name}_rgb'
        generate_figure(rgb_image, filename, folder)

    elif band_indices:
        # Display specific bands
        for idx in band_indices:
            filename = f'{name}_band_{idx}'
            band_image = hsi_cube[:, :, idx]
            generate_figure(band_image, filename, folder)

    elif eigen_indices:
        # Perform PCA to get eigenimages
        reshaped_hsi = hsi_cube.reshape(-1, num_bands)
        pca = PCA(n_components=num_bands)
        pca_result = pca.fit_transform(reshaped_hsi)

        # Reshape back to image shape
        eigenimages = pca_result.reshape(height, width, num_bands)

        # Display specific eigenimages
        for idx in eigen_indices:
            if idx < num_bands:
                eigenimage = eigenimages[:, :, idx]
                filename = f'{name}_eigenimage_{idx}'
                generate_figure(eigenimage, filename, folder)
            else:
                print(f"Eigenindex {idx} is out of bounds for the number of bands {num_bands}.")

    else:
        raise ValueError("Specify either 'rgb_indices', 'band_indices', 'eigen_indices', or set 'use_pca' to True.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    """
    <tocomplete>
    define the parser argument
    </tocomplete>
    
    """

    args = parser.parse_args()

    rich.print(
            '[bold green] <tocomplete> prompt a relevant print </tocomplete>')

    folder = args.storage_path
    root = zarr.open(f'{folder}/results.zarr', mode='r')
    data_idx = 

    ds = load_dataset(args.data_path)

    if args.noisy:
        residual_noise = root["residual_noise"]
        noise_map = root["noise_map"]
        noise_level = root.attrs["noise_level"]
        gt =         
        gt += 1e-2*residual_noise/np.linalg.norm(gt)

        noisy = gt + sqrt(noise_level)*noise_map
        
        visualize_hyperspectral_image()
    if args.reconstructed:
        reconstructed = root['reconstructed'][args.image_idx]

        visualize_hyperspectral_image()
    if args.ground_truth:

        visualize_hyperspectral_image()
    # get reconstructed images






