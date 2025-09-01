import os
import torch

from scipy.io import loadmat

def load_pavia(hsi_filename: str = "PaviaU.mat"):
    """
    Load the Indian Pines hyperspectral dataset.
    
    Parameters:
    -----------
    hsi_filename : str
        Filename of the hyperspectral data

        
    Returns:
    --------
    torch.tensor
        HSI data (1 x B x H x W)
    """
    # Load hyperspectral data
    hsi_mat = loadmat(hsi_filename)
    hsi_data = torch.tensor(hsi_mat['pavia'], dtype=torch.float32).swapaxes(-1,0).swapaxes(1,2).unsqueeze(0)

    

    print(f"Loaded HSI data with shape: {hsi_data.shape}")
    print(f"Number of spectral bands: {hsi_data.shape[1]}")

    
    return hsi_data

