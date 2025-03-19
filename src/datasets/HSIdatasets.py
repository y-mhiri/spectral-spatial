import torch
import zarr

from torch.utils import data
from torch.linalg import svd

def get_eigenimages(hsi_data, return_eigenvalues=False):

    assert len(hsi_data.shape) == 4, "Input must be a 4D tensor of shape [batch, channels, height, width]"

    # Reshape HSI cubes to matrices of size [number of bands, number of pixels]
    x_mat = hsi_data.reshape(hsi_data.shape[1], -1)
    # Singular Value Decomposition of noisy and true HSI
    U, s, V = svd(x_mat, full_matrices=False)
    # Eigenimages are the coefficient images of each HSI in the basis formed by its eigenvectors
    Z_mat = torch.diag(s) @ V
    Z = Z_mat.reshape(hsi_data.shape)

    if return_eigenvalues:
        return Z, s
    
    return Z

class HSIDataset(data.Dataset):

    def __init__(self, root_dir, split='train', transform=None, normalize=False):

        self.transform = transform
        self.normalize = normalize
        
        if root_dir.endswith('.zarr'):
            self.file = zarr.open(root_dir, mode="r")


        self.split = split
        self.rgb_index = self.file.attrs['rgb']
        self.wavenumbers = self.file.attrs['spectral_range']
        self.spatial_resolution = self.file.attrs['spatial_resolution (m)']
        self.spectral_resolution = self.file.attrs['spectral_resolution (nm)']
        self.nband = self.file[self.split][0][:].shape[2]
        self.height = self.file[self.split][0][:].shape[0]
        self.width = self.file[self.split][0][:].shape[1]

        pass

    def get_panchromatic(self, index):
        hsi_data = self.file[self.split][index][:]
        panchromatic = hsi_data.mean(axis=-1)
        return panchromatic

    def get_spectrum(self, index, row=None, col=None):
        hsi_data = self.file[self.split][index][:]
        if row is not None and col is not None:
            spectrum = hsi_data[row, col]
        else:
            row, col = hsi_data.shape[0] // 2, hsi_data.shape[1] // 2
            spectrum = hsi_data[row, col]
        return spectrum

    def get_rgb(self, index):
        hsi_data = self.file[self.split][index][:]
        rgb = hsi_data[:,:,self.rgb_index]
        return rgb
    
    def get_eigenimages(self, index):
        pass
        
    def __getitem__(self, index):

        if index >= len(self.file[self.split]):
            raise IndexError("Index out of range")
        
        hsi_data = self.file[self.split][index][:]

        if self.transform:
            hsi_data = self.transform(hsi_data)

        if self.normalize:
            hsi_data = (hsi_data - torch.min(hsi_data)) / (torch.max(hsi_data) - torch.min(hsi_data))

        return hsi_data       

    
    def __len__(self) -> int:
        return len(self.file[self.split])
    


    def __init__(self, root_dir, split='train', transform=None, normalize=False):

        self.transform = transform
        self.normalize = normalize
        
        if root_dir.endswith('.zarr'):
            self.file = zarr.open(root_dir, mode="r")


        self.split = split
        self.rgb_index = self.file.attrs['rgb']
        self.wavenumbers = self.file.attrs['spectral_range']
        self.spatial_resolution = self.file.attrs['spatial_resolution (m)']
        self.spectral_resolution = self.file.attrs['spectral_resolution (nm)']
        self.nband = self.file[self.split][0][:].shape[2]
        self.height = self.file[self.split][0][:].shape[0]
        self.width = self.file[self.split][0][:].shape[1]

        pass

    def get_panchromatic(self, index):
        hsi_data = self.file[self.split][index][:]
        panchromatic = hsi_data.mean(axis=-1)
        return panchromatic

    def get_spectrum(self, index, row=None, col=None):
        hsi_data = self.file[self.split][index][:]
        if row is not None and col is not None:
            spectrum = hsi_data[row, col]
        else:
            row, col = hsi_data.shape[0] // 2, hsi_data.shape[1] // 2
            spectrum = hsi_data[row, col]
        return spectrum

    def get_rgb(self, index):
        hsi_data = self.file[self.split][index][:]
        rgb = hsi_data[:,:,self.rgb_index]
        return rgb
    
    def get_eigenimages(self, index):
        pass
        
    def __getitem__(self, index):

        if index >= len(self.file[self.split]):
            raise IndexError("Index out of range")
        
        hsi_data = self.file[self.split][index][:]

        if self.transform:
            hsi_data = self.transform(hsi_data)

        if self.normalize:
            hsi_data = (hsi_data - torch.min(hsi_data)) / (torch.max(hsi_data) - torch.min(hsi_data))

        return hsi_data       

    
    def __len__(self) -> int:
        return len(self.file[self.split])
    

