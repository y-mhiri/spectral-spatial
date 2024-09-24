"""

Define the HSIDataset class used to create hyperspectral dataset objects.


"""

import torch
import zarr

from torch.utils import data
from torch.linalg import svd

def get_eigenimages(hsi_data):

    b,c,h,w = hsi_data.shape
    assert b == 1, "Batch size must be 1"

    hsi_data = hsi_data.reshape(b*c,h*w)
    u, s, v = svd(hsi_data, full_matrices=False)
    acp = (u[:, :3] @ torch.diag(s[:3]) @ v[:3]).reshape(hsi_data.shape)
    # acp = (acp - acp.min()) / (acp.max() - acp.min())

    return acp.reshape(b,c,h,w), s 

class HSIDataset(data.Dataset):

    def __init__(self, root_dir, split='train', transform=None):

        self.transform = transform
        
        if root_dir.endswith('.zarr'):
            self.file = zarr.open(root_dir, mode="r")


        self.split = split
        self.rgb_index = self.file.attrs['rgb']
        self.wavenumbers = self.file.attrs['spectral_range']
        self.spatial_resolution = self.file.attrs['spatial_resolution (m)']
        self.spectral_resolution = self.file.attrs['spectral_resolution (nm)']
        self.nband = len(self.wavenumbers)

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
        hsi_data = self.file[self.split][index][:]

        if self.transform:
            hsi_data = self.transform(hsi_data)

        return hsi_data       

    
    def __len__(self) -> int:
        return len(self.file[self.split])
    

