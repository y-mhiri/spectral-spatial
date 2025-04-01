import torch
import torch.nn as nn
from deepinv.physics import Blur, Downsampling
from torchvision.transforms.functional import gaussian_blur
import zarr
from torch.utils import data

class PANDatasetImproved(data.Dataset):
    def __init__(self, root_dir, split='train', transform=None, normalize=False, 
                 scale=4, sigma=1.0, device="cpu"):
        super().__init__()
        
        # Chargement des données originales
        self.file = zarr.open(root_dir, mode='r')
        self.split = split
        self.transform = transform
        self.normalize = normalize
        
        # Paramètres de traitement
        self.scale = scale
        self.sigma = sigma
        self.device = device
        
        # Métadonnées
        self.rgb_index = self.file.attrs['rgb']
        self.wavenumbers = self.file.attrs['spectral_range']
        self.spatial_resolution = self.file.attrs['spatial_resolution (m)']
        self.spectral_resolution = self.file.attrs['spectral_resolution (nm)']
        self.nband = self.file[self.split][0][:].shape[2]
        self.height = self.file[self.split][0][:].shape[0]
        self.width = self.file[self.split][0][:].shape[1]
        
        # Initialisation des opérateurs DeepInv
        self._init_operators()

    def _init_operators(self):
        """Initialise les opérateurs de deepinv avec les bons paramètres"""
        img_size = (self.nband, self.height, self.width)
        
        # Opérateur de flou
        self.blur_op = Blur(
            filter=self._create_gaussian_kernel(),
            padding='circular',
            device=self.device
        )
        
        # Opérateur de sous-échantillonnage
        self.downsample_op = Downsampling(
            img_size=img_size,
            filter='gaussian',
            factor=self.scale,
            padding='circular',
            device=self.device
        )

    def _create_gaussian_kernel(self):
        """Crée un filtre gaussien pour le flou"""
        kernel_size = 5
        sigma = (self.sigma, self.sigma)
        return gaussian_blur(torch.ones(1, 1, kernel_size, kernel_size), 
                            kernel_size=[kernel_size, kernel_size], 
                            sigma=sigma)

    def __len__(self):
        return len(self.file[self.split])

    def __getitem__(self, idx):
        # Chargement des données
        img = torch.from_numpy(self.file[self.split][idx][:]).float()
        img = img.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        
        # Normalisation si nécessaire
        if self.normalize:
            img = (img - img.min()) / (img.max() - img.min())
            
        if self.transform:
            img = self.transform(img)
            
        return img

    # Opérateurs améliorés
    def blur(self, input_image):
        """Flou gaussien optimisé avec deepinv"""
        return self.blur_op(input_image)

    def sub_sample(self, input_image):
        """Sous-échantillonnage avec filtre anti-aliasing"""
        return self.downsample_op(input_image)

    def S_up(self, input_image):
        """Sur-échantillonnage avec interpolation bilinéaire"""
        return torch.nn.functional.interpolate(
            input_image, 
            scale_factor=self.scale, 
            mode='bilinear',
            align_corners=False
        )

    def simule_low_hsi(self, input_image):
        """Flou + sous-échantillonnage"""
        if input_image.ndim != 4:
            raise ValueError("L'image doit être un tenseur 4D [b,c,h,w]")
        return self.sub_sample(self.blur(input_image))

    def get_panchromatic(self, input_image):
        """Moyenne spectrale normalisée"""
        return input_image.mean(dim=1, keepdim=True)

    def simule_low_hsi_adjoint(self, input_image):
        """Opérateur adjoint exact"""
        if input_image.ndim != 4:
            raise ValueError("L'image doit être un tenseur 4D [b,c,h,w]")
        
        # Utilisation de l'adjoint de Downsampling
        upsampled = self.downsample_op.A_adjoint(input_image)
        return self.blur_op(upsampled)

    @staticmethod
    def spectral(input_image):
        """Opérateur spectral (moyenne spatiale)"""
        _, c, h, w = input_image.shape
        return torch.ones(1, c, device=input_image.device) / (h * w)

    @staticmethod
    def spectral_trans(input_image):
        """Transposé de l'opérateur spectral"""
        return PANDatasetImproved.spectral(input_image).t()