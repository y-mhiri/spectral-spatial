import torch
import torch.nn as nn
from deepinv.physics import Blur, Downsampling
from torchvision.transforms.functional import gaussian_blur
import zarr
from torch.utils import data
import deepinv as dinv
from math import sqrt
from deepinv.physics import GaussianNoise, Denoising

class PANDataset(data.Dataset):
    """
    Dataset amélioré pour le pansharpening hyperspectral
    
    Attributes:
        file (zarr.Group): Accès aux données Zarr
        split (str): Partition des données ('train'/'test'/'val')
        transform (callable): Transformations à appliquer aux données
        normalize (bool): Si True, normalise les données entre 0 et 1
        scale (int): Facteur de sous-échantillonnage
        sigma_blur (float): Paramètre du flou gaussien
        device (str): Device pour les opérations ('cpu' ou 'cuda')
        nband (int): Nombre de bandes spectrales
        height (int): Hauteur des images
        width (int): Largeur des images
        rgb_index (list): Indices des bandes RGB
        wavenumbers: Plage spectrale des données
        spatial_resolution: Résolution spatiale en mètres
        spectral_resolution: Résolution spectrale en nm
        blur_op (deepinv.physics.Blur): Opérateur de flou
        downsample_op (deepinv.physics.Downsampling): Opérateur de sous-échantillonnage
    """

    def __init__(self, root_dir, split='train', transform=None, normalize=False, 
                 scale=4, sigma_blur=0.001, noise_level = 0.001, device="cpu", seed=0):
        """
        Args:
            root_dir (str): Chemin vers le fichier Zarr
            split (str): Partition des données ('train'/'test'/'val')
            transform (callable): Transformations à appliquer
            normalize (bool): Normalisation [0,1] si True
            scale (int): Facteur de sous-échantillonnage
            sigma_blur (float): Paramètre du flou gaussien
            noise_level (float) : noise level in dB; converted internally as 10^(-noise_level/20)
            device (str): Device pour les opérations
        """
        super().__init__()
        # Initialisation des attributs
        self.file = zarr.open(root_dir, mode='r')
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.scale = scale
        self.sigma_blur = sigma_blur
        self.device = device
        self.seed = seed
        
        # Métadonnées
        self.rgb_index = self.file.attrs['rgb']
        self.wavenumbers = self.file.attrs['spectral_range']
        self.spatial_resolution = self.file.attrs['spatial_resolution (m)']
        self.spectral_resolution = self.file.attrs['spectral_resolution (nm)']

        # Dimensions des données
        self.nband = self.file[self.split][str(0)][:].shape[2]
        self.height = self.file[self.split][str(0)][:].shape[0]
        self.width = self.file[self.split][str(0)][:].shape[1]
        img_size = (self.nband, self.height, self.width)

        self.noise_level =  10**(-noise_level/20) 
        
        self.R = (1/self.nband)*torch.ones(1,self.nband, device=self.device) 
        self.blur_op = Blur(
            filter=dinv.physics.blur.gaussian_blur(sigma=(self.sigma_blur,self.sigma_blur), angle=0.0),
            padding='circular',
            device=self.device
        )
        self.downsample_op = Downsampling(
            img_size=img_size,
            filter='gaussian',
            factor=self.scale,
            padding='circular',
            device=self.device
        )
    
    def __len__(self):
        """
        Returns:
            int: Nombre d'échantillons dans le dataset
        """
        return len(self.file[self.split])

    def __getitem__(self, idx):
        """
        Charge et transforme un échantillon
        
        Args:
            idx (int): Index de l'échantillon
            
        Returns:
            torch.Tensor: Image hyperspectrale [C, H, W]
        """
        img = torch.from_numpy(self.file[self.split][str(idx)][:]).float()
        img = img.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        
            
        if self.transform:
            img = self.transform(img)

        if self.normalize:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        return img.to(self.device)
    

    def get_operators(self):

        A = lambda x : self.downsample_op(self.blur_op(x))
        A_adj = lambda x : self.blur_op(self.downsample_op.A_adjoint(x))

        R = lambda x : self.spectral_op(x)
        R_adj = lambda x : self.spectral_op_t(x)

        return A, A_adj, R, R_adj


    def simulate_low_res_hsi(self, input_image, noise=True):
        """
        Simule une acquisition basse résolution (flou + sous-échantillonnage)
        
        Args:
            input_image (torch.Tensor): Image HR [b,c,h,w]
            
        Returns:
            torch.Tensor: Image LR [b,c,h//scale,w//scale]
        """
        if input_image.ndim != 4:
            raise ValueError("L'image doit être un tenseur 4D [b,c,h,w]")
        return self.noise(self.downsample_op(self.blur_op(input_image))) if noise else self.downsample_op(self.blur_op(input_image))

    def simulate_panchromatic(self, input_image, noise=True):
        """
        Calcule l'image panchromatique par moyenne spectrale
        
        Args:
            input_image (torch.Tensor): Image hyperspectrale [b,c,h,w]
            
        Returns:
            torch.Tensor: Image panchromatique [b,1,h,w]
        """
        return self.noise(self.spectral_op(input_image)) if noise else self.spectral_op(input_image)
   
    def spectral_op(self,input_image):
        """
        Calcule la signature spectrale moyenne
        
        Args:
            input_image (torch.Tensor): Image [b,c,h,w]
            
        Returns:
            torch.Tensor: Vecteur spectral moyen [1,c]
        """
        X = input_image.reshape(1,self.nband, -1)
        RX = torch.matmul(self.R, X.squeeze(0)).unsqueeze(0)
        RX = RX.reshape(1, 1, self.height,self.width)
        return RX
    

    def spectral_op_t(self,input_image):
        """
        Calcule la signature spectrale moyenne
        
        Args:
            input_image (torch.Tensor): Image [b,c,h,w]
            
        Returns:
            torch.Tensor: Vecteur spectral moyen [1,c]
        """
        Y = input_image.reshape(1, 1, -1)
        RtX = torch.matmul(self.R.t(), Y)
        RtX = RtX.reshape(1,self.nband, self.height,self.width)
        return RtX

    def noise(self,input_image):
        noise_model = GaussianNoise(self.noise_level)
        physics = Denoising(noise_model=noise_model)
        observation = physics(input_image)
        return observation
