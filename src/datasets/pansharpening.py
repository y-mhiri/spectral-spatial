import torch
import torch.nn as nn
from deepinv.physics import Blur, Downsampling
from torchvision.transforms.functional import gaussian_blur
import zarr
from torch.utils import data
import deepinv as dinv
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
        sigma (float): Paramètre du flou gaussien
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
                 scale=4, sigma=0.001,sigma1 = 0.001, device="cpu", size=None, seed=0):
        """
        Args:
            root_dir (str): Chemin vers le fichier Zarr
            split (str): Partition des données ('train'/'test'/'val')
            transform (callable): Transformations à appliquer
            normalize (bool): Normalisation [0,1] si True
            scale (int): Facteur de sous-échantillonnage
            sigma (float): Paramètre du flou gaussien
            device (str): Device pour les opérations
            size (int): Taille de redimensionnement optionnelle
        """
        super().__init__()
        # Initialisation des attributs
        self.file = zarr.open(root_dir, mode='r')
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.scale = scale
        self.sigma = sigma
        self.sigma1 = sigma1
        self.device = device
        self.seed = seed
        
        # Métadonnées
        self.rgb_index = self.file.attrs['rgb']
        self.wavenumbers = self.file.attrs['spectral_range']
        self.spatial_resolution = self.file.attrs['spatial_resolution (m)']
        self.spectral_resolution = self.file.attrs['spectral_resolution (nm)']

        # Dimensions des données
        self.nband = self.file[self.split][0][:].shape[2]
        self.height = size if size else self.file[self.split][0][:].shape[0]
        self.width = size if size else self.file[self.split][0][:].shape[1]
        
        self._init_operators()

    def _init_operators(self):
        """Initialise les opérateurs de flou et sous-échantillonnage"""
        img_size = (self.nband, self.height, self.width)
        self.blur_op = Blur(
            filter=self._kernel_gaussien(),
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
        self.R = self.create_spectral_matrix()

    def _kernel_gaussien(self):
        """
        Crée un filtre gaussien 2D
        
        Returns:
            torch.Tensor: Kernel gaussien de shape [1, 1, k, k]
        """
        return dinv.physics.blur.gaussian_blur(sigma=(self.sigma,self.sigma), angle=0.0)
    
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
        img = torch.from_numpy(self.file[self.split][idx][:]).float()
        img = img.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        
        if self.normalize:
            img = (img - img.min()) / (img.max() - img.min())
            
        if self.transform:
            img = self.transform(img)
                
        return img
    

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

    def get_panchromatic(self, input_image, noise=True):
        """
        Calcule l'image panchromatique par moyenne spectrale
        
        Args:
            input_image (torch.Tensor): Image hyperspectrale [b,c,h,w]
            
        Returns:
            torch.Tensor: Image panchromatique [b,1,h,w]
        """
        return self.noise(self.spectral_op(input_image)) if noise else self.spectral_op(input_image)
    
    def create_spectral_matrix(self):
        """
        Calcule la signature spectrale moyenne
        
        Args:
            input_image (torch.Tensor): Image [b,c,h,w]
            
        Returns:
            torch.Tensor: Vecteur spectral moyen [1,c]
        """
        c = self.nband
        return (1/c)*torch.ones(1,c, device=self.device)
    
    def spectral_op(self,input_image):
        """
        Calcule la signature spectrale moyenne
        
        Args:
            input_image (torch.Tensor): Image [b,c,h,w]
            
        Returns:
            torch.Tensor: Vecteur spectral moyen [1,c]
        """
        input_image = input_image.contiguous()
        U_flat = input_image.view(1,self.nband, -1)
        RU = torch.matmul(self.R, U_flat.squeeze(0)).unsqueeze(0)
        RU = RU.view(1, 1, self.height,self.width)
        return RU
    

    def spectral_op_t(self,imput_image):
        """
        Calcule la signature spectrale moyenne
        
        Args:
            input_image (torch.Tensor): Image [b,c,h,w]
            
        Returns:
            torch.Tensor: Vecteur spectral moyen [1,c]
        """
        imput_image = imput_image.view(1, 1, -1)
        RU_t = torch.matmul(self.R.t(), imput_image)
        RU_t = RU_t.view(1,self.nband, self.height,self.width)
        return RU_t

    def noise(self,imput_image):
        noise_model = GaussianNoise(self.sigma1)
        physics = Denoising(device=imput_image.device, noise_model=noise_model)
        observation = physics(imput_image)
        return observation
