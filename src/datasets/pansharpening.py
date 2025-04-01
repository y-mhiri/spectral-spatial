"""
Define the HSIDataset class used to create hyperspectral dataset objects.
"""

import sys
path = '/home/ndiayem/Documents/spectral-spatial/src/'
sys.path.append(f'{path}/datasets')

import torch
import zarr
from torch.utils import data
from torch.linalg import svd
import numpy as np
from scipy.ndimage import gaussian_filter
from torchvision.transforms.functional import gaussian_blur
from HSIdatasets import HSIDataset
import torchvision.transforms as transforms


class PANDataset(HSIDataset):
    """
    Cette classe permet de recuperer les données et de simuler les images hyperspectral base résolution,
    l'image panchromatic et les opérateur flou et sous échantillonnage et suréchantillonnage 

    ...

    Attributes
    ----------
    split: data
    rgb_index : vecteur
    wavenumbers : entier
    spatial_resolution : entier
    spectral_resolution : entier
    nband : entier
    height : entier
    width : entier
    sigma : entier
    scale : entier

    Methods
    ---
    blurr(image,sigma)
        cette fonction applique un flou gaussien à image avec un niveau de bruit de bruit sigma
    sub_sample(image ,scale)
        cette fonction applique un sous échantillonnage à l'image avec un facteur scale 
    up_sample(image,scale)
        cette fonction applique un suréchantillonnage à l'image avec un facteur scale 
    simule_low_hsi(image,scale)
        cette fonction applique un flou gaussien à image suivi d'une sous échantillonage avec un facteur scale a fin d'avoir 
        l'image hyperspectral de base résolution 
    simule_low_hsi_adjoint(image,scale)
        cette fontion applique une suréchantillonnage suivi d'un flou gaussien a fin de récupérer l'image l'opérateur adjoint
    get_panchromatique(image)
        cette fonction récupére l'image panchromtique en faisant une moyenne de image selon les bandes 
    """ 
    def __init__(self, root_dir, split='train', transform=None, normalize=False, scale=8, sigma=2):
        super(PANDataset, self).__init__(root_dir, split=split, transform=transform, normalize=normalize)

        self.split = split
        self.scale = scale
        self.sigma = sigma 
        self.rgb_index = self.file.attrs['rgb']
        self.wavenumbers = self.file.attrs['spectral_range']
        self.spatial_resolution = self.file.attrs['spatial_resolution (m)']
        self.spectral_resolution = self.file.attrs['spectral_resolution (nm)']
        self.nband = self.file[self.split][0][:].shape[2]
        self.height = self.file[self.split][0][:].shape[0]
        self.width = self.file[self.split][0][:].shape[1]

    def blur(self, input_image):
        """Applique un flou gaussien à l'image d'entrée.
    
        Parameters:
            input_image (torch.tensor) : l'image hyperspectrale originale de taille [b,c,h,w]

        Returns:
            blurred (torch.tensor) : l'image floutée de taille [b,c,h,w]
        """
        # Appliquer le flou gaussien sur l'image entière
        gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=self.sigma)

        

        # Appliquer le flou
        blurred = gaussian_blur(input_image)

        

        return blurred
    
    def sub_sample(self, input_image):
        """Effectue un sous-échantillonnage de l'image (torch.tensor)
        
        Parameters : 
            input_image (torch.tensor) : l'image hyperspectrale floutée de taille [b,c,h,w]

        Returns : 
            input_image (torch.tensor) : l'image hyperspectrale floutée et sous-échantillonnée de taille [b,c,h//self.scale, w//self.scale]
        """
        return input_image[:,:,::self.scale, ::self.scale]
    
    def S_up(self, input_image):
        """Effectue un sur-échantillonnage de l'image d'entrée 

        Parameters : 
            input_image (torch.tensor) : l'image sous échantillonnée de taille [h //self.scale, w //self.scale,c]

        Returns : 
            result_image (torch.tensor) : l'image sur-échantillonnée de taille [b,c,h*self.scale, w*self.scale] 
        """
        b,c,h,w = input_image.shape
        result_image = torch.zeros((b,c,h * self.scale, w * self.scale))

        result_image[:,:,::self.scale, ::self.scale] = input_image[:,:, :, :]

        return result_image
    
    def simule_low_hsi(self, input_image):
        """Applique l'opérateur de dégradation flou et sous-échantillonnage de l'image d'entrée [b,c,h,w] 

        Parameters : 
            input_image (torch.tensor) : l'image hyperspectrale originale de taille [b,c,h,w] 

        Returns :
            degraded_image (torch.tensor) : l'image sous-échantillonnée et floutée de taille [b,c,h // self.scale, w // self.scale].
        """
        # Vérifiez que l'image est en 4D
        if input_image.ndim != 4:
            raise ValueError("L'image hyperspectrale doit être un tableau 4D.")

        # Appliquer l'opérateur de dégradation à chaque bande
        blurred = self.blur(input_image)

        # Sous-échantillonnage
        degraded_image = self.sub_sample(blurred)

        return degraded_image
    
    def get_panchromatic(self,input_image):
        """Fais une moyenne des pixels le long des canaux

        Parameters :
            x (int) : l'index de l'image de taille [b,c,h,w] qu'on veut récupérer

        Returns :
            panchromatic (torch.tensor) : l'image obtenue en faisant la moyenne par canal des pixels de taille [b,1,h,w]
        """
        
        panchromatic = input_image.mean(axis=1)
        panchromatic = panchromatic.unsqueeze(1)
        return panchromatic
    
    def simule_low_hsi_adjoint(self, input_image):
        """Applique l'opérateur de dégradation adjoint à l'image hyperspectrale 

        Parameters :
            input_image (torch.tensor) : l'image floutée et sous-échantillonnée de taille [b,c,h//scale,w//scale].

        Returns : 
            result_image (torch.tensor) : l'image sur-échantillonnée et défloutée de taille [b,c,h,w]
        """
        # Vérifiez que l'image est en 4D
        if input_image.ndim != 4:
            raise ValueError("L'image hyperspectrale doit être un tableau 4D.")

        # Suréchantillonner l'image
        result_image = self.S_up(input_image)

        # Appliquer le flou gaussien
        result_image = self.blur(result_image)

        return result_image
    
    @staticmethod
    def spectral(input_image):
        _,c,h,w = input_image.shape
        scalaire = 1/(h*w)
        mat_R = torch.ones((1, c))
        mat_R = scalaire * mat_R 

        return  mat_R
    
    @staticmethod
    def spectral_trans(input_image):
        _,c,h,w = input_image.shape
        scalaire = 1/(h*w)
        mat_R = torch.ones((1, c))
        mat_R = scalaire * mat_R 

        return  mat_R.t