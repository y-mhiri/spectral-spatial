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
    def __init__(self, root_dir, split='train', transform=None, normalize=False, scale =8,sigma=2):
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
        """Applique un flou gaussien à l'image d'entrée (torch.tensor) [h, w,c]."""


        h, w, c = input_image.shape
        blurred = torch.zeros((h, w, c))
        for i in range(c):
            # Appliquer le flou
            blurred[:,:,i] = gaussian_filter(input=input_image[:, :, i], sigma=self.sigma, mode='mirror')
        return blurred
    
    def sub_sample(self, input_image):
        """Effectue un sous-échantillonnage de l'image (torch.tensor) [h,w,c]."""
        return input_image[:, ::self.scale, ::self.scale]
    

    def S_up(self, input_image):
        """Effectue un sur-échantillonnage de l'image (torch.tensor) [h //self.scale, w //self.scale,c]"""
        h, w,c = input_image.shape
        result_image = np.zeros((h * self.scale, w * self.scale,c))(h * self.scale, w * self.scale, c)

        result_image[:,::self.scale, ::self.scale] = input_image[:,:,:]

        return result_image
    

    def simule_low_hsi(self, input_image):
        """Applique l'opérateur de dégradation de l'image d'entrée (torch.tensor) [h,w,c] ."""
        # Vérifiez que l'image est en 3D
        if input_image.ndim != 3:
            raise ValueError("L'image hyperspectrale doit être un tableau 3D.")

        # Initialiser un tableau pour stocker les résultats
        h, w,c = input_image.shape
        degraded_image = np.zeros((h // self.scale, w // self.scale, c))

        # Appliquer l'opérateur de dégradation à chaque bande

        blurred = self.blur(input_image)

            # Sous-échantillonnage
        degraded_image = self.sub_sample(blurred, self.scale)

        return degraded_image
    
    def get_panchromatic(self, index):
        """
        index (entier) 
        """
        hsi_data = self.file[self.split][index][:]
        panchromatic = hsi_data.mean(axis=-1)
        return panchromatic
    

    def simule_low_hsi_adjoint(self, input_image):
        """
        Applique l'opérateur de dégradation SB adjoint  de l'image hyperspectrale (torch.tensor) [(h//scale,w//scale,c)].
        """
        # Vérifiez que l'image est en 3D
        if input_image != 3:
            raise ValueError("L'image hyperspectrale doit être un tableau 3D.")

        # Initialiser un tableau pour stocker les résultats
        h, w, c = input_image.shape
        result_image = np.zeros((h * self.scale, w * self.scale, c))  # Dimensions de l'image d'origine

            # Suréchantillonner la bande
        result_image[:, :, :] = self.S_up(input_image, self.scale)
            # Appliquer le flou gaussien
        result_image[:, :, :] = self.blur(input_image,self.sigma)  # Utilise la fonction blur déjà définie

        return result_image


