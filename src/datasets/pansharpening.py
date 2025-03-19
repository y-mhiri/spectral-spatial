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
    def __init__(self, root_dir, split='train', transform=None, normalize=False):
        super(PANDataset, self).__init__(root_dir, split='train', transform=None, normalize=False)

    def matrices():
        modofi
        return H,B,S,R

    def blur(self, input_image, sigma=2):
        """Applique un flou gaussien à l'image d'entrée."""
        output = gaussian_filter(input=input_image, sigma=sigma, mode='mirror')
        return output

    def sub_sample(self, img, scale):
        """Effectue un sous-échantillonnage de l'image."""
        return img[::scale, ::scale]

    def process_hyperspectral_image(self, hyperspectral_image, scale):
        """Applique l'opérateur de dégradation à chaque bande d'une image hyperspectrale."""
        # Vérifiez que l'image est en 3D
        if hyperspectral_image.ndim != 3:
            raise ValueError("L'image hyperspectrale doit être un tableau 3D.")

        # Initialiser un tableau pour stocker les résultats
        h, w, c = hyperspectral_image.shape
        degraded_image = np.zeros((h // scale, w // scale, c))

        # Appliquer l'opérateur de dégradation à chaque bande
        for i in range(c):
            # Appliquer le flou
            blurred_band = self.blur(hyperspectral_image[:, :, i])
            print(f"Forme après flou (bande {i}): {blurred_band.shape}")

            # Sous-échantillonnage
            degraded_band = self.sub_sample(blurred_band, scale)
            print(f"Forme après sous-échantillonnage (bande {i}): {degraded_band.shape}")

            # Vérifiez que les dimensions sont correctes
            if degraded_band.shape != (h // scale, w // scale):
                raise ValueError(f"Erreur de dimension pour la bande {i}: attendue {(h // scale, w // scale)}, obtenue {degraded_band.shape}")

            # Assignation à l'image dégradée
            degraded_image[:, :, i] = degraded_band

        return degraded_image

    def simulate_panchromatic(self, index):
        """
        Simule une image panchromatique à partir de l'image hyperspectrale.

        Paramètres :
        - index : Indice de l'image dans le dataset.

        Retour :
        - Y_M : Image panchromatique de forme (hauteur, largeur).
        """
        # Charger l'image hyperspectrale originale
        hsi_data = self.file[self.split][index][:]
        hsi_data = torch.tensor(hsi_data, dtype=torch.float32)

        # Calculer la moyenne sur les bandes spectrales
        Y_M = hsi_data.mean(dim=-1)

        return Y_M

    def S_up(self, hyperspectral_image, scale):
        """
        Effectue un suréchantillonnage d'une image hyperspectrale canal par canal.

        Args:
            hyperspectral_image (ndarray) : Image hyperspectrale d'entrée à suréchantillonner.
            scale (int) : Facteur d'échantillonnage.

        Retourne:
            ndarray : Image suréchantillonnée.
        """
        # Vérifiez que l'image est en 3D
        if hyperspectral_image.ndim != 3:
            raise ValueError("L'image hyperspectrale doit être un tableau 3D.")

        h, w, c = hyperspectral_image.shape
        # Initialiser un tableau pour stocker les résultats
        result_image = np.zeros((h * scale, w * scale, c))

        # Appliquer le suréchantillonnage à chaque bande
        for i in range(c):
            # Suréchantillonner la bande i
            result_image[:, :, i] = np.zeros((h * scale, w * scale))
            result_image[::scale, ::scale, i] = hyperspectral_image[:, :, i]

        return result_image

    def process_hyperspectral_image_adjoint(self, hyperspectral_image, scale):
        """
        Applique l'opérateur de dégradation SB adjoint à chaque bande d'une image hyperspectrale.

        Args:
            hyperspectral_image (ndarray) : Image hyperspectrale d'entrée.
            scale (int) : Facteur d'échantillonnage.

        Retourne:
            ndarray : Image transformée.
        """
        # Vérifiez que l'image est en 3D
        if hyperspectral_image.ndim != 3:
            raise ValueError("L'image hyperspectrale doit être un tableau 3D.")

        # Initialiser un tableau pour stocker les résultats
        h, w, c = hyperspectral_image.shape
        result_image = np.zeros((h, w, c))  # Dimensions de l'image d'origine

        # Appliquer l'opérateur adjoint à chaque bande
        for i in range(c):
            # Suréchantillonner la bande
            result_image[:, :, i] = self.S_up(hyperspectral_image[:, :, i], scale)
            # Appliquer le flou gaussien
            result_image[:, :, i] = self.blur(result_image[:, :, i])  # Utilise la fonction blur déjà définie

        return result_image