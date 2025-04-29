import sys
sys.path.append('src/datasets')
sys.path.append('src/algorithms')


import torch

import matplotlib.pyplot as plt

from pans import PANDatasetImproved
from gradient_prox import PANProximalGradient
from alg1 import PANTVGradProj
from torchvision import transforms
from math import sqrt


if __name__ == '__main__':

    # Define device (default is "cpu")
    device = "cpu" 

    # Define dtype
    dtype = torch.float64

    # Define random seed
    seed = 42
    torch.manual_seed(seed)

    # Define data path
    data_path = '/home/ndiayem/Documents/spectral-spatial/data/harvard.zarr'
    crop_size = 128

    val_transform  =  transforms.Compose([ transforms.CenterCrop(crop_size)])
    dataset =PANDatasetImproved(root_dir=data_path, split='train' ,transform=val_transform,normalize=True,scale=2,sigma=1.5,device=device,size =crop_size)
    idx = 17
    X = dataset[idx]
    X = X.unsqueeze(0)

    # Matrice de transformation
    Y_H = dataset.simule_low_hsi(X)  
    Y_M = dataset.get_panchromatic(X)                                                            # Transposée de B 
    

    solver_gp = PANTVGradProj(
    A=dataset.simule_low_hsi,
    Aadj=dataset.simule_low_hsi_adjoint,
    up = dataset.S_up,
    R=dataset.spectral,
    max_iter=900,
    lmbda=1e-2,
    lmbda_m=2,
    tau=0.08,
    tol=1e-7,
    scale=4,
    verbose=True,
    max_iter_gp=100
    )
    
    
    # Exécution de l'optimisation
    U_gp, costs_gp = solver_gp(Y_H, Y_M)

    print('OK.')