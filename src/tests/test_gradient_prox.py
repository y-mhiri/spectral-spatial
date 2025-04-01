import sys
sys.path.append('src/datasets')
sys.path.append('src/algorithms')


import torch

import matplotlib.pyplot as plt

from pansharpening import PANDataset
from gradient_prox_1 import ProximalGradient
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


    val_transform = transforms.Compose([transforms.ToTensor()]) # Transforms a the input data to torch tensors
    dataset = PANDataset(root_dir=data_path, split='train', transform= val_transform)


    idx = 17
    X = dataset[idx]
    X = X.unsqueeze(0)

    # Matrice de transformation
    Y_H = dataset.simule_low_hsi(X)  
    Y_M = dataset.get_panchromatic(X)
    Y_M = Y_M                                                             # Transposée de B 
    

    params = {
        'max_iter_cb' : 1000
        'theta_cb' : 1
        ...
    }
    
    pansharpening_model = ProximalGradient(dataset.simule_low_hsi,dataset.simule_low_hsi_adjoint ,max_iter=100, lmbda=0.1, lmbda_m=1, tau=0.1,tol=1e-7 ,scale = 8,verbose=True, **param_cb)

    # Exécution de l'optimisation
    U_result = pansharpening_model(Y_H, Y_M)

    print('OK.')