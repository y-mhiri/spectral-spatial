import sys
sys.path.append('../datasets')
sys.path.append('../algorithms')


import torch

import matplotlib.pyplot as plt

from pansharpening import PANDataset
from gradient_prox import GradientProximal
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
    data_path = '/home/mhiriy/data/harvard.zarr'


    #val_transform = transforms.Compose([transforms.ToTensor()]) # Transforms a the input data to torch tensors
    dataset = PANDataset(root_dir=data_path, split='train', transform= None)


    idx = 17
    X = torch.tensor(dataset[idx])
    X = X.unsqueeze(0).to(device=device, dtype=dtype)

    # Matrice de transformation
    Y_H = dataset.process_hyperspectral_image(X,8)  
    Y_M = dataset.get_panchromatic(X)                                                             # Transposée de B
    H, B, R = dataset.matrices() 
    


    pansharpening_model = ProximalGradient(max_iter=100, lmbda=0.1, lmbda_m=1, tau=0.1, verbose=True)

    # Exécution de l'optimisation
    U_result = pansharpening_model.forward(Y_H, Y_M, B, B_t, R)