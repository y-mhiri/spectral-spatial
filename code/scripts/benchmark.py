"""



"""

import sys
sys.path.append('../algorithms')
sys.path.append('../datasets')
sys.path.append('../metrics')


import torch
import zarr

import matplotlib.pyplot as plt

from datasets import HSIDataset
from tv_plus_grad_alignement import TVGradAlignement
from nabla import nabla

from metrics import compute_metrics

from torchvision import transforms
from math import sqrt
import time


###################################################
###################################################

######
# Setup parameters
######

# Define device (default is "cpu")
device = "cpu"
cuda = "cuda"


# Define dtype
dtype = torch.float64

# Define random seed
seed = 42
torch.manual_seed(seed)

# Define data path
data_path = '/home/y/Documents/Data/HSI/datasets/harvard.zarr'
out_path = '../results/harvard/tv_plus_grad_alignement.zarr'

dataset_name = data_path.split('/')[-1].split('.')[0]
algorithm = 'tv_plus_grad_alignement'

# Choose subset of data
n_data = 4
data_idx = [6,17,22,42,8,43]

# Noise level
SNR = 10

# Chambolle-Pock hyperparameters
max_iter = 1
sigma = 0.99
tau = 0.99
theta = 1.0

# lambda_tv and mu range

lmbda_range = [0.1]
mu_range = [10.0]

###################################################
###################################################



######
# Define Zarr output file
######

root = zarr.open(out_path, mode='w')

root.attrs['dataset'] = dataset_name
root.attrs['algorithm'] = algorithm
root.attrs['seed'] = seed
root.attrs['device'] = device
root.attrs['dtype'] = str(dtype)


root.attrs['lambda_range'] = lmbda_range
root.attrs['mu_range'] = mu_range
root.attrs['max_iter'] = max_iter
root.attrs['noise_level'] = SNR



######
# Load data
######

# Load Datasets 
val_transform = transforms.Compose([transforms.ToTensor()]) # Transforms a the input data to torch tensors
dataset = HSIDataset(root_dir=data_path, split='train', transform=val_transform)
subset = torch.utils.data.Subset(dataset, data_idx)

######
# Define the solver parameters
######

params = {}

params['compute_L'] = {'nband': dataset.nband}
params['K'] = {}
params['K_adjoint'] = {}
params['prox_sigma_g_conj'] = {'eps': 1e-12}
params['loss_fn'] = {'sigma2': 1}


######
# Run loop
######

residual_noise = torch.randn_like(dataset[0], device=device, dtype=dtype)

trial_idx = 0
for lmbda in root.attrs['lambda_range']:
    trial_idx += 1
    for mu in root.attrs['mu_range']:
        print(f"Running trial {trial_idx} : lambda = {lmbda}, mu = {mu}")
        print('-----------------------------------')  
        group = root.create_group(f'trial_{trial_idx:02d}')
        group.attrs['lambda'] = lmbda
        group.attrs['mu'] = mu


        metrics = {}
        reconstructed_ar = torch.zeros([len(subset), dataset.nband, dataset.height, dataset.width], device=device, dtype=dtype)
        
        loss_ar = torch.zeros([len(subset), max_iter], device=device, dtype=dtype)
        for j,data in enumerate(subset):
            print(f"Running data {j+1} on {len(subset)}")

            # import image to device (cpu or gpu), sizes of x is [1,number of bands, width, height]
            x = data.unsqueeze(0).to(device=device,dtype=dtype) 

            # Adds a small amount of white gaussian noise (sigma^2 = 1e-4) to avoid numerical issues
            x += 1e-2*residual_noise

            # Compute the panchromatic image from the ground truth HSI
            panc = torch.sum(x, dim=1).unsqueeze(1)/x.shape[1]
            grad_panc = nabla(panc)
            
            # Adds noise to the input HSI
            sigma2 = 10**(-SNR/10) * torch.norm(x, dim=[2,3])**2 / x.shape[2] / x.shape[3]
            sigma2 = sigma2.unsqueeze(0).unsqueeze(1).reshape(1, sigma2.numel(), 1, 1)
            sigma2 = sigma2.repeat(1, 1, x.shape[2], x.shape[3])

            y = x + torch.sqrt(sigma2)*torch.randn_like(x, device=device, dtype=dtype)

            # Define optimization object
            params['prox_tau_f'] = {'y': y.to(device=cuda,dtype=dtype), 'sigma2': 1}
            optim = TVGradAlignement(max_iter=max_iter, 
                                    mu=mu, 
                                    lmbda=lmbda, 
                                    theta=theta, 
                                    sigma=sigma, 
                                    tau=tau, 
                                    grad_panc=grad_panc.to(device=cuda,dtype=dtype))


            start_time = time.time()

            reconstructed, loss = optim(y.to(device=cuda,dtype=dtype), init=None, verbose=False, params=params)
            
            compute_time = time.time() - start_time
            group.attrs['time'] = compute_time
            
            reconstructed_ar[j] = reconstructed
            loss_ar[j] = loss




            sample_metrics = compute_metrics(gt=x.to(device=cuda,dtype=dtype), est=reconstructed, numpy=True)
            for metric in sample_metrics:
                if metric in metrics:
                    metrics[metric].append(sample_metrics[metric])
                else:
                    metrics[metric] = [sample_metrics[metric]]

            torch.cuda.empty_cache()



        for metric in metrics:
            group.attrs[metric] = metrics[metric]

        group.create_dataset(f'reconstructed', data=reconstructed_ar.cpu().numpy())
        group.create_dataset(f'loss', data=loss_ar.cpu().numpy())
        

        print(f"Finished lambda_tv_{lmbda}_mu_{mu}")

