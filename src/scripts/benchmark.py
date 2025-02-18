"""



"""

import sys
path = '/home/mhiriy/spectral-spatial/code'
sys.path.append(f'{path}/algorithms')
sys.path.append(f'{path}/datasets')
sys.path.append(f'{path}/metrics')


import torch
import zarr

import matplotlib.pyplot as plt

from datasets import HSIDataset
from tv_plus_grad_alignement import TVGradAlignement
from nabla import nabla
from math import sqrt

from metrics import compute_metrics

from torchvision import transforms
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
dtype = torch.float32

# Define random seed
seed = 42
torch.manual_seed(seed)

# Define data path
data_path = '/home/mhiriy/data/harvard.zarr'
out_path = '/home/mhiriy/spectral-spatial/code/results/harvard/exp1.zarr'

dataset_name = data_path.split('/')[-1].split('.')[0]
algorithm = 'tv_plus_grad_alignement'

# Choose subset of data
data_idx = [6, 17]#  6,42,8,43]
crop = True 
crop_size = 256 

# Noise level
## SNR_range = [10, 15, 20]
noise_levels = [1e-1, 1e-2, 1e-3, 1e-4]

# Chambolle-Pock hyperparameters

## global
max_iter = 100000
tol = 1e-12
theta = 1.0
init = 'true'

## lambda_tv and mu range
lmbda_range = [1e-4,1e-3,1e-2,1e-1,1] 
mu_range = [0, 1.0]

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
root.attrs['data_idx'] = data_idx
root.attrs['crop'] = crop
root.attrs['crop_size'] = crop_size
root.attrs['init'] = init


root.attrs['lambda_range'] = lmbda_range
root.attrs['mu_range'] = mu_range
root.attrs['max_iter'] = max_iter
root.attrs['tol'] = tol
root.attrs['noise_levels'] = noise_levels



######
# Load data
######

# Load Datasets 
# val_transform = transforms.Compose([transforms.ToTensor()]) # Transforms a the input data to torch tensors
crop_transform = transforms.Compose([ transforms.ToTensor(), transforms.CenterCrop(crop_size)])
dataset = HSIDataset(root_dir=data_path, split='train', transform=crop_transform, normalize=True)
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

residual_noise = torch.randn_like(subset[0], device=device, dtype=dtype)
noise_map = torch.randn_like(subset[0], device=device, dtype=dtype)

root.create_dataset('residual_noise', data=residual_noise.cpu().numpy())
root.create_dataset('noise_map', data=noise_map.cpu().numpy())


noise_variances = []

trial_idx = 0

for sigma2 in root.attrs['noise_levels']:
    for lmbda in root.attrs['lambda_range']:

        lmbda_eff = lmbda
        gain = 0.5/lmbda_eff
        sigma = 0.99*gain
        tau = 0.99/gain

        for mu in root.attrs['mu_range']:
            trial_idx += 1
            print(f"Running trial {trial_idx} : lambda = {lmbda}, mu = {mu}")
            print('-----------------------------------')  
            group = root.create_group(f'trial_{trial_idx:02d}')
            group.attrs['lambda'] = lmbda
            group.attrs['mu'] = mu
            group.attrs['noise_level'] = sigma2

            metrics = {}
            reconstructed_ar = torch.zeros([len(subset), dataset.nband, crop_size, crop_size], device=device, dtype=dtype)
            
            loss_ar = torch.zeros([len(subset), max_iter], device=device, dtype=dtype)
            rel_ar = torch.zeros([len(subset), max_iter], device=device, dtype=dtype)
            for j,data in enumerate(subset):
                print(f"Running data {j+1} on {len(subset)}")

                # import image to device (cpu or gpu), sizes of x is [1,number of bands, width, height]
                x = data.unsqueeze(0).to(device=device,dtype=dtype) 

                # Adds a small amount of white gaussian noise to avoid numerical issues
                x += 1e-2*residual_noise/torch.norm(x)

                y = x + sqrt(sigma2)*noise_map

                # Compute the panchromatic image from the noisy HSI
                panc = torch.sum(x, dim=1).unsqueeze(1)/y.shape[1]
                grad_panc = nabla(panc)

                # Define optimization object
                params['prox_tau_f'] = {'y': y.to(device=cuda,dtype=dtype), 'sigma2': 1}
                optim = TVGradAlignement(max_iter=max_iter, 
                                        thresh=mu, 
                                        lmbda=lmbda_eff, 
                                        theta=theta, 
                                        sigma=sigma, 
                                        tau=tau, 
                                        tol = tol,
                                        grad_panc=grad_panc.to(device=cuda,dtype=dtype))


                start_time = time.time()

                reconstructed, loss, rel = optim(y.to(device=cuda,dtype=dtype), init=x.to(device=cuda, dtype=dtype), verbose=True, params=params)
                
                compute_time = time.time() - start_time
                group.attrs['time'] = compute_time
                
                reconstructed_ar[j] = reconstructed
                loss_ar[j] = loss
                rel_ar[j] = rel




                sample_metrics = compute_metrics(gt=x.to(device=cuda,dtype=dtype), est=reconstructed, numpy=True)
                for metric in sample_metrics:
                    if metric in metrics:
                        metrics[metric].append(sample_metrics[metric])
                    else:
                        metrics[metric] = [sample_metrics[metric]]

                torch.cuda.empty_cache()

            # if trial_idx == 1:
                # root.attrs['noise_variances'] = noise_variances

            for metric in metrics:
                group.attrs[metric] = metrics[metric]

            group.create_dataset(f'reconstructed', data=reconstructed_ar.cpu().numpy())
            group.create_dataset(f'loss', data=loss_ar.cpu().numpy())
            group.create_dataset(f'relvar', data=rel_ar.cpu().numpy())
            

            print(f"Finished lambda_tv_{lmbda}_mu_{mu}_sigma2_{sigma2}")

