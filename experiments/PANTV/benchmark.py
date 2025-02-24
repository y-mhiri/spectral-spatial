import sys
path = '/home/mhiriy/projects/spectral-spatial/src'
sys.path.append(f'{path}/algorithms')
sys.path.append(f'{path}/datasets')
sys.path.append(f'{path}/metrics')

import argparse

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




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cuda_avail", action='store_true')
    parser.add_argument("--dtype", type=str, default="float32")

    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--theta", type=float, default=1)

    parser.add_argument("--image_idx", nargs="*", type=int, default=[6, 17])
    parser.add_argument("--crop_center", action='store_true')
    parser.add_argument("--crop_size", type=int, default=256)

    parser.add_argument("--noise_level",type=float, required=True)
    parser.add_argument("--lmbda", type=float, required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--true_init", action='store_true')


    args = parser.parse_args()

    print(f"I am a run. Everything done here will be save to {args.storage_path}.")
    print("Here are all the args called : ")

    args_dict = vars(args)
    for key in args_dict:
        print(f'{key} : {args_dict[key]}')

    print('\n ------------------------------------------------- \n')
    ###################################################
    ###################################################

    ######
    # Setup parameters
    ######

    # Define device (default is "cpu")
    device = args.device
    cuda = "cuda" if args.cuda_avail else "cpu"


    # Define dtype
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # Define random seed
    seed = args.seed
    torch.manual_seed(seed)

    # Define data path
    data_path = args.dataset_path
    out_path = args.storage_path

    dataset_name = data_path.split('/')[-1].split('.')[0]
    algorithm = 'tv_plus_grad_alignement'

    # Choose subset of data
    data_idx = args.image_idx # [6, 17]#  6,42,8,43]
    crop = args.crop_center 
    crop_size = args.crop_size 

    # Noise level
    noise_level = args.noise_level

    # Chambolle-Pock hyperparameters

    ## global
    max_iter = args.max_iter
    tol = args.tol
    theta = args.theta
    ground_truth_init = True if args.true_init else False

    ## lambda_tv and mu
    lmbda = args.lmbda
    mu = args.mu

    ###################################################
    ###################################################

    ######
    # Define Zarr output file
    ######

    root = zarr.open(f'{out_path}/results.zarr', mode='w')


    ######
    # Load data
    ######

    # Load Datasets 


    val_transform = transforms.Compose([transforms.ToTensor()]) # Transforms a the input data to torch tensors
    crop_transform = transforms.Compose([ transforms.ToTensor(), transforms.CenterCrop(crop_size)])
    if crop:
        dataset = HSIDataset(root_dir=data_path, split='train', transform=crop_transform, normalize=True)
    else:
        dataset = HSIDataset(root_dir=data_path, split='train', transform=val_transform, normalize=True)

    subset = torch.utils.data.Subset(dataset, data_idx)
    
    root.attrs['crop'] = crop
    root.attrs['crop_size'] = crop_size
    root.attrs['data_idx'] = data_idx

    ######
    # Define the solver parameters
    ######

    params = {}

    params['compute_L'] = {'nband': dataset.nband}
    params['K'] = {}
    params['K_adjoint'] = {}
    params['prox_sigma_g_conj'] = {'eps': tol}
    params['loss_fn'] = {'sigma2': 1}


    ######
    # Run loop
    ######

    residual_noise = torch.randn_like(subset[0], device=device, dtype=dtype)
    noise_map = torch.randn_like(subset[0], device=device, dtype=dtype)

    root.create_dataset('residual_noise', data=residual_noise.cpu().numpy())
    root.create_dataset('noise_map', data=noise_map.cpu().numpy())

    gain = 0.5/lmbda
    sigma = 0.99*gain
    tau = 0.99/gain

    print(f"Running experiment : lambda = {lmbda}, mu = {mu}, noise_level = {noise_level}")
    print('-----------------------------------')  
    root.attrs['lambda'] = lmbda
    root.attrs['mu'] = mu
    root.attrs['noise_level'] = noise_level

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

        y = x + sqrt(noise_level)*noise_map

        # Compute the panchromatic image from the noisy HSI
        panc = torch.sum(x, dim=1).unsqueeze(1)/y.shape[1]
        grad_panc = nabla(panc)

        # Define optimization object
        params['prox_tau_f'] = {'y': y.to(device=cuda,dtype=dtype), 'sigma2': 1}
        optim = TVGradAlignement(max_iter=max_iter, 
                                thresh=mu, 
                                lmbda=lmbda, 
                                theta=theta, 
                                sigma=sigma, 
                                tau=tau, 
                                tol = tol,
                                grad_panc=grad_panc.to(device=cuda,dtype=dtype))


        start_time = time.time()

        init = x.to(device=cuda, dtype=dtype) if ground_truth_init else None
        reconstructed, loss, rel = optim(y.to(device=cuda,dtype=dtype), init=init, verbose=True, params=params)
        
        compute_time = time.time() - start_time
        root.attrs['time'] = compute_time
        
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

    for metric in metrics:
        root.attrs[metric] = metrics[metric]

    root.create_dataset(f'reconstructed', data=reconstructed_ar.cpu().numpy())
    root.create_dataset(f'loss', data=loss_ar.cpu().numpy())
    root.create_dataset(f'relvar', data=rel_ar.cpu().numpy())
    
    
    print(f"All done here.")

