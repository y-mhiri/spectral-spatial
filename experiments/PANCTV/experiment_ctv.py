import sys
import os 

path = os.path.join(os.getenv("HOME"), 'spectral-spatial/src')
sys.path.append(os.path.join(path, 'datasets'))
sys.path.append(os.path.join(path, 'algorithms'))
sys.path.append(os.path.join(path, 'metrics'))
print(sys.path)
import argparse
import torch
import zarr
import time

import matplotlib.pyplot as plt
from pansharpening import PANDataset
from alg2 import PANTVCB
from torchvision import transforms
from math import sqrt
import yaml
from datetime import datetime

from metrics import compute_metrics

from torchvision import transforms




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cuda_avail", type=bool, default=True)
    parser.add_argument("--dtype", type=str, default="float32")

    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="/home/ndiayem/Documents/spectral-spatial/data/harvard.zarr")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--max_iter_cp", type=int, default=100)
    parser.add_argument("--sigma_cp", type=float, default=2.0)
    parser.add_argument("--theta_cp", type=float, default=1.0)
    

    parser.add_argument("--tol", type=float, default=1e-12)

    parser.add_argument("--image_idx", nargs="+", type=int, default=[6, 17])
    parser.add_argument("--crop_center", type=bool, default=True)
    parser.add_argument("--crop_size", type=int, default=256)

    parser.add_argument("--lmbda", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--lmbda_m", type=float, required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--noise_level",type=float, required=True)
    parser.add_argument("--sigma",type=float, required=True)
    parser.add_argument("--scale",type=int, required=True)




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
    algorithm = 'PANCTV'

    # Choose subset of data
    data_idx = args.image_idx # [6, 17]#  6,42,8,43]
    crop = args.crop_center 
    crop_size = args.crop_size 
    # Noise level
    noise_level = args.noise_level
    sigma = args.sigma
    scale = args.scale

      

    # CTV hyperparameters

    ## global
    max_iter = args.max_iter
    tol = args.tol

    ## lambda_tv ,lambda_m , p , q et r
    lmbda = args.lmbda
    alpha = args.alpha
    lmbda_m = args.lmbda_m
    p = args.p
    q = args.q 
    r = args.r


    # CP params

    sigma_cp = args.sigma_cp
    theta_cp  = args.theta_cp
    max_iter_cp = args.max_iter_cp
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


    
    crop_transform = transforms.Compose([transforms.CenterCrop(crop_size)])
    if crop:
        dataset = dataset =PANDataset(root_dir=data_path, split='train' ,transform=crop_transform,normalize=True,scale= scale,sigma= sigma/(crop_size*crop_size),sigma1 = noise_level/(crop_size*crop_size),device=device,size=crop_size,seed =seed)
    else:
        dataset = dataset =PANDataset(root_dir=data_path, split='train' ,transform=None,normalize=True,scale= scale,sigma= sigma,sigma1 = noise_level,device=device,size=None,seed =seed)

    subset = torch.utils.data.Subset(dataset, data_idx)

    # Operators parameters 
    A,A_adj, R, R_adj = dataset.get_operators()


    


    ######
    # Run loop
    ######

    Hsi_noisy = dataset.simulate_low_res_hsi(subset[0].unsqueeze(0))
    Pan_noisy = dataset.get_panchromatic(subset[0].unsqueeze(0))

    root.create_dataset('hsi_noise', data=Hsi_noisy.cpu().numpy())
    root.create_dataset('pan_noise ', data=Pan_noisy.cpu().numpy())


    print(f"Running experiment : lambda = {lmbda}, lambda_m = {lmbda_m},p = {p},q = {q} ,r = {r}, noise_level = {noise_level},sigma = {sigma},scale = {scale},data_idx = {data_idx},crop = {crop},crop_size = {crop_size},device = {device},seed = {seed}")
    print('-----------------------------------')  
    root.attrs['lambda'] = lmbda
    root.attrs['lambda_m'] = lmbda_m
    root.attrs['p'] = p
    root.attrs['q'] = q
    root.attrs['r'] = r
    root.attrs['noise_level'] = noise_level
    root.attrs['sigma'] = sigma
    root.attrs['scale'] = scale
    root.attrs['data_idx'] = data_idx  # Sauvegarde les indices des images traitées
    root.attrs['crop'] = crop
    root.attrs['crop_size'] = crop_size
    root.attrs['device'] = device
    root.attrs['seed'] = seed

    # parametre chambolle 
    params = {
    'max_iter': max_iter_cp,         # niters → max_iter (nom attendu par TVPrior)
    'lmbda': lmbda,          # paramètre supplémentaire
    'theta_cp': theta_cp ,            # paramètre de régularisation
    'sigma_cp': sigma_cp,                  # sigma = gain (gain=2)   
    'tau': 0.99/sigma_cp              # tau = 0.99 / gain (calculé)             
    }


    metrics = {}
    reconstructed_ar = torch.zeros([len(subset), dataset.nband, crop_size, crop_size], device=device, dtype=dtype)
    
    loss_ar = torch.zeros([len(subset), max_iter], device=device, dtype=dtype)
    for j,data in enumerate(subset):
        print(f"Running data {j+1} on {len(subset)}")

        # import image to device (cpu or gpu), sizes of x is [1,number of bands, width, height]
        X = data.unsqueeze(0).to(device=device,dtype=dtype)
        #  simulate_low_res_hsi image + noise 
        Y_H = dataset.simulate_low_res_hsi(data.unsqueeze(0)).to(device=device,dtype=dtype)
        # get_panchromatic image + noise 
        Y_M = dataset.get_panchromatic(data.unsqueeze(0)).to(device=device,dtype=dtype)

        optim = PANTVCB(
                    A=A,
                    Aadj=A_adj,
                    spectral_op = R,
                    spectral_op_t = R_adj,
                    max_iter=max_iter,
                    lmbda=lmbda,
                    alpha = alpha,
                    lmbda_m=lmbda_m,
                    tol=tol,
                    scale=dataset.scale,
                    p = p,
                    q = q,
                    r = r,
                    verbose=True,
                    params = params)


        start_time = time.time()

        
        reconstructed, loss = optim(Y_H,Y_M)
        
        compute_time = time.time() - start_time
        root.attrs['time'] = compute_time
        
        reconstructed_ar[j] = reconstructed
        loss_ar[j] = loss
        




        sample_metrics = compute_metrics(gt=X, est=reconstructed, numpy=True)
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

    info = {
        'datasets': {
            dataset_name: data_path  # Ex: 'harvard': '/chemin/vers/harvard.zarr'
        },
        'experiment': {
            'date': datetime.now().isoformat(),
            'parameters': {
                'lambda': lmbda,
                'lambda_m': lmbda_m,
                'max_iter' : max_iter,
                'p': p,
                'q': q,
                'r': r,
                'noise_level': noise_level,
                'sigma': sigma,
                'scale': scale
            }
        }
    }

    with open(os.path.join(out_path, 'info.yaml'), 'w') as f:
        yaml.safe_dump(info, f)
    
    
    print(f"All done here.")


