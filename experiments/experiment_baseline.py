import argparse
import torch
import zarr
import os
import time

# Setup paths and imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from experiment_helpers import *
from src.datasets.pandataset import PANDataset
from src.algorithms.pan_ctv import PANCTV
from src.algorithms.pan_ctv_grad_align import PANCTVGradAlignment
from src.algorithms.nabla import nabla
from src.metrics.metrics import compute_metrics

def main():
    parser = argparse.ArgumentParser()
    
    # Core parameters
    parser.add_argument("--algorithm", type=str, required=True, choices=["CTV", "GradAlign"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    
    # Simulation parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_idx", type=str, default=None)
    parser.add_argument("--noise_level", type=float, required=True)
    parser.add_argument("--sigma_blur", type=float, required=True)
    parser.add_argument("--clean_pan", type='store_true')
    parser.add_argument("--scale", type=int, required=True)
    
    # Algorithm parameters
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--lmbda", type=float, required=True)
    parser.add_argument("--lmbda_m", type=float, required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--r", type=float, required=True)
    
    # Chambolle-Pock parameters
    parser.add_argument("--max_iter_cp", type=int, default=50)
    parser.add_argument("--sigma_cp", type=float, default=2.0)
    parser.add_argument("--theta_cp", type=float, default=1.0)
    
    # PANTVGradAlign specific
    parser.add_argument("--threshold_softness", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=None)
    
    args = parser.parse_args()
    print_experiment_info(args, args.algorithm)
    
    # Setup experiment
    device = args.device
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(args.seed)

    # Open dataset
    dataset = PANDataset(
        root_dir=args.dataset_path,
        split='train',
        normalize=True,
        scale=args.scale,
        sigma_blur=args.sigma_blur,
        noise_level=args.noise_level,
        device=device,
        seed=args.seed
    )

    # Select a subset of images 
    if image_idx is None:
        subset = dataset
    else:
        image_idx = [int(idx) for idx in args.image_idx.split(' ')]
        subset = torch.utils.data.Subset(dataset, image_idx)

    # Export forward and ajoint operators
    A, A_adj, R, R_adj = dataset.get_operators()
        
    # Create output directory and zarr file
    os.makedirs(args.storage_path, exist_ok=True)
    root = zarr.open(f'{args.storage_path}/results.zarr', mode='w')
    
    # Setup algorithm-specific parameters
    sigma2 = dataset.noise_level**2
    chambolle_params = {
                            'max_iter': args.max_iter_cp,
                            'lmbda': args.lmbda*sigma2,
                            'theta': args.theta_cp,
                            'sigma': args.sigma_cp,
                            'tau': 0.99 / args.sigma_cp
                        }

    
    # Initialization
    metrics = {}
    reconstructed_ar = torch.zeros([len(subset), dataset.nband, args.crop_size, args.crop_size], 
                                  device=device, dtype=dtype)
    loss_ar = torch.zeros([len(subset), args.max_iter], device=device, dtype=dtype)
    relval_ar = torch.zeros([len(subset), args.max_iter], device=device, dtype=dtype)
    total_time = 0
    
    # Run experiments
    for j, data in enumerate(subset):
        print(f"Processing image {j+1}/{len(subset)}")
        
        # Prepare data
        X = data.unsqueeze(0).to(device=device, dtype=dtype)
        Y_H = dataset.simulate_low_res_hsi(X).to(device=device, dtype=dtype)
        Y_M = dataset.simulate_panchromatic(X, noise=(not args.clean_pan)).to(device=device, dtype=dtype)
        
        # Setup optimizer
        if args.algorithm == 'CTV':
            optim = PANCTV(A=A, Aadj=A_adj, spectral_op=R, spectral_op_t=R_adj,
                           max_iter=args.max_iter, lmbda=args.lmbda,
                           lmbda_m=args.lmbda_m, tol=args.tol, scale=dataset.scale,
                           p=args.p, q=args.q, r=args.r, verbose=True, params=chambolle_params)
        elif args.algorithm == 'GradAlign':

            grad_panc = nabla(Y_M)
            chambolle_params.update({
                'grad_panc': grad_panc,
                'threshold_softness': args.threshold_softness,
                'threshold': args.threshold
            })
            optim = PANCTVGradAlignment(A=A, Aadj=A_adj, spectral_op=R, spectral_op_t=R_adj,
                                     max_iter=args.max_iter, lmbda=args.lmbda, 
                                     lmbda_m=args.lmbda_m, tol=args.tol, scale=dataset.scale,
                                     p=args.p, q=args.q, r=args.r, verbose=True, params=chambolle_params)
        else: 
            raise NotImplementedError(f'{args.algorithms} not implemented. Try either CTV or GradAlign.')        


        # Run optimization
        start_time = time.time()
        reconstructed, loss, relval = optim(Y_H, Y_M)
        compute_time = time.time() - start_time
        total_time += compute_time

        # Store results
        reconstructed_ar[j] = reconstructed
        loss_ar[j] = loss
        relval_ar[j] = relval
        
        # Compute metrics
        sample_metrics = compute_metrics(gt=X, est=reconstructed, numpy=True)
        for metric in sample_metrics:
            if metric in metrics:
                metrics[metric].append(sample_metrics[metric])
            else:
                metrics[metric] = [sample_metrics[metric]]
        
        torch.cuda.empty_cache()
    
    # Save results
    store_results(root, args, reconstructed_ar, loss_ar, relval_ar, metrics, total_time, args.algorithm)
    save_experiment_info(args.storage_path, args, total_time, metrics, args.algorithm)
    
    print(f"Experiment completed in {total_time:.2f}s")
    print(f"Average PSNR: {sum(metrics.get('PSNR', [0]))/len(metrics.get('PSNR', [1])):.2f}")

if __name__ == "__main__":
    main()