import argparse
import torch
import zarr
import os

# Setup paths and imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from experiment_helpers import *

setup_paths()
from pansharpening import PANDataset
from pan_ctv import PANTVCB
from pan_ctv_grad_align import PANTVGradAlignment
from nabla import nabla
from metrics import compute_metrics

def main():
    parser = argparse.ArgumentParser()
    
    # Core parameters
    parser.add_argument("--algorithm", type=str, required=True, choices=["PANTVCB", "PANTVGradAlign"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cuda_avail", type=bool, default=True)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    
    # Data parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_idx", type=str, default="6")
    parser.add_argument("--crop_center", type=bool, default=True)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--noise_level", type=float, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--pan", type=str, default='noisy')
    parser.add_argument("--scale", type=int, required=True)
    
    # Algorithm parameters
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--lmbda", type=float, required=True)
    parser.add_argument("--lmbda_m", type=float, required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alpha1", type=float, default=1.0)
    
    # Chambolle-Pock parameters
    parser.add_argument("--max_iter_cp", type=int, default=50)
    parser.add_argument("--sigma_cp", type=float, default=2.0)
    parser.add_argument("--theta_cp", type=float, default=1.0)
    
    # PANTVGradAlign specific
    parser.add_argument("--threshold_type", type=str, default="soft", choices=["hard", "soft"])
    parser.add_argument("--threshold_param", type=float, default=0.5)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    # Setup
    device, dtype = setup_device_and_dtype(args)
    dataset = create_dataset(args, device, dtype)

    image_idx = [int(idx) for idx in args.image_idx.split(' ')]

    subset = torch.utils.data.Subset(dataset, image_idx)
    A, A_adj, R, R_adj = dataset.get_operators()
    clean_pan = False if args.pan == 'clean' else True

    print_experiment_info(args, args.algorithm)
    
    # Create output directory and zarr file
    os.makedirs(args.storage_path, exist_ok=True)
    root = zarr.open(f'{args.storage_path}/results.zarr', mode='w')
    
    # Setup algorithm-specific parameters
    chambolle_params = setup_chambolle_params(args)
    if args.algorithm == 'PANTVGradAlign':
        first_pan = dataset.get_panchromatic(subset[0].unsqueeze(0))
        grad_panc = nabla(first_pan)
        chambolle_params.update({
            'grad_panc': grad_panc,
            'threshold_type': args.threshold_type,
            'alpha': args.alpha,
            'threshold_param': args.threshold_param
        })
    
    # Initialize result storage
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
        Y_H = dataset.simulate_low_res_hsi(data.unsqueeze(0)).to(device=device, dtype=dtype)
        Y_M = dataset.get_panchromatic(data.unsqueeze(0), noise=clean_pan).to(device=device, dtype=dtype)
        
        # Setup optimizer
        if args.algorithm == 'PANTVCB':
            optim = PANTVCB(A=A, Aadj=A_adj, spectral_op=R, spectral_op_t=R_adj,
                           max_iter=args.max_iter, lmbda=args.lmbda, alpha=args.alpha,
                           lmbda_m=args.lmbda_m, tol=args.tol, scale=dataset.scale,
                           p=args.p, q=args.q, r=args.r, verbose=True, params=chambolle_params)
        else:

            grad_panc = nabla(Y_M)
            chambolle_params.update({
                'grad_panc': grad_panc,
                'threshold_type': args.threshold_type,
                'alpha': args.alpha,
                'threshold_param': args.threshold_param
            })
            optim = PANTVGradAlignment(A=A, Aadj=A_adj, spectral_op=R, spectral_op_t=R_adj,
                                     max_iter=args.max_iter, lmbda=args.lmbda, alpha=args.alpha1,
                                     lmbda_m=args.lmbda_m, tol=args.tol, scale=dataset.scale,
                                     p=args.p, q=args.q, r=args.r, verbose=True, params=chambolle_params)
        
        # Run optimization
        reconstructed, loss, relval, compute_time = run_optimization(optim, Y_H, Y_M)
        total_time += compute_time
        
        # Store results
        reconstructed = torch.clamp(min=0, max=1)
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