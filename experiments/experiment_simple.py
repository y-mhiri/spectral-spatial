#!/usr/bin/env python3
"""
Simplified experiment runner for hyperspectral pansharpening algorithms.

This provides a more user-friendly interface with sensible defaults and better organization.
"""

import argparse
import torch
import zarr
import os
import time
import sys
from pathlib import Path

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.datasets.pandataset import PANDataset
from src.algorithms.pan_ctv import PANCTV
from src.algorithms.pan_ctv_grad_align import PANCTVGradAlignment
from src.algorithms.nabla import nabla
from src.metrics.metrics import compute_metrics
from experiments.experiment_helpers import save_experiment_info, store_results, print_experiment_info


def get_default_params(algorithm_name):
    """Return sensible defaults for each algorithm."""
    defaults = {
        # Common parameters
        'device': 'cpu',
        'dtype': 'float32',
        'seed': 42,
        'max_iter': 50,
        'tol': 1e-8,
        'scale': 4,
        'noise_level': 0.01,  # -20dB
        'sigma_blur': 1.0,
        
        # Algorithm-specific parameters
        'CTV': {
            'lmbda': 0.1,      # Regularization weight
            'lmbda_m': 1.0,    # Panchromatic weight
            'p': 2.0,          # CTV norm parameters
            'q': 2.0,
            'r': 1.0,
            'max_iter_cp': 50,
            'sigma_cp': 2.0,
            'theta_cp': 1.0
        },
        'GradAlign': {
            'lmbda': 0.1,
            'lmbda_m': 1.0,
            'p': 2.0,
            'q': 2.0,
            'r': 1.0,
            'max_iter_cp': 50,
            'sigma_cp': 2.0,
            'theta_cp': 1.0,
            'threshold_softness': 1e-5,
            'threshold': None  # Auto-computed if None
        }
    }
    
    return {**defaults['Common'], **defaults.get(algorithm_name, {})}


def setup_experiment(args):
    """Setup experiment environment and parameters."""
    # Set up device and data type
    device = args.device
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.storage_path, exist_ok=True)
    
    return device, dtype


def run_experiment(args):
    """Run the main experiment."""
    device, dtype = setup_experiment(args)
    
    # Load dataset
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
    
    # Get operators
    A, A_adj, R, R_adj = dataset.get_operators()
    
    # Setup algorithm-specific parameters
    chambolle_params = {
        'max_iter': args.max_iter_cp,
        'lmbda': args.lmbda,
        'theta': args.theta_cp,
        'sigma': args.sigma_cp,
        'tau': 0.99 / args.sigma_cp
    }
    
    # Initialize results storage
    metrics = {}
    reconstructed_ar = torch.zeros([len(dataset), dataset.nband, dataset.width, dataset.height],
                                  device=device, dtype=dtype)
    loss_ar = torch.zeros([len(dataset), args.max_iter], device=device, dtype=dtype)
    relval_ar = torch.zeros([len(dataset), args.max_iter], device=device, dtype=dtype)
    
    # Run algorithm on each image
    for j in range(len(dataset)):
        print(f"Processing image {j+1}/{len(dataset)}")
        
        # Prepare data
        X = dataset[j].unsqueeze(0).to(device=device, dtype=dtype)
        Y_H = dataset.simulate_low_res_hsi(X).to(device=device, dtype=dtype)
        Y_M = dataset.simulate_panchromatic(X, noise=True).to(device=device, dtype=dtype)
        
        # Setup optimizer
        if args.algorithm == 'CTV':
            optim = PANCTV(A=A, Aadj=A_adj, spectral_op=R, spectral_op_t=R_adj,
                         max_iter=args.max_iter, lmbda=args.lmbda,
                         lmbda_m=args.lmbda_m, tol=args.tol, scale=dataset.scale,
                         p=args.p, q=args.q, r=args.r, verbose=True, init_params=chambolle_params)
        else:  # GradAlign
            grad_panc = nabla(Y_M)
            chambolle_params.update({
                'grad_panc': grad_panc,
                'threshold_softness': args.threshold_softness,
                'threshold': args.threshold
            })
            optim = PANCTVGradAlignment(A=A, Aadj=A_adj, spectral_op=R, spectral_op_t=R_adj,
                                     max_iter=args.max_iter, lmbda=args.lmbda,
                                     lmbda_m=args.lmbda_m, tol=args.tol, scale=dataset.scale,
                                     p=args.p, q=args.q, r=args.r, verbose=True, init_params=chambolle_params)
        
        # Run optimization
        start_time = time.time()
        reconstructed, loss, relval = optim(Y_H, Y_M)
        compute_time = time.time() - start_time
        
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
    root = zarr.open(f'{args.storage_path}/results.zarr', mode='w')
    store_results(root, args, reconstructed_ar, loss_ar, relval_ar, metrics, compute_time, args.algorithm)
    save_experiment_info(args.storage_path, args, compute_time, metrics, args.algorithm)
    
    print(f"Experiment completed successfully!")
    print(f"Results saved to: {args.storage_path}")
    if 'PSNR' in metrics:
        print(f"Average PSNR: {sum(metrics['PSNR'])/len(metrics['PSNR']):.2f} dB")


def main():
    parser = argparse.ArgumentParser(
        description="Simplified hyperspectral pansharpening experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core parameters
    parser.add_argument("--algorithm", type=str, required=True, 
                       choices=["CTV", "GradAlign"], 
                       help="Algorithm to use")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset Zarr file")
    parser.add_argument("--storage_path", type=str, required=True,
                       help="Directory to save results")
    
    # Device and precision
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--dtype", type=str, default="float32",
                       choices=["float32", "float64"], help="Data type")
    parser.add_argument("--seed",type=int, default=42, help="Seed for random number generation.")
    
    # Dataset parameters
    parser.add_argument("--scale", type=int, default=4,
                       help="Downsampling factor")
    parser.add_argument("--noise_level", type=float, default=0.01,
                       help="Noise level (variance)")
    parser.add_argument("--sigma_blur", type=float, default=1.0,
                       help="Blur standard deviation")
    
    # Algorithm parameters (with sensible defaults)
    parser.add_argument("--max_iter", type=int, default=50,
                       help="Maximum iterations")
    parser.add_argument("--tol", type=float, default=1e-8,
                       help="Convergence tolerance")
    parser.add_argument("--lmbda", type=float, default=0.1,
                       help="TV regularization weight")
    parser.add_argument("--lmbda_m", type=float, default=1.0,
                       help="Panchromatic data weight")
    parser.add_argument("--p", type=float, default=2.0,
                       help="CTV norm parameter p")
    parser.add_argument("--q", type=float, default=2.0,
                       help="CTV norm parameter q")
    parser.add_argument("--r", type=float, default=1.0,
                       help="CTV norm parameter r")
    
    # Chambolle-Pock parameters
    parser.add_argument("--max_iter_cp", type=int, default=50,
                       help="Chambolle-Pock max iterations")
    parser.add_argument("--sigma_cp", type=float, default=2.0,
                       help="Chambolle-Pock sigma parameter")
    parser.add_argument("--theta_cp", type=float, default=1.0,
                       help="Chambolle-Pock theta parameter")
    
    # GradAlign-specific parameters
    parser.add_argument("--threshold_softness", type=float, default=1e-5,
                       help="GradAlign threshold softness (GradAlign only)")
    parser.add_argument("--threshold", type=float, default=None,
                       help="GradAlign threshold (auto if None)")
    
    # Add sweep arguments
    parser.add_argument("--sweep", action="store_true",
                       help="Enable parameter sweep mode")
    parser.add_argument("--lmbda_sweep", type=float, nargs='+',
                       help="Lambda values to sweep")
    parser.add_argument("--max_iter_cp_sweep", type=int, nargs='+',
                       help="Chambolle-Pock max iterations to sweep")
    parser.add_argument("--noise_level_sweep", type=float, nargs='+',
                       help="Noise levels to sweep")
    parser.add_argument("--sigma_blur_sweep", type=float, nargs='+',
                       help="Blur sigma values to sweep")

    args = parser.parse_args()
    
    # Check if sweep mode is requested
    if hasattr(args, 'sweep') and args.sweep:
        # Parse sweep parameters
        sweep_params = {}
        for param in ['lmbda', 'max_iter_cp', 'noise_level', 'sigma_blur']:
            if hasattr(args, f'{param}_sweep'):
                values = getattr(args, f'{param}_sweep')
                if values:
                    sweep_params[param] = values
        
        if sweep_params:
            print(f"Running parameter sweep with: {sweep_params}")
            results = run_parameter_sweep(args, sweep_params, args.storage_path)
            
            # Save sweep summary
            import json
            summary_file = os.path.join(args.storage_path, "sweep_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Sweep summary saved to: {summary_file}")
        else:
            print("No sweep parameters specified. Running single experiment.")
            run_experiment(args)
    else:
        # Print experiment info
        print_experiment_info(args, args.algorithm)
        
        # Run experiment
        run_experiment(args)


if __name__ == "__main__":
    main()


def run_parameter_sweep(base_args, param_sweep, output_dir):
    """
    Run multiple experiments with parameter variations.
    
    Args:
        base_args: Base arguments (parsed args object)
        param_sweep: Dict of {param_name: [values]} to sweep
        output_dir: Base output directory for sweep results
    
    Returns:
        List of results summaries
    """
    import time
    import os
    from itertools import product
    
    # Convert args to dict for manipulation
    base_params = vars(base_args)
    
    # Generate all parameter combinations
    param_names = list(param_sweep.keys())
    param_values = list(param_sweep.values())
    combinations = list(product(*param_values))
    
    results = []
    
    print(f"Starting parameter sweep: {len(combinations)} combinations")
    print(f"Parameters: {param_names}")
    print(f"Values: {param_values}")
    print("-" * 50)
    
    for i, combo in enumerate(combinations):
        # Create parameter dict for this combination
        params = base_params.copy()
        for name, value in zip(param_names, combo):
            params[name] = value
        
        # Create unique output directory
        combo_str = "_".join([f"{name}{value}" for name, value in zip(param_names, combo)])
        run_output_dir = os.path.join(output_dir, f"run_{i:03d}_{combo_str}")
        os.makedirs(run_output_dir, exist_ok=True)
        
        # Update args object
        for key, value in params.items():
            setattr(base_args, key, value)
        base_args.storage_path = run_output_dir
        
        print(f"Run {i+1}/{len(combinations)}: {combo_str}")
        
        # Run experiment
        start_time = time.time()
        run_experiment(base_args)
        compute_time = time.time() - start_time
        
        # Load and store results summary
        try:
            from analysis.data_io import load_experiment_data
            from analysis.report import generate_text_summary
            
            zarr_path = os.path.join(run_output_dir, "results.zarr")
            if os.path.exists(zarr_path):
                data = load_experiment_data(zarr_path)
                summary = generate_text_summary(data, base_args.algorithm)
                results.append({
                    'parameters': params,
                    'summary': summary,
                    'time': compute_time,
                    'output_dir': run_output_dir
                })
        except Exception as e:
            print(f"Warning: Could not load results for {combo_str}: {e}")
            results.append({
                'parameters': params,
                'summary': f"Error: {e}",
                'time': compute_time,
                'output_dir': run_output_dir
            })
        
        print(f"Completed {i+1}/{len(combinations)}")
        print("-" * 30)
    
    print("Parameter sweep completed!")
    print(f"Total time: {sum(r['time'] for r in results):.1f} seconds")
    print(f"Results saved to: {output_dir}")
    
    return results