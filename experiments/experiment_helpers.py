import sys
import os
import torch
import zarr
import time
import yaml
from datetime import datetime
from torchvision import transforms

def setup_paths():
    """Setup system paths for imports"""
    path = os.path.join(os.getenv("HOME_DATA"), 'spectral-spatial/src')
    sys.path.append(os.path.join(path, 'datasets'))
    sys.path.append(os.path.join(path, 'algorithms'))
    sys.path.append(os.path.join(path, 'metrics'))
    sys.path.append(os.path.join(os.getenv("HOME_DATA"), 'spectral-spatial', 'experiments'))

def setup_device_and_dtype(args):
    """Setup device and data type"""
    device = args.device
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(args.seed)
    return device, dtype

def create_dataset(args, device, dtype):
    """Create dataset with appropriate transforms and normalization"""
    crop_transform = transforms.Compose([transforms.CenterCrop(args.crop_size)]) if args.crop_center else None
    
    # Normalize sigma by crop size for consistency
    sigma_norm = args.sigma / (args.crop_size * args.crop_size) if args.crop_center else args.sigma
    noise_norm = args.noise_level / (args.crop_size * args.crop_size) if args.crop_center else args.noise_level
    
    from pansharpening import PANDataset
    return PANDataset(
        root_dir=args.dataset_path,
        split='train',
        transform=crop_transform,
        normalize=True,
        scale=args.scale,
        sigma=sigma_norm,
        sigma1=noise_norm,
        device=device,
        size=args.crop_size if args.crop_center else None,
        seed=args.seed
    )

def setup_chambolle_params(args):
    """Setup Chambolle-Pock parameters"""
    return {
        'max_iter': args.max_iter_cp,
        'lmbda': args.lmbda,
        'theta': args.theta_cp,
        'sigma': args.sigma_cp,
        'tau': 0.99 / args.sigma_cp
    }

def save_experiment_info(out_path, args, total_time, metrics, algorithm_name):
    """Save experiment information to YAML"""
    dataset_name = args.dataset_path.split('/')[-1].split('.')[0]
    
    info = {
        'experiment': {
            'algorithm': algorithm_name,
            'dataset': dataset_name,
            'date': datetime.now().isoformat(),
            'total_time': total_time,
            'parameters': vars(args),
            'final_metrics': {k: float(sum(v)/len(v)) for k, v in metrics.items()}
        }
    }
    
    with open(os.path.join(out_path, 'info.yaml'), 'w') as f:
        yaml.safe_dump(info, f)

def store_results(root, args, reconstructed_ar, loss_ar, metrics, total_time, algorithm_name):
    """Store all results in zarr format"""
    # Store attributes
    root.attrs['algorithm'] = algorithm_name
    root.attrs['total_time'] = total_time
    for key, value in vars(args).items():
        root.attrs[key] = value
    
    # Store metrics
    for metric in metrics:
        root.attrs[metric] = metrics[metric]
    
    # Store arrays
    root.create_dataset('reconstructed', data=reconstructed_ar.cpu().numpy())
    root.create_dataset('loss', data=loss_ar.cpu().numpy())

def run_optimization(optim, Y_H, Y_M):
    """Run optimization and return results with timing"""
    start_time = time.time()
    reconstructed, loss = optim(Y_H, Y_M)
    compute_time = time.time() - start_time
    return reconstructed, loss, compute_time

def print_experiment_info(args, algorithm_name):
    """Print experiment information"""
    print(f"Running {algorithm_name} experiment")
    print(f"Results will be saved to: {args.storage_path}")
    print("Parameters:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 50)