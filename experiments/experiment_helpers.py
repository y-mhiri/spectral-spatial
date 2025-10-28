import os
import yaml
from datetime import datetime



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

def store_results(root, args, reconstructed_ar, loss_ar, relval_ar, metrics, total_time, algorithm_name):
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
    root.create_dataset('reconstructed', data=reconstructed_ar.cpu().numpy(), shape=reconstructed_ar.shape)
    root.create_dataset('loss', data=loss_ar.cpu().numpy(),shape=loss_ar.shape)
    root.create_dataset('relval', data=relval_ar.cpu().numpy(),shape=relval_ar.shape)


def print_experiment_info(args, algorithm_name):
    """Print experiment information"""
    print(f"Running {algorithm_name} experiment")
    print(f"Results will be saved to: {args.storage_path}")
    print("Parameters:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 50)