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

def compute_convergence_metrics(loss_array, tol=1e-8):
    """
    Compute convergence metrics from loss curve.
    
    Args:
        loss_array: Array of loss values
        tol: Convergence tolerance
        
    Returns:
        Dictionary of convergence metrics
    """
    import numpy as np

    if loss_array.ndim == 1:
        loss_array = loss_array[np.newaxis, :]  # (1, n_iter)

    n_samples, n_iter = loss_array.shape
    conv_iters = []
    converged_flags = []

    for s in range(n_samples):
        curve = loss_array[s]
        converged = False
        conv_iter = n_iter
        for i in range(n_iter - 5):
            window = curve[i:i+5]
            if np.max(window) - np.min(window) < tol:
                converged = True
                conv_iter = i + 5
                break
        converged_flags.append(converged)
        conv_iters.append(conv_iter)

    return {
        'converged_fraction': float(np.mean(converged_flags)),
        'iteration_median': int(np.median(conv_iters)),
        'iteration_max': int(np.max(conv_iters)),
        'final_loss_mean': float(loss_array[:, -1].mean()),
        'loss_reduction_mean': float((loss_array[:, 0] - loss_array[:, -1]).mean()),
    }


def store_results(root, args, reconstructed_ar, loss_ar, relval_ar, metrics, total_time, algorithm_name):
    """Store all results in zarr format"""
    # Store attributes
    root.attrs['algorithm'] = algorithm_name
    root.attrs['total_time'] = total_time
    for key, value in vars(args).items():
        # Convert non-scalar values to strings for Zarr attributes
        if hasattr(value, '__len__') and not isinstance(value, str):
            root.attrs[key] = str(value)
        else:
            root.attrs[key] = value
    
    # Store metrics - convert lists to mean values since Zarr attributes can't store lists
    for metric in metrics:
        metric_values = metrics[metric]
        if isinstance(metric_values, list):
            # Store mean value for scalar metrics
            if len(metric_values) > 0:
                mean_value = sum(metric_values) / len(metric_values)
                std_value = (sum((x - mean_value) ** 2 for x in metric_values) / len(metric_values)) ** 0.5
                root.attrs[f'{metric}_mean'] = float(mean_value)
                root.attrs[f'{metric}_std'] = float(std_value)  # Added: standard deviation
                root.attrs[f'{metric}_count'] = len(metric_values)
        else:
            # Store scalar value directly
            root.attrs[metric] = metric_values
    
    # Store convergence metrics - Added
    convergence_metrics = compute_convergence_metrics(loss_ar.cpu().numpy())
    for metric, value in convergence_metrics.items():
        root.attrs[f'convergence_{metric}'] = value
    
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