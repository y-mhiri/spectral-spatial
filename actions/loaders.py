import sys
import zarr
import numpy as np
import pandas as pd
import yaml
import os
from glob import glob
from pathlib import Path

def setup_paths():
    """Setup system paths for imports"""
    path = os.path.join(os.getenv("HOME_DATA"), 'spectral-spatial/src')
    sys.path.append(os.path.join(path, 'datasets'))
    sys.path.append(os.path.join(path, 'algorithms'))
    sys.path.append(os.path.join(path, 'metrics'))
    sys.path.append(os.path.join(os.getenv("HOME_DATA"), 'spectral-spatial', 'experiments'))


# =============================================================================
# CORE DATA LOADING FUNCTIONS - MEMORY EFFICIENT
# =============================================================================

def find_all_runs(experiment_dir):
    """Find all run_X directories in experiment directory"""
    run_pattern = os.path.join(experiment_dir, "run_*")
    run_dirs = glob(run_pattern)
    return sorted(run_dirs)

def find_all_groups(run_path):
    """Find all group_X directories in a run directory"""
    group_pattern = os.path.join(run_path, "group_*")
    group_dirs = glob(group_pattern)
    return sorted(group_dirs)

def load_group_info(group_path):
    """Load group_info.yaml from a group directory"""
    group_info_path = os.path.join(group_path, "info.yaml")
    
    if not os.path.exists(group_info_path):
        raise FileNotFoundError(f"info.yaml not found in {group_path}")
    
    with open(group_info_path, 'r') as f:
        group_info = yaml.safe_load(f)
    
    return group_info['experiment']

def load_zarr_metadata(group_path):
    """Load only metadata (attributes) from results.zarr - LIGHTWEIGHT"""
    zarr_path = os.path.join(group_path, "results.zarr")
    
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"results.zarr not found in {group_path}")
    
    root = zarr.open(zarr_path, mode='r')
    
    # Extract only attributes (parameters and scalar metrics)
    metadata = {}
    for key in root.attrs.keys():
        metadata[key] = root.attrs[key]
    
    # Add array shape information without loading arrays
    if 'loss' in root:
        metadata['loss_shape'] = root['loss'].shape
    if 'reconstructed' in root:
        metadata['reconstructed_shape'] = root['reconstructed'].shape
    
    return metadata

def load_zarr_arrays(group_path, arrays=None):
    """Load specific arrays from results.zarr - HEAVY, use sparingly"""
    zarr_path = os.path.join(group_path, "results.zarr")
    
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"results.zarr not found in {group_path}")
    
    root = zarr.open(zarr_path, mode='r')
    
    arrays_data = {}
    
    # If no specific arrays requested, load all available
    if arrays is None:
        arrays = []
        if 'loss' in root:
            arrays.append('loss')
        if 'reconstructed' in root:
            arrays.append('reconstructed')
    
    # Load requested arrays
    for array_name in arrays:
        if array_name in root:
            if array_name == 'loss':
                arrays_data['loss_curves'] = np.array(root['loss'])
            elif array_name == 'reconstructed':
                arrays_data['reconstructed'] = np.array(root['reconstructed'])
            else:
                arrays_data[array_name] = np.array(root[array_name])
    
    return arrays_data

def load_group_metadata(group_path):
    """Load lightweight metadata only (group info + zarr attributes)"""
    try:
        group_info = load_group_info(group_path)
        zarr_metadata = load_zarr_metadata(group_path)
        
        # Combine into single metadata dictionary
        metadata = {
            'group_path': group_path,
            'group_info': group_info,
            'zarr_metadata': zarr_metadata
        }
        
        # Extract parameters for easier access
        if 'parameters' in group_info:
            metadata['parameters'] = group_info['parameters']
        
        return metadata
        
    except Exception as e:
        print(f"Warning: Could not load metadata from {group_path}: {e}")
        return None

def load_group_complete(group_path, arrays=None):
    """Load complete group data (metadata + specific arrays)"""
    metadata = load_group_metadata(group_path)
    if metadata is None:
        return None
    
    # Load arrays if requested
    if arrays is not None:
        try:
            arrays_data = load_zarr_arrays(group_path, arrays)
            metadata['arrays'] = arrays_data
        except Exception as e:
            print(f"Warning: Could not load arrays from {group_path}: {e}")
            metadata['arrays'] = {}
    
    return metadata

def load_experiment_metadata(experiment_dir):
    """Load lightweight metadata from all groups in experiment - EFFICIENT"""
    run_dirs = find_all_runs(experiment_dir)
    
    all_metadata = []
    for run_dir in run_dirs:
        group_dirs = find_all_groups(run_dir)
        for group_dir in group_dirs:
            metadata = load_group_metadata(group_dir)
            if metadata is not None:
                all_metadata.append(metadata)
    
    return all_metadata

def load_specific_arrays(group_paths, arrays):
    """Load specific arrays from multiple groups - USE CAREFULLY"""
    results = []
    
    for group_path in group_paths:
        try:
            arrays_data = load_zarr_arrays(group_path, arrays)
            arrays_data['group_path'] = group_path
            results.append(arrays_data)
        except Exception as e:
            print(f"Warning: Could not load arrays from {group_path}: {e}")
    
    return results

# =============================================================================
# PARAMETER EXTRACTION FUNCTIONS
# =============================================================================

def extract_parameter_value(metadata, param_name, default=None):
    """Extract a specific parameter value from metadata"""
    # Try parameters dict first
    if 'parameters' in metadata and param_name in metadata['parameters']:
        return metadata['parameters'][param_name]
    
    # Try zarr metadata (stored as attributes)
    if 'zarr_metadata' in metadata and param_name in metadata['zarr_metadata']:
        return metadata['zarr_metadata'][param_name]
    
    # Handle parameter name variations (e.g., 'lmbda' vs 'lambda')
    param_variations = {
        'lambda': ['lmbda', 'lambda'],
        'lmbda': ['lmbda', 'lambda'],
        'lambda_m': ['lmbda_m', 'lambda_m'],
        'lmbda_m': ['lmbda_m', 'lambda_m']
    }
    
    if param_name in param_variations:
        for variation in param_variations[param_name]:
            if 'parameters' in metadata and variation in metadata['parameters']:
                return metadata['parameters'][variation]
            if 'zarr_metadata' in metadata and variation in metadata['zarr_metadata']:
                return metadata['zarr_metadata'][variation]
    
    return default

def extract_metric_value(metadata, metric_name):
    """Extract a specific metric value from zarr metadata"""
    if 'zarr_metadata' not in metadata:
        return None
    
    zarr_metadata = metadata['zarr_metadata']
    
    if metric_name in zarr_metadata:
        metric_val = zarr_metadata[metric_name]
        
        # Handle different metric formats
        if isinstance(metric_val, list):
            return metric_val
        elif isinstance(metric_val, (int, float)):
            return [metric_val]  # Make it a list for consistency
        else:
            return metric_val
    
    return None

def get_unique_parameter_values(metadata_list, param_name):
    """Get all unique values for a parameter across metadata"""
    values = []
    for metadata in metadata_list:
        val = extract_parameter_value(metadata, param_name)
        if val is not None:
            values.append(val)
    
    return sorted(list(set(values)))

def get_parameter_combinations(metadata_list, param_names):
    """Get all unique parameter combinations"""
    combinations = []
    
    for metadata in metadata_list:
        combo = {}
        valid = True
        
        for param_name in param_names:
            val = extract_parameter_value(metadata, param_name)
            if val is None:
                valid = False
                break
            combo[param_name] = val
        
        if valid:
            combinations.append(combo)
    
    # Remove duplicates
    unique_combinations = []
    for combo in combinations:
        if combo not in unique_combinations:
            unique_combinations.append(combo)
    
    return unique_combinations

# =============================================================================
# DATA FILTERING FUNCTIONS - WORK ON METADATA ONLY
# =============================================================================

def filter_by_parameter(metadata_list, param_name, param_value):
    """Filter metadata by a specific parameter value"""
    filtered = []
    
    for metadata in metadata_list:
        val = extract_parameter_value(metadata, param_name)
        if val == param_value:
            filtered.append(metadata)
    
    return filtered

def filter_by_parameters(metadata_list, param_dict):
    """Filter metadata by multiple parameter values"""
    filtered = metadata_list.copy()
    
    for param_name, param_value in param_dict.items():
        filtered = filter_by_parameter(filtered, param_name, param_value)
    
    return filtered

def filter_by_algorithm(metadata_list, algorithm_name):
    """Filter metadata by algorithm name"""
    return filter_by_parameter(metadata_list, 'algorithm', algorithm_name)

def filter_by_noise_level(metadata_list, noise_levels):
    """Filter metadata by noise level(s)"""
    if not isinstance(noise_levels, list):
        noise_levels = [noise_levels]
    
    filtered = []
    for metadata in metadata_list:
        noise_val = extract_parameter_value(metadata, 'noise_level')
        if noise_val in noise_levels:
            filtered.append(metadata)
    
    return filtered

def filter_by_image_idx(metadata_list, image_indices):
    """Filter metadata by image indices"""
    if not isinstance(image_indices, list):
        image_indices = [image_indices]
    
    filtered = []
    for metadata in metadata_list:
        img_idx = extract_parameter_value(metadata, 'image_idx')
        
        # image_idx might be a list itself
        if isinstance(img_idx, list):
            # Check if any of the desired indices are in this result
            if any(idx in img_idx for idx in image_indices):
                filtered.append(metadata)
        else:
            if img_idx in image_indices:
                filtered.append(metadata)
    
    return filtered

def filter_successful_runs(metadata_list):
    """Filter to only include runs that completed successfully"""
    filtered = []
    
    for metadata in metadata_list:
        # Check if essential metrics exist
        psnr = extract_metric_value(metadata, 'PSNR')
        total_time = extract_parameter_value(metadata, 'total_time')
        
        if psnr is not None and total_time is not None:
            filtered.append(metadata)
    
    return filtered

# =============================================================================
# METRIC EXTRACTION AND AGGREGATION FUNCTIONS
# =============================================================================

def extract_metric_statistics(metadata_list, metric_name):
    """Extract statistics for a metric across multiple metadata entries"""
    all_values = []
    
    for metadata in metadata_list:
        metric_vals = extract_metric_value(metadata, metric_name)
        if metric_vals is not None:
            if isinstance(metric_vals, list):
                all_values.extend(metric_vals)
            else:
                all_values.append(metric_vals)
    
    if not all_values:
        return None
    
    return {
        'values': all_values,
        'mean': np.mean(all_values),
        'std': np.std(all_values),
        'min': np.min(all_values),
        'max': np.max(all_values),
        'median': np.median(all_values),
        'count': len(all_values)
    }

def create_parameter_grid(metadata_list, param1_name, param2_name, metric_name):
    """Create a parameter grid for heatmap visualization"""
    grid_data = []
    
    for metadata in metadata_list:
        param1_val = extract_parameter_value(metadata, param1_name)
        param2_val = extract_parameter_value(metadata, param2_name)
        metric_stats = extract_metric_statistics([metadata], metric_name)
        
        if param1_val is not None and param2_val is not None and metric_stats is not None:
            grid_data.append({
                param1_name: param1_val,
                param2_name: param2_val,
                f'{metric_name}_mean': metric_stats['mean'],
                f'{metric_name}_std': metric_stats['std'],
                'total_time': extract_parameter_value(metadata, 'total_time', 0)
            })
    
    return pd.DataFrame(grid_data)

def find_best_parameters(metadata_list, metric_name, maximize=True):
    """Find the metadata entry that optimizes a metric"""
    best_metadata = None
    best_metric = None
    
    for metadata in metadata_list:
        metric_stats = extract_metric_statistics([metadata], metric_name)
        if metric_stats is None:
            continue
        
        metric_val = metric_stats['mean']
        
        if best_metric is None:
            best_metadata = metadata
            best_metric = metric_val
        else:
            if (maximize and metric_val > best_metric) or (not maximize and metric_val < best_metric):
                best_metadata = metadata
                best_metric = metric_val
    
    return best_metadata, best_metric

def aggregate_by_parameters(metadata_list, group_params, metric_name):
    """Group metadata by parameter values and aggregate metrics"""
    # Group metadata by parameter combination
    groups = {}
    
    for metadata in metadata_list:
        # Create key from parameter values
        key_parts = []
        for param in group_params:
            val = extract_parameter_value(metadata, param)
            key_parts.append(f"{param}={val}")
        key = "_".join(key_parts)
        
        if key not in groups:
            groups[key] = []
        groups[key].append(metadata)
    
    # Aggregate metrics for each group
    aggregated = []
    for key, group_metadata in groups.items():
        # Extract parameter values for this group
        params = {}
        for param in group_params:
            params[param] = extract_parameter_value(group_metadata[0], param)
        
        # Calculate metric statistics
        metric_stats = extract_metric_statistics(group_metadata, metric_name)
        
        if metric_stats is not None:
            entry = params.copy()
            entry.update({
                f'{metric_name}_mean': metric_stats['mean'],
                f'{metric_name}_std': metric_stats['std'],
                f'{metric_name}_count': metric_stats['count']
            })
            aggregated.append(entry)
    
    return pd.DataFrame(aggregated)

# =============================================================================
# ARRAY LOADING UTILITIES - USE WHEN NEEDED
# =============================================================================

def get_loss_curves(metadata_list, max_groups=None):
    """Load loss curves from selected groups"""
    if max_groups is not None:
        metadata_list = metadata_list[:max_groups]
    
    group_paths = [m['group_path'] for m in metadata_list]
    return load_specific_arrays(group_paths, ['loss'])

def get_reconstructed_images(metadata_list, max_groups=None):
    """Load reconstructed images from selected groups"""
    if max_groups is not None:
        metadata_list = metadata_list[:max_groups]
    
    group_paths = [m['group_path'] for m in metadata_list]
    return load_specific_arrays(group_paths, ['reconstructed'])

def get_best_reconstruction(metadata_list, metric_name='PSNR'):
    """Get reconstructed image from best performing group"""
    best_metadata, _ = find_best_parameters(metadata_list, metric_name, maximize=True)
    
    if best_metadata is None:
        return None
    
    arrays = load_zarr_arrays(best_metadata['group_path'], ['reconstructed'])
    return arrays.get('reconstructed')

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_experiment_structure(experiment_dir):
    """Validate that experiment directory has expected structure"""
    if not os.path.exists(experiment_dir):
        return False, f"Experiment directory does not exist: {experiment_dir}"
    
    run_dirs = find_all_runs(experiment_dir)
    if not run_dirs:
        return False, f"No run directories found in {experiment_dir}"
    
    # Check first run for proper structure
    first_run = run_dirs[0]
    group_dirs = find_all_groups(first_run)
    
    if not group_dirs:
        return False, f"No group directories found in {first_run}"
    
    # Check first group for required files
    first_group = group_dirs[0]
    required_files = ['group_info.yaml', 'results.zarr']
    
    for req_file in required_files:
        if not os.path.exists(os.path.join(first_group, req_file)):
            return False, f"Required file {req_file} not found in {first_group}"
    
    return True, f"Valid experiment structure with {len(run_dirs)} runs"

def check_experiment_completion(experiment_dir):
    """Check which groups completed successfully"""
    total_groups = 0
    successful_groups = 0
    failed_groups = []
    
    run_dirs = find_all_runs(experiment_dir)
    for run_dir in run_dirs:
        group_dirs = find_all_groups(run_dir)
        total_groups += len(group_dirs)
        
        for group_dir in group_dirs:
            try:
                load_group_metadata(group_dir)
                successful_groups += 1
            except Exception as e:
                failed_groups.append((group_dir, str(e)))
    
    return {
        'total_groups': total_groups,
        'successful_groups': successful_groups,
        'failed_groups': failed_groups,
        'success_rate': successful_groups / total_groups if total_groups > 0 else 0
    }

def get_memory_usage_estimate(metadata_list):
    """Estimate memory usage if all arrays were loaded"""
    total_elements = 0

    for metadata in metadata_list:
        if 'zarr_metadata' in metadata:
            # Estimate from shape information
            if 'loss_shape' in metadata['zarr_metadata']:
                loss_shape = metadata['zarr_metadata']['loss_shape']
                total_elements += np.prod(loss_shape)
            
            if 'reconstructed_shape' in metadata['zarr_metadata']:
                recon_shape = metadata['zarr_metadata']['reconstructed_shape']
                total_elements += np.prod(recon_shape)
    
    # Assume float32 (4 bytes per element)
    estimated_bytes = total_elements * 4
    estimated_gb = estimated_bytes / (1024**3)
    estimated_mb = estimated_bytes / (1024**2)
    
    print(total_elements)
    return {
        'total_elements': total_elements,
        'estimated_bytes': estimated_bytes,
        'estimated_gb': estimated_gb,
        'estimated_mb': estimated_mb,
    }