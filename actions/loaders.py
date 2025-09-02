import sys
import zarr
import numpy as np
import pandas as pd
import yaml
import os
from glob import glob

def setup_paths():
    """Setup system paths for imports"""
    path = os.path.join(os.getenv("HOME_DATA"), 'spectral-spatial/src')
    sys.path.append(os.path.join(path, 'datasets'))
    sys.path.append(os.path.join(path, 'algorithms'))
    sys.path.append(os.path.join(path, 'metrics'))
    sys.path.append(os.path.join(os.getenv("HOME_DATA"), 'spectral-spatial', 'experiments'))

# =============================================================================
# CORE DATA LOADING FUNCTIONS
# =============================================================================

def find_all_groups(storage_path):
    """Find all group_X directories in storage path"""
    group_pattern = os.path.join(storage_path, "group_*")
    group_dirs = glob(group_pattern)
    return sorted(group_dirs)

def load_group_info(group_path):
    """Load info.yaml from a group directory"""
    info_path = os.path.join(group_path, "info.yaml")
    
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"info.yaml not found in {group_path}")
    
    with open(info_path, 'r') as f:
        group_info = yaml.safe_load(f)
    
    return group_info['experiment']

def load_zarr_metadata(group_path):
    """Load only attributes from results.zarr - LIGHTWEIGHT"""
    zarr_path = os.path.join(group_path, "results.zarr")
    
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"results.zarr not found in {group_path}")
    
    root = zarr.open(zarr_path, mode='r')
    
    # Extract attributes
    metadata = dict(root.attrs)
    
    # Add array shape info without loading arrays
    if 'loss' in root:
        metadata['loss_shape'] = root['loss'].shape
    if 'reconstructed' in root:
        metadata['reconstructed_shape'] = root['reconstructed'].shape
    
    return metadata

def load_zarr_arrays(group_path, arrays=None):
    """Load specific arrays from results.zarr - HEAVY"""
    zarr_path = os.path.join(group_path, "results.zarr")
    
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"results.zarr not found in {group_path}")
    
    root = zarr.open(zarr_path, mode='r')
    arrays_data = {}
    
    # Default: load all available arrays
    if arrays is None:
        arrays = [key for key in root.keys()]
    
    for array_name in arrays:
        if array_name in root:
            if array_name == 'loss':
                arrays_data['loss_curves'] = np.array(root['loss'])
            else:
                arrays_data[array_name] = np.array(root[array_name])
    
    return arrays_data

def load_group_metadata(group_path):
    """Load lightweight metadata (info + zarr attributes)"""
    try:
        group_info = load_group_info(group_path)
        zarr_metadata = load_zarr_metadata(group_path)
        
        metadata = {
            'group_path': group_path,
            'group_info': group_info,
            'zarr_metadata': zarr_metadata,
            'parameters': group_info.get('parameters', {})
        }
        
        return metadata
        
    except Exception as e:
        print(f"Warning: Could not load {group_path}: {e}")
        return None

def load_experiment_metadata(storage_path):
    """Load metadata from all groups in experiment"""
    all_metadata = []
    group_dirs = find_all_groups(storage_path)
    
    for group_dir in group_dirs:
        metadata = load_group_metadata(group_dir)
        if metadata is not None:
            all_metadata.append(metadata)
    
    return all_metadata

# =============================================================================
# PARAMETER EXTRACTION
# =============================================================================

def extract_parameter_value(metadata, param_name, default=None):
    """Extract parameter value from metadata"""
    # Try parameters first
    if 'parameters' in metadata and param_name in metadata['parameters']:
        return metadata['parameters'][param_name]
    
    # Try zarr metadata
    if 'zarr_metadata' in metadata and param_name in metadata['zarr_metadata']:
        return metadata['zarr_metadata'][param_name]
    
    # Handle variations
    variations = {
        'lambda': ['lmbda', 'lambda'],
        'lmbda': ['lmbda', 'lambda'],
        'lambda_m': ['lmbda_m', 'lambda_m'],
        'lmbda_m': ['lmbda_m', 'lambda_m']
    }
    
    if param_name in variations:
        for var in variations[param_name]:
            if 'parameters' in metadata and var in metadata['parameters']:
                return metadata['parameters'][var]
            if 'zarr_metadata' in metadata and var in metadata['zarr_metadata']:
                return metadata['zarr_metadata'][var]
    
    return default

def extract_metric_value(metadata, metric_name):
    """Extract metric value from zarr metadata"""
    if 'zarr_metadata' not in metadata:
        return None
    
    zarr_metadata = metadata['zarr_metadata']
    if metric_name in zarr_metadata:
        return zarr_metadata[metric_name]
    
    return None

# =============================================================================
# FILTERING FUNCTIONS
# =============================================================================

def filter_by_parameter(metadata_list, param_name, param_value):
    """Filter by parameter value"""
    filtered = []
    for metadata in metadata_list:
        if extract_parameter_value(metadata, param_name) == param_value:
            filtered.append(metadata)
    return filtered

def filter_by_algorithm(metadata_list, algorithm_name):
    """Filter by algorithm name"""
    return filter_by_parameter(metadata_list, 'algorithm', algorithm_name)

def filter_by_noise_level(metadata_list, noise_level):
    """Filter by noise level"""
    return filter_by_parameter(metadata_list, 'noise_level', noise_level)

def get_unique_parameter_values(metadata_list, param_name):
    """Get unique values for a parameter"""
    values = []
    for metadata in metadata_list:
        val = extract_parameter_value(metadata, param_name)
        if val is not None:
            values.append(val)
    return sorted(list(set(values)))