#!/usr/bin/env python3
"""
Data I/O functions for hyperspectral pansharpening visualization.

Handles loading and querying experiment data for visualization purposes.
"""

import zarr
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_experiment_data(zarr_path: str) -> Dict:
    """
    Load complete experiment data from Zarr file.
    
    Args:
        zarr_path: Path to results.zarr file
        
    Returns:
        Dictionary containing all experiment data
    """
    root = zarr.open(zarr_path, mode='r')
    
    # Load arrays
    data = {
        'reconstructed': root['reconstructed'][:],
        'loss': root['loss'][:],
        'relval': root['relval'][:],
    }
    
    # Load attributes
    for key in root.attrs:
        data[key] = root.attrs[key]
    
    return data


def load_noisy_inputs(dataset_path: str, indices: List[int] = None) -> Dict:
    """
    Load noisy input images from dataset for visualization.
    
    Args:
        dataset_path: Path to dataset Zarr file
        indices: List of image indices to load (None for all)
        
    Returns:
        Dictionary with noisy inputs for each image
    """
    # This is a placeholder - in practice, we would need to re-run the simulation
    # or store the noisy inputs during the experiment
    
    # For now, return empty structure
    return {
        'low_res_hsi': [],
        'panchromatic': [],
        'original': []
    }


def get_experiment_parameters(data: Dict) -> Dict:
    """
    Extract experiment parameters from loaded data.
    
    Args:
        data: Loaded experiment data
        
    Returns:
        Dictionary of experiment parameters
    """
    params = {}
    
    # Extract common parameters
    param_keys = ['algorithm', 'scale', 'noise_level', 'sigma_blur', 
                  'lmbda', 'lmbda_m', 'p', 'q', 'r', 'max_iter']
    
    for key in param_keys:
        if key in data:
            params[key] = data[key]
    
    return params


def get_metrics_from_data(data: Dict) -> Dict:
    """
    Extract metrics from experiment data.
    
    Args:
        data: Loaded experiment data
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Common metric names
    metric_names = ['PSNR', 'SSIM', 'SAM', 'RNMSE', 'CC']
    
    for name in metric_names:
        if name in data:
            metrics[name] = data[name]
    
    return metrics


def query_image_data(
    data: Dict,
    image_index: int = 0
) -> Dict:
    """
    Query data for a specific image from experiment results.
    
    Args:
        data: Loaded experiment data
        image_index: Index of image to query
        
    Returns:
        Dictionary with data for the specified image
    """
    return {
        'reconstructed': data['reconstructed'][image_index],
        'loss': data['loss'][image_index],
        'relval': data['relval'][image_index],
        'image_index': image_index
    }


def load_dataset_metadata(dataset_path: str) -> Dict:
    """
    Load metadata from dataset Zarr file.
    
    Args:
        dataset_path: Path to dataset Zarr file
        
    Returns:
        Dictionary with dataset metadata
    """
    root = zarr.open(dataset_path, mode='r')
    
    metadata = {
        'rgb_indices': root.attrs.get('rgb', [0, 1, 2]),
        'wavenumbers': root.attrs.get('spectral_range', None),
        'spatial_resolution': root.attrs.get('spatial_resolution (m)', None),
        'spectral_resolution': root.attrs.get('spectral_resolution (nm)', None),
        'nband': root.attrs.get('nband', None),
        'height': root.attrs.get('height', None),
        'width': root.attrs.get('width', None)
    }
    
    return metadata


def save_visualization_data(
    data: Dict,
    output_dir: str,
    prefix: str = "vis"
) -> Path:
    """
    Save visualization data to numpy files for later use.
    
    Args:
        data: Data to save
        output_dir: Output directory
        prefix: File prefix
        
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy file
    output_path = output_dir / f"{prefix}_data.npz"
    np.savez(output_path, **data)
    
    return output_path