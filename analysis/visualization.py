#!/usr/bin/env python3
"""
Enhanced visualization module for hyperspectral pansharpening experiments.

Provides standardized plotting functions with consistent styling and automatic
result organization for publication-quality figures.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import zarr
from typing import List, Dict, Optional, Union
from pathlib import Path

# Add current directory to Python path to ensure analysis module can be found
if __name__ == "__main__":
    # When running as script, add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our new modules
from analysis.plots import plot_single_image, plot_rgb_comparison, plot_convergence_curve, plot_error_map
from analysis.data_io import load_experiment_data, get_experiment_parameters, get_metrics_from_data, query_image_data
from analysis.report import generate_experiment_summary, create_visualization_index

# Set consistent styling
plt.style.use('seaborn-v0_8-darkgrid')
plt.rc('font', family='serif', size=12)
plt.rc('axes', titlesize=14, labelsize=12)
plt.rc('figure', titlesize=16)
plt.rc('legend', fontsize=10)


def setup_plot_directory(output_dir: str) -> Path:
    """Create and validate output directory for plots."""
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def plot_noisy_inputs_comparison(
    original: np.ndarray,
    low_res_hsi: np.ndarray,
    panchromatic: np.ndarray,
    output_dir: str,
    title: str = "Noisy Inputs Comparison",
    rgb_indices: Optional[List[int]] = None
) -> None:
    """
    Plot comparison of original vs noisy input images.
    
    Args:
        original: Original high-resolution hyperspectral image
        low_res_hsi: Simulated low-resolution hyperspectral image
        panchromatic: Simulated panchromatic image
        output_dir: Directory to save plots
        title: Plot title
        rgb_indices: Indices for RGB visualization
    """
    plot_dir = setup_plot_directory(output_dir)
    
    # Create figure with 4 columns: Original, Low-res HSI, Panchromatic, Error
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Normalize function for consistent scaling
    def normalize_image(img, rgb_idx):
        if rgb_idx is not None:
            img_rgb = img[rgb_idx]
        else:
            img_rgb = img[:3] if img.shape[0] >= 3 else img
        img_min = img_rgb.min()
        img_max = img_rgb.max()
        return (img_rgb - img_min) / (img_max - img_min + 1e-8)
    
    # Find global min/max for consistent scaling
    all_images = [original, low_res_hsi, panchromatic]
    all_values = []
    
    for img in all_images:
        if rgb_indices is not None:
            # Safe RGB band selection with bounds checking
            n_bands = img.shape[0]
            if n_bands == 1:
                img_rgb = img[0]  # Single band (panchromatic)
            else:
                # Use valid indices, clamp to available bands
                valid_indices = [min(idx, n_bands-1) for idx in rgb_indices[:min(3, n_bands)]]
                img_rgb = img[valid_indices]
        else:
            img_rgb = img[:3] if img.shape[0] >= 3 else img
        all_values.extend([img_rgb.min(), img_rgb.max()])
    
    global_min = min(all_values)
    global_max = max(all_values)
    
    # Plot original
    orig_rgb = original[rgb_indices] if rgb_indices is not None else original[:3]
    orig_normalized = (orig_rgb - global_min) / (global_max - global_min + 1e-8)
    
    if orig_normalized.ndim == 3:
        axes[0].imshow(np.transpose(orig_normalized, (1, 2, 0)))
    else:
        axes[0].imshow(orig_normalized, cmap='viridis')
    axes[0].set_title('Original HR HSI')
    axes[0].axis('off')
    
    # Plot low-res HSI
    lr_rgb = low_res_hsi[rgb_indices] if rgb_indices is not None else low_res_hsi[:3]
    lr_normalized = (lr_rgb - global_min) / (global_max - global_min + 1e-8)
    
    if lr_normalized.ndim == 3:
        axes[1].imshow(np.transpose(lr_normalized, (1, 2, 0)))
    else:
        axes[1].imshow(lr_normalized, cmap='viridis')
    axes[1].set_title('Simulated LR HSI (Noisy)')
    axes[1].axis('off')
    
    # Plot panchromatic (special handling for typically single-band image)
    if panchromatic.shape[0] == 1:
        pan_rgb = panchromatic[0]  # Single band for grayscale display
    else:
        pan_rgb = panchromatic[rgb_indices] if rgb_indices is not None else panchromatic[:3]
    pan_normalized = (pan_rgb - global_min) / (global_max - global_min + 1e-8)
    
    if pan_normalized.ndim == 3:
        axes[2].imshow(np.transpose(pan_normalized, (1, 2, 0)))
    else:
        axes[2].imshow(pan_normalized, cmap='viridis')
    axes[2].set_title('Simulated Panchromatic (Noisy)')
    axes[2].axis('off')
    
    # Plot error map (difference between original and low-res HSI)
    error = np.abs(orig_rgb - lr_rgb).mean(axis=0)
    error_img = axes[3].imshow(error, cmap='viridis')
    axes[3].set_title('Error: Original vs LR HSI')
    axes[3].axis('off')
    plt.colorbar(error_img, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    # Save plot
    filename = plot_dir / "noisy_inputs_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved noisy inputs comparison plot: {filename}")


def plot_convergence(
    loss_data: Union[np.ndarray, List[np.ndarray]],
    output_dir: str,
    algorithm_name: str = "Algorithm",
    max_iter: Optional[int] = None,
    labels: Optional[List[str]] = None,
    ylim: Optional[tuple] = None
) -> None:
    """
    Plot convergence curves with consistent styling.
    
    Args:
        loss_data: Array of loss values, shape (n_runs, n_iterations) or list of such arrays
        output_dir: Directory to save plots
        algorithm_name: Name of algorithm for title/filename
        max_iter: Maximum iterations to plot
        labels: Custom labels for each curve
        ylim: Y-axis limits (min, max)
    """
    plot_dir = setup_plot_directory(output_dir)
    
    plt.figure(figsize=(10, 6))
    
    # Convert to consistent format
    if isinstance(loss_data, list):
        loss_arrays = loss_data
    else:
        loss_arrays = list(loss_data)
    
    markers = ['o', 's', 'D', '^', 'v', 'p', '*']
    colors = plt.cm.tab10.colors
    
    for i, loss_array in enumerate(loss_arrays):
        if max_iter is not None:
            loss_to_plot = loss_array[:max_iter]
        else:
            loss_to_plot = loss_array
            
        label = labels[i] if labels else f'Run {i+1}'
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.semilogy(loss_to_plot, label=label, 
                    marker=marker, markersize=6, 
                    markevery=max(1, len(loss_to_plot)//10),
                    color=color, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Objective Function (log scale)', fontsize=12)
    plt.title(f'{algorithm_name} Convergence', fontsize=14, pad=20)
    
    if ylim:
        plt.ylim(ylim)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    
    # Save plot
    filename = plot_dir / f"convergence_{algorithm_name.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved convergence plot: {filename}")


def plot_metrics_comparison(
    metrics_dict: Dict[str, List[float]],
    output_dir: str,
    algorithm_names: Optional[List[str]] = None
) -> None:
    """
    Plot comparison of multiple metrics across algorithms.
    
    Args:
        metrics_dict: Dictionary of {metric_name: [values]}
        output_dir: Directory to save plots
        algorithm_names: Names of algorithms for legend
    """
    if not metrics_dict:
        print("No metrics to plot")
        return
        
    plot_dir = setup_plot_directory(output_dir)
    
    n_metrics = len(metrics_dict)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = plt.cm.tab10.colors
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        ax = axes[i]
        
        # Convert to numpy array
        values_array = np.array(values)
        
        # Plot as boxplot if multiple values, or bar if single value
        if len(values_array) > 1:
            ax.boxplot(values_array, patch_artist=True,
                      boxprops=dict(facecolor=colors[0], alpha=0.7),
                      whiskerprops=dict(color=colors[0]),
                      capprops=dict(color=colors[0]),
                      medianprops=dict(color='red'))
            ax.set_ylabel(metric_name)
        else:
            ax.bar([metric_name], values_array, color=colors[0], alpha=0.7)
            ax.set_ylim(0, values_array[0] * 1.2 if values_array[0] > 0 else values_array[0] * 0.8)
            ax.set_ylabel(metric_name)
        
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # Add mean value annotation
        mean_val = np.mean(values_array)
        ax.annotate(f'Mean: {mean_val:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    filename = plot_dir / "metrics_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved metrics comparison plot: {filename}")


def plot_reconstruction_example(
    original: np.ndarray,
    reconstructed: np.ndarray,
    output_dir: str,
    title: str = "Reconstruction Example",
    rgb_indices: Optional[List[int]] = None
) -> None:
    """
    Plot original vs reconstructed image comparison.
    
    Args:
        original: Original image array
        reconstructed: Reconstructed image array
        output_dir: Directory to save plots
        title: Plot title
        rgb_indices: Indices for RGB visualization
    """
    plot_dir = setup_plot_directory(output_dir)
    
    # Select RGB bands if not provided
    if rgb_indices is None:
        if original.shape[0] >= 3:
            rgb_indices = [0, 1, 2]  # First 3 bands
        else:
            rgb_indices = list(range(original.shape[0]))
    
    # Normalize for display
    def normalize_image(img):
        img = img[rgb_indices]  # Select RGB bands
        img_min = img.min()
        img_max = img.max()
        return (img - img_min) / (img_max - img_min + 1e-8)
    
    orig_rgb = normalize_image(original)
    recon_rgb = normalize_image(reconstructed)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(np.transpose(orig_rgb, (1, 2, 0)))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Reconstructed image
    axes[1].imshow(np.transpose(recon_rgb, (1, 2, 0)))
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    # Error map
    error = np.abs(orig_rgb - recon_rgb).mean(axis=0)
    error_img = axes[2].imshow(error, cmap='viridis')
    axes[2].set_title('Absolute Error')
    axes[2].axis('off')
    
    plt.colorbar(error_img, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    # Save plot
    filename = plot_dir / "reconstruction_example.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved reconstruction example plot: {filename}")





def visualize_experiment_results(
    zarr_path: str,
    output_dir: str,
    rgb_indices: Optional[List[int]] = None,
    dataset_path: Optional[str] = None
) -> None:
    """
    Complete visualization pipeline for experiment results.
    
    Args:
        zarr_path: Path to results.zarr file
        output_dir: Directory to save visualizations
        rgb_indices: Indices for RGB visualization
        dataset_path: Optional path to original dataset for noisy inputs visualization
    """
    # Load results using new data IO
    results = load_experiment_data(zarr_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot convergence
    if 'loss' in results:
        plot_convergence(
            results['loss'],
            output_dir,
            algorithm_name=results.get('algorithm', 'Algorithm')
        )
    
    # Plot metrics if available
    metrics = get_metrics_from_data(results)
    if metrics:
        plot_metrics_comparison(
            metrics,
            output_dir,
            algorithm_names=[results.get('algorithm', 'Algorithm')]
        )
    
    # Plot reconstruction example (first image)
    if 'reconstructed' in results and len(results['reconstructed']) > 0:
        image_data = query_image_data(results, image_index=0)
        
        # For demo, use reconstructed as "original" (in practice, load real original)
        original = image_data['reconstructed']
        reconstructed = image_data['reconstructed']
        
        plot_reconstruction_example(
            original,
            reconstructed,
            output_dir,
            title=f"{results.get('algorithm', 'Algorithm')} Reconstruction",
            rgb_indices=rgb_indices
        )
    
    # Plot noisy inputs if dataset path is provided
    if dataset_path is not None:
        try:
            # Load dataset to simulate noisy inputs
            from src.datasets.pandataset import PANDataset
            import torch
            
            # Create minimal dataset just for simulation
            dataset = PANDataset(
                root_dir=dataset_path,
                split='train',
                normalize=True,
                scale=results.get('scale', 4),
                sigma_blur=results.get('sigma_blur', 1.0),
                noise_level=results.get('noise_level', 0.01),
                device='cpu',
                seed=42
            )
            
            # Get first image and simulate noisy inputs
            X = dataset[0].unsqueeze(0)
            Y_H = dataset.simulate_low_res_hsi(X, noise=True)
            Y_M = dataset.simulate_panchromatic(X, noise=True)
            
            # Convert to numpy for visualization
            original_np = X.squeeze(0).cpu().numpy()
            low_res_np = Y_H.squeeze(0).cpu().numpy()
            pan_np = Y_M.squeeze(0).cpu().numpy()
            
            plot_noisy_inputs_comparison(
                original_np,
                low_res_np,
                pan_np,
                output_dir,
                title="Noisy Inputs Visualization",
                rgb_indices=rgb_indices
            )
            
        except Exception as e:
            print(f"Could not generate noisy inputs visualization: {e}")
    
    # Generate comprehensive reports
    generate_experiment_summary(
        results,
        output_dir,
        algorithm_name=results.get('algorithm', 'Algorithm')
    )
    
    # Create HTML index
    image_count = len(results.get('reconstructed', [0]))
    create_visualization_index(
        output_dir,
        algorithm_name=results.get('algorithm', 'Algorithm'),
        image_count=image_count
    )
    
    print(f"Visualization complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize hyperspectral pansharpening experiment results"
    )
    parser.add_argument("--zarr_path", type=str, required=True,
                       help="Path to results.zarr file")
    parser.add_argument("--output_dir", type=str, default="visualization_results",
                       help="Directory to save visualizations")
    parser.add_argument("--rgb_indices", type=int, nargs='+', default=None,
                       help="RGB band indices for visualization")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to original dataset for noisy inputs visualization")
    
    args = parser.parse_args()
    
    visualize_experiment_results(
        args.zarr_path,
        args.output_dir,
        args.rgb_indices,
        args.dataset_path
    )