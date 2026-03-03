#!/usr/bin/env python3
"""
Enhanced visualization module for hyperspectral pansharpening experiments.

Provides standardized plotting functions with consistent styling and automatic
result organization for publication-quality figures.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import zarr
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path

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
        loss_arrays = [loss_data]
    
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


def save_experiment_summary(
    results: Dict,
    output_dir: str,
    algorithm_name: str = "Algorithm"
) -> None:
    """
    Save experiment summary as JSON and markdown.
    
    Args:
        results: Dictionary containing experiment results
        output_dir: Directory to save summary
        algorithm_name: Name of algorithm
    """
    summary_dir = Path(output_dir)
    summary_dir.mkdir(exist_ok=True)
    
    # Save JSON summary
    json_file = summary_dir / "summary.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save markdown summary
    md_file = summary_dir / "summary.md"
    with open(md_file, 'w') as f:
        f.write(f"# {algorithm_name} Experiment Summary\n\n")
        f.write(f"**Date**: {results.get('date', 'N/A')}\n\n")
        
        f.write("## Parameters\n")
        for key, value in results.get('parameters', {}).items():
            f.write(f"- **{key}**: {value}\n")
        
        f.write("\n## Metrics\n")
        if 'metrics' in results:
            for metric, values in results['metrics'].items():
                if isinstance(values, list):
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    f.write(f"- **{metric}**: {mean_val:.4f} ± {std_val:.4f}\n")
                else:
                    f.write(f"- **{metric}**: {values:.4f}\n")
        
        f.write("\n## Performance\n")
        if 'performance' in results:
            for key, value in results['performance'].items():
                f.write(f"- **{key}**: {value:.2f}\n")
    
    print(f"Saved experiment summary: {json_file} and {md_file}")


def load_results_from_zarr(zarr_path: str) -> Dict:
    """
    Load results from Zarr file for visualization.
    
    Args:
        zarr_path: Path to results.zarr file
        
    Returns:
        Dictionary containing loaded results
    """
    root = zarr.open(zarr_path, mode='r')
    
    results = {
        'reconstructed': root['reconstructed'][:],
        'loss': root['loss'][:],
        'relval': root['relval'][:],
        'algorithm': root.attrs.get('algorithm', 'unknown'),
        'parameters': {k: v for k, v in root.attrs.items() if not k.startswith('_')}
    }
    
    # Add metrics from attributes
    metrics = {}
    for key in root.attrs:
        if key in ['PSNR', 'SSIM', 'SAM', 'RNMSE', 'CC']:
            metrics[key] = root.attrs[key]
    
    if metrics:
        results['metrics'] = metrics
    
    return results


def visualize_experiment_results(
    zarr_path: str,
    output_dir: str,
    rgb_indices: Optional[List[int]] = None
) -> None:
    """
    Complete visualization pipeline for experiment results.
    
    Args:
        zarr_path: Path to results.zarr file
        output_dir: Directory to save visualizations
        rgb_indices: Indices for RGB visualization
    """
    # Load results
    results = load_results_from_zarr(zarr_path)
    
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
    if 'metrics' in results:
        plot_metrics_comparison(
            results['metrics'],
            output_dir,
            algorithm_names=[results.get('algorithm', 'Algorithm')]
        )
    
    # Plot reconstruction example (first image)
    if 'reconstructed' in results and len(results['reconstructed']) > 0:
        # Create synthetic original for demo (in practice, load from dataset)
        original = results['reconstructed'][0]  # Placeholder
        reconstructed = results['reconstructed'][0]
        
        plot_reconstruction_example(
            original,
            reconstructed,
            output_dir,
            title=f"{results.get('algorithm', 'Algorithm')} Reconstruction",
            rgb_indices=rgb_indices
        )
    
    # Save summary
    save_experiment_summary(
        results,
        output_dir,
        algorithm_name=results.get('algorithm', 'Algorithm')
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
    
    args = parser.parse_args()
    
    visualize_experiment_results(
        args.zarr_path,
        args.output_dir,
        args.rgb_indices
    )