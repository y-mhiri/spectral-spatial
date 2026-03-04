#!/usr/bin/env python3
"""
Low-level plotting functions for hyperspectral pansharpening visualization.

Provides basic plotting primitives for consistent visualization across the codebase.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict

# Consistent styling
plt.style.use('seaborn-v0_8-darkgrid')
plt.rc('font', family='serif', size=12)
plt.rc('axes', titlesize=14, labelsize=12)
plt.rc('figure', titlesize=16)


def plot_convergence_comparison(
    loss_curves: List[np.ndarray],
    param_values: List,
    param_name: str,
    output_dir: str,
    title: str = "Convergence Comparison"
) -> None:
    """
    Compare convergence curves across parameter values.
    
    Args:
        loss_curves: List of loss arrays [n_curves, n_iterations]
        param_values: Parameter values for each curve
        param_name: Name of parameter being varied
        output_dir: Directory to save plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    for loss_curve, param_value in zip(loss_curves, param_values):
        plt.semilogy(loss_curve, label=f"{param_name}={param_value}", linewidth=2)
    
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Loss (log scale)', fontsize=14)
    plt.title(title, fontsize=16, pad=20)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = f"{output_dir}/convergence_{param_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_parameter_impact(
    param_values: List,
    metric_values: List,
    param_name: str,
    metric_name: str,
    output_dir: str,
    title: str = "Parameter Impact"
) -> None:
    """
    Plot metric vs parameter with error bars.
    
    Args:
        param_values: Parameter values (x-axis)
        metric_values: Metric values (y-axis)
        param_name: Name of parameter
        metric_name: Name of metric
        output_dir: Directory to save plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Simple line plot (error bars would need multiple runs)
    plt.plot(param_values, metric_values, 'o-', linewidth=2, markersize=8)
    
    plt.xlabel(param_name, fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.title(title, fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = f"{output_dir}/impact_{param_name}_{metric_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_algorithm_comparison(
    algorithm_results: Dict[str, List[float]],
    metric_name: str,
    output_dir: str,
    title: str = "Algorithm Comparison"
) -> None:
    """
    Boxplot comparison of algorithms.
    
    Args:
        algorithm_results: {algorithm_name: [metric_values]}
        metric_name: Name of metric being compared
        output_dir: Directory to save plot
        title: Plot title
    """
    data = list(algorithm_results.values())
    algorithms = list(algorithm_results.keys())
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=algorithms, patch_artist=True,
                boxprops=dict(facecolor='skyblue', alpha=0.7),
                whiskerprops=dict(color='navy'),
                capprops=dict(color='navy'),
                medianprops=dict(color='red'))
    
    plt.ylabel(metric_name, fontsize=14)
    plt.title(title, fontsize=16, pad=20)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = f"{output_dir}/comparison_{metric_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_single_image(
    image: np.ndarray,
    ax,
    title: str = "",
    cmap: str = 'viridis',
    show_colorbar: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> None:
    """
    Plot a single image on given axes.
    
    Args:
        image: 2D or 3D image array
        ax: Matplotlib axes to plot on
        title: Title for the plot
        cmap: Colormap to use
        show_colorbar: Whether to show colorbar
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
    """
    if image.ndim == 3 and image.shape[0] == 3:
        # RGB image
        img_plot = np.transpose(image, (1, 2, 0))
        im = ax.imshow(img_plot)
    else:
        # Single channel or grayscale
        if image.ndim == 3:
            image = image[0]  # Take first channel
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax.set_title(title)
    ax.axis('off')
    
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_rgb_comparison(
    images: List[np.ndarray],
    titles: List[str],
    rgb_indices: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot RGB comparison of multiple images.
    
    Args:
        images: List of image arrays [C, H, W]
        titles: List of titles for each image
        rgb_indices: Indices for RGB bands
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    # Normalize all images using the same scaling
    def normalize_image(img, rgb_idx):
        img_rgb = img[rgb_idx] if rgb_idx is not None else img[:3]
        img_min = img_rgb.min()
        img_max = img_rgb.max()
        return (img_rgb - img_min) / (img_max - img_min + 1e-8)
    
    # Find global min/max for consistent scaling
    all_values = []
    for img in images:
        img_rgb = img[rgb_indices] if rgb_indices is not None else img[:3]
        all_values.extend([img_rgb.min(), img_rgb.max()])
    
    global_min = min(all_values)
    global_max = max(all_values)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        # Select RGB bands
        if rgb_indices is not None:
            img_rgb = img[rgb_indices]
        else:
            img_rgb = img[:3] if img.shape[0] >= 3 else img
        
        # Normalize
        img_normalized = (img_rgb - global_min) / (global_max - global_min + 1e-8)
        
        # Plot
        if img_normalized.ndim == 3:
            axes[i].imshow(np.transpose(img_normalized, (1, 2, 0)))
        else:
            axes[i].imshow(img_normalized, cmap='viridis')
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_convergence_curve(
    loss_values: np.ndarray,
    ax,
    label: str = "Loss",
    color: str = None,
    marker: str = 'o',
    markersize: int = 6,
    linewidth: int = 2
) -> None:
    """
    Plot convergence curve on given axes.
    
    Args:
        loss_values: Array of loss values
        ax: Matplotlib axes
        label: Label for the curve
        color: Color for the curve
        marker: Marker style
        markersize: Marker size
        linewidth: Line width
    """
    markevery = max(1, len(loss_values) // 10)
    ax.semilogy(
        loss_values,
        label=label,
        marker=marker,
        markersize=markersize,
        markevery=markevery,
        color=color,
        linewidth=linewidth
    )

def plot_error_map(
    original: np.ndarray,
    reconstructed: np.ndarray,
    ax,
    title: str = "Absolute Error",
    rgb_indices: Optional[List[int]] = None
) -> None:
    """
    Plot error map between original and reconstructed images.
    
    Args:
        original: Original image array
        reconstructed: Reconstructed image array
        ax: Matplotlib axes
        title: Title for error map
        rgb_indices: Indices for RGB bands
    """
    # Select RGB bands if specified
    if rgb_indices is not None:
        orig_rgb = original[rgb_indices]
        recon_rgb = reconstructed[rgb_indices]
    else:
        orig_rgb = original[:3] if original.shape[0] >= 3 else original
        recon_rgb = reconstructed[:3] if reconstructed.shape[0] >= 3 else reconstructed
    
    # Compute error
    error = np.abs(orig_rgb - recon_rgb).mean(axis=0)
    
    # Plot error map
    im = ax.imshow(error, cmap='viridis')
    ax.set_title(title)
    ax.axis('off')
    
    return im

def plot_spectral_signature(
    image: np.ndarray,
    ax,
    title: str = "Spectral Signature",
    x_label: str = "Band Index",
    y_label: str = "Intensity"
) -> None:
    """
    Plot spectral signature (mean across spatial dimensions).
    
    Args:
        image: Hyperspectral image [C, H, W]
        ax: Matplotlib axes
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
    """
    # Compute mean spectrum
    spectrum = image.mean(axis=(1, 2))  # Mean over H, W
    
    # Plot
    ax.plot(spectrum, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
