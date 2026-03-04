#!/usr/bin/env python3
"""
Low-level plotting functions for hyperspectral pansharpening visualization.

Provides basic plotting primitives for consistent visualization across the codebase.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

# Consistent styling
plt.style.use('seaborn-v0_8-darkgrid')
plt.rc('font', family='serif', size=12)
plt.rc('axes', titlesize=14, labelsize=12)
plt.rc('figure', titlesize=16)

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
