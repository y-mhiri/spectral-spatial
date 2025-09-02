import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def setup_plot_style():
    """Setup consistent plot styling"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

def save_plot(fig, output_path, dpi=150):
    """Save matplotlib figure"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {output_path}")

def create_heatmap(data, x_col, y_col, z_col, title="Parameter Heatmap", output_path=None):
    """Create parameter heatmap"""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pivot_data = data.pivot(index=y_col, columns=x_col, values=z_col)
    
    im = ax.imshow(pivot_data.values, cmap='viridis', aspect='auto',
                   extent=[pivot_data.columns.min(), pivot_data.columns.max(),
                          pivot_data.index.min(), pivot_data.index.max()])
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=z_col)
    
    if output_path:
        save_plot(fig, output_path)
    return fig

def create_degradation_curve(data, x_col, y_col, title="Performance vs Noise", output_path=None):
    """Create performance degradation curve"""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if f'{y_col}_std' in data.columns:
        ax.errorbar(data[x_col], data[y_col], yerr=data[f'{y_col}_std'],
                   marker='o', capsize=5, linewidth=2, markersize=8)
    else:
        ax.plot(data[x_col], data[y_col], 'o-', linewidth=2, markersize=8)
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if output_path:
        save_plot(fig, output_path)
    return fig

def create_box_plot(data, x_col, y_col, title="Group Comparison", output_path=None):
    """Create box plot for group comparison"""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(data, pd.DataFrame):
        sns.boxplot(data=data, x=x_col, y=y_col, ax=ax)
    else:
        ax.boxplot(data, labels=x_col if isinstance(x_col, list) else [x_col])
        ax.set_ylabel(y_col)
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if output_path:
        save_plot(fig, output_path)
    return fig

def create_convergence_plot(loss_data, title="Convergence Analysis", output_path=None):
    """Create convergence plot from loss curves"""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (label, curve) in enumerate(loss_data.items()):
        ax.semilogy(curve, label=label, alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_path:
        save_plot(fig, output_path)
    return fig

def create_scatter_plot(data, x_col, y_col, color_col=None, title="Scatter Plot", output_path=None):
    """Create scatter plot with optional color mapping"""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if color_col and color_col in data.columns:
        scatter = ax.scatter(data[x_col], data[y_col], c=data[color_col], 
                           s=60, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label=color_col)
    else:
        ax.scatter(data[x_col], data[y_col], s=60, alpha=0.7)
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if output_path:
        save_plot(fig, output_path)
    return fig