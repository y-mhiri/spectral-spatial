#!/usr/bin/env python3
"""
Analysis package for hyperspectral pansharpening experiments.

This package provides visualization, plotting, data I/O, and reporting
functionality for analyzing experiment results.
"""

# Make submodules available for import
from .visualization import visualize_experiment_results, plot_convergence, plot_metrics_comparison, plot_reconstruction_example
from .plots import plot_single_image, plot_rgb_comparison, plot_convergence_curve, plot_error_map
from .data_io import load_experiment_data, get_experiment_parameters, get_metrics_from_data, query_image_data, load_dataset_metadata
from .report import generate_experiment_summary, create_visualization_index

__all__ = [
    # Visualization functions
    'visualize_experiment_results', 'plot_convergence', 'plot_metrics_comparison', 'plot_reconstruction_example',
    # Plotting functions
    'plot_single_image', 'plot_rgb_comparison', 'plot_convergence_curve', 'plot_error_map',
    # Data I/O functions
    'load_experiment_data', 'get_experiment_parameters', 'get_metrics_from_data', 'query_image_data', 'load_dataset_metadata',
    # Report functions
    'generate_experiment_summary', 'create_visualization_index'
]

__version__ = "1.0.0"
